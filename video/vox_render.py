"""Voxel-grid ray caster for the videos: CUDA 3D DDA (via CuPy) for the traversal, shading in torch.

    vr = VoxRender(lo, v, (nx, ny, nz), colors=None)      # colors: [nx*ny*nz, 3] float16 for class 1, or None
    rgb, alpha, z = vr.raycast(cls, eye, tgt, w, h, fov, layers=1, alpha={5: 0.6})

cls is a uint8 [nx,ny,nz] grid: 0 empty, 1 voxel coloured from `colors`, other values from PAL. eye/tgt are in the
level frame (z up), same units as lo and v. Faces are lit by a fixed sun, darkened by a 3x3x3 ambient occlusion
of the cell in front of the face, and get edge lines that fade out when a voxel is under ~4 px. The image is
rendered at ss x the size and box-filtered. Returns premultiplied rgb [h,w,3], alpha [h,w] and the z-depth of the
first hit [h,w] (inf where the ray hits nothing).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

PAL = {  # class: fill rgb, edge rgb
    2: ((0.85, 0.12, 0.08), (1.00, 0.60, 0.50)),     # floaters / removed
    4: ((0.06, 0.50, 0.25), (0.70, 1.00, 0.75)),     # walkable floor
    5: ((0.03, 0.38, 0.48), (0.50, 0.98, 1.00)),     # collision shell
    6: ((0.95, 0.62, 0.10), (1.00, 0.88, 0.45)),     # camera volume
    7: ((0.90, 0.16, 0.10), (1.00, 0.62, 0.50)),     # removed from the camera volume
}
LIGHT = np.array([0.45, 0.30, 0.84])
LIGHT /= np.linalg.norm(LIGHT)
_KERNEL = r'''
extern "C" __global__ void dda(const unsigned char* cls, int nx, int ny, int nz, float ox, float oy, float oz,
                               const float* dirs, int n, int L, int* oidx, signed char* oax, float* ot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int l = 0; l < L; l++) oidx[i * L + l] = -1;
    float d[3] = {dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2]};
    float o[3] = {ox, oy, oz};
    int dim[3] = {nx, ny, nz};
    float inv[3], tn = -1e30f, tf = 1e30f;
    int ax = 0;
    for (int k = 0; k < 3; k++) {
        float dk = fabsf(d[k]) < 1e-9f ? (d[k] < 0.f ? -1e-9f : 1e-9f) : d[k];
        d[k] = dk;
        inv[k] = 1.f / dk;
        float t0 = (0.f - o[k]) * inv[k], t1 = ((float)dim[k] - o[k]) * inv[k];
        float lo_ = fminf(t0, t1), hi_ = fmaxf(t0, t1);
        if (lo_ > tn) { tn = lo_; ax = k; }
        tf = fminf(tf, hi_);
    }
    float t = fmaxf(tn, 0.f);
    if (tf <= t) return;
    int cell[3], step[3];
    float tm[3], td[3];
    for (int k = 0; k < 3; k++) {
        int ci = (int)floorf(o[k] + d[k] * (t + 1e-4f));
        cell[k] = ci < 0 ? 0 : (ci >= dim[k] ? dim[k] - 1 : ci);
        step[k] = d[k] > 0.f ? 1 : -1;
        tm[k] = ((float)(cell[k] + (step[k] > 0)) - o[k]) * inv[k];
        td[k] = fabsf(inv[k]);
    }
    int hits = 0;
    while (true) {
        int idx = (cell[0] * ny + cell[1]) * nz + cell[2];
        if (cls[idx]) {
            oidx[i * L + hits] = idx;
            oax[i * L + hits] = (signed char)(ax * 2 + (step[ax] > 0));
            ot[i * L + hits] = t;
            if (++hits >= L) break;
        }
        int k = (tm[0] < tm[1]) ? (tm[0] < tm[2] ? 0 : 2) : (tm[1] < tm[2] ? 1 : 2);
        t = tm[k];
        cell[k] += step[k];
        tm[k] += td[k];
        ax = k;
        if (cell[k] < 0 || cell[k] >= dim[k] || t > tf) break;
    }
}'''


def basis(eye, tgt):
    f = np.asarray(tgt, np.float64) - eye
    f /= np.linalg.norm(f)
    r_ = np.cross(f, [0.0, 0.0, 1.0])
    r_ /= np.linalg.norm(r_)
    return r_, np.cross(f, r_), f


def pixel_dirs(eye, tgt, w, h, fov, dev="cuda"):
    """Level-frame ray directions with unit forward component [h,w,3] (point = eye + z_depth * dir), and fx."""
    fx = 0.5 * w / math.tan(math.radians(fov) / 2)
    r_, d_, f = (torch.tensor(q, dtype=torch.float32, device=dev) for q in basis(eye, tgt))
    yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    xn, yn = (xx + 0.5 - w / 2) / fx, (yy + 0.5 - h / 2) / fx
    return xn[..., None] * r_ + yn[..., None] * d_ + f, fx


class VoxRender:
    def __init__(self, lo, v, dims, colors=None, ss=2, dev="cuda"):
        import cupy as cp
        self.cp = cp
        self.lo, self.v = np.asarray(lo, np.float64), float(v)
        self.nx, self.ny, self.nz = (int(d) for d in dims)
        self.colors, self.ss, self.dev = colors, ss, dev
        self.kernel = cp.RawKernel(_KERNEL, "dda")
        self._ao = (None, None)

    def ao_grid(self, cls):
        if self._ao[0] is not cls:
            occ = (cls > 0).half()[None, None]
            cnt = F.conv3d(occ, torch.ones((1, 1, 3, 3, 3), dtype=torch.half, device=self.dev), padding=1)[0, 0]
            self._ao = (cls, (1.0 - 0.55 * ((cnt.float() - 9.0) / 9.0).clamp(0, 1)).half().view(-1))
        return self._ao[1]

    def dda(self, cls_flat, og, dirs, layers):
        cp = self.cp
        n = dirs.shape[0]
        oidx = torch.empty((n, layers), dtype=torch.int32, device=self.dev)
        oax = torch.empty((n, layers), dtype=torch.int8, device=self.dev)
        ot = torch.empty((n, layers), dtype=torch.float32, device=self.dev)
        dirs = dirs.contiguous()
        self.kernel(((n + 255) // 256,), (256,),
                    (cp.asarray(cls_flat), cp.int32(self.nx), cp.int32(self.ny), cp.int32(self.nz),
                     cp.float32(og[0]), cp.float32(og[1]), cp.float32(og[2]), cp.asarray(dirs),
                     cp.int32(n), cp.int32(layers), cp.asarray(oidx), cp.asarray(oax), cp.asarray(ot)))
        return oidx, oax, ot

    @torch.no_grad()
    def raycast(self, cls, eye, tgt, w, h, fov, layers=1, alpha=None, edges=1.0, edge_dark=0.30, ss=None):
        """alpha: per-class opacity (default 1 = opaque, first hit only); translucent classes let `layers` hits through."""
        dev, nx, ny, nz, v = self.dev, self.nx, self.ny, self.nz, self.v
        ss = self.ss if ss is None else ss
        W2, H2 = w * ss, h * ss
        D, fx = pixel_dirs(eye, tgt, W2, H2, fov, dev)
        dirs = D.view(-1, 3)
        nrm = dirs.norm(dim=1)
        dirs = dirs / nrm[:, None]
        og = (np.asarray(eye, np.float64) - self.lo) / v
        oidx, oax, ot = self.dda(cls.view(-1), og, dirs, layers)
        am = torch.ones(16, device=dev)
        for k_, al_ in (alpha or {}).items():
            am[k_] = al_
        fill_c, edge_c = torch.zeros((16, 3), device=dev), torch.zeros((16, 3), device=dev)
        for k_, (fc, ec) in PAL.items():
            fill_c[k_], edge_c[k_] = torch.tensor(fc), torch.tensor(ec)
        Lt = torch.tensor(LIGHT, dtype=torch.float32, device=dev)
        flat = cls.view(-1)
        aog = self.ao_grid(cls)
        ogt = torch.tensor(og, dtype=torch.float32, device=dev)
        nl = torch.tensor([nx, ny, nz], device=dev)
        n = dirs.shape[0]
        acc = torch.zeros((n, 3), device=dev)
        TA = torch.ones(n, device=dev)
        for l in range(layers):
            idx = oidx[:, l].long()
            hh = (idx >= 0).nonzero()[:, 0]
            if len(hh) == 0:
                break
            idx = idx[hh]
            code = oax[hh, l].long()
            ax, sgn = code // 2, (code % 2) * 2 - 1
            te = ot[hh, l]
            cell = torch.stack([idx // (ny * nz), (idx // nz) % ny, idx % nz], 1)
            p = ogt + dirs[hh] * te[:, None]
            fr_ = p - p.floor()
            oth = torch.stack([(ax + 1) % 3, (ax + 2) % 3], 1)
            uw = torch.gather(fr_, 1, oth)
            ed = torch.minimum(uw, 1 - uw).min(1).values
            pxv = (te / fx).clamp(min=1e-4)                                   # pixel size in voxels
            lw = (1.1 * pxv).clamp(0.035, 0.22)
            e = edges * (1 - (ed - 0.5 * lw) / lw).clamp(0, 1) * ((1 / pxv - 1.5) / 3).clamp(0, 1)
            shade = 0.64 + 0.44 * (-sgn.float() * Lt[ax]).clamp(min=0)          # face normal . light
            prev = cell.clone()
            prev.scatter_add_(1, ax[:, None], -sgn[:, None])
            inside = ((prev >= 0) & (prev < nl[None])).all(1)
            pidx = (prev[:, 0].clamp(0, nx - 1) * ny + prev[:, 1].clamp(0, ny - 1)) * nz + prev[:, 2].clamp(0, nz - 1)
            ao = torch.where(inside, aog[pidx].float(), torch.ones_like(shade))
            ch = flat[idx].long()
            colored = ch == 1
            base = fill_c[ch]
            if self.colors is not None:
                base = torch.where(colored[:, None], self.colors[idx].float(), base)
            lit = base * (shade * ao)[:, None]
            col = torch.where(colored[:, None], lit * (1 - edge_dark * e)[:, None],
                              lit * (1 - e[:, None]) + edge_c[ch] * e[:, None])
            a_c = am[ch]
            a_f = a_c if l == 0 else 0.4 * a_c
            al = torch.where(a_c >= 0.999, torch.ones_like(e), a_f * (1 - e) + torch.clamp(a_c * 1.6, max=0.95) * e)
            acc[hh] += (TA[hh] * al)[:, None] * col
            TA[hh] = TA[hh] * (1 - al)
        z = torch.where(oidx[:, 0] >= 0, ot[:, 0] * v / nrm, torch.full_like(nrm, float("inf")))
        rgb, alp, z = acc.view(H2, W2, 3), (1 - TA).view(H2, W2), z.view(H2, W2)
        if ss > 1:
            rgb = F.avg_pool2d(rgb.permute(2, 0, 1)[None], ss)[0].permute(1, 2, 0)
            alp = F.avg_pool2d(alp[None, None], ss)[0, 0]
            z = -F.max_pool2d(-z[None, None], ss)[0, 0]
        return rgb, alp, z
