"""Showcase renders of a voxel run: the voxelised scene (voxels coloured from the splat) beside the 3DGS.

    py.bat video/showcase.py out/final_vox bake        # voxel colours from splat renders -> <run>/voxel_rgb.npz (once)
    py.bat video/showcase.py out/final_vox orbit       # 360 deg orbit of the voxelised scene, low -> high -> low
    py.bat video/showcase.py out/final_vox wide        # high wide-angle orbit; one half of the world 3DGS, the other voxels
    py.bat video/showcase.py out/final_vox split       # screen split: 3DGS left of a moving divider, voxels right
    py.bat video/showcase.py out/final_vox compare     # 2x2: 3DGS | voxels | collision shell | camera volume
    ... --still 0,6,12 [--scale 0.5]                   # preview frames instead of the video
Writes <run>/showcase/<mode>.mp4 (H.264 via imageio-ffmpeg) or <run>/showcase/<mode>_t<sec>.jpg.

The voxels are the run's solid voxels (stages.npz solid_clean, 10 cm), ray-cast through the grid (3D DDA,
first hit, face shading + ambient occlusion + edge lines). Their colours are the splat as seen from ~250
views, back-projected onto the first solid voxel along each pixel's ray; voxels no view sees (inside
the ground slab, bushes cut by the crop) take the nearest seen colour, darkened with depth.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from gs_core import load_ply, look_at, intrinsics, render

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("mode", choices=["bake", "orbit", "wide", "split", "compare"])
ap.add_argument("--still", default=None, help="comma-separated times (s): JPEG previews instead of the video")
ap.add_argument("--scale", type=float, default=1.0, help="resolution scale of 1920x1080")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--crf", type=int, default=20)
ap.add_argument("--ss", type=int, default=2, help="voxel supersampling (anti-aliasing)")
a = ap.parse_args()
dev = "cuda"
SS = a.ss
W, H = int(round(1920 * a.scale / 2) * 2), int(round(1080 * a.scale / 2) * 2)

meta = json.load(open(f"{a.run}/meta.json"))
c, R = np.array(meta["c"]), np.array(meta["R"])
lo, v = np.array(meta["lo"]), float(meta["voxel_units"])
nx, ny, nz = meta["dims"]
N = nx * ny * nz
mpu = meta["mpu"]
hi = lo + np.array([nx, ny, nz]) * v
st = np.load(f"{a.run}/stages.npz")
un = lambda k: torch.from_numpy(np.unpackbits(st[k])[:N].reshape(nx, ny, nz).astype(bool)).to(dev)
solid = un("solid_clean")
seed = [int(x) for x in st["seed"]]
z_floor = lo[2] + seed[2] * v
CTR = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z_floor])      # middle of the crop, on the floor
STATUE = np.array([-2.0, -5.0, z_floor])
to_w = lambda p: c + np.asarray(p, dtype=np.float64) @ R
os.makedirs(f"{a.run}/showcase", exist_ok=True)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def basis(eye, tgt):
    f = np.asarray(tgt, np.float64) - eye
    f /= np.linalg.norm(f)
    r_ = np.cross(f, [0.0, 0.0, 1.0])
    r_ /= np.linalg.norm(r_)
    return r_, np.cross(f, r_), f


def pixel_dirs(eye, tgt, w, h, fov):
    """Level-frame ray directions with unit forward component [h,w,3] (so point = eye + z_depth * dir)."""
    fx = 0.5 * w / math.tan(math.radians(fov) / 2)
    r_, d_, f = (torch.tensor(q, dtype=torch.float32, device=dev) for q in basis(eye, tgt))
    yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    xn, yn = (xx + 0.5 - w / 2) / fx, (yy + 0.5 - h / 2) / fx
    return xn[..., None] * r_ + yn[..., None] * d_ + f, fx


def splat(g, eye, tgt, w, h, fov):
    """Premultiplied rgb [h,w,3], z-depth [h,w] (level units), alpha [h,w]."""
    M = look_at(to_w(eye), to_w(tgt), R[2])
    rgb, depth, alpha = render(g, M, intrinsics(w, h, fov), w, h, near=0.2, far=1e4)
    return rgb, depth, alpha


# ==== bake: voxel colours from the splat ===========================================================
if a.mode == "bake":
    g = load_ply(meta["ply"], dev)
    views = []
    for rad, hgt, n, off in [(100, 75, 40, 0.0), (70, 32, 48, 0.3), (40, 12, 40, 0.1), (18, 6, 24, 0.5)]:
        for k in range(n):
            an = 2 * math.pi * (k + off) / n
            views.append((CTR + [rad * math.cos(an), rad * math.sin(an), hgt], CTR + [0, 0, 2.0]))
    for x in np.linspace(-45, 45, 4):
        for y in np.linspace(-49, 41, 4):
            views.append((np.array([x, y + 1.0, z_floor + 60]), np.array([x, y, z_floor])))
    feet = np.load(f"{a.run}/reach.npz")["feet"].astype(np.float64)
    rs = np.random.default_rng(0)
    for i in rs.choice(len(feet), 120, replace=False):
        an = rs.uniform(0, 2 * math.pi)
        eye = feet[i] + [0, 0, 1.6 / mpu]
        views.append((eye, eye + [10 * math.cos(an), 10 * math.sin(an), -2.5]))
    Wb, Hb, fov_b = 960, 540, 70.0
    acc = torch.zeros((N, 3), device=dev)
    wsum = torch.zeros(N, device=dev)
    occ = solid.view(-1)
    lo_t = torch.tensor(lo, dtype=torch.float32, device=dev)
    dims_t = torch.tensor([nx, ny, nz], device=dev)
    offs = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], device=dev) * v
    t0 = time.time()
    for k, (eye, tgt) in enumerate(views):
        rgb, dep, al = splat(g, eye, tgt, Wb, Hb, fov_b)
        D, _ = pixel_dirs(eye, tgt, Wb, Hb, fov_b)
        ok = (al > 0.5) & (dep > 0.3)
        P = torch.tensor(eye, dtype=torch.float32, device=dev) + dep[..., None] * D
        P, Dn, col, w = P[ok], (D / D.norm(dim=-1, keepdim=True))[ok], (rgb / al[..., None].clamp(min=1e-3))[ok], al[ok]
        Q = P[None] + offs[:, None, None] * Dn[None]                          # [K,M,3] candidates along the ray
        ijk = ((Q - lo_t) / v).floor().long()
        inb = ((ijk >= 0) & (ijk < dims_t)).all(-1)
        idx = (ijk[..., 0].clamp(0, nx - 1) * ny + ijk[..., 1].clamp(0, ny - 1)) * nz + ijk[..., 2].clamp(0, nz - 1)
        hit = inb & occ[idx]
        anyh = hit.any(0)
        first = hit.float().argmax(0)
        sel = idx.gather(0, first[None])[0][anyh]
        acc.index_add_(0, sel, col[anyh].clamp(0, 1) * w[anyh, None])
        wsum.index_add_(0, sel, w[anyh])
        if k % 50 == 0:
            print(f"view {k}/{len(views)}: {int(anyh.sum())} of {int(ok.sum())} pixels landed on a voxel", flush=True)
    seen = (wsum > 0).view(nx, ny, nz) & solid
    rgb = (acc / wsum.clamp(min=1e-9)[:, None]).view(nx, ny, nz, 3).cpu().numpy()
    seen_np = seen.cpu().numpy()
    sol_np = solid.cpu().numpy()
    print(f"{len(views)} views in {time.time() - t0:.0f}s; solid voxels seen {seen_np.sum()} of {sol_np.sum()}", flush=True)
    d, ind = ndimage.distance_transform_edt(~seen_np, return_indices=True)
    near = rgb[ind[0], ind[1], ind[2]]
    keep = np.exp(-d / 3.0)[..., None]
    soil = np.array([0.20, 0.16, 0.12])
    fill = near * (0.55 + 0.45 * keep) * keep + soil * (1 - keep)
    out = np.where(seen_np[..., None], rgb, fill)[sol_np]
    np.savez_compressed(f"{a.run}/voxel_rgb.npz", rgb=(out.clip(0, 1) * 255).astype(np.uint8),
                        note="colours of the solid_clean voxels in np.flatnonzero order")
    print("wrote", f"{a.run}/voxel_rgb.npz", flush=True)
    sys.exit(0)

# ==== voxel ray casting =============================================================================
vrgb = np.load(f"{a.run}/voxel_rgb.npz")["rgb"]
COL = torch.zeros((N, 3), dtype=torch.float16, device=dev)
COL[torch.from_numpy(np.flatnonzero(solid.cpu().numpy())).to(dev)] = torch.from_numpy(vrgb).to(dev).half() / 255
PAL = {  # class: fill rgb, edge rgb (class 1 = coloured from the splat)
    5: ((0.03, 0.38, 0.48), (0.50, 0.98, 1.00)),     # collision shell
    4: ((0.06, 0.50, 0.25), (0.70, 1.00, 0.75)),     # walkable floor
    6: ((0.95, 0.62, 0.10), (1.00, 0.88, 0.45)),     # camera volume
}
LIGHT = np.array([0.45, 0.30, 0.84])
LIGHT /= np.linalg.norm(LIGHT)
_ao_cache = {}


def ao_grid(cls):
    key = id(cls)
    if key not in _ao_cache:
        occ = (cls > 0).half()[None, None]
        cnt = F.conv3d(occ, torch.ones((1, 1, 3, 3, 3), dtype=torch.half, device=dev), padding=1)[0, 0]
        _ao_cache.clear()
        _ao_cache[key] = (1.0 - 0.55 * ((cnt.float() - 9.0) / 9.0).clamp(0, 1)).half().view(-1)
    return _ao_cache[key]


_DDA = None


def dda(cls_flat, og, dirs, layers):
    """First `layers` occupied cells along each ray (CUDA, via CuPy). og: origin in grid units; dirs [n,3] unit.
    Returns cell index [n,L] int32 (-1 = none), entry face [n,L] int8 (axis*2 + step>0), distance [n,L] (voxels)."""
    global _DDA
    import cupy as cp
    if _DDA is None:
        _DDA = cp.RawKernel(r'''
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
}''', "dda")
    n = dirs.shape[0]
    oidx = torch.empty((n, layers), dtype=torch.int32, device=dev)
    oax = torch.empty((n, layers), dtype=torch.int8, device=dev)
    ot = torch.empty((n, layers), dtype=torch.float32, device=dev)
    dirs = dirs.contiguous()
    _DDA(((n + 255) // 256,), (256,), (cp.asarray(cls_flat), cp.int32(nx), cp.int32(ny), cp.int32(nz),
                                       cp.float32(og[0]), cp.float32(og[1]), cp.float32(og[2]), cp.asarray(dirs),
                                       cp.int32(n), cp.int32(layers), cp.asarray(oidx), cp.asarray(oax), cp.asarray(ot)))
    return oidx, oax, ot


@torch.no_grad()
def raycast(cls, eye, tgt, w, h, fov, layers=1, alpha=None, edge_dark=0.30, ss=None, edges=1.0):
    """cls uint8 [nx,ny,nz] (0 empty, 1 coloured voxel, else PAL). alpha: per-class opacity (default 1).
    Rendered at ss x the resolution and box-filtered (anti-aliasing).
    Returns premultiplied rgb [h,w,3], alpha [h,w], z-depth of the first hit [h,w] (inf = none)."""
    ss = SS if ss is None else ss
    W2, H2 = w * ss, h * ss
    D, fx = pixel_dirs(eye, tgt, W2, H2, fov)
    dirs = D.view(-1, 3)
    nrm = dirs.norm(dim=1)
    dirs = dirs / nrm[:, None]
    og = (eye - lo) / v
    oidx, oax, ot = dda(cls.view(-1), og, dirs, layers)
    am = torch.ones(8, device=dev)
    for k_, al_ in (alpha or {}).items():
        am[k_] = al_
    fill_c, edge_c = torch.zeros((8, 3), device=dev), torch.zeros((8, 3), device=dev)
    for k_, (fc, ec) in PAL.items():
        fill_c[k_], edge_c[k_] = torch.tensor(fc), torch.tensor(ec)
    Lt = torch.tensor(LIGHT, dtype=torch.float32, device=dev)
    flat = cls.view(-1)
    aog = ao_grid(cls)
    ogt = torch.tensor(og, dtype=torch.float32, device=dev)
    nl = torch.tensor([nx, ny, nz], device=dev)
    pix_ang = 1.0 / fx
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
        pxv = (te * pix_ang).clamp(min=1e-4)                             # pixel size in voxels
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
        base = torch.where(colored[:, None], COL[idx].float(), fill_c[ch])
        lit = base * (shade * ao)[:, None]
        col = torch.where(colored[:, None], lit * (1 - edge_dark * e)[:, None],
                          lit * (1 - e[:, None]) + edge_c[ch] * e[:, None])
        a_c = am[ch]
        al = torch.where(a_c >= 0.999, torch.ones_like(e), ((0.42 if l == 0 else 0.16) * (1 - e) + 0.95 * e) * a_c)
        acc[hh] += (TA[hh] * al)[:, None] * col
        TA[hh] = TA[hh] * (1 - al)
    z = torch.where(oidx[:, 0] >= 0, ot[:, 0] * v / nrm, torch.full_like(nrm, float("inf")))
    rgb, alp, z = acc.view(H2, W2, 3), (1 - TA).view(H2, W2), z.view(H2, W2)
    if ss > 1:
        rgb = F.avg_pool2d(rgb.permute(2, 0, 1)[None], ss)[0].permute(1, 2, 0)
        alp = F.avg_pool2d(alp[None, None], ss)[0, 0]
        z = -F.max_pool2d(-z[None, None], ss)[0, 0]
    return rgb, alp, z


# ==== shared drawing ===============================================================================
_bg = {}


def background(w, h):
    if (w, h) not in _bg:
        y = torch.linspace(0, 1, h, device=dev)[:, None, None]
        top, bot = torch.tensor([0.10, 0.12, 0.16], device=dev), torch.tensor([0.035, 0.04, 0.05], device=dev)
        _bg[(w, h)] = (top * (1 - y) + bot * y).expand(h, w, 3).contiguous()
    return _bg[(w, h)]


def to8(img):
    return (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


FONT_T = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", max(12, int(44 * a.scale)))
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", max(10, int(26 * a.scale)))
FONT_L = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", max(12, int(34 * a.scale)))


def text(dr, xy, s, font, fill=(255, 255, 255, 255), anchor="la"):
    dr.text(xy, s, font=font, fill=fill, anchor=anchor, stroke_width=max(1, int(2 * a.scale)), stroke_fill=(0, 0, 0, 200))


def title(img8, t1, t2=None, corner="tl", x0=None, y0=None):
    pil = Image.fromarray(img8)
    dr = ImageDraw.Draw(pil, "RGBA")
    m = int(44 * a.scale)
    if corner == "tl":
        text(dr, (x0 or m, y0 or m), t1, FONT_T)
        if t2:
            text(dr, (x0 or m, (y0 or m) + int(62 * a.scale)), t2, FONT_S, (225, 233, 242, 255))
    return np.asarray(pil)


# ==== modes ========================================================================================
g = load_ply(meta["ply"], dev)
n_gauss = len(g.means)
cls_vox = solid.to(torch.uint8)                       # class 1 = coloured voxel
n_vox = int(solid.sum())
mode = a.mode

if mode == "orbit":
    T_END, FOV = 24.0, 62.0

    def cam_raw(t, rad):
        u = 0.5 - 0.5 * math.cos(2 * math.pi * t / T_END)
        an = math.radians(-150.0) + 2 * math.pi * t / T_END
        hgt = 9.0 + 33.0 * u
        return STATUE + [rad * math.cos(an), rad * math.sin(an), hgt], STATUE + [0, 0, 6.0 - 3.0 * u]

    # around the statue inside the clearing: pull the camera in wherever trees would block the view
    TG = np.arange(0, T_END, 0.1)
    rads = np.arange(34.0, 17.0, -1.0)
    ii, jj = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * v, lo[1] + (np.arange(ny) + 0.5) * v, indexing="ij")
    near_statue = torch.from_numpy(np.hypot(ii - STATUE[0], jj - STATUE[1]) < 24.0).to(dev)
    clear = np.full(len(TG), rads[-1])
    for i, t in enumerate(TG):
        eyes, tgts = zip(*[cam_raw(t, r) for r in rads])
        eyes, tgts = np.array(eyes), np.array(tgts)
        dd = torch.tensor(tgts - eyes, dtype=torch.float32, device=dev)
        dist = dd.norm(dim=1)
        hit_t = torch.full_like(dist, float("inf"))
        for j in range(len(rads)):
            oi, _, ot_ = dda(cls_vox.view(-1), (eyes[j] - lo) / v, (dd[j:j + 1] / dist[j]), 1)
            h_ = int(oi[0, 0])
            if h_ >= 0 and not bool(near_statue[h_ // (ny * nz), (h_ // nz) % ny]):
                hit_t[j] = ot_[0, 0] * v
        ok = (hit_t > dist).cpu().numpy()                          # nothing but the statue in between
        clear[i] = rads[np.argmax(ok)] if ok.any() else rads[-1]
    clear = ndimage.gaussian_filter1d(ndimage.minimum_filter1d(clear, 15, mode="wrap"), 6, mode="wrap") - 1.0
    print(f"orbit radius {clear.min():.0f}..{clear.max():.0f} units", flush=True)

    def cam(t):
        return cam_raw(t, float(np.interp(t, TG, clear, period=T_END)))

    def frame(t):
        eye, tgt = cam(t)
        vr, va, _ = raycast(cls_vox, eye, tgt, W, H, FOV)
        img = vr + (1 - va[..., None]) * background(W, H)
        return title(to8(img), "体素化 · 全视角", f"{n_vox / 1e4:.0f} 万个 10 cm 体素，颜色取自 3DGS")

elif mode == "wide":
    T_END, FOV = 24.0, 60.0
    XS = STATUE[0] + 1.0                                          # the cut: level x = XS (through the statue)
    P_l = (g.means.double() - torch.tensor(c, device=dev)) @ torch.tensor(R, device=dev).T
    inbox = ((P_l[:, 0] > lo[0]) & (P_l[:, 0] < hi[0]) & (P_l[:, 1] > lo[1]) & (P_l[:, 1] < hi[1]) &
             (P_l[:, 2] > lo[2]) & (P_l[:, 2] < hi[2]))
    g_half = g.subset(inbox & (P_l[:, 0] < XS))
    cls_half = cls_vox.clone()
    cls_half[: int(math.ceil((XS - lo[0]) / v))] = 0
    del P_l

    def cam(t):
        s = t / T_END
        an = math.radians(-115.0) + math.radians(230.0) * s
        rad, hgt = 84.0 - 10.0 * smooth(s), 128.0 - 22.0 * smooth(s)
        return CTR + [rad * math.cos(an), rad * math.sin(an), hgt], CTR + [0, 0, -4.0]

    def frame(t):
        eye, tgt = cam(t)
        vr, va, vz = raycast(cls_half, eye, tgt, W, H, FOV)
        sr, sz, sa = splat(g_half, eye, tgt, W, H, FOV)
        bg = background(W, H)
        sp_front = (sa > 0.02) & ((va < 0.5) | (sz < vz))
        under_v = vr + (1 - va[..., None]) * bg
        under_s = sr + (1 - sa[..., None]) * bg
        img = torch.where(sp_front[..., None], sr + (1 - sa[..., None]) * under_v, vr + (1 - va[..., None]) * under_s)
        # glowing seam where the cut plane meets the visible surface
        D, fxw = pixel_dirs(eye, tgt, W, H, FOV)
        zz = torch.nan_to_num(torch.where(sp_front, sz, vz), nan=0.0, posinf=0.0)
        xw = eye[0] + zz * D[..., 0]
        foot = zz * torch.sqrt(D[..., 0] ** 2 + 1e-6) / fxw            # pixel footprint along level x (approx)
        seam = (1 - (xw - XS).abs() / (1.6 * foot.clamp(min=0.02) + 0.06)).clamp(0, 1)
        seam = seam * (((sa > 0.3) & sp_front) | ((va > 0.5) & ~sp_front)).float()
        glow = torch.from_numpy(cv2.GaussianBlur(seam.cpu().numpy(), (0, 0), 4 * a.scale)).to(dev)
        img = img + (0.8 * seam + 1.6 * glow)[..., None] * torch.tensor([0.35, 0.85, 1.0], device=dev)
        im8 = title(to8(img), "一半 3DGS，一半体素", "同一场景沿雕像中线切开：左侧为高斯泼溅，右侧为 10 cm 体素")
        return im8

elif mode == "split":
    T_END, FOV = 20.0, 58.0

    def cam(t):
        s = t / T_END
        an = math.radians(20.0) + math.radians(110.0) * s
        rad, hgt = 32.0 - 4.0 * smooth(s), 22.0 - 8.0 * smooth(s)
        return STATUE + [rad * math.cos(an), rad * math.sin(an), hgt], STATUE + [0, 0, 4.0]

    def frame(t):
        eye, tgt = cam(t)
        sr, _, sa = splat(g, eye, tgt, W, H, FOV)
        vr, va, _ = raycast(cls_vox, eye, tgt, W, H, FOV)
        bg = background(W, H)
        left = sr + (1 - sa[..., None]) * bg
        right = vr + (1 - va[..., None]) * (0.25 * sr + (1 - sa[..., None]) * bg)
        xd = W * (0.5 + 0.32 * math.sin(2 * math.pi * t / 10.0))         # divider sweeps across
        xs = torch.arange(W, device=dev, dtype=torch.float32)[None, :, None] + 0.5
        k = ((xs - xd) / 1.5 + 0.5).clamp(0, 1)
        img = left * (1 - k) + right * k
        line = torch.exp(-((xs - xd) / (2.0 * a.scale + 0.5)) ** 2) + 0.35 * torch.exp(-((xs - xd) / (14 * a.scale)) ** 2)
        img = img + line * torch.tensor([0.75, 0.95, 1.0], device=dev)
        pil = Image.fromarray(to8(img))
        dr = ImageDraw.Draw(pil, "RGBA")
        m = int(44 * a.scale)
        if xd > W * 0.16:
            text(dr, (xd - m * 0.6, m), "3DGS", FONT_L, anchor="ra")
        if xd < W * 0.84:
            text(dr, (xd + m * 0.6, m), "体素化", FONT_L)
        return np.asarray(pil)

elif mode == "compare":
    T_END, FOV = 20.0, 60.0
    w2, h2 = W // 2, H // 2
    shell, free, camv = un("shell"), un("free"), un("camera")
    kk = torch.arange(nz, device=dev)
    reach = un("reach")
    has = reach.any(2).cpu().numpy()
    kf = torch.where(reach, kk, torch.full_like(kk, nz)).amin(2).float().cpu().numpy()
    num, den = np.zeros(kf.shape), np.zeros(kf.shape)
    for sig in (2.0, 8.0, 32.0):
        n_ = ndimage.gaussian_filter(np.where(has, kf, 0.0), sig)
        d_ = ndimage.gaussian_filter(has.astype(float), sig)
        t_ = (den < 0.05) & (d_ > 1e-9)
        num[t_], den[t_] = n_[t_], d_[t_]
    kfloor = torch.from_numpy(np.where(den > 0, num / np.maximum(den, 1e-12), 0)).float().to(dev)
    near_free = F.max_pool3d(free[None, None].half(), 3, stride=1, padding=1)[0, 0] > 0
    cap = 3.0 / (v * mpu)                                          # walls shown up to 3 m above the floor
    shell_surf = shell & near_free & ((kk[None, None, :].float() - kfloor[:, :, None]) <= cap)
    cls_shell = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=dev)
    cls_shell[shell_surf] = 5
    ero = -F.max_pool3d(-camv[None, None].half(), 3, stride=1, padding=1)[0, 0] > 0
    cls_cam = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=dev)
    cls_cam[camv & ~ero] = 6
    info = json.load(open(f"{a.run}/ue/scene_ue.json"))
    tris = {L["name"]: L["tris"] for L in info["layers"]}
    vol_m3 = meta["camera"]["volume_m3"]
    LAB = [("3DGS", f"{n_gauss / 1e4:.0f} 万个高斯"),
           ("体素化", f"{n_vox / 1e4:.0f} 万个 10 cm 体素"),
           ("角色碰撞外壳", f"{tris['VoxelShell'] / 1e4:.1f} 万三角形 · 掉不下去、走不出去"),
           ("相机体积", f"{vol_m3:.0f} m³ · 相机不进物体、不出场景")]

    def cam(t):
        s = t / T_END
        an = math.radians(-100.0) + math.radians(140.0) * s
        rad, hgt = 80.0 - 8 * smooth(s), 112.0 - 10 * smooth(s)
        return CTR + [rad * math.cos(an), rad * math.sin(an), hgt], CTR + [0, 0, -4.0]

    def frame(t):
        eye, tgt = cam(t)
        bg = background(w2, h2)
        sr, _, sa = splat(g, eye, tgt, w2, h2, FOV)
        p1 = sr + (1 - sa[..., None]) * bg
        vr, va, _ = raycast(cls_vox, eye, tgt, w2, h2, FOV)
        p2 = vr + (1 - va[..., None]) * bg
        dim = 0.32 * sr + (1 - sa[..., None]) * bg
        er, ea, _ = raycast(cls_shell, eye, tgt, w2, h2, FOV, layers=1, alpha={5: 0.999}, edges=0.0)
        er, ea = er * 0.6, ea * 0.6
        p3 = er + (1 - ea[..., None]) * dim
        cr, ca, _ = raycast(cls_cam, eye, tgt, w2, h2, FOV, layers=2, alpha={6: 0.5}, edges=0.0)
        p4 = cr + (1 - ca[..., None]) * (0.6 * sr + (1 - sa[..., None]) * bg)
        img = torch.cat([torch.cat([p1, p2], 1), torch.cat([p3, p4], 1)], 0)
        img[h2 - 1:h2 + 1] = 0.9
        img[:, w2 - 1:w2 + 1] = 0.9
        pil = Image.fromarray(to8(img))
        dr = ImageDraw.Draw(pil, "RGBA")
        m = int(30 * a.scale)
        for i, (t1, t2) in enumerate(LAB):
            x, y = (i % 2) * w2 + m, (i // 2) * h2 + m
            text(dr, (x, y), t1, FONT_L)
            text(dr, (x, y + int(48 * a.scale)), t2, FONT_S, (225, 233, 242, 255))
        return np.asarray(pil)

# ==== output =======================================================================================
if a.still:
    for ts in a.still.split(","):
        t0 = time.time()
        im = frame(float(ts))
        fn = f"{a.run}/showcase/{mode}_t{float(ts):04.1f}.jpg"
        cv2.imwrite(fn, im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{fn} in {time.time() - t0:.1f}s", flush=True)
else:
    import imageio_ffmpeg
    out = f"{a.run}/showcase/{mode}.mp4"
    wr = imageio_ffmpeg.write_frames(out, (W, H), fps=a.fps, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", str(a.crf), "-preset", "medium"], macro_block_size=2)
    wr.send(None)
    n = int(round(T_END * a.fps))
    t0 = time.time()
    for i in range(n):
        wr.send(np.ascontiguousarray(frame(i / a.fps)).tobytes())
        if i % 60 == 0:
            el = time.time() - t0
            print(f"frame {i}/{n}  {el:.0f}s elapsed, eta {el / max(i, 1) * (n - i):.0f}s", flush=True)
    wr.close()
    print("wrote", out, flush=True)
