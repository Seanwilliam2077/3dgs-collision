"""Dense TSDF volume (torch), surface-nets meshing, SDF ray-marching, and mesh clean-up helpers.

The volume lives in the level frame: voxel (i,j,k) has its centre at  lo + (i,j,k) * voxel  and
p_level = R @ (p_world - c).  All mesh outputs are converted back to the original world frame.
"""
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class TSDF:
    def __init__(self, c, R, lo, dims, voxel, trunc, device="cuda", max_weight=64.0):
        self.dev = device
        self.c = torch.tensor(c, dtype=torch.float64)
        self.R = torch.tensor(R, dtype=torch.float64)
        self.lo = torch.tensor(lo, dtype=torch.float64)
        self.dims = tuple(int(d) for d in dims)
        self.voxel, self.trunc, self.max_weight = float(voxel), float(trunc), max_weight
        self.tsdf = torch.ones(self.dims, dtype=torch.float32, device=device)
        self.w = torch.zeros(self.dims, dtype=torch.float32, device=device)

    # ---- coordinate helpers (numpy, float64) -------------------------------------------------
    def index_to_world(self, idx):
        pl = self.lo.numpy() + np.asarray(idx, dtype=np.float64) * self.voxel
        return self.c.numpy() + pl @ self.R.numpy()

    def world_to_index(self, pw):
        pl = (np.asarray(pw, dtype=np.float64) - self.c.numpy()) @ self.R.numpy().T
        return (pl - self.lo.numpy()) / self.voxel

    # ---- integration -------------------------------------------------------------------------
    @torch.no_grad()
    def integrate(self, depth, pix_ok, viewmat, K, max_range, near=1e-4, weight=1.0, max_vox=6_000_000, free_weight=1.0):
        """Projective TSDF update from one z-depth map.

        depth:  [H,W] z-depth; use a huge value for rays known to be empty (they carve free space)
        pix_ok: [H,W] bool, pixels that may update the volume
        """
        H, W = depth.shape
        vm = torch.as_tensor(np.asarray(viewmat), dtype=torch.float64)
        Kd = np.asarray(K, dtype=np.float64)
        Rwc, twc = vm[:3, :3], vm[:3, 3]
        M = (Rwc @ self.R.T) * self.voxel                      # index -> camera (linear part)
        b = Rwc @ (self.c + self.R.T @ self.lo) + twc          # index -> camera (offset)

        # frustum AABB in index space
        us = np.array([0.0, W, 0.0, W]); vs = np.array([0.0, 0.0, H, H])
        rays = np.stack([(us - Kd[0, 2]) / Kd[0, 0], (vs - Kd[1, 2]) / Kd[1, 1], np.ones(4)], 1)
        pts_c = torch.tensor(np.concatenate([rays * near, rays * max_range], 0))
        pts_i = (pts_c - b) @ torch.linalg.inv(M).T
        i0 = torch.clamp(torch.floor(pts_i.min(0).values) - 1, min=0).long()
        i1 = torch.minimum(torch.ceil(pts_i.max(0).values).long() + 2, torch.tensor(self.dims))
        if (i1 <= i0).any():
            return 0
        (a0, b0, c0), (a1, b1, c1) = i0.tolist(), i1.tolist()
        dev = self.dev
        Mf, bf = M.float().to(dev), b.float().to(dev)
        ii = torch.arange(a0, a1, device=dev, dtype=torch.float32)
        jj = torch.arange(b0, b1, device=dev, dtype=torch.float32)
        d_flat = depth.reshape(-1)
        ok_flat = pix_ok.reshape(-1)
        fx, fy, cx, cy = float(Kd[0, 0]), float(Kd[1, 1]), float(Kd[0, 2]), float(Kd[1, 2])
        slab = max(1, int(max_vox // max(1, (a1 - a0) * (b1 - b0))))
        n_upd = 0
        for k0 in range(c0, c1, slab):
            k1 = min(k0 + slab, c1)
            kk = torch.arange(k0, k1, device=dev, dtype=torch.float32)

            def axis(r):
                return (Mf[r, 0] * ii)[:, None, None] + (Mf[r, 1] * jj)[None, :, None] \
                    + (Mf[r, 2] * kk)[None, None, :] + bf[r]

            zc = axis(2)
            front = zc > near
            zs = torch.where(front, zc, torch.ones_like(zc))
            u = torch.floor(fx * axis(0) / zs + cx)
            v = torch.floor(fy * axis(1) / zs + cy)
            inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            lin = (v.clamp(0, H - 1) * W + u.clamp(0, W - 1)).long()
            del u, v, zs
            sdf = d_flat[lin] - zc
            upd = inb & ok_flat[lin] & (sdf > -self.trunc) & (zc < max_range)
            del lin, inb, front
            t_new = (sdf / self.trunc).clamp(max=1.0)
            # full weight within the truncation band in front of the surface, linear fall-off behind
            # it (keeps thin walls), and free_weight for pure free-space evidence further in front
            w_new = torch.where(sdf >= self.trunc, torch.full_like(sdf, free_weight),
                                torch.where(sdf >= 0, torch.ones_like(sdf), 1.0 + sdf / self.trunc)) * weight
            w_new = torch.where(upd, w_new, torch.zeros_like(w_new))
            sub_t = self.tsdf[a0:a1, b0:b1, k0:k1]
            sub_w = self.w[a0:a1, b0:b1, k0:k1]
            w_sum = sub_w + w_new
            sub_t.copy_(torch.where(w_new > 0, (sub_t * sub_w + t_new * w_new) / w_sum.clamp(min=1e-6), sub_t))
            sub_w.copy_(w_sum.clamp(max=self.max_weight))
            n_upd += int(upd.sum())
        return n_upd

    @torch.no_grad()
    def smoothed(self, sigma, min_weight):
        """Gaussian-smoothed copy of the TSDF (sigma in voxels), ignoring unobserved voxels.

        Normalised convolution: observed values are averaged with Gaussian weights. A voxel counts
        as observed afterwards if it was observed before or if at least 30% of its kernel mass was
        (which closes pinholes without eroding the thin observed shell behind surfaces). Removes
        voxel-scale roughness; structures thinner than ~2 sigma get eroded. Returns (tsdf, obs)."""
        obs = (self.w >= min_weight).float()
        if sigma <= 0:
            return self.tsdf, obs > 0
        r = int(np.ceil(3 * sigma))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=self.dev, dtype=torch.float32) / sigma) ** 2)
        k = k / k.sum()

        def blur(v):
            v = v[None, None]
            for axis in range(3):
                shape = [1, 1, 1, 1, 1]
                shape[2 + axis] = -1
                pad = [0, 0, 0, 0, 0, 0]
                pad[(2 - axis) * 2] = pad[(2 - axis) * 2 + 1] = r
                v = F.conv3d(F.pad(v, pad, mode="replicate"), k.view(shape))
            return v[0, 0]

        den = blur(obs)
        num = blur(self.tsdf * obs)
        return torch.where(den > 1e-3, num / den.clamp(min=1e-3), self.tsdf), (obs > 0) | (den > 0.3)

    # ---- queries -----------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, idx, mode="bilinear", what="tsdf"):
        """Sample the volume at float index coords [N,3] (torch, on device)."""
        vol = self.tsdf if what == "tsdf" else self.w
        d = torch.tensor(self.dims, device=idx.device, dtype=torch.float32)
        g = 2.0 * idx / (d - 1) - 1.0
        g = g[:, [2, 1, 0]].view(1, -1, 1, 1, 3)                 # grid_sample wants (x=W, y=H, z=D)
        out = F.grid_sample(vol[None, None], g, mode=mode, padding_mode="border", align_corners=True)
        out = out.view(-1)
        inside = ((idx >= 0) & (idx <= d - 1)).all(-1)
        return out, inside


# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def surface_nets(tsdf, obs):
    """Naive surface nets on a dense grid. tsdf [nx,ny,nz] float, obs bool (voxel observed).

    Returns vertices in index coordinates [M,3] float64 and triangles [T,3] int64, wound so
    normals point from negative (inside) to positive (free space).
    """
    tsdf = tsdf.cpu()
    obs = obs.cpu()
    neg = tsdf < 0
    nx, ny, nz = tsdf.shape

    def corner(a, dx, dy, dz):
        return a[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz]

    offs = list(itertools.product((0, 1), repeat=3))
    cnt = torch.zeros((nx - 1, ny - 1, nz - 1), dtype=torch.int8)
    allobs = torch.ones((nx - 1, ny - 1, nz - 1), dtype=torch.bool)
    for o in offs:
        cnt += corner(neg, *o).to(torch.int8)
        allobs &= corner(obs, *o)
    active = allobs & (cnt > 0) & (cnt < 8)
    del cnt, allobs
    idx = active.nonzero()                                        # [M,3]
    m = idx.shape[0]
    if m == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    vals = torch.stack([tsdf[idx[:, 0] + o[0], idx[:, 1] + o[1], idx[:, 2] + o[2]] for o in offs], 1).double()
    acc = torch.zeros((m, 3), dtype=torch.float64)
    num = torch.zeros((m, 1), dtype=torch.float64)
    offs_t = torch.tensor(offs, dtype=torch.float64)
    for a in range(8):
        for b_ in range(a + 1, 8):
            if sum(abs(offs[a][q] - offs[b_][q]) for q in range(3)) != 1:
                continue
            va, vb = vals[:, a], vals[:, b_]
            cross = (va < 0) != (vb < 0)
            t = (va / (va - vb).where(cross, torch.ones_like(va))).clamp(0, 1)
            p = offs_t[a][None] + t[:, None] * (offs_t[b_] - offs_t[a])[None]
            acc += p * cross[:, None]
            num += cross[:, None]
    verts = idx.double() + acc / num.clamp(min=1)

    vid = torch.full((nx - 1, ny - 1, nz - 1), -1, dtype=torch.int32)
    vid[active] = torch.arange(m, dtype=torch.int32)
    del active
    tris = []

    def emit(c0, c1, c2, c3, lower_neg, normal_positive_if_ccw):
        ok = (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
        c0, c1, c2, c3, lower_neg = c0[ok].long(), c1[ok].long(), c2[ok].long(), c3[ok].long(), lower_neg[ok]
        # c0..c3 is CCW around the +axis (or -axis when normal_positive_if_ccw is False)
        keep = lower_neg if normal_positive_if_ccw else ~lower_neg
        a = torch.stack([c0, c1, c2], 1); b_ = torch.stack([c0, c2, c3], 1)
        af = torch.stack([c0, c2, c1], 1); bf = torch.stack([c0, c3, c2], 1)
        tris.append(torch.where(keep[:, None], a, af))
        tris.append(torch.where(keep[:, None], b_, bf))

    # x edges: voxel (i,j,k) -> (i+1,j,k)
    e = (neg[:-1, 1:-1, 1:-1] != neg[1:, 1:-1, 1:-1]).nonzero()
    i, j, k = e[:, 0], e[:, 1] + 1, e[:, 2] + 1
    emit(vid[i, j - 1, k - 1], vid[i, j, k - 1], vid[i, j, k], vid[i, j - 1, k], neg[i, j, k], True)
    # y edges: voxel (i,j,k) -> (i,j+1,k)   (c0..c3 is CCW around -y)
    e = (neg[1:-1, :-1, 1:-1] != neg[1:-1, 1:, 1:-1]).nonzero()
    i, j, k = e[:, 0] + 1, e[:, 1], e[:, 2] + 1
    emit(vid[i - 1, j, k - 1], vid[i, j, k - 1], vid[i, j, k], vid[i - 1, j, k], neg[i, j, k], False)
    # z edges: voxel (i,j,k) -> (i,j,k+1)
    e = (neg[1:-1, 1:-1, :-1] != neg[1:-1, 1:-1, 1:]).nonzero()
    i, j, k = e[:, 0] + 1, e[:, 1] + 1, e[:, 2]
    emit(vid[i - 1, j - 1, k], vid[i, j - 1, k], vid[i, j, k], vid[i - 1, j, k], neg[i, j, k], True)
    faces = torch.cat(tris, 0).numpy()
    return verts.numpy(), faces


def compact(V, Fc):
    used = np.zeros(len(V), dtype=bool)
    used[Fc.reshape(-1)] = True
    remap = np.cumsum(used) - 1
    return V[used], remap[Fc]


def tri_area(V, Fc):
    return 0.5 * np.linalg.norm(np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]]), axis=1)


def drop_components(V, Fc, min_area, height=None, max_float=None):
    """Remove connected components that are small (area < min_area, same units as V) or floating
    (lowest vertex more than max_float above the terrain; `height` = per-vertex height above it)."""
    n = len(V)
    e = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]]])
    A = coo_matrix((np.ones(len(e), dtype=bool), (e[:, 0], e[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    flab = lab[Fc[:, 0]]
    comp_area = np.bincount(flab, weights=tri_area(V, Fc), minlength=ncomp)
    small = comp_area < min_area
    floating = np.zeros(ncomp, dtype=bool)
    if height is not None and max_float is not None:
        low = np.full(ncomp, np.inf)
        np.minimum.at(low, lab, height)
        floating = (low > max_float) & ~small & (comp_area > 0)
    bad = small | floating
    stats = dict(components=int((comp_area > 0).sum()), kept=int(((comp_area > 0) & ~bad).sum()),
                 area_total=float(comp_area.sum()), area_small=float(comp_area[small].sum()),
                 floating=int(floating.sum()), area_floating=float(comp_area[floating].sum()))
    V2, F2 = compact(V, Fc[~bad[flab]])
    return V2, F2, stats


def drop_small_components(V, Fc, min_area):
    return drop_components(V, Fc, min_area)


def cluster_decimate(V, Fc, cell):
    """Vertex clustering on a regular grid of size `cell` (same units as V)."""
    key = np.floor(V / cell).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uniq, inv = np.unique(lin, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    Vn = np.stack([np.bincount(inv, weights=V[:, a], minlength=len(uniq)) / cnt for a in range(3)], 1)
    Fn = inv[Fc]
    good = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
    Fn = Fn[good]
    _, first = np.unique(np.sort(Fn, axis=1), axis=0, return_index=True)
    Fn = Fn[np.sort(first)]
    return compact(Vn, Fn)


# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def raymarch(vol, viewmat, K, W, H, t_max, step, min_weight, t_near=0.0, chunk=400_000):
    """First observed + -> - zero crossing along each camera ray.

    Returns z-depth [H,W] (0 where no hit) and world-space normals [H,W,3].
    """
    dev = vol.dev
    vm = np.asarray(viewmat, dtype=np.float64)
    Kd = np.asarray(K, dtype=np.float64)
    Rcw = vm[:3, :3].T
    eye = -Rcw @ vm[:3, 3]
    vs, us = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    dirs_c = np.stack([(us - Kd[0, 2]) / Kd[0, 0], (vs - Kd[1, 2]) / Kd[1, 1], np.ones_like(us)], -1).reshape(-1, 3)
    # index-space origin and direction per unit z-depth
    Rl, c, lo = vol.R.numpy(), vol.c.numpy(), vol.lo.numpy()
    o_i = torch.tensor(((eye - c) @ Rl.T - lo) / vol.voxel, dtype=torch.float32, device=dev)
    d_i = torch.tensor((dirs_c @ Rcw.T) @ Rl.T / vol.voxel, dtype=torch.float32, device=dev)   # per unit z
    n_ray = d_i.shape[0]
    depth = torch.zeros(n_ray, device=dev)
    normal = torch.zeros(n_ray, 3, device=dev)
    # clip the march to the part of each ray that lies inside the volume
    hi_i = torch.tensor(vol.dims, dtype=torch.float32, device=dev) - 1
    inv = 1.0 / torch.where(d_i.abs() < 1e-9, torch.full_like(d_i, 1e-9), d_i)
    ta, tb = (0 - o_i[None]) * inv, (hi_i[None] - o_i[None]) * inv
    t_in = torch.minimum(ta, tb).amax(1).clamp(min=t_near)
    t_out = torch.maximum(ta, tb).amin(1).clamp(max=t_max)
    crosses = t_out > t_in
    if not bool(crosses.any()):
        return depth.view(H, W), normal.view(H, W, 3)
    t_near = max(t_near, float(t_in[crosses].min()) - step)
    t_max = min(t_max, float(t_out[crosses].max()) + step)
    n_step = int(np.ceil((t_max - t_near) / step))
    for s0 in range(0, n_ray, chunk):
        d = d_i[s0:s0 + chunk]
        n = d.shape[0]
        hit_t = torch.zeros(n, device=dev)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        prev_v = torch.ones(n, device=dev)
        prev_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        for s in range(n_step + 1):
            t = t_near + s * step
            p = o_i[None] + d * t
            v, inside = vol.sample(p, "bilinear", "tsdf")
            wv, _ = vol.sample(p, "nearest", "w")
            ok = inside & (wv >= min_weight)
            hit = (~done) & ok & prev_ok & (prev_v > 0) & (v <= 0)
            frac = prev_v / (prev_v - v).clamp(min=1e-6)
            hit_t = torch.where(hit, (t - step) + frac * step, hit_t)
            done |= hit
            prev_v, prev_ok = v, ok
            if s % 64 == 0 and bool(done.all()):
                break
        p = o_i[None] + d * hit_t[:, None]
        g = []
        for a in range(3):
            e = torch.zeros(1, 3, device=dev); e[0, a] = 1.0
            g.append(vol.sample(p + e, "bilinear")[0] - vol.sample(p - e, "bilinear")[0])
        g = torch.stack(g, 1)
        nw = F.normalize(g @ torch.tensor(Rl, dtype=torch.float32, device=dev), dim=-1)   # level -> world
        depth[s0:s0 + chunk] = torch.where(done, hit_t, torch.zeros_like(hit_t))
        normal[s0:s0 + chunk] = nw * done[:, None]
    return depth.view(H, W), normal.view(H, W, 3)
