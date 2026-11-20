"""Explicit 3DGS PLY -> depth from virtual cameras -> TSDF fusion -> collision mesh.

No training cameras are needed: an aerial rig (nadir + obliques) is flown over the chunk, then
street-level rigs are dropped wherever the first pass proved the space to be empty.

Outputs (all in the ORIGINAL splat coordinates, so they overlay the splat without any transform):
  collision_full.ply   surface-nets mesh at voxel resolution, small floating components removed
  collision_lite.glb   vertex-clustered version for use as a collision proxy (+ .ply)
  ground_net.glb       smoothed terrain sheet slightly below the ground: catches falls through holes
  tsdf.npz + meta.json fused volume and frame, for SDF-style camera collision
  debug/               DSM/DTM, coverage map, splat-vs-TSDF comparison images

Units: every length argument is in metres and converted with --mpu (metres per scene unit).
"""
import argparse
import json
import os
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy import ndimage

from gs_core import load_ply, estimate_frame, look_at, intrinsics, render, render_median_depth
from tsdf_core import TSDF, surface_nets, drop_components, cluster_decimate, tri_area, raymarch


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply")
    ap.add_argument("--out", default="out/run")
    ap.add_argument("--mpu", type=float, default=1.0, help="metres per scene unit")
    ap.add_argument("--up", type=float, nargs=3, default=None, help="world up vector (default: estimated)")
    ap.add_argument("--center", type=float, nargs=2, default=None, help="chunk centre, level-frame x y in SCENE UNITS")
    ap.add_argument("--size", type=float, nargs=2, default=None, help="chunk size x y in metres")
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--trunc", type=float, default=4.0, help="truncation distance in voxels")
    ap.add_argument("--aerial", default="60:45,25:35", help="tiers as altitude_m:pitch_deg, first tier also gets nadir views")
    ap.add_argument("--aerial-spacing", type=float, default=0.6, help="node spacing as a fraction of altitude")
    ap.add_argument("--street", type=int, default=0, help="1: add a street-level pass (experimental; only useful if the splat was trained on street views)")
    ap.add_argument("--street-spacing", type=float, default=5.0)
    ap.add_argument("--street-range", type=float, default=25.0)
    ap.add_argument("--street-weight", type=float, default=1.0)
    ap.add_argument("--eye", type=float, default=1.6)
    ap.add_argument("--slices", type=int, default=96, help="depth slices per view for the median depth")
    ap.add_argument("--dsm-slices", type=int, default=160)
    ap.add_argument("--plane-depth", type=int, default=1, help="1: tangent-plane depth per Gaussian, 0: centre depth")
    ap.add_argument("--min-alpha", type=float, default=0.3, help="total opacity a ray needs to count as hitting a surface")
    ap.add_argument("--alpha-lo", type=float, default=0.05, help="rays with less total opacity carve free space")
    ap.add_argument("--max-float", type=float, default=3.0, help="m; pieces whose lowest point is higher above terrain are removed")
    ap.add_argument("--edge", type=float, default=0.10, help="relative depth jump treated as a silhouette")
    ap.add_argument("--free-weight", type=float, default=0.3,
                    help="weight of free-space evidence relative to surface evidence (1 = classic TSDF; lower keeps "
                         "surfaces that only some views see, at the cost of weaker floater removal)")
    ap.add_argument("--min-weight", type=float, default=2.0)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Gaussian smoothing of the fused TSDF before meshing, in voxels. Off by default: on the test scene it "
                         "removed more floor than it repaired")
    ap.add_argument("--min-comp-area", type=float, default=5.0, help="m^2; smaller floating pieces are removed")
    ap.add_argument("--lite", type=float, default=3.0, help="vertex-cluster cell for the lite mesh, in voxels")
    ap.add_argument("--net-offset", type=float, default=0.25)
    ap.add_argument("--net-max-rise", type=float, default=4.0,
                    help="m; the net is cut where the local terrain model sits this far above bare earth (= under large roofs)")
    ap.add_argument("--n-check", type=int, default=4, help="held-out views per kind for the comparison images")
    ap.add_argument("--save-tsdf", type=int, default=1)
    return ap.parse_args()


def depth_masks(depth, alpha, a_lo, edge_thr):
    """-> (depth_for_tsdf, pix_ok): surfaces keep their depth, provably empty rays carve.

    depth is the median depth (0 = opacity never reached 0.5 inside the view's range)."""
    has = depth > 0
    pad = F.pad(depth[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    nb = torch.stack([pad[1:-1, :-2], pad[1:-1, 2:], pad[:-2, 1:-1], pad[2:, 1:-1]])
    jump = ((nb - depth[None]).abs().amax(0) / depth.clamp(min=1e-9) > edge_thr) & has
    jump = F.max_pool2d(jump[None, None].float(), 3, 1, 1)[0, 0] > 0
    surf = has & ~jump
    free = (~has) & (alpha < a_lo)
    return torch.where(surf, depth, torch.full_like(depth, 1e9)), surf | free


def colorize(x, lo, hi, mask=None, cmap=cv2.COLORMAP_TURBO):
    v = np.clip((np.nan_to_num(x, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cmap)
    if mask is not None:
        img[~mask] = 0
    return img


def main():
    a = parse_args()
    os.makedirs(f"{a.out}/debug", exist_ok=True)
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    t0 = time.time()
    U = 1.0 / a.mpu                                           # scene units per metre
    voxel, trunc = a.voxel * U, a.trunc * a.voxel * U
    near = 0.05 * U
    tiers = [tuple(float(x) for x in t.split(":")) for t in a.aerial.split(",")]

    g_all = load_ply(a.ply)
    xyz = g_all.means.cpu().numpy().astype(np.float64)
    c, R = estimate_frame(xyz, 3.0, a.up)
    e1, e2, up = R
    pl = (xyz - c) @ R.T
    core_lo, core_hi = np.percentile(pl, [3, 97], axis=0)
    cx, cy = a.center if a.center else 0.5 * (core_lo[:2] + core_hi[:2])
    sx, sy = (np.array(a.size) * U) if a.size else (core_hi[:2] - core_lo[:2])
    lo_xy, hi_xy = np.array([cx - sx / 2, cy - sy / 2]), np.array([cx + sx / 2, cy + sy / 2])
    margin = max(alt / np.tan(np.radians(p)) + 0.5 * alt for alt, p in tiers) * U
    print(f"[frame] up={up.round(4)}  chunk centre=({cx:.3f},{cy:.3f}) size={sx * a.mpu:.0f}x{sy * a.mpu:.0f} m  "
          f"camera margin={margin * a.mpu:.0f} m")

    # ---- crop Gaussians to chunk + camera margin -------------------------------------------
    r_lo, r_hi = lo_xy - margin, hi_xy + margin
    in_chunk = ((pl[:, :2] >= lo_xy) & (pl[:, :2] <= hi_xy)).all(1)
    in_region = ((pl[:, :2] >= r_lo) & (pl[:, :2] <= r_hi)).all(1)
    z_med = np.median(pl[in_chunk, 2])
    keep = ((pl[:, :2] >= r_lo - 30 * U) & (pl[:, :2] <= r_hi + 30 * U)).all(1)
    keep &= (pl[:, 2] > z_med - 100 * U) & (pl[:, 2] < z_med + 400 * U)
    g = g_all.subset(torch.from_numpy(keep).to(g_all.means.device))
    del g_all
    print(f"[crop] {keep.sum()} / {len(keep)} Gaussians kept ({in_chunk.sum()} inside the chunk)")

    # ---- DSM (ortho nadir, median depth) and DTM -------------------------------------------
    px = max(voxel, float((r_hi - r_lo).max()) / 3000)        # units per DSM pixel
    Wd, Hd = int(np.ceil((r_hi[0] - r_lo[0]) / px)), int(np.ceil((r_hi[1] - r_lo[1]) / px))
    zs_region = pl[in_region & keep, 2]
    z_a, z_b = np.percentile(zs_region, 0.2) - 2 * U, np.percentile(zs_region, 99.5) + 3 * U
    z_cam = z_b + 1 * U
    eye = c + R.T @ np.array([0.5 * (r_lo[0] + r_hi[0]), 0.5 * (r_lo[1] + r_hi[1]), z_cam])
    M = np.eye(4)
    M[:3, :3] = np.stack([e1, -e2, -up], 0)                    # image x=+e1, image down=-e2, forward=-up
    M[:3, 3] = -M[:3, :3] @ eye
    Ko = np.array([[1 / px, 0, Wd / 2], [0, 1 / px, Hd / 2], [0, 0, 1]])
    d, _ = render_median_depth(g, M, Ko, Wd, Hd, z_cam - np.linspace(z_b, z_a, a.dsm_slices), near=1e-6, ortho=True,
                               min_alpha=a.min_alpha, plane=bool(a.plane_depth))
    d = d.cpu().numpy()[::-1]                                  # row index now grows with +y
    dsm_ok = d > 0
    dsm = np.where(dsm_ok, z_cam - d, np.nan)
    rgb, _, _ = render(g, M, Ko, Wd, Hd, near=1e-6, ortho=True)
    cv2.imwrite(f"{a.out}/debug/top_rgb.jpg", (rgb.cpu().numpy()[..., ::-1] * 255).astype(np.uint8))

    cell = max(1, int(round(1.0 * U / px)))                    # ~1 m cells for the terrain model
    Hc, Wc = Hd // cell, Wd // cell
    blk = dsm[:Hc * cell, :Wc * cell].reshape(Hc, cell, Wc, cell)
    low = np.nanpercentile(blk.transpose(0, 2, 1, 3).reshape(Hc, Wc, -1), 10, axis=2)
    hole = np.isnan(low)
    if hole.any():
        _, (iy, ix) = ndimage.distance_transform_edt(hole, return_indices=True)
        low = low[iy, ix]
    win = max(3, int(round(12.0 * U / (cell * px))) | 1)
    p10 = ndimage.percentile_filter(low, 10, size=win)         # erodes cars, trees, small buildings
    # junk below the ground (seen through holes in the splat) would dig pits into the terrain:
    # never let it drop more than 3 m below the regional median level
    big = max(win, int(round(60.0 * U / (cell * px))) | 1)
    dtm = ndimage.gaussian_filter(np.maximum(p10, ndimage.median_filter(p10, size=big) - 3 * U), 1.5)
    # bare-earth estimate with large buildings removed too; only used to tell street level from roofs
    p10b = ndimage.percentile_filter(low, 10, size=big)
    earth = ndimage.gaussian_filter(np.maximum(p10b, ndimage.median_filter(p10b, size=big) - 3 * U), 1.5)

    def grid_lookup(arr, step, x, y):
        ix = np.clip(((np.asarray(x) - r_lo[0]) / step).astype(int), 0, arr.shape[1] - 1)
        iy = np.clip(((np.asarray(y) - r_lo[1]) / step).astype(int), 0, arr.shape[0] - 1)
        return arr[iy, ix]

    def ground_at(x, y):
        return grid_lookup(dtm, cell * px, x, y)

    def dsm_at(x, y):
        return grid_lookup(dsm, px, x, y)

    def earth_at(x, y):
        return grid_lookup(earth, cell * px, x, y)

    ys, xs = np.mgrid[0:Hd, 0:Wd]
    xw, yw = r_lo[0] + (xs + 0.5) * px, r_lo[1] + (ys + 0.5) * px
    inch = (xw >= lo_xy[0]) & (xw <= hi_xy[0]) & (yw >= lo_xy[1]) & (yw <= hi_xy[1]) & dsm_ok
    g_min = float(ground_at(xw[inch], yw[inch]).min())
    h_top = float(np.percentile(dsm[inch], 99.9))
    print(f"[dsm] {Wd}x{Hd} px at {px * a.mpu:.2f} m, surface seen in {dsm_ok.mean() * 100:.1f}% of pixels; chunk relief "
          f"{(h_top - g_min) * a.mpu:.1f} m (ground {g_min * a.mpu:.1f} m, top {h_top * a.mpu:.1f} m in level z)")
    vis_lo, vis_hi = np.nanpercentile(dsm, [1, 99.5])
    cv2.imwrite(f"{a.out}/debug/dsm.jpg", colorize(dsm, vis_lo, vis_hi, dsm_ok)[::-1])
    cv2.imwrite(f"{a.out}/debug/dtm.jpg", cv2.resize(colorize(dtm, vis_lo, vis_hi), (Wd, Hd), interpolation=cv2.INTER_NEAREST)[::-1])
    cv2.imwrite(f"{a.out}/debug/earth.jpg", cv2.resize(colorize(earth, vis_lo, vis_hi), (Wd, Hd), interpolation=cv2.INTER_NEAREST)[::-1])

    # ---- TSDF volume -------------------------------------------------------------------------
    z_lo, z_hi = g_min - 3 * U, h_top + 3 * U
    lo = np.array([lo_xy[0], lo_xy[1], z_lo])
    dims = [int(x) for x in np.ceil(np.array([sx, sy, z_hi - z_lo]) / voxel).astype(int) + 1]
    nvox = int(np.prod(dims))
    print(f"[tsdf] grid {tuple(dims)} = {nvox / 1e6:.1f} M voxels, voxel {a.voxel} m, trunc {a.trunc * a.voxel:.2f} m, "
          f"~{nvox * 8 / 2 ** 30:.2f} GB")
    if nvox > 140e6:
        raise SystemExit("grid too large for one dense chunk: use a smaller --size or a larger --voxel")
    vol = TSDF(c, R, lo, dims, voxel, trunc)

    def to_world(p_l):
        return c + np.asarray(p_l, dtype=np.float64) @ R

    def make_view(eye_l, yaw, pitch, fov, W, H, max_range, kind):
        if pitch >= 89.9:
            Mv = np.eye(4)
            Mv[:3, :3] = np.stack([e1, -e2, -up], 0)
            Mv[:3, 3] = -Mv[:3, :3] @ to_world(eye_l)
        else:
            yr, pr = np.radians(yaw), np.radians(pitch)
            dir_l = np.array([np.cos(pr) * np.cos(yr), np.cos(pr) * np.sin(yr), -np.sin(pr)])
            Mv = look_at(to_world(eye_l), to_world(eye_l + dir_l), up)
        return dict(M=Mv, K=intrinsics(W, H, fov), W=W, H=H, max_range=max_range, kind=kind, eye_l=np.asarray(eye_l, dtype=np.float64))

    box_lo = lo - trunc
    box_hi = lo + (np.array(dims) - 1) * voxel + trunc

    def far_planes(v, K=None, W=None, H=None):
        """Depth slices spanning only the part of the frustum that lies inside the TSDF volume.

        Everything nearer falls into the first bracket; the thinner the brackets inside the
        volume, the more consistent the median depth is across views."""
        K = v["K"] if K is None else K
        W, H = (v["W"], v["H"]) if W is None else (W, H)
        Rcw = v["M"][:3, :3].T
        uu, vv = np.meshgrid(np.linspace(0, W, 9), np.linspace(0, H, 9))
        dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1).reshape(-1, 3)
        dl = (dc @ Rcw.T) @ R.T                                 # level-frame direction per unit z-depth
        dl = np.where(np.abs(dl) < 1e-12, 1e-12, dl)
        ta, tb = (box_lo - v["eye_l"]) / dl, (box_hi - v["eye_l"]) / dl
        t_in = np.minimum(ta, tb).max(1)
        t_out = np.maximum(ta, tb).min(1)
        ok = (t_out > t_in) & (t_out > 0)
        if not ok.any():                                         # frustum misses the volume
            return np.geomspace(0.5 * U, v["max_range"], 8)
        d0 = max(float(t_in[ok].min()), 0.5 * U)
        d1 = min(float(t_out[ok].max()), v["max_range"])
        return np.geomspace(d0, max(d1, d0 * 1.05), a.slices)

    def clear_above(x, y, z, radius, clearance):
        r = int(np.ceil(radius / px))
        ix = int(np.clip((x - r_lo[0]) / px, 0, Wd - 1)); iy = int(np.clip((y - r_lo[1]) / px, 0, Hd - 1))
        patch = dsm[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
        return (not np.isfinite(patch).any()) or np.nanmax(patch) + clearance < z

    def aerial_views(yaw_shift=0.0, pitch_shift=0.0, node_shift=0.0, nadir=True):
        views, skipped = [], 0
        for ti, (alt, pitch) in enumerate(tiers):
            reach = alt / np.tan(np.radians(pitch)) * U
            step = alt * a.aerial_spacing * U
            gx = np.arange(lo_xy[0] - reach, hi_xy[0] + reach + 1e-9, step) + node_shift * step
            gy = np.arange(lo_xy[1] - reach, hi_xy[1] + reach + 1e-9, step) + node_shift * step
            rng = alt / np.sin(np.radians(pitch)) * 2.5 * U
            for i, x in enumerate(gx):
                for j, y in enumerate(gy):
                    z = float(ground_at(x, y)) + alt * U
                    if not clear_above(x, y, z, 4 * U, 3 * U):
                        skipped += 1
                        continue
                    if ti == 0 and nadir:
                        views.append(make_view([x, y, z], 0, 90, 70, 800, 600, alt * 2.0 * U, "nadir"))
                    for k in range(4):
                        yaw = 90 * k + (45 if (i + j) % 2 else 0) + yaw_shift
                        views.append(make_view([x, y, z], yaw, pitch + pitch_shift, 70, 800, 600, rng, f"aerial{ti}"))
        return views, skipped

    def integrate(views, weight=1.0, tag=""):
        t = time.time()
        n_upd = 0
        for vi, v in enumerate(views):
            d, al = render_median_depth(g, v["M"], v["K"], v["W"], v["H"], far_planes(v), near=near, min_alpha=a.min_alpha, plane=bool(a.plane_depth))
            dd, ok = depth_masks(d, al, a.alpha_lo, a.edge)
            n_upd += vol.integrate(dd, ok, v["M"], v["K"], v["max_range"], near=near, weight=weight, max_vox=16_000_000,
                                   free_weight=a.free_weight)
            if vi < 4:
                rgb, ed, _ = render(g, v["M"], v["K"], v["W"], v["H"], near=near)
                surf = (ok & (dd < 1e8)).cpu().numpy()
                dn = d.cpu().numpy()
                rng = np.percentile(dn[surf], [1, 99]) if surf.any() else (0, 1)
                row = np.hstack([(rgb.cpu().numpy()[..., ::-1] * 255).astype(np.uint8), colorize(ed.cpu().numpy(), rng[0], rng[1]),
                                 colorize(dn, rng[0], rng[1], surf)])
                cv2.imwrite(f"{a.out}/debug/in_{tag}_{vi}.jpg", row, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"[integrate:{tag}] {len(views)} views, {n_upd / 1e6:.0f} M voxel updates, {time.time() - t:.0f}s")

    views, skipped = aerial_views()
    print(f"[aerial] {len(views)} views ({skipped} nodes skipped: too close to geometry)")
    integrate(views, tag="aerial")

    # ---- street-level rigs where the aerial pass proved the space empty --------------------
    def street_candidates(shift=0.0):
        step = a.street_spacing * U
        gx = np.arange(lo_xy[0] + step / 2, hi_xy[0], step) + shift * step
        gy = np.arange(lo_xy[1] + step / 2, hi_xy[1], step) + shift * step
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        X, Y = X.ravel(), Y.ravel()
        with np.errstate(invalid="ignore"):
            near_ground = (dsm_at(X, Y) - earth_at(X, Y)) < 2.5 * U
        P = np.stack([X, Y, dsm_at(X, Y) + a.eye * U], 1)[near_ground]
        if len(P) == 0:
            return P
        offs = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) * 0.4 * U
        okk = np.ones(len(P), dtype=bool)
        for o in offs:
            idx = torch.tensor((P + o - lo) / voxel, dtype=torch.float32, device=vol.dev)
            tv, inside = vol.sample(idx, "nearest", "tsdf")
            wv, _ = vol.sample(idx, "nearest", "w")
            okk &= (inside & (wv >= a.min_weight) & (tv > 0.5)).cpu().numpy()
        return P[okk]

    def street_views(P, yaw_shift=0.0):
        views = []
        for n, p in enumerate(P):
            for k in range(4):
                views.append(make_view(p, 90 * k + 37.0 * n + yaw_shift, 15, 90, 448, 448, a.street_range * U, "street"))
        return views

    P_street = np.zeros((0, 3))
    if a.street:
        P_street = street_candidates()
        sv = street_views(P_street)
        print(f"[street] {len(P_street)} eye positions -> {len(sv)} views")
        integrate(sv, weight=a.street_weight, tag="street")

    # ---- mesh extraction ---------------------------------------------------------------------
    t = time.time()
    ts, obs = vol.smoothed(a.smooth, a.min_weight)
    vol.tsdf = ts                                              # everything downstream uses the smoothed field
    vol.w = torch.where(obs, vol.w.clamp(min=a.min_weight), torch.zeros_like(vol.w))
    Vi, Fc = surface_nets(vol.tsdf, obs)
    V = vol.index_to_world(Vi)
    area_m2 = tri_area(V, Fc).sum() * a.mpu ** 2
    print(f"[mesh] surface nets: {len(V)} verts, {len(Fc)} tris, {area_m2:.0f} m^2, {time.time() - t:.0f}s")
    Vl = (V - c) @ R.T
    V, Fc, cs = drop_components(V, Fc, a.min_comp_area * U ** 2, Vl[:, 2] - ground_at(Vl[:, 0], Vl[:, 1]), a.max_float * U)
    print(f"[mesh] components {cs['components']} -> kept {cs['kept']}; removed {cs['area_small'] * a.mpu ** 2:.0f} m^2 of small "
          f"pieces and {cs['floating']} floating pieces ({cs['area_floating'] * a.mpu ** 2:.0f} m^2) of {cs['area_total'] * a.mpu ** 2:.0f} m^2")
    Vl = (V - c) @ R.T
    Vl2, F2 = cluster_decimate(Vl, Fc, a.lite * voxel)
    V2 = c + Vl2 @ R
    print(f"[mesh] lite: {len(V2)} verts, {len(F2)} tris (cell {a.lite * a.voxel:.2f} m)")

    # ground net: terrain sheet over the chunk
    step = cell * px
    gx = np.arange(lo_xy[0], hi_xy[0] + step, step); gy = np.arange(lo_xy[1], hi_xy[1] + step, step)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    GZ = ground_at(GX, GY) - a.net_offset * U
    Vn = to_world(np.stack([GX, GY, GZ], -1).reshape(-1, 3))
    ii, jj = np.meshgrid(np.arange(len(gx) - 1), np.arange(len(gy) - 1), indexing="ij")
    v00 = (ii * len(gy) + jj).ravel(); v10 = v00 + len(gy); v01 = v00 + 1; v11 = v10 + 1
    Fn = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
    # under a large roof the 12 m terrain model climbs to roof level; a sheet there would be a ramp
    net_ok = ((ground_at(GX, GY) - earth_at(GX, GY)) < a.net_max_rise * U).ravel()
    Fn = Fn[net_ok[Fn].all(1)]

    trimesh.Trimesh(V, Fc, process=False).export(f"{a.out}/collision_full.ply")
    lite = trimesh.Trimesh(V2, F2, process=False)
    lite.export(f"{a.out}/collision_lite.ply")
    lite.export(f"{a.out}/collision_lite.glb")
    trimesh.Trimesh(Vn, Fn, process=False).export(f"{a.out}/ground_net.glb")

    # ---- coverage: which columns have any upward-facing surface (walkability.py goes further) ----
    up_cross = obs[:, :, :-1] & obs[:, :, 1:] & (vol.tsdf[:, :, :-1] < 0) & (vol.tsdf[:, :, 1:] >= 0)
    has_any = up_cross.any(2).cpu().numpy()
    del up_cross
    CX, CY = np.meshgrid(lo[0] + np.arange(dims[0]) * voxel, lo[1] + np.arange(dims[1]) * voxel, indexing="ij")
    cov_any = float(has_any.mean())
    cov = np.zeros((dims[0], dims[1], 3), dtype=np.uint8)
    cov[has_any] = (200, 200, 200)
    cov[~has_any] = (0, 140, 255)                             # orange: no floor of any kind in the column
    for p in P_street:
        i_, j_ = int((p[0] - lo[0]) / voxel), int((p[1] - lo[1]) / voxel)
        cov[max(0, i_ - 1):i_ + 2, max(0, j_ - 1):j_ + 2] = (255, 120, 0)   # blue: street-level eye positions
    cv2.imwrite(f"{a.out}/debug/coverage.png", np.ascontiguousarray(cov.transpose(1, 0, 2)[::-1]))
    print(f"[coverage] columns with an upward-facing surface: {cov_any * 100:.1f}%")

    # ---- held-out comparison: splat median depth vs fused surface --------------------------
    rng = np.random.default_rng(0)
    checks = []
    av, _ = aerial_views(yaw_shift=22.5, pitch_shift=-8.0, node_shift=0.5, nadir=False)
    av = [v for v in av if ((v["eye_l"][:2] > lo_xy) & (v["eye_l"][:2] < hi_xy)).all()]
    for kind in sorted({v["kind"] for v in av}):
        pool = [v for v in av if v["kind"] == kind]
        checks += [pool[i] for i in rng.choice(len(pool), min(a.n_check, len(pool)), replace=False)]
    if a.street:
        Pc = street_candidates(shift=0.5)
        if len(Pc):
            pool = street_views(Pc[rng.choice(len(Pc), min(a.n_check, len(Pc)), replace=False)], yaw_shift=45.0)
            checks += [pool[i] for i in rng.choice(len(pool), min(a.n_check, len(pool)), replace=False)]
    Rt = torch.tensor(R, dtype=torch.float32, device=vol.dev)
    ct, lot = torch.tensor(c, dtype=torch.float32, device=vol.dev), torch.tensor(lo, dtype=torch.float32, device=vol.dev)
    light = torch.tensor(R.T @ np.array([0.35, 0.25, 0.9]) / np.linalg.norm([0.35, 0.25, 0.9]), dtype=torch.float32, device=vol.dev)
    report = {}
    for n, v in enumerate(checks):
        W, H = (640, 480) if v["kind"] != "street" else (448, 448)
        K = intrinsics(W, H, 70 if v["kind"] != "street" else 90)
        rgb, _, _ = render(g, v["M"], K, W, H, near=near)
        d, al = render_median_depth(g, v["M"], K, W, H, far_planes(v, K, W, H), near=near, min_alpha=a.min_alpha, plane=bool(a.plane_depth))
        dm, nm = raymarch(vol, v["M"], K, W, H, v["max_range"], 0.6 * voxel, a.min_weight, t_near=near)
        dd, ok = depth_masks(d, al, a.alpha_lo, a.edge)
        surf = ok & (dd < 1e8)
        # only compare pixels whose splat surface lies inside the fused volume
        vm = torch.tensor(v["M"], dtype=torch.float32, device=vol.dev)
        vs_, us_ = torch.meshgrid(torch.arange(H, device=vol.dev) + 0.5, torch.arange(W, device=vol.dev) + 0.5, indexing="ij")
        pc = torch.stack([(us_ - K[0, 2]) / K[0, 0] * d, (vs_ - K[1, 2]) / K[1, 1] * d, d], -1)
        pidx = ((((pc - vm[:3, 3]) @ vm[:3, :3]) - ct) @ Rt.T - lot) / voxel
        inside = ((pidx > 1) & (pidx < torch.tensor(dims, dtype=torch.float32, device=vol.dev) - 2)).all(-1) & surf
        hit = dm > 0
        both = inside & hit
        err = ((dm - d).abs() * a.mpu)[both]
        has = len(err) > 0
        stats = dict(kind=v["kind"], pixels=int(inside.sum()), hit_rate=float(both.sum() / max(1, int(inside.sum()))),
                     median_m=float(err.median()) if has else None, p90_m=float(err.quantile(0.9)) if has else None,
                     within_2vox=float((err < 2 * a.voxel).float().mean()) if has else None)
        report[f"check_{n}"] = stats
        shade = (0.25 + 0.75 * (nm @ light).clamp(0, 1))[..., None] * (0.5 + 0.5 * (nm @ Rt.T))
        shade = (shade.clamp(0, 1) * hit[..., None]).cpu().numpy()
        dn, s_np = d.cpu().numpy(), surf.cpu().numpy()
        r_ = np.percentile(dn[s_np], [1, 99]) if s_np.any() else (0, 1)
        e_img = colorize(((dm - d).abs() * a.mpu).cpu().numpy(), 0, 1.0, both.cpu().numpy(), cv2.COLORMAP_INFERNO)
        e_img[(inside & ~hit).cpu().numpy()] = (255, 0, 255)   # magenta: splat surface with no fused surface
        row = np.hstack([(rgb.cpu().numpy()[..., ::-1] * 255).astype(np.uint8), colorize(dn, r_[0], r_[1], s_np),
                         (shade[..., ::-1] * 255).astype(np.uint8), e_img])
        cv2.imwrite(f"{a.out}/debug/check_{n}_{v['kind']}.jpg", row, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if has:
            print(f"[check {n}:{v['kind']}] fused surface found for {stats['hit_rate'] * 100:.1f}% of splat-surface pixels; "
                  f"|dz| median {stats['median_m']:.3f} m, p90 {stats['p90_m']:.3f} m, within 2 voxels {stats['within_2vox'] * 100:.1f}%")
        else:
            print(f"[check {n}:{v['kind']}] no overlapping pixels")

    meta = dict(ply=a.ply, mpu=a.mpu, voxel_units=voxel, trunc_units=trunc, c=c.tolist(), R=R.tolist(), lo=lo.tolist(),
                dims=dims, args=vars(a), n_aerial=len(views), n_street_eyes=int(len(P_street)),
                mesh=dict(full_tris=int(len(Fc)), lite_tris=int(len(F2)), area_m2=float(area_m2), **cs),
                coverage_any=cov_any, checks=report, seconds=round(time.time() - t0, 1))
    with open(f"{a.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    if a.save_tsdf:
        np.savez_compressed(f"{a.out}/tsdf.npz", tsdf=vol.tsdf.cpu().numpy().astype(np.float16),
                            weight=vol.w.clamp(max=255).cpu().numpy().astype(np.uint8),
                            # ground-net height per column (level z); -inf where the net was cut
                            net_z=np.where((ground_at(CX, CY) - earth_at(CX, CY)) < a.net_max_rise * U,
                                           ground_at(CX, CY) - a.net_offset * U, -np.inf).astype(np.float32))
    print(f"[done] {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
