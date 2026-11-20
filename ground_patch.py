"""Ground patch: a collision-only heightfield exactly on the walkable ground, filling its holes.

    py.bat ground_patch.py out/chunk_a          (after walkability.py, which writes walk_cells.npz)

Why: the fused mesh's ground is full of sub-metre to metre holes. walkability.py hides them
(a 0.3 m cell counts as standable if any of its 15 cm columns is), but a character capsule drops
into them: in UE about 25% of the ground-set cells had no mesh floor within 0.5 m of their centre.

What: take the floor heights of the main ground set (set 0), fill unknown cells up to
--fill metres away from it by normalised Gaussian interpolation, skip filled cells where the
volume has something solid at body height above the new floor (cars, walls, kerbs > step), and
triangulate the cell centres as a heightfield --offset metres below the floor. Known heights are
refined to the TSDF zero crossing (sub-voxel) and median-filtered. A real step stays a steep,
unwalkable face (a wall to the character), not a slit; only rises above --max-rise are dropped.

Also writes an objects mesh: collision_lite minus every triangle lying on the patched ground (all
vertices over the patch and at most --ground-band above it, or below it). Patch + objects is a
"heightfield ground + objects" collision set: a smooth floor with no holes or ankle-height bumps, and
walls, vehicles and stacks kept from the fused mesh.

Also writes ground_net2: the ground net lifted to just under the extrapolated ground near the walkable
area, so that falling through a remaining hole is a short step down instead of a pit.

Writes <run>/ground_patch.ply, <run>/ground_net2.ply and <run>/collision_objects.ply (splat frame), <run>/ground_cells.npy
(clean ground cells, level frame) and <run>/debug/ground_patch.png
(green = ground set, cyan = filled, red = left open because something is in the way).
"""
import argparse
import json

import cv2
import numpy as np
import trimesh
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--fill", type=float, default=5.0, help="fill holes up to this far from known ground (m)")
ap.add_argument("--offset", type=float, default=0.03, help="patch sits this far below the floor (m)")
ap.add_argument("--max-rise", type=float, default=3.0,
                help="drop triangles with more height change than this (m); smaller rises stay as steep, unwalkable "
                     "faces so that a step never becomes a slit")
ap.add_argument("--outlier", type=float, default=0.3, help="ground cells this far off the local median are holes (m)")
ap.add_argument("--ground-band", type=float, default=0.3,
                help="objects mesh: drop lite triangles lying within this height above the patch, or below it (m)")
ap.add_argument("--smooth", type=int, default=5, help="median window (cells) on the known ground heights")
ap.add_argument("--net-reach", type=float, default=10.0, help="net v2 follows the ground up to this far from it (m)")
ap.add_argument("--net-offset", type=float, default=0.25, help="net v2 sits this far below the extrapolated ground (m)")
ap.add_argument("--height", type=float, default=1.8)
ap.add_argument("--step", type=float, default=0.45)
ap.add_argument("--min-weight", type=float, default=2.0)
a = ap.parse_args()

meta = json.load(open(f"{a.run}/meta.json"))
vu, lo, mpu = meta["voxel_units"], np.array(meta["lo"]), meta["mpu"]
c_, R = np.array(meta["c"]), np.array(meta["R"])
voxel_m = meta["args"]["voxel"]
c = max(1, int(round(meta["walkability"]["cell"] / voxel_m)))
nx, ny, nz = meta["dims"]
cx_n, cy_n = nx // c, ny // c

cells = np.load(f"{a.run}/walk_cells.npz")
g = cells["xyz"][cells["set"] == 0]
ci = np.round(((g[:, 0] - lo[0]) / vu - (c - 1) / 2) / c).astype(int)
cj = np.round(((g[:, 1] - lo[1]) / vu - (c - 1) / 2) / c).astype(int)
F = np.full((cx_n, cy_n), np.nan)
F[ci, cj] = g[:, 2]
# cells whose only floor in the set is far from their neighbours' (the bottom of a hole) are holes too
ref = ndimage.generic_filter(F, np.nanmedian, size=7, mode="constant", cval=np.nan)
outlier = ~np.isnan(F) & (np.abs(F - ref) * mpu > a.outlier)
F[outlier] = np.nan
known = ~np.isnan(F)
print(f"[patch] {outlier.sum()} of {len(g)} ground cells are > {a.outlier} m off their neighbourhood and treated as holes")

# sub-voxel floor heights: walkability heights are quantised to whole voxels (15 cm), which turns a
# smooth road into 45-degree facets between cells. Use the TSDF zero crossing nearest to each cell's
# floor in each of its columns, averaged over the columns, then a 3x3 median against single-cell spikes.
z = np.load(f"{a.run}/tsdf.npz")
tsdf, wgt = z["tsdf"], z["weight"]
ki, kj = np.nonzero(known)
kf = np.round((F[ki, kj] - lo[2]) / vu - 0.5).astype(int)                      # solid voxel under the floor
acc, cnt = np.zeros(len(ki)), np.zeros(len(ki))
for di in range(c):
    for dj in range(c):
        X, Y = ki * c + di, kj * c + dj
        best = np.full(len(ki), np.inf)
        for dk in (-1, 0, 1):
            k0_ = np.clip(kf + dk, 0, nz - 2)
            t0, t1 = tsdf[X, Y, k0_].astype(np.float32), tsdf[X, Y, k0_ + 1].astype(np.float32)
            ok = (t0 < 0) & (t1 >= 0) & (wgt[X, Y, k0_] >= a.min_weight) & (wgt[X, Y, k0_ + 1] >= a.min_weight)
            zc = lo[2] + (k0_ + t0 / np.where(ok, t0 - t1, 1.0)) * vu
            better = ok & (np.abs(zc - F[ki, kj]) < np.abs(best - F[ki, kj]))
            best[better] = zc[better]
        f = np.isfinite(best)
        acc[f] += best[f]
        cnt[f] += 1
del tsdf, wgt, z
F[ki[cnt > 0], kj[cnt > 0]] = acc[cnt > 0] / cnt[cnt > 0]
F = np.where(known, ndimage.generic_filter(F, np.nanmedian, size=a.smooth, mode="constant", cval=np.nan), np.nan)
d = np.abs(np.diff(F, axis=0)) * mpu
d = d[~np.isnan(d)]
print(f"[patch] sub-voxel heights for {(cnt > 0).mean() * 100:.0f}% of cells; neighbour height steps p50/p90/p99 "
      f"{np.percentile(d, [50, 90, 99]).round(3)} m")

# ---- interpolate heights into nearby unknown cells ------------------------------------------------
dist = ndimage.distance_transform_edt(~known) * c * voxel_m                     # metres to known ground
cand = ~known & (dist <= a.fill)
def interp(sigmas):
    """Normalised Gaussian interpolation of the known heights, finest scale that has support."""
    Z = np.where(known, F, 0.0)
    num, den = np.zeros_like(Z), np.zeros_like(Z)
    for sigma in sigmas:                                                        # cells; coarser where sparse
        n_ = ndimage.gaussian_filter(Z * known, sigma)
        d_ = ndimage.gaussian_filter(known.astype(float), sigma)
        take = (den < 0.05) & (d_ > 1e-6)
        num[take], den[take] = n_[take], d_[take]
    return np.where(known, F, np.where(den > 0, num / np.maximum(den, 1e-12), np.nan))


Zi = interp((1.0, 2.0, 4.0, 8.0))

# ---- obstacle check for filled cells: anything solid between step height and head height ----------
z = np.load(f"{a.run}/tsdf.npz")
solid = (z["weight"] >= a.min_weight) & (z["tsdf"] < 0)
net_z = z["net_z"]
del z
cs = np.zeros((nx, ny, nz + 1), np.int32)
np.cumsum(solid, axis=2, dtype=np.int32, out=cs[:, :, 1:])
del solid
ii, jj = np.nonzero(cand)
k_floor = np.floor((Zi[ii, jj] - lo[2]) / vu).astype(int)                       # voxel just below the floor
k_lo = np.clip(k_floor + 1 + int(round(a.step / voxel_m)), 0, nz)
k_hi = np.clip(k_floor + 1 + int(np.ceil(a.height / voxel_m)), 0, nz)
blocked = np.zeros(len(ii), bool)
for di in range(c):
    for dj in range(c):
        X, Y = ii * c + di, jj * c + dj
        blocked |= (cs[X, Y, k_hi] - cs[X, Y, k_lo]) > 0
del cs
filled = np.zeros_like(known)
filled[ii[~blocked], jj[~blocked]] = True
open_ = np.zeros_like(known)
open_[ii[blocked], jj[blocked]] = True
patch = known | filled
print(f"[patch] ground set {known.sum() * (c * voxel_m) ** 2:.0f} m^2, filled {filled.sum() * (c * voxel_m) ** 2:.0f} m^2, "
      f"left open (obstacle at body height) {open_.sum() * (c * voxel_m) ** 2:.0f} m^2")

# ---- heightfield meshes ---------------------------------------------------------------------------
def heightfield(mask, Zc, path, label, max_rise=a.max_rise):
    """Triangulate the cell centres of `mask` at heights Zc (level frame), write in the splat frame."""
    vid = -np.ones((cx_n, cy_n), np.int64)
    pi, pj = np.nonzero(mask)
    vid[pi, pj] = np.arange(len(pi))
    Vl = np.stack([lo[0] + (pi * c + (c - 1) / 2) * vu, lo[1] + (pj * c + (c - 1) / 2) * vu, Zc[pi, pj]], 1)
    q00, q10, q01, q11 = vid[:-1, :-1], vid[1:, :-1], vid[:-1, 1:], vid[1:, 1:]
    full = (q00 >= 0) & (q10 >= 0) & (q01 >= 0) & (q11 >= 0)
    tris = [np.stack([q00[full], q10[full], q11[full]], 1), np.stack([q00[full], q11[full], q01[full]], 1)]
    for A, B, C_, miss in [(q00, q10, q11, q01), (q00, q11, q01, q10), (q00, q10, q01, q11), (q10, q11, q01, q00)]:
        m = (A >= 0) & (B >= 0) & (C_ >= 0) & (miss < 0)                      # three corners: one triangle
        tris.append(np.stack([A[m], B[m], C_[m]], 1))
    Fc = np.concatenate(tris)
    T = Vl[Fc]
    nz_ = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])[:, 2]
    Fc[nz_ < 0] = Fc[nz_ < 0][:, ::-1]                                         # all facing up
    rise = np.ptp(Vl[Fc, 2], axis=1) * mpu
    Fc = Fc[rise <= max_rise]
    print(f"[{label}] {len(Vl)} verts, {len(Fc)} tris ({(rise > max_rise).sum()} dropped for rising > {max_rise} m)")
    trimesh.Trimesh(c_ + Vl @ R, Fc, process=False).export(path)            # level -> splat frame


heightfield(patch, Zi - a.offset / mpu, f"{a.run}/ground_patch.ply", "patch")

# ---- ground net v2: just under the ground the character walks on --------------------------------
# The DTM-based ground_net sits 1-3 m under the real ground where the reconstruction is poor; a
# character falling through a hole there lands in a pit it cannot climb out of. Within --net-reach of
# the clean ground, lift the net to the extrapolated ground height minus --net-offset (a fall of at
# most ~0.3 m, which the character can step back up from); further out keep the DTM net.
old = np.full((cx_n, cy_n), -np.inf)
for di in range(c):
    for dj in range(c):
        old = np.maximum(old, net_z[di:cx_n * c:c, dj:cy_n * c:c][:cx_n, :cy_n])
ext = interp((1.0, 2.0, 4.0, 8.0, 16.0, 32.0))
reach = (dist <= a.net_reach) & np.isfinite(ext)
net2 = np.where(reach, np.maximum(old, np.nan_to_num(ext, nan=-np.inf) - a.net_offset / mpu), old)
heightfield(np.isfinite(net2), net2, f"{a.run}/ground_net2.ply", "net2", max_rise=np.inf)   # a net with gaps is no net
np.savez(f"{a.run}/ground_grid.npz", patch_z=np.where(patch, Zi, np.nan), net_z=net2, old_net_z=old, known=known,
         filled=filled, open=open_, cell_m=c * voxel_m)                         # level frame, for diagnostics
lift = (net2 - old)[reach & np.isfinite(old)] * mpu
print(f"[net2] within {a.net_reach} m of ground: lifted {(lift > 0.05).mean() * 100:.0f}% of cells, "
      f"lift p50/p90 {np.percentile(lift, [50, 90]).round(2)} m; {(reach & ~np.isfinite(old)).sum() * (c * voxel_m) ** 2:.0f} m^2 "
      f"added where the DTM net was cut")
# ---- objects mesh: the collision mesh minus its ground, which the patch replaces ------------------
lite = trimesh.load(f"{a.run}/collision_lite.ply", process=False)
VL = (np.asarray(lite.vertices) - c_) @ R.T
vi = np.clip(np.round(((VL[:, 0] - lo[0]) / vu - (c - 1) / 2) / c).astype(int), 0, cx_n - 1)
vj = np.clip(np.round(((VL[:, 1] - lo[1]) / vu - (c - 1) / 2) / c).astype(int), 0, cy_n - 1)
low = patch[vi, vj] & (VL[:, 2] <= Zi[vi, vj] + a.ground_band / mpu)          # on or below the patched ground
keep = ~low[lite.faces].all(1)
trimesh.Trimesh(lite.vertices, lite.faces[keep], process=False).export(f"{a.run}/collision_objects.ply")
print(f"[objects] {keep.sum()} of {len(keep)} lite triangles kept; {(~keep).sum()} were ground within "
      f"{a.ground_band} m of the patch or below it")
ki, kj = np.nonzero(known)                                                      # clean ground cells, e.g. spawn points
np.save(f"{a.run}/ground_cells.npy", np.stack([lo[0] + (ki * c + (c - 1) / 2) * vu, lo[1] + (kj * c + (c - 1) / 2) * vu, F[ki, kj]], 1))

img = np.full((cx_n, cy_n, 3), 25, np.uint8)
img[known] = (90, 190, 90)
img[filled] = (220, 200, 60)
img[open_] = (60, 60, 230)
cv2.imwrite(f"{a.run}/debug/ground_patch.png",
            cv2.resize(np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
meta["ground_patch"] = dict(ground_m2=float(known.sum() * (c * voxel_m) ** 2), filled_m2=float(filled.sum() * (c * voxel_m) ** 2),
                            open_m2=float(open_.sum() * (c * voxel_m) ** 2), fill=a.fill, offset=a.offset, max_rise=a.max_rise)
json.dump(meta, open(f"{a.run}/meta.json", "w"), indent=1)
