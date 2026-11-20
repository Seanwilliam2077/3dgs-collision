"""Walkability check on the fused volume at capsule scale: where can a character stand, what is
connected, and where would it fall through.

Per voxel column, a floor (solid below, not solid above) is standable if nothing solid sits in the
body band between step height and `--height` above it. Columns are pooled into cells of
`--cell` metres (about a capsule radius): a cell is standable at a height if any of its columns is.
Cells are connected to their 4 neighbours when standable heights differ by at most `--step`.
The largest connected set is reported as reachable. A leak is a reachable cell with a neighbour
that offers neither a standable height within step nor anything solid at body height: a character
stepping there falls through the mesh.

`--net 1` adds ground_net as a floor, i.e. evaluates mesh + net together.
Only surfaces that exist in the mesh count; unobserved space has no collision and is passable.
Writes <run>/debug/walkable[_net].png and <run>/walk_cells[_net].npz (standable cells of the largest
sets, level frame). Not a navmesh, just the check.
"""
import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--height", type=float, default=1.8)
ap.add_argument("--step", type=float, default=0.45)
ap.add_argument("--cell", type=float, default=0.3)
ap.add_argument("--min-weight", type=float, default=2.0)
ap.add_argument("--net", type=int, default=0)
a = ap.parse_args()

meta = json.load(open(f"{a.run}/meta.json"))
voxel_m = meta["args"]["voxel"]
z = np.load(f"{a.run}/tsdf.npz")
tsdf = torch.from_numpy(z["tsdf"].astype(np.float32)).cuda()
obs = torch.from_numpy(z["weight"]).cuda() >= a.min_weight
nx, ny, nz = tsdf.shape
solid = obs & (tsdf < 0)
if a.net:
    zc = meta["lo"][2] + torch.arange(nz, device="cuda") * meta["voxel_units"]
    solid |= zc[None, None, :] <= torch.from_numpy(z["net_z"]).cuda()[:, :, None]
del tsdf, obs
Hc = int(np.ceil(a.height / voxel_m))
S = int(round(a.step / voxel_m))
c = max(1, int(round(a.cell / voxel_m)))

floor = torch.zeros_like(solid)
floor[:, :, :-1] = solid[:, :, :-1] & ~solid[:, :, 1:]                  # index k: solid at k, not at k+1
cs = torch.zeros((nx, ny, nz + 1), dtype=torch.int16, device="cuda")
cs[:, :, 1:] = torch.cumsum(solid.to(torch.int16), 2)
kmax = nz - Hc - 1
stand = torch.zeros_like(solid)
# nothing solid in k+1+S .. k+Hc
stand[:, :, :kmax] = floor[:, :, :kmax] & ((cs[:, :, Hc + 1:Hc + 1 + kmax] - cs[:, :, 1 + S:1 + S + kmax]) == 0)

# pool to capsule-sized cells
def pool_any(v):
    return F.max_pool2d(v.permute(2, 0, 1)[None].float(), c, c)[0].permute(1, 2, 0) > 0

stand_c = pool_any(stand)
cx_n, cy_n = stand_c.shape[:2]
# body-band blockage per cell and level: something solid between k+1+S and k+Hc in any column of the cell
band = torch.zeros_like(solid)
band[:, :, :kmax] = (cs[:, :, Hc + 1:Hc + 1 + kmax] - cs[:, :, 1 + S:1 + S + kmax]) > 0
band_c = pool_any(band)
del band, cs, floor

# connectivity: thicken in z by about half a step each way so neighbours within a step overlap
h = max(1, S // 2)
wd = stand_c.clone()
for s in range(1, h + 1):
    wd[:, :, s:] |= stand_c[:, :, :-s]
    wd[:, :, :-s] |= stand_c[:, :, s:]
ks = wd.any(0).any(0).nonzero()[:, 0]
k0, k1 = int(ks.min()), int(ks.max()) + 1
lab, n = ndimage.label(wd[:, :, k0:k1].cpu().numpy())
lab_t = torch.from_numpy(lab).cuda()
sb = stand_c[:, :, k0:k1]
cnt = torch.bincount(lab_t[sb], minlength=n + 1)
cnt[0] = 0
order = torch.argsort(cnt, descending=True)
cell_m2 = (c * voxel_m) ** 2
foot = lambda m: int(m.any(2).sum()) * cell_m2
print(f"standable area: {foot(stand_c):.0f} m^2 of {cx_n * cy_n * cell_m2:.0f} m^2; {int((cnt > 0).sum())} connected sets"
      f"{' (mesh + ground net)' if a.net else ' (mesh only)'}")
sets = []
for r in range(min(4, n)):
    i = int(order[r])
    m = sb & (lab_t == i)
    kk_mean = float(m.nonzero()[:, 2].float().mean()) + k0
    sets.append((i, foot(m), kk_mean))
    print(f"  set {r}: {foot(m):.0f} m^2, mean height {(meta['lo'][2] + kk_mean * meta['voxel_units']) * meta['mpu']:.1f} m (level z)")

img = np.full((cx_n, cy_n, 3), 25, dtype=np.uint8)
img[pool_any(solid[:, :, :-1] & ~solid[:, :, 1:]).any(2).cpu().numpy()] = (70, 70, 70)      # surface, not standable
img[stand_c.any(2).cpu().numpy()] = (150, 110, 60)                                           # standable, small set
colors = [(90, 190, 90), (60, 170, 220), (200, 120, 200), (200, 200, 90)]
report = []
cells_xyz, cells_set = [], []
kk = torch.arange(k1 - k0, device="cuda")
wn = stand_c.clone()
for s in range(1, S + 1):
    wn[:, :, s:] |= stand_c[:, :, :-s]
    wn[:, :, :-s] |= stand_c[:, :, s:]
for r, (i, area, _) in enumerate(sets):
    reach = sb & (lab_t == i)
    r2 = reach.any(2)
    img[r2.cpu().numpy()] = colors[r]
    k_r = torch.where(reach, kk[None, None, :], torch.full_like(lab_t, 10 ** 6)).amin(2) + k0
    ci, cj = r2.nonzero(as_tuple=True)
    vu = meta["voxel_units"]
    # A set can hold several floors in one cell (e.g. the bottom of a hole in the ground, reached through
    # a chain of small steps). For the saved height take the floor closest to the local median of the
    # cells' top floors, i.e. the surface a character actually walks on.
    k_top = torch.where(reach, kk[None, None, :], torch.full_like(lab_t, -1)).amax(2).float()
    top = np.where(r2.cpu().numpy(), k_top.cpu().numpy(), np.nan)
    ref = ndimage.generic_filter(top, np.nanmedian, size=7, mode="constant", cval=np.nan)
    ref_t = torch.from_numpy(np.nan_to_num(ref, nan=0.0)).cuda().float()
    k_sel = torch.where(reach, (kk[None, None, :] - ref_t[:, :, None]).abs(), torch.full_like(ref_t[:, :, None], 1e9)).argmin(2) + k0
    cells_xyz.append(np.stack([meta["lo"][0] + (ci.cpu().numpy() * c + (c - 1) / 2) * vu,
                               meta["lo"][1] + (cj.cpu().numpy() * c + (c - 1) / 2) * vu,
                               meta["lo"][2] + (k_sel[ci, cj].cpu().numpy() + 0.5) * vu], 1))  # level frame, top of floor
    cells_set.append(np.full(len(ci), r))
    valid = r2.clone()
    valid[0, :] = valid[-1, :] = False
    valid[:, 0] = valid[:, -1] = False                    # the chunk border is not a leak
    ii, jj = valid.nonzero(as_tuple=True)
    k = k_r[ii, jj]
    leak2d = torch.zeros((cx_n, cy_n), dtype=torch.bool, device="cuda")
    n_leak = 0
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ni, nj = ii + di, jj + dj
        lk = ~wn[ni, nj, k] & ~band_c[ni, nj, k.clamp(max=kmax - 1)]
        leak2d[ii[lk], jj[lk]] = True
        n_leak += int(lk.sum())
    img[leak2d.cpu().numpy()] = (0, 0, 255)
    edge_m = n_leak * c * voxel_m
    report.append(dict(area_m2=area, leak_m=edge_m))
    print(f"  set {r}: leak edges {edge_m:.0f} m ({int(leak2d.sum()) / max(1, int(r2.sum())) * 100:.1f}% of its cells border a hole)")
cv2.imwrite(f"{a.run}/debug/walkable{'_net' if a.net else ''}.png",
            cv2.resize(np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
# standable cells of the largest sets (level frame, scene units), e.g. spawn points for a walk test
np.savez(f"{a.run}/walk_cells{'_net' if a.net else ''}.npz", xyz=np.concatenate(cells_xyz), set=np.concatenate(cells_set))
meta["walkability_net" if a.net else "walkability"] = dict(standable_m2=foot(stand_c), sets=report, height=a.height, step=a.step, cell=a.cell)
json.dump(meta, open(f"{a.run}/meta.json", "w"), indent=1)
