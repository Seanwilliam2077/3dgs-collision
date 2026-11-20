"""Where do agents get trapped or blocked, relative to the ground patch and net?

    py.bat ue/diag_walk.py out/chunk_a [phase]
Background: hillshaded patch heights (grey), net-only cells (dark blue), no floor at all (black),
patch steps > step height between neighbouring cells (red). Overlay for one phase (default: last):
trapped (white ring), blocked by the patch (orange), blocked by objects (blue), snag (yellow),
fell onto the net (magenta). Writes <run>/ue/diag_<phase>.png, cropped to the patch, 4 px per cell.
"""
import json
import sys

import cv2
import numpy as np

run = sys.argv[1]
meta = json.load(open(f"{run}/meta.json"))
info = json.load(open(f"{run}/ue/scene_ue.json"))
res = json.load(open(f"{run}/ue/walk_result.json"))
g = np.load(f"{run}/ground_grid.npz")
ph = res["phases"][-1] if len(sys.argv) < 3 else [p for p in res["phases"] if p["name"] == sys.argv[2]][0]
vu, lo, mpu = meta["voxel_units"], meta["lo"], meta["mpu"]
c = int(round(float(g["cell_m"]) / meta["args"]["voxel"]))
Z = g["patch_z"] * mpu                       # metres
N = g["net_z"] * mpu
S = 4

# hillshade
zf = np.nan_to_num(Z, nan=np.nanmin(Z))
gx, gy = np.gradient(zf, float(g["cell_m"]))
shade = np.clip(0.6 + 0.4 * (-(gx - gy) / np.sqrt(1 + gx ** 2 + gy ** 2)), 0, 1)
img = np.zeros(Z.shape + (3,), np.uint8)
has_p = ~np.isnan(Z)
img[has_p] = (np.stack([shade] * 3, -1)[has_p] * 200).astype(np.uint8)
img[~has_p & np.isfinite(N)] = (90, 40, 20)
step = 0.45
ex = np.zeros(Z.shape, bool)
dx = np.abs(np.diff(Z, axis=0)) > step
dy = np.abs(np.diff(Z, axis=1)) > step
ex[:-1][dx] = True; ex[1:][dx] = True; ex[:, :-1][dy] = True; ex[:, 1:][dy] = True
img[ex] = (40, 40, 230)
img = cv2.resize(np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
cy_n = Z.shape[1]


def px(xy_cm):
    xy = np.asarray(xy_cm, np.float64).reshape(-1, 2)
    x = xy[:, 0] / (100 * mpu) + info["x0"]
    y = -xy[:, 1] / (100 * mpu) + info["y0"]
    u, v = (x - lo[0]) / vu, (y - lo[1]) / vu
    return np.round(np.stack([(u + 0.5) / c * S, (cy_n - (v + 0.5) / c) * S], 1)).astype(int)


for e in ph["events"]:
    if e["kind"] == "stuck":
        col = {"GroundPatch": (0, 140, 255), "CollisionObjects": (255, 120, 0)}.get(e.get("by"), (0, 220, 255) if e["what"] == "snag" else (200, 200, 200))
        cv2.circle(img, tuple(px(e["at"][:2])[0]), 3, col, -1)
    elif e["kind"] == "drop" and e.get("surface") == "GroundNet":
        cv2.drawMarker(img, tuple(px(e["from_xy"])[0]), (255, 0, 255), cv2.MARKER_DIAMOND, 10, 2)
for e in ph["events"]:
    if e["kind"] == "trapped":
        q = tuple(px(e["at"][:2])[0])
        cv2.circle(img, q, 10, (255, 255, 255), 2)
cols = np.nonzero(has_p.any(1))[0]
x0, x1 = max(0, cols.min() * S - 10), min(img.shape[1], (cols.max() + 1) * S + 10)
out = img[:, x0:x1]
cv2.putText(out, f"{ph['name']}: red = patch step > {step} m", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(f"{run}/ue/diag_{ph['name']}.png", out)
print("wrote", f"{run}/ue/diag_{ph['name']}.png", out.shape)
