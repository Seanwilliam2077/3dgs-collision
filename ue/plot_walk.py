"""Draw the UE walk test (walk_result.json) over the walkability map of the same run.

    py.bat ue/plot_walk.py out/chunk_a
Writes <run>/ue/walk_map.png: one panel per phase. Background = debug/walkable.png (mesh only).
White = tracks, red X = where an agent left the floor before falling out of the world (void),
magenta = fell onto the ground net, orange = fell onto the ground patch, cyan = dropped >= 1 m onto
the mesh, yellow = snag (stuck with nothing in front at knee or waist height), blue = blocked,
red disc in a white ring = trapped (agent respawned).
"""
import json
import sys

import cv2
import numpy as np

run = sys.argv[1]
meta = json.load(open(f"{run}/meta.json"))
info = json.load(open(f"{run}/ue/scene_ue.json"))
res = json.load(open(f"{run}/ue/walk_result.json"))
vu, lo = meta["voxel_units"], meta["lo"]
if meta.get("kind") == "voxel":                   # voxel run: background = reachable floor height, 1 px per voxel
    c, S = 1, 2
    bg = cv2.imread(f"{run}/debug/reach_top.png")
    bg = cv2.resize(bg, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
else:                                             # TSDF run: walkability map, 3 px per cell
    c, S = max(1, int(round(meta["walkability"]["cell"] / meta["args"]["voxel"]))), 3
    bg = cv2.imread(f"{run}/debug/walkable.png")
cy_n = meta["dims"][1] // c
mpu, x0, y0 = info["mpu"], info["x0"], info["y0"]


def px(xy_cm):
    xy = np.asarray(xy_cm, np.float64).reshape(-1, 2)
    x = xy[:, 0] / (100 * mpu) + x0
    y = -xy[:, 1] / (100 * mpu) + y0
    u, v = (x - lo[0]) / vu, (y - lo[1]) / vu
    return np.stack([(u + 0.5) / c * S, (cy_n - (v + 0.5) / c) * S], 1)


panels = []
for ph in res["phases"]:
    img = (bg * 0.75).astype(np.uint8)
    for tr in ph["tracks"]:
        if len(tr) < 2:
            continue
        t = np.asarray(tr, np.float64)
        p = px(t[:, :2])
        jump = np.r_[False, np.linalg.norm(np.diff(t[:, :2], axis=0), axis=1) > 400]   # respawns
        for k in range(1, len(p)):
            if not jump[k]:
                cv2.line(img, tuple(np.round(p[k - 1]).astype(int)), tuple(np.round(p[k]).astype(int)), (235, 235, 235), 1, cv2.LINE_AA)
    for e in ph["events"]:
        if e["kind"] == "void":
            q = px(e.get("from_xy") or e["at"][:2])[0].round().astype(int)
            cv2.drawMarker(img, tuple(q), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        elif e["kind"] == "drop":
            q = px(e["from_xy"])[0].round().astype(int)
            col = {"GroundNet": (255, 0, 255), "GroundPatch": (0, 165, 255)}.get(e.get("surface"), (255, 255, 0))
            cv2.drawMarker(img, tuple(q), col, cv2.MARKER_DIAMOND, 10, 2)
        elif e["kind"] == "trapped":
            q = px(e["at"][:2])[0].round().astype(int)
            cv2.circle(img, tuple(q), 9, (255, 255, 255), 2)
            cv2.circle(img, tuple(q), 6, (0, 0, 200), -1)
        elif e["kind"] == "stuck":
            q = px(e["at"][:2])[0].round().astype(int)
            cv2.circle(img, tuple(q), 4, (0, 220, 255) if e["what"] == "snag" else (255, 120, 0), -1)
    s = ph["summary"]
    txt = f"{ph['name']}: {ph['dist_cm'] / 1e5:.1f} km walked, " + ", ".join(f"{k} {v}" for k, v in sorted(s.items()))
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    panels.append(img)
cv2.imwrite(f"{run}/ue/walk_map.png", np.concatenate(panels, 1))
print("wrote", f"{run}/ue/walk_map.png")
