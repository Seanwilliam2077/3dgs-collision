"""Vertical cross-sections of the fused volume (x-z at fixed y) with the ground-net height overlaid.

usage: python xsection.py <run>   ->  <run>/debug/xsection.png  (dark = solid, light = free, grey = never observed, red = net)
"""
import json, sys
import cv2, numpy as np
run = sys.argv[1]
meta = json.load(open(f"{run}/meta.json"))
z = np.load(f"{run}/tsdf.npz")
tsdf, w, net = z["tsdf"].astype(np.float32), z["weight"], z["net_z"]
nx, ny, nz = tsdf.shape
vox, lo = meta["voxel_units"], meta["lo"]
mpu = meta["mpu"]
rows = []
for frac in (0.2, 0.5, 0.8):
    j = int(ny * frac)
    t, ww = tsdf[:, j, :], w[:, j, :]
    obs = ww >= 2
    img = np.full((nx, nz, 3), 90, np.uint8)                    # unobserved: grey
    img[obs & (t >= 0)] = (235, 235, 235)                       # observed free: light
    img[obs & (t >= 0) & (t < 0.25)] = (200, 215, 235)          # near-surface free
    img[obs & (t < 0)] = (40, 40, 40)                           # solid: dark
    fin = np.isfinite(net[:, j])
    k_net = np.clip(((np.where(fin, net[:, j], lo[2]) - lo[2]) / vox).astype(int), 0, nz - 1)
    img[np.arange(nx)[fin], k_net[fin]] = (0, 0, 255)                     # ground net: red
    im = np.ascontiguousarray(img.transpose(1, 0, 2)[::-1])     # z up
    im = cv2.resize(im, (nx * 2, nz * 2), interpolation=cv2.INTER_NEAREST)
    cv2.putText(im, f"y index {j} ({frac:.0%} of chunk), x left->right 100 m, height {nz * vox * mpu:.0f} m", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)
    rows.append(im)
    # floor statistics along this line: lowest upward crossing per column vs net
    up = obs[:, :-1] & obs[:, 1:] & (t[:, :-1] < 0) & (t[:, 1:] >= 0)
    has = up.any(1)
    k_floor = np.where(has, up.argmax(1), -1)
    dz = (lo[2] + k_floor * vox - np.where(fin, net[:, j], 0)) * mpu
    both = has & fin
    print(f"y {frac:.0%}: columns with a floor {has.mean()*100:.0f}%, with a net {fin.mean()*100:.0f}%; where both exist, lowest floor "
          f"minus net: median {np.median(dz[both]):.2f} m, p10 {np.percentile(dz[both],10):.2f}, p90 {np.percentile(dz[both],90):.2f} "
          f"(object tops count as floors); never-observed voxels {100*(~obs).mean():.0f}%")
cv2.imwrite(f"{run}/debug/xsection.png", np.vstack(rows))
