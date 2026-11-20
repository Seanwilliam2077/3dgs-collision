"""Cross-view consistency of centre depth vs tangent-plane depth.   usage: py.bat depth_consistency.py <run> [slices]

For pairs of views of the same target, back-project view A's depth to 3D, project into view B and
compare with B's own depth. Lower disagreement = better input for TSDF fusion.
"""
import json, sys, itertools
import numpy as np, torch
sys.path.insert(0, ".")
from gs_core import load_ply, look_at, intrinsics, render_median_depth

meta = json.load(open(f"{sys.argv[1]}/meta.json"))
c, R, mpu = np.array(meta["c"]), np.array(meta["R"]), meta["mpu"]
U = 1 / mpu
up = R[2]
g = load_ply(meta["ply"])
pl = (g.means.cpu().numpy() - c) @ R.T
keep = (np.abs(pl[:, 0] + 0.55) < 4) & (np.abs(pl[:, 1] + 0.1) < 4) & (pl[:, 2] > -2) & (pl[:, 2] < 7)
g = g.subset(torch.from_numpy(keep).cuda())
W, H = 800, 600
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 96
K = intrinsics(W, H, 70)
Kt = torch.tensor(K, dtype=torch.float32, device="cuda")
vv, uu = torch.meshgrid(torch.arange(H, device="cuda") + 0.5, torch.arange(W, device="cuda") + 0.5, indexing="ij")

# NOTE: target points (level-frame x, y in scene units) and ground_z are specific to with_aerial.ply
targets = {"road": (-0.80, -0.20), "yard": (-1.10, -0.45), "roof": (-0.10, 0.10), "road_n": (-0.80, 0.35)}
ground_z = -0.07
res = {False: [], True: []}
for tname, (tx, ty) in targets.items():
    cams = []
    for yaw, pitch, dist in [(0, 45, 45), (90, 45, 45), (180, 45, 45), (270, 45, 45), (45, 30, 40), (225, 30, 40), (135, 70, 55)]:
        yr, pr = np.radians(yaw), np.radians(pitch)
        dl = np.array([np.cos(pr) * np.cos(yr), np.cos(pr) * np.sin(yr), -np.sin(pr)])
        eye_l = np.array([tx, ty, ground_z]) - dl * dist * U
        cams.append(look_at(c + eye_l @ R, c + (eye_l + dl) @ R, up))
    for plane in (False, True):
        depths = [render_median_depth(g, M, K, W, H, np.geomspace(8 * U, 150 * U, NS), near=0.05 * U, plane=plane)[0] for M in cams]
        errs = []
        for ia, ib in itertools.permutations(range(len(cams)), 2):
            Ma = torch.tensor(cams[ia], dtype=torch.float32, device="cuda"); Mb = torch.tensor(cams[ib], dtype=torch.float32, device="cuda")
            da = depths[ia]
            # central part of A only (the common target area)
            m = (da > 0) & ((uu - W / 2).abs() < W / 4) & ((vv - H / 2).abs() < H / 4)
            pc = torch.stack([(uu - Kt[0, 2]) / Kt[0, 0] * da, (vv - Kt[1, 2]) / Kt[1, 1] * da, da], -1)[m]
            pw = (pc - Ma[:3, 3]) @ Ma[:3, :3]
            pb = pw @ Mb[:3, :3].T + Mb[:3, 3]
            ub = (Kt[0, 0] * pb[:, 0] / pb[:, 2] + Kt[0, 2]).long(); vb = (Kt[1, 1] * pb[:, 1] / pb[:, 2] + Kt[1, 2]).long()
            inb = (pb[:, 2] > 0) & (ub >= 0) & (ub < W) & (vb >= 0) & (vb < H)
            db = depths[ib][vb.clamp(0, H - 1), ub.clamp(0, W - 1)]
            ok = inb & (db > 0)
            # signed: positive = B sees something farther than A's point (A's point floats in front of B's surface
            # or is simply occluded-from-B); use the |.| of non-occluded part: db > pb_z - 1.5 m
            diff = (db - pb[:, 2])[ok] * mpu
            vis = diff > -1.5
            errs.append(diff[vis].abs())
        e = torch.cat(errs)
        res[plane].append(e)
        print(f"{tname:7s} {'plane ' if plane else 'centre'}: cross-view |dz| median {e.median():.3f} m  mean {e.mean():.3f} m  "
              f"within 0.15 m {(e < 0.15).float().mean() * 100:.1f}%  within 0.30 m {(e < 0.30).float().mean() * 100:.1f}%")
for plane in (False, True):
    e = torch.cat(res[plane])
    print(f"ALL     {'plane ' if plane else 'centre'}: median {e.median():.3f} m  mean {e.mean():.3f} m  within 0.15 m {(e < 0.15).float().mean() * 100:.1f}%  "
          f"within 0.30 m {(e < 0.30).float().mean() * 100:.1f}%  within 0.60 m {(e < 0.60).float().mean() * 100:.1f}%")
