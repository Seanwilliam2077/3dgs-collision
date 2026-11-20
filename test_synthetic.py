"""Self-test of the TSDF / surface-nets / ray-march code on an analytic scene (no gsplat needed).

Scene (level frame): ground plane z=0 plus a box [-2,2]x[-1,1]x[0,3]. The level frame is a
rotated + shifted copy of the world frame so that every coordinate transform is exercised.
"""
import numpy as np
import torch

from tsdf_core import TSDF, surface_nets, raymarch, tri_area, drop_small_components, cluster_decimate

rng = np.random.default_rng(1)
A = rng.normal(size=(3, 3))
Q, _ = np.linalg.qr(A)
if np.linalg.det(Q) < 0:
    Q[2] *= -1
R, c = Q, np.array([3.0, -2.0, 7.0])            # p_level = R @ (p_world - c)
e1, e2, up = R
BOX_LO, BOX_HI = np.array([-2.0, -1.0, 0.0]), np.array([2.0, 1.0, 3.0])


def to_world(p):
    return c + np.asarray(p, dtype=np.float64) @ R


def look_at(eye, target, upv):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, upv); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    M = np.eye(4); M[:3, :3] = np.stack([r, d, f]); M[:3, 3] = -M[:3, :3] @ eye
    return M


def analytic_depth(M, K, W, H):
    """z-depth of plane+box for every pixel (inf where nothing is hit)."""
    Rcw = M[:3, :3].T
    eye_l = R @ ((-Rcw @ M[:3, 3]) - c)
    vs, us = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    dc = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    dl = (dc @ Rcw.T) @ R.T                      # level-frame direction per unit z-depth
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = np.where(dl[..., 2] < 0, -eye_l[2] / dl[..., 2], np.inf)
        t0 = (BOX_LO - eye_l) / dl; t1 = (BOX_HI - eye_l) / dl
        tn = np.minimum(t0, t1).max(-1); tf = np.maximum(t0, t1).min(-1)
    t_box = np.where((tn <= tf) & (tf > 0), np.maximum(tn, 0), np.inf)
    return np.minimum(t_plane, t_box)


voxel = 0.1
lo = np.array([-5.013, -4.027, -1.033]); dims = (np.array([10.0, 8.0, 5.5]) / voxel).astype(int) + 1   # off-grid on purpose
vol = TSDF(c, R, lo, dims, voxel, 4 * voxel)
W, H = 320, 240
K = np.array([[260.0, 0, W / 2], [0, 260.0, H / 2], [0, 0, 1]])
views = []
for yaw in np.arange(0, 360, 30):
    for (rad, h) in [(9.0, 7.0), (7.0, 2.5)]:
        eye = to_world([rad * np.cos(np.radians(yaw)), rad * np.sin(np.radians(yaw)), h])
        views.append(look_at(eye, to_world([0, 0, 1.0]), up))
views.append(look_at(to_world([0.3, 0.2, 12.0]), to_world([0.3, 0.2, 0.0]), e2))
for M in views:
    d = analytic_depth(M, K, W, H)
    ok = np.ones_like(d, dtype=bool)
    dt = torch.tensor(np.where(np.isfinite(d), d, 1e9), dtype=torch.float32, device="cuda")
    vol.integrate(dt, torch.tensor(ok, device="cuda"), M, K, max_range=40.0, near=0.05)

obs = vol.w >= 2
Vi, F = surface_nets(vol.tsdf, obs)
V = vol.index_to_world(Vi)
Vl = (V - c) @ R.T
# distance of every vertex to the true surface (plane outside the box footprint, or box faces)
inside_fp = (Vl[:, 0] > BOX_LO[0] - 0.05) & (Vl[:, 0] < BOX_HI[0] + 0.05) & (Vl[:, 1] > BOX_LO[1] - 0.05) & (Vl[:, 1] < BOX_HI[1] + 0.05)
d_plane = np.abs(Vl[:, 2])
q = np.maximum(BOX_LO - Vl, Vl - BOX_HI)
d_box = np.abs(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0))
dist = np.where(inside_fp, d_box, np.minimum(d_plane, d_box))
print(f"verts {len(V)}  tris {len(F)}  dist to truth: mean {dist.mean():.4f}  p99 {np.percentile(dist, 99):.4f}  max {dist.max():.4f} (voxel {voxel})")

# winding: normals must point away from the solid (up on the ground, outward on the box)
n = np.cross(Vl[F[:, 1]] - Vl[F[:, 0]], Vl[F[:, 2]] - Vl[F[:, 0]])
n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
cen = Vl[F].mean(1)
ground = (np.abs(cen[:, 2]) < 0.06) & ((np.abs(cen[:, 0]) > 2.3) | (np.abs(cen[:, 1]) > 1.3))
roof = (np.abs(cen[:, 2] - 3) < 0.06) & (np.abs(cen[:, 0]) < 1.7) & (np.abs(cen[:, 1]) < 0.7)
wall = (np.abs(cen[:, 0] - 2) < 0.06) & (np.abs(cen[:, 1]) < 0.7) & (cen[:, 2] > 0.4) & (cen[:, 2] < 2.6)
print(f"normals: ground up {np.mean(n[ground, 2] > 0.9):.3f} (n={ground.sum()})  roof up {np.mean(n[roof, 2] > 0.9):.3f} (n={roof.sum()})  "
      f"+x wall out {np.mean(n[wall, 0] > 0.9):.3f} (n={wall.sum()})")
area = tri_area(Vl, F).sum()
print(f"area {area:.1f} (expected >= box 4*1+... visible part; ground patch 80 - 8 + box sides 36 + roof 8 = 116)")

# ray-march from a held-out view
M = look_at(to_world([6.5, -6.0, 5.0]), to_world([0, 0, 1.0]), up)
d_true = analytic_depth(M, K, W, H)
dm, nm = raymarch(vol, M, K, W, H, 40.0, 0.6 * voxel, 2.0, t_near=0.05)
dm = dm.cpu().numpy()
m = (dm > 0) & np.isfinite(d_true)
err = np.abs(dm - d_true)[m]
print(f"raymarch: hit {m.mean():.3f} of pixels, |dz| median {np.median(err):.4f}  p95 {np.percentile(err, 95):.4f}")

V2, F2, cs = drop_small_components(V, F, 0.5)
V3, F3 = cluster_decimate((V2 - c) @ R.T, F2, 3 * voxel)
print(f"components {cs['components']} kept {cs['kept']};  lite {len(V3)} verts {len(F3)} tris, area {tri_area(V3, F3).sum():.1f}")

# winding sanity: no triangle on the ground / roof may face down, none on the +x wall may face -x
print(f"flipped: ground {np.mean(n[ground, 2] < 0):.4f}  roof {np.mean(n[roof, 2] < 0):.4f}  wall {np.mean(n[wall, 0] < 0):.4f}; "
      f"ground n_z p5 {np.percentile(n[ground, 2], 5):.3f}")
