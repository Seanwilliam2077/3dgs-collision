"""Voxel collision for a 3DGS scene: voxelise the Gaussians, keep only what a character can walk to.

    py.bat voxel_collision.py <ply> --out out/vox_a --mpu 58 --center -0.55 -0.1 --size 100 100 --mode outdoor

No cameras or rendering needed; works the same way indoors and outdoors (only parameters differ).

1. Frame and chunk as in splat_to_collision.py (level frame from the Gaussians, chunk in XY).
2. Density: each Gaussian adds  opacity * exp(-1/2 d^T S'^-1 d)  to the voxels around it, where S' is
   its covariance widened by --inflate voxels (peak kept) so thin surfels are not missed between voxel
   centres. Two thresholds: floor-solid = density >= --opacity (0.1: real ground surfaces are thin and
   often only 0.1-0.5), obstacle-solid = density >= --obstacle-opacity (0.5: semi-transparent fog and
   floaters, which are just as faint as thin ground, must not wall the walkable area off).
3. Floaters: solid blobs under --min-blob m^3 are dropped.
4. Standable voxels: on top of floor-solid, with the character capsule (--radius, --height) clear of
   obstacle-solid, ignoring the lowest --step (it steps over that). Reachable = standable voxels
   connected to the seed by steps of at most --step (8-neighbourhood).
   Outdoor mode: floor under a large roof (> --roof-area) is cut away: an outdoor capture does not see
   building interiors, the splat only has occluded junk there. Feet more than --max-sink below the
   local ground are cut too: through holes in the ground the flood would walk down into the junk.
5. Floor fill (--floor-fill R metres, outdoor default 5): columns without a reachable floor within R
   of reachable ones get a floor at the interpolated height, unless something solid is at body height
   there; then step 4 is repeated.
6. Free space = the capsule swept over all reachable positions, extended upwards to the first obstacle,
   plus open air inside the walkable footprint above the topmost obstacle (and at least a body height
   above the local floor). Everything else is solid: holes in the floor, the space below it, unreached
   areas and everything outside the reconstruction. The boundary of the free space is the collision
   mesh: a character starting on it cannot fall through or walk out, by construction.
7. Camera volume (--camera 1): the free space at least --cam-clear from anything solid, at most
   --cam-max above the walkable floor and --cam-margin inside the walkable footprint, connected to a
   character's eye. A camera kept inside never shows the splat from inside a surface, from under the
   ground, from high above or from past the edge of the capture.

Outputs (splat frame, like the TSDF route): collision.ply (the shell: characters), camera.ply (the
camera volume boundary: cameras only), visual.ply (surface of the voxelised splat, for display),
camera_sdf.npz (signed distance to the camera volume, level frame; camera_volume.py samples and clamps
with it), reach.npz (reachable feet, level frame), voxels.npz, meta.json, debug/*.png.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy import ndimage

from gs_core import load_ply, estimate_frame, look_at, intrinsics, contribution
from tsdf_core import surface_nets, cluster_decimate, compact
from voxel_mesh import merged_faces, smooth_binary
try:
    import fast_simplification
except ImportError:
    fast_simplification = None

ap = argparse.ArgumentParser()
ap.add_argument("ply")
ap.add_argument("--out", required=True)
ap.add_argument("--mpu", type=float, default=1.0, help="metres per scene unit")
ap.add_argument("--up", type=float, nargs=3, default=None)
ap.add_argument("--center", type=float, nargs=2, default=None, help="chunk centre, level-frame x y in SCENE UNITS")
ap.add_argument("--size", type=float, nargs=2, default=[100.0, 100.0], help="chunk size in metres")
ap.add_argument("--mode", choices=["outdoor", "indoor", "object"], default="outdoor")
ap.add_argument("--source", choices=["density", "carve"], default="density",
                help="density: voxelise the Gaussians themselves (no cameras); carve: solid = everything views did not "
                     "see as free, from a TSDF run of splat_to_collision.py (--carve-run)")
ap.add_argument("--carve-run", default=None, help="run directory of splat_to_collision.py (tsdf.npz + meta.json)")
ap.add_argument("--carve-weight", type=float, default=2.0, help="carve: views needed to call a voxel seen")
ap.add_argument("--coarsen", type=int, default=1,
                help="carve: merge f x f x f voxels of the TSDF grid into one (majority solid), e.g. 2 for 0.3 m from 0.15 m")
ap.add_argument("--voxel", type=float, default=0.15, help="metres")
ap.add_argument("--opacity", type=float, default=0.1, help="density threshold for something to stand on")
ap.add_argument("--obstacle-opacity", type=float, default=0.5,
                help="density threshold for something that blocks the body; higher, so that semi-transparent fog "
                     "and floaters do not wall off the walkable area")
ap.add_argument("--inflate", type=float, default=0.5, help="widen each Gaussian by this many voxels (std)")
ap.add_argument("--min-blob", type=float, default=0.5, help="drop solid blobs smaller than this (m^3)")
ap.add_argument("--radius", type=float, default=0.35, help="character capsule radius (m)")
ap.add_argument("--height", type=float, default=1.8, help="character height (m)")
ap.add_argument("--step", type=float, default=0.45, help="max step height (m)")
ap.add_argument("--floor-fill", type=float, default=None, help="fill floor holes up to this far from reachable floor (m); "
                                                                "default 5 outdoor, 1 indoor, 0 object")
ap.add_argument("--regions", choices=["seed", "all"], default=None,
                help="walkable = the region reachable from --seed, or all standable regions of at least --min-region "
                     "(default: all outdoors, where areas are often joined only by drops; seed indoors)")
ap.add_argument("--min-region", type=float, default=50.0, help="smallest walkable region for --regions all (m^2)")
ap.add_argument("--seed", type=float, nargs="+", default=None,
                help="start position, level-frame x y [z] in SCENE UNITS; default: the most common floor near the centre")
ap.add_argument("--roof-area", type=float, default=None,
                help="outdoor: floor under a dense covered area larger than this (m^2) is a building interior "
                     "and not walkable; 0 = off (default 100 for density + outdoor, else off)")
ap.add_argument("--eave", type=float, default=2.0, help="outdoor: keep this much (m) under the edge of a roof")
ap.add_argument("--max-sink", type=float, default=None,
                help="outdoor: feet this far (m) below the median ground of their ~4 m surroundings are holes; 0 = off "
                     "(default 2 for density + outdoor, else off)")
ap.add_argument("--visibility", choices=["none", "aerial"], default="none",
                help="drop Gaussians no view sees before voxelising; 'aerial' = a virtual drone rig over the chunk")
ap.add_argument("--vis-min", type=float, default=0.5, help="minimum summed contribution (pixels) to count as seen")
ap.add_argument("--max-voxels", type=float, default=2.0e8)
ap.add_argument("--save-stages", type=int, default=0,
                help="1 = also write stages.npz (raw / cleaned / filled / standable / reach / free / shell), e.g. for videos")
ap.add_argument("--free", choices=["walk", "air"], default=None,
                help="free space = capsule swept over walkable feet (walk), or a capsule flood through the air above a "
                     "ground slab under the walkable floor (air: joins areas a character can only drop between, and "
                     "suits cameras). Default: air outdoors, walk indoors")
ap.add_argument("--mesh", choices=["faces", "qem", "smooth"], default="faces",
                help="collision mesh: faces = merged voxel faces (exact; walks best: UE's character steps up voxel "
                     "terraces cleanly but snags on smoothed bumps; small for flat rooms, large for terrain); "
                     "qem = smoothed surface + quadric decimation within --qem-tol (needs fast-simplification; "
                     "~5x lighter outdoors, more snags); smooth = smoothed surface + vertex clustering")
ap.add_argument("--qem-tol", type=float, default=0.11,
                help="qem: 99.9th percentile distance (m) allowed between the decimated and the smoothed surface")
ap.add_argument("--visual-reduction", type=float, default=0.95,
                help="fixed QEM reduction of the display mesh (when fast-simplification is installed)")
ap.add_argument("--smooth-shell", type=int, default=1,
                help="faces: rounds of 3x3x3 opening + closing on the shell before meshing (removes voxel noise)")
ap.add_argument("--decimate", type=float, default=3.0, help="smooth meshes: vertex clustering cell in voxels (0 = off)")
ap.add_argument("--camera", type=int, default=1, help="1 = also write the camera volume (camera.ply, camera_sdf.npz)")
ap.add_argument("--cam-clear", type=float, default=0.3,
                help="camera: minimum distance (m) from anything solid, so the near plane never enters the splat")
ap.add_argument("--cam-opacity", type=float, default=0.03,
                help="camera: density counted as visible for --cam-clear (density source); lower than --opacity, so the "
                     "camera also keeps out of sparse foliage and fog, faint per voxel but blurring the view up close "
                     "(final.ply: 0.1 -> 0.03 takes enclosed spots from 2.5%% to 1%% of the volume)")
ap.add_argument("--cam-max", type=float, default=None,
                help="camera: highest position (m) above the local walkable floor (highest reachable foot, interpolated); "
                     "0 = no cap (default 5 outdoors, where the capture thins out above the ground; else 0)")
ap.add_argument("--cam-margin", type=float, default=0.0,
                help="camera: stay this far (m) inside the walkable footprint horizontally (0: right up to its edge, "
                     "where a character can stand; with 1 m on final.ply only 85%% of the eyes stay inside, so a camera "
                     "following a character at the edge starts blocked)")
ap.add_argument("--cam-eye", type=float, default=None, help="camera: pivot (eye) height above a foot (m), default 0.9 x --height")
a = ap.parse_args()
if a.floor_fill is None:
    a.floor_fill = {"outdoor": 5.0, "indoor": 1.0, "object": 0.0}[a.mode]
if a.regions is None:
    a.regions = "all" if a.mode == "outdoor" else "seed"
if a.free is None:
    a.free = "air" if a.mode == "outdoor" else "walk"
dens_out = a.source == "density" and a.mode == "outdoor"        # heuristics only the density source needs
if a.roof_area is None:
    a.roof_area = 100.0 if dens_out else 0.0
if a.max_sink is None:
    a.max_sink = 2.0 if dens_out else 0.0
if a.cam_max is None:
    a.cam_max = 5.0 if a.mode == "outdoor" else 0.0
if a.cam_eye is None:
    a.cam_eye = 0.9 * a.height
if a.source == "carve" and not a.carve_run:
    raise SystemExit("--source carve needs --carve-run")
os.makedirs(f"{a.out}/debug", exist_ok=True)
dev = "cuda"
T0 = time.time()
U = 1.0 / a.mpu
v = a.voxel * U                                                          # voxel, scene units


def log(*s):
    print(f"[{time.time() - T0:6.1f}s]", *s, flush=True)


if a.source == "density":
    # ---- 1. frame and grid ---------------------------------------------------------------------------
    g = load_ply(a.ply, dev)
    xyz_np = g.means.cpu().numpy()
    c, R = estimate_frame(xyz_np, 3.0, a.up)
    Rt = torch.tensor(R, dtype=torch.float32, device=dev)
    ct = torch.tensor(c, dtype=torch.float32, device=dev)
    ml = (g.means - ct) @ Rt.T                                               # level frame
    cx, cy = a.center if a.center is not None else (0.0, 0.0)
    hx, hy = a.size[0] / 2 * U, a.size[1] / 2 * U
    inxy = (ml[:, 0] > cx - hx - 3 * U) & (ml[:, 0] < cx + hx + 3 * U) & (ml[:, 1] > cy - hy - 3 * U) & (ml[:, 1] < cy + hy + 3 * U)
    zs = ml[inxy, 2]
    zlo = float(torch.quantile(zs[torch.randperm(len(zs), device=dev)[:2_000_000]], 0.002)) - 2 * U
    zhi = float(torch.quantile(zs[torch.randperm(len(zs), device=dev)[:2_000_000]], 0.998)) + 4 * U
    lo = np.array([cx - hx, cy - hy, zlo])
    dims = np.ceil(np.array([2 * hx, 2 * hy, zhi - zlo]) / v).astype(int) + 1
    if np.prod(dims) > a.max_voxels:
        dims[2] = int(a.max_voxels // (dims[0] * dims[1]))
        log(f"grid capped in z to {dims[2] * a.voxel:.0f} m")
    nx, ny, nz = (int(d) for d in dims)
    log(f"{len(g)} Gaussians, {int(inxy.sum())} in the chunk; grid {nx}x{ny}x{nz} = {nx * ny * nz / 1e6:.0f}M voxels "
        f"at {a.voxel} m, z {zlo * a.mpu:.1f} .. {(zlo + nz * v) * a.mpu:.1f} m (level)")

    # ---- 1b. visibility: drop Gaussians that no view sees ---------------------------------------------
    # Optimisation leaves big Gaussians wherever nothing constrains them: under the ground, inside
    # buildings an outdoor capture never saw, behind walls. They render invisibly but are as dense as real
    # surfaces. Weight each Gaussian by what it actually contributes to a set of views and drop the unseen.
    vis_kept = None
    if a.visibility == "aerial":
        zg = float(torch.quantile(zs[torch.randperm(len(zs), device=dev)[:2_000_000]], 0.3))   # rough ground level
        e1, e2, upv = R[0], R[1], R[2]
        to_w = lambda p: c + np.asarray(p, dtype=np.float64) @ R
        Wv, Hv = 480, 360
        Kv = intrinsics(Wv, Hv, 70)
        mats = []
        for ti, (alt, pitch) in enumerate([(60.0, 45.0), (25.0, 35.0)]):
            reach_ = alt / np.tan(np.radians(pitch)) * U
            step = alt * 0.6 * U
            for i_, x in enumerate(np.arange(cx - hx - reach_, cx + hx + reach_ + 1e-9, step)):
                for j_, y in enumerate(np.arange(cy - hy - reach_, cy + hy + reach_ + 1e-9, step)):
                    eye = np.array([x, y, zg + alt * U])
                    if ti == 0:
                        M = np.eye(4)
                        M[:3, :3] = np.stack([e1, -e2, -upv], 0)
                        M[:3, 3] = -M[:3, :3] @ to_w(eye)
                        mats.append(M)
                    for k4 in range(4):
                        yr, pr = np.radians(90 * k4 + (45 if (i_ + j_) % 2 else 0)), np.radians(pitch)
                        d_l = np.array([np.cos(pr) * np.cos(yr), np.cos(pr) * np.sin(yr), -np.sin(pr)])
                        mats.append(look_at(to_w(eye), to_w(eye + d_l), upv))
        sel_v = inxy | True                                                      # all Gaussians can occlude
        VM = torch.tensor(np.stack(mats), dtype=torch.float32, device=dev)
        KM = torch.tensor(Kv, dtype=torch.float32, device=dev)[None].expand(len(mats), 3, 3).contiguous()
        contrib = contribution(g, VM, KM, Wv, Hv, near=0.5 * U)
        vis_kept = contrib >= a.vis_min
        nin = int(inxy.sum())
        log(f"visibility: {len(mats)} aerial views; in the chunk {int((vis_kept & inxy).sum())} of {nin} Gaussians seen "
            f"(contribution >= {a.vis_min} px), {nin - int((vis_kept & inxy).sum())} dropped")

    # ---- 2. density ----------------------------------------------------------------------------------
    keep = inxy & (ml[:, 2] > zlo - 3 * U) & (ml[:, 2] < zlo + nz * v + 3 * U) & (g.opac > a.opacity / 8)
    if vis_kept is not None:
        keep &= vis_kept
    mu, q, s, al = ml[keep], g.quats[keep], g.scales[keep], g.opac[keep]
    w_, x_, y_, z_ = q.unbind(-1)
    Rg = torch.stack([1 - 2 * (y_ ** 2 + z_ ** 2), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_),
                      2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ ** 2 + z_ ** 2), 2 * (y_ * z_ - w_ * x_),
                      2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ ** 2 + y_ ** 2)], -1).view(-1, 3, 3)
    Rl = Rt[None] @ Rg
    cov = Rl @ torch.diag_embed(s ** 2) @ Rl.transpose(1, 2) + torch.eye(3, device=dev) * (a.inflate * v) ** 2
    icov = torch.linalg.inv(cov)
    ksig = torch.sqrt(2 * torch.log(al / (a.opacity / 8))).clamp(max=3.5)
    half = torch.ceil(ksig[:, None] * torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2)) / v).long()   # voxels
    loT = torch.tensor(lo, dtype=torch.float32, device=dev)
    ci = torch.round((mu - loT) / v).long()
    B = torch.tensor([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64], device=dev)
    bxy = torch.bucketize(torch.maximum(half[:, 0], half[:, 1]).clamp(max=64), B)
    bz = torch.bucketize(half[:, 2].clamp(max=64), B)
    log(f"{len(mu)} Gaussians splatted; {int((half.max(1).values > 64).sum())} truncated at 64 voxels")
    dens = torch.zeros(nx * ny * nz, dtype=torch.float32, device=dev)
    work = 0
    for i_b in range(len(B)):
        for j_b in range(len(B)):
            sel = ((bxy == i_b) & (bz == j_b)).nonzero()[:, 0]
            if len(sel) == 0:
                continue
            rx, rz = int(B[i_b]), int(B[j_b])
            ax = torch.arange(-rx, rx + 1, device=dev)
            az = torch.arange(-rz, rz + 1, device=dev)
            O = torch.stack(torch.meshgrid(ax, ax, az, indexing="ij"), -1).view(-1, 3)
            bs = max(1, int(16_000_000 // len(O)))
            for s0 in range(0, len(sel), bs):
                ids = sel[s0:s0 + bs]
                idx = ci[ids, None, :] + O[None]                                 # [b, n, 3]
                d = loT + idx.float() * v - mu[ids, None, :]
                qf = (torch.einsum("bni,bij->bnj", d, icov[ids]) * d).sum(-1)
                val = al[ids, None] * torch.exp(-0.5 * qf)
                ok = (val > a.opacity / 8) & (idx[..., 0] >= 0) & (idx[..., 0] < nx) & (idx[..., 1] >= 0) & (idx[..., 1] < ny) \
                    & (idx[..., 2] >= 0) & (idx[..., 2] < nz)
                flat = (idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]
                dens.index_add_(0, flat[ok], val[ok])
                work += idx.shape[0] * idx.shape[1]
    dens = dens.view(nx, ny, nz)
    solid = dens >= a.opacity
    log(f"density done ({work / 1e9:.2f}G evaluations); floor-solid {solid.float().mean() * 100:.2f}%, "
        f"obstacle-solid {(dens >= a.obstacle_opacity).float().mean() * 100:.2f}% of voxels")
    del mu, q, s, al, Rg, Rl, cov, icov, ci, g

    # ---- 3. floaters ---------------------------------------------------------------------------------
    solid_raw = solid.clone() if a.save_stages else None                     # for videos: before clean-up
    lab, n = ndimage.label(solid.cpu().numpy())
    cnt = np.bincount(lab.ravel(), minlength=n + 1)
    cnt[0] = 0
    small = cnt * a.voxel ** 3 < a.min_blob
    solid = torch.from_numpy(~small[lab] & (lab > 0)).to(dev)
    log(f"blobs: {n} -> kept {int((~small[1:]).sum())}; dropped {cnt[small].sum() * a.voxel ** 3:.0f} m^3 in small ones")
    del lab
    obst = solid & (dens >= a.obstacle_opacity)                               # what blocks the body
else:                                                                        # carve: seen-free space from a TSDF run
    tm = json.load(open(f"{a.carve_run}/meta.json"))
    assert abs(tm["mpu"] - a.mpu) < 1e-9, "--mpu differs from the carve run"
    c, R, lo = np.array(tm["c"]), np.array(tm["R"]), np.array(tm["lo"])
    nx, ny, nz = (int(d) for d in tm["dims"])
    v = tm["voxel_units"]
    a.voxel = v * a.mpu
    cx, cy = tm["args"]["center"]
    a.center, a.size = [cx, cy], tm["args"]["size"]
    hx, hy = nx * v / 2, ny * v / 2
    zt = np.load(f"{a.carve_run}/tsdf.npz")
    # free = seen in front of the fused surface by enough views; everything else (behind surfaces, never
    # seen: under the ground, inside buildings, beyond the capture) is solid
    seen_free = (zt["weight"] >= a.carve_weight) & (zt["tsdf"].astype(np.float32) >= 0)
    del zt
    solid = torch.from_numpy(~seen_free).to(dev)
    del seen_free
    if a.coarsen > 1:
        # f x f x f blocks; a block is solid when at least half of its voxels are (surfaces stay centred)
        f_ = a.coarsen
        cnx, cny, cnz = nx // f_, ny // f_, nz // f_
        blk = solid[:cnx * f_, :cny * f_, :cnz * f_].reshape(cnx, f_, cny, f_, cnz, f_)
        solid = blk.sum((1, 3, 5), dtype=torch.int16) * 2 >= f_ ** 3
        del blk
        lo = lo + (f_ - 1) / 2 * v                                           # centre of the first block
        nx, ny, nz, v = cnx, cny, cnz, v * f_
        a.voxel = v * a.mpu
    obst = solid
    dens = None
    log(f"carve source {a.carve_run}: grid {nx}x{ny}x{nz} at {a.voxel:.3f} m; seen free {(~solid).float().mean() * 100:.1f}% of voxels")

if a.save_stages:
    solid_clean = solid.clone()                                              # after clean-up, before floor fill
    if globals().get("solid_raw") is None:
        solid_raw = solid_clean

# ---- 4. standable and reachable --------------------------------------------------------------------
S = int(a.step / a.voxel + 1e-6)                                            # whole voxels the character can step
Hc = int(np.ceil(a.height / a.voxel))
rv = a.radius / a.voxel
kr = int(np.ceil(rv))
yy, xx = np.mgrid[-kr:kr + 1, -kr:kr + 1]
disk = torch.tensor((xx ** 2 + yy ** 2 <= rv ** 2 + 0.25).astype(np.float32), device=dev)[None, None]


def dilate_xy(m):
    """Dilate a [nx,ny,nz] bool volume in XY by the capsule radius (per z slice)."""
    out = torch.empty_like(m)
    for k0 in range(0, nz, 64):
        sl = m[:, :, k0:k0 + 64].permute(2, 0, 1)[:, None].half()
        out[:, :, k0:k0 + 64] = (F.conv2d(sl, disk.half(), padding=kr)[:, 0] > 0).permute(1, 2, 0)
    return out


def standable_of(sol, obs):
    """Feet stand on the top of floor-solid `sol`; the body must be clear of obstacle-solid `obs`."""
    near = dilate_xy(obs)                                                    # obstacle within the capsule radius
    cs = torch.zeros((nx, ny, nz + 1), dtype=torch.int16, device=dev)
    cs[:, :, 1:] = torch.cumsum(near.to(torch.int16), 2)
    st = torch.zeros_like(sol)
    kk = nz - Hc
    # foot voxel k: solid at k-1, free at k, nothing near in k+S .. k+Hc-1
    st[:, :, 1:kk] = sol[:, :, 0:kk - 1] & ~sol[:, :, 1:kk] & ((cs[:, :, 1 + Hc:kk + Hc] - cs[:, :, 1 + S:kk + S]) == 0)
    return st


def reach_from(st, seed_idx):
    """Flood fill over standable voxels: 8-neighbours in XY, up to S voxels up or down."""
    r = torch.zeros((1, 1, nx, ny, nz), dtype=torch.float16, device=dev)
    r[0, 0, seed_idx[0], seed_idx[1], seed_idx[2]] = 1
    stf = st[None, None].half()
    prev, it = -1, 0
    while True:
        for _ in range(25):
            r = F.max_pool3d(r, (3, 3, 2 * S + 1), stride=1, padding=(1, 1, S)) * stf
        it += 25
        cur = int((r > 0).sum())
        if cur == prev:
            break
        prev = cur
    return r[0, 0] > 0, it


def pick_seed(st):
    if a.seed is not None and len(a.seed) == 3:
        p = (np.array(a.seed) - lo) / v
        cand = st.nonzero()
        d = ((cand.float() - torch.tensor(p, device=dev).float()) ** 2).sum(1)
        return cand[d.argmin()].tolist()
    sx, sy = (a.seed if a.seed is not None else (cx, cy))
    i0, j0 = int((sx - lo[0]) / v), int((sy - lo[1]) / v)
    rr = int(6.0 / a.voxel)                                                  # look within 6 m
    sub = st[max(0, i0 - rr):i0 + rr, max(0, j0 - rr):j0 + rr]
    ks = sub.nonzero()
    if len(ks) == 0:
        raise SystemExit("no standable voxel near the seed; pass --seed x y z")
    hist = torch.bincount(ks[:, 2], minlength=nz).float()
    hist = F.conv1d(hist[None, None], torch.ones(1, 1, 2 * S + 1, device=dev), padding=S)[0, 0]
    kmode = int(hist.argmax())                                               # the most common floor level
    ks = ks[(ks[:, 2] - kmode).abs() <= S]
    d = (ks[:, 0] + max(0, i0 - rr) - i0) ** 2 + (ks[:, 1] + max(0, j0 - rr) - j0) ** 2
    k = ks[d.argmin()]
    return [int(k[0]) + max(0, i0 - rr), int(k[1]) + max(0, j0 - rr), int(k[2])]


def regions_of(st_):
    """All walkable regions of at least --min-region m^2: connected components of the standable voxels
    (8-neighbours, steps up to S), with feet thickened by one voxel up and down and 26-connectivity."""
    th = st_.clone()
    th[:, :, 1:] |= st_[:, :, :-1]
    th[:, :, :-1] |= st_[:, :, 1:]
    lab3, n3 = ndimage.label(th.cpu().numpy(), structure=np.ones((3, 3, 3)))
    lab3 = torch.from_numpy(lab3).to(dev)
    del th
    lf = lab3[st_]
    nzc = st_.nonzero()
    colid = nzc[:, 0] * ny + nzc[:, 1]
    # area of a region = number of distinct columns among its feet
    key = torch.unique(lf.long() * (nx * ny) + colid)
    area_ = torch.bincount(key // (nx * ny), minlength=n3 + 1).float() * a.voxel ** 2
    keep_ = area_ >= a.min_region
    keep_[0] = False
    out = st_ & keep_[lab3]
    return out, int(keep_.sum()), area_


def reach_all(st_, seed_idx):
    if a.regions == "all":
        r_, nk, ar = regions_of(st_)
        return r_, f"{nk} regions >= {a.min_region} m^2"
    r_, it_ = reach_from(st_, seed_idx)
    return r_, f"{it_} iterations from the seed"


st = standable_of(solid, obst)
seed = pick_seed(st)
reach, its = reach_all(st, seed)
col = reach.any(2)
log(f"standable {int(st.any(2).sum()) * a.voxel ** 2:.0f} m^2 (columns); seed {seed} at "
    f"{(lo[2] + seed[2] * v) * a.mpu:.1f} m; reachable {int(col.sum()) * a.voxel ** 2:.0f} m^2 ({its})")

# ---- 4b. outdoor: not under large roofs -----------------------------------------------------------
# An outdoor capture does not see inside buildings; what is there is occluded junk. Reachable floor
# with a surface above head height over a large connected area (> --roof-area) is a building
# interior and is cut away, except within --eave of the open. (Large tree canopies count too.)
roofed = torch.zeros((nx, ny), dtype=torch.bool, device=dev)
if a.mode == "outdoor" and a.roof_area > 0:
    kk = torch.arange(nz, device=dev)
    kfeet = torch.where(reach, kk, torch.full_like(kk, -1)).amax(2).clamp(min=0)  # highest reachable foot
    # (not the lowest: a pit under a hole in the street would see the street itself as its roof)
    # any surface counts as cover (roofs are thin and often < --obstacle-opacity, like the ground)
    above = torch.flip(torch.cumsum(torch.flip(solid.to(torch.int16), [2]), 2), [2]) > 0  # surface at or above k
    khead = (kfeet + Hc).clamp(max=nz - 1)
    covered = col & torch.gather(above, 2, khead[:, :, None])[:, :, 0]
    del above
    # a roof is continuous; fog and floaters above a street are patchy: open with a 2 m disk first
    rr2 = int(round(2.0 / a.voxel))
    yy2, xx2 = np.mgrid[-rr2:rr2 + 1, -rr2:rr2 + 1]
    covered = ndimage.binary_opening(covered.cpu().numpy(), structure=(xx2 ** 2 + yy2 ** 2 <= rr2 ** 2))
    lab2, n2 = ndimage.label(covered, structure=np.ones((3, 3)))
    area2 = np.bincount(lab2.ravel(), minlength=n2 + 1) * a.voxel ** 2
    area2[0] = 0
    big = (area2 > a.roof_area)[lab2]
    deep = ndimage.distance_transform_edt(big) * a.voxel > a.eave
    roofed = torch.from_numpy(deep).to(dev)
    st &= ~roofed[:, :, None]
    reach, its = reach_all(st, seed)
    col = reach.any(2)
    log(f"roof cut: {int((area2 > a.roof_area).sum())} covered areas > {a.roof_area} m^2, {deep.sum() * a.voxel ** 2:.0f} m^2 "
        f"under them cut; reachable now {int(col.sum()) * a.voxel ** 2:.0f} m^2")

# ---- 4c. outdoor: no sinking into holes -----------------------------------------------------------
# Under a hole in the ground the splat often has occluded junk with sloping tops; a character can walk
# down into it in small steps. Feet more than --max-sink below the median ground of their surroundings
# (~4 m) are not standable; floor fill then closes those holes at the surrounding height.
sunk_m2 = 0.0
sunk = torch.zeros_like(st)
if a.mode == "outdoor" and a.max_sink > 0:
    kk = torch.arange(nz, device=dev)
    htop = torch.where(reach, kk, torch.full_like(kk, -1)).amax(2).float().cpu().numpy()
    htop[htop < 0] = np.nan
    bsz = 4                                                                   # 0.6 m blocks
    px_, py_ = (-nx) % bsz, (-ny) % bsz
    hp = np.pad(htop, ((0, px_), (0, py_)), constant_values=np.nan)
    blk = hp.reshape(hp.shape[0] // bsz, bsz, hp.shape[1] // bsz, bsz).transpose(0, 2, 1, 3).reshape(hp.shape[0] // bsz, hp.shape[1] // bsz, -1)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        cm = np.nanmedian(blk, axis=2)
        ref = ndimage.generic_filter(cm, np.nanmedian, size=7, mode="constant", cval=np.nan)
    ref = np.repeat(np.repeat(ref, bsz, 0), bsz, 1)[:nx, :ny]
    lim = torch.from_numpy(np.nan_to_num(ref - a.max_sink / a.voxel, nan=-1e9)).to(dev).float()
    sunk = st & (kk[None, None, :] < lim[:, :, None])
    sunk_m2 = int((reach & sunk).any(2).sum()) * a.voxel ** 2
    st &= ~sunk
    reach, its = reach_all(st, seed)
    col = reach.any(2)
    log(f"sink cut: feet > {a.max_sink} m below the local ground in {sunk_m2:.0f} m^2 of columns; "
        f"reachable now {int(col.sum()) * a.voxel ** 2:.0f} m^2")

# ---- 5. floor fill ---------------------------------------------------------------------------------
filled_m2 = 0.0
if a.floor_fill > 0:
    kk = torch.arange(nz, device=dev)
    kf = torch.where(reach, kk, torch.full_like(kk, nz)).amin(2).float()     # lowest reachable foot per column
    known = col.cpu().numpy()
    Fh = np.where(known, kf.cpu().numpy(), 0.0)
    num, den = np.zeros_like(Fh), np.zeros_like(Fh)
    for sig in (1.0, 2.0, 4.0, 8.0, 16.0):
        n_ = ndimage.gaussian_filter(Fh * known, sig)
        d_ = ndimage.gaussian_filter(known.astype(float), sig)
        take = (den < 0.05) & (d_ > 1e-6)
        num[take], den[take] = n_[take], d_[take]
    dist = ndimage.distance_transform_edt(~known) * a.voxel
    cand = ~known & (dist <= a.floor_fill) & (den > 0) & ~roofed.cpu().numpy()
    kfill = np.round(num / np.maximum(den, 1e-12)).astype(int)
    ii, jj = np.nonzero(cand)
    kf_ = np.clip(kfill[ii, jj], 1, nz - Hc - 1)
    near = dilate_xy(obst).cpu().numpy()
    cs = np.zeros((nx, ny, nz + 1), np.int16)
    np.cumsum(near, axis=2, dtype=np.int16, out=cs[:, :, 1:])
    del near
    blocked = (cs[ii, jj, kf_ + Hc] - cs[ii, jj, kf_ + S]) > 0                 # something at body height
    del cs
    ii, jj, kf_ = ii[~blocked], jj[~blocked], kf_[~blocked]
    sol = solid.cpu().numpy()
    if a.save_stages:
        filled_vox = np.zeros_like(sol)
        filled_vox[ii, jj, kf_ - 1] = ~sol[ii, jj, kf_ - 1]
    sol[ii, jj, kf_ - 1] = True                                              # one-voxel floor under the new foot
    solid = torch.from_numpy(sol).to(dev)
    del sol
    filled_m2 = len(ii) * a.voxel ** 2
    st = standable_of(solid, obst) & ~roofed[:, :, None] & ~sunk
    reach, its = reach_all(st, seed)
    col = reach.any(2)
    log(f"floor fill (<= {a.floor_fill} m): {filled_m2:.0f} m^2 of floor added, {int(blocked.sum()) * a.voxel ** 2:.0f} m^2 "
        f"left open (obstacle at body height); reachable now {int(col.sum()) * a.voxel ** 2:.0f} m^2")

# ---- 6. free space and shell ---------------------------------------------------------------------
kk = torch.arange(nz, device=dev)
foot2 = col.cpu().numpy()
kf2 = torch.where(reach, kk, torch.full_like(kk, nz)).amin(2).float().cpu().numpy()   # lowest walkable foot


def extrapolate(h, known, sigmas):
    """Normalised Gaussian interpolation of h over the whole grid, finest scale with support; nan where none."""
    Fh = np.where(known, h, 0.0)
    num, den = np.zeros_like(Fh), np.zeros_like(Fh)
    for sig in sigmas:
        n_ = ndimage.gaussian_filter(Fh * known, sig)
        d_ = ndimage.gaussian_filter(known.astype(float), sig)
        take = (den < 0.05) & (d_ > 1e-9)
        num[take], den[take] = n_[take], d_[take]
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


kfloor = extrapolate(kf2, foot2, (2.0, 8.0, 32.0))

if a.free == "air":
    # Ground slab: everything below the walkable floor is solid, extrapolated over the whole chunk
    # (columns with no walkable floor within ~20 m are solid to the top). Unlike filling each column
    # from below up to its first surface, this does not stop at occluded junk under a hole in the ground.
    kslab = np.where(foot2, kf2, np.where(np.isnan(kfloor), nz, np.round(np.nan_to_num(kfloor))))
    blocked = obst | (kk[None, None, :] < torch.from_numpy(kslab).to(dev)[:, :, None])
    near = dilate_xy(blocked)
    del blocked
    cs = torch.zeros((nx, ny, nz + 1), dtype=torch.int16, device=dev)
    cs[:, :, 1:] = torch.cumsum(near.to(torch.int16), 2)
    del near
    clear = torch.zeros((nx, ny, nz), dtype=torch.bool, device=dev)
    clear[:, :, :nz - Hc + 1] = (cs[:, :, Hc:] - cs[:, :, :nz - Hc + 1]) == 0  # capsule at k .. k+Hc-1 fits
    del cs
    b = 2 + kr                                                                # keep capsules off the chunk border
    clear[:b], clear[-b:], clear[:, :b], clear[:, -b:] = False, False, False, False
    i0, j0, k0 = seed
    ks_ = clear[i0, j0, k0:].nonzero()
    if len(ks_) == 0:
        raise SystemExit("air carve: no room for the capsule above the seed")
    r = torch.zeros((1, 1, nx, ny, nz), dtype=torch.float16, device=dev)
    r[0, 0, i0, j0, k0 + int(ks_[0])] = 1
    clf = clear[None, None].half()
    prev, it = -1, 0
    while True:                                                               # capsule flood through the air
        for _ in range(25):
            r = F.max_pool3d(r, 3, stride=1, padding=1) * clf
        it += 25
        cur = int((r > 0).sum())
        if cur == prev:
            break
        prev = cur
    del clf, clear
    body = F.max_pool3d(F.pad(r, (Hc - 1, 0)), (1, 1, Hc), stride=1)[0, 0] > 0   # capsule .. capsule+H
    del r
    free = dilate_xy(body)
    del body
    log(f"air carve: capsule flood from the seed in {it} iterations")
else:
    # walk: the capsule swept over the walkable feet, open sky above them
    body = F.max_pool3d(F.pad(reach[None, None].half(), (Hc - 1, 0)), (1, 1, Hc), stride=1)[0, 0] > 0
    free = dilate_xy(body)
    del body
    up = free.clone()
    for k in range(1, nz):                                                   # open sky stays open
        up[:, :, k] |= up[:, :, k - 1] & ~obst[:, :, k]
    free = up
    del up
    # Open air over everything inside the walkable footprint (cars, trees, roofs are boxes on the ground,
    # not pillars to the sky): above the topmost obstacle and at least a body height above the local
    # floor, so that a floor hole stays closed and only becomes a head-high bump.
    inside = ndimage.binary_fill_holes(ndimage.binary_closing(foot2, structure=np.ones((3, 3)), iterations=3))
    kfl = np.nan_to_num(kfloor, nan=nz)
    ktop = torch.where(obst, kk, torch.full_like(kk, -1)).amax(2).cpu().numpy()
    kair = np.maximum(ktop + 1, np.ceil(kfl + Hc))
    # smooth the air boundary (it only matters for cameras): ~1 m grey opening + closing drops thin spikes
    # and small dips, which would otherwise make a ragged ceiling of fog and canopy tops
    kair = ndimage.grey_closing(ndimage.grey_opening(kair, size=7), size=7)
    kair = np.maximum(kair, np.ceil(kfl + Hc)).astype(np.int64)
    kair = np.where(inside, kair, nz)
    free |= kk[None, None, :] >= torch.from_numpy(kair).to(dev)[:, :, None]
free[:2], free[-2:], free[:, :2], free[:, -2:] = False, False, False, False   # walls on the chunk border
shell = ~free
log(f"free space {free.float().mean() * 100:.1f}% of the grid ({a.free})")

# ---- 6b. camera volume -----------------------------------------------------------------------------
# A camera is not a character: it flies, its near plane must stay out of the splat, and it must stay
# where the capture is good (not high above the scene, not past its edges). Allowed = free space, at
# least --cam-clear from anything visible (floor-solid too: thin ground, foliage), at most --cam-max
# above the walkable floor, over the walkable footprint (minus --cam-margin), and only the parts
# connected to a character's eye (pockets no camera can get to are dropped). Its boundary is camera.ply (collision
# for the camera only), its signed distance camera_sdf.npz (for clamping a camera without an engine).
cam, cam_stats = None, {}


def dilate_ball(m, r_):
    """Dilate a [nx,ny,nz] bool volume by a ball of radius r_ voxels (chunks along x)."""
    rb = int(np.ceil(r_ - 1e-6))
    if rb <= 0:
        return m.clone()
    ax_ = np.arange(-rb, rb + 1)
    X_, Y_, Z_ = np.meshgrid(ax_, ax_, ax_, indexing="ij")
    ker = torch.tensor((X_ ** 2 + Y_ ** 2 + Z_ ** 2 <= r_ ** 2 + 0.25).astype(np.float32), device=dev).half()[None, None]
    out = torch.empty_like(m)
    for i0 in range(0, nx, 64):
        i1 = min(nx, i0 + 64)
        a0, a1 = max(0, i0 - rb), min(nx, i1 + rb)
        sl = F.pad(m[a0:a1].half()[None, None], (rb, rb, rb, rb, rb - (i0 - a0), rb - (a1 - i1)))
        out[i0:i1] = F.conv3d(sl, ker)[0, 0] > 0
    return out


if a.camera:
    vox3 = a.voxel ** 3
    # clearance from what is visible (floor-solid and obstacles); the edge of the free space itself is a
    # reachability bound, already a capsule radius from obstacles, so the camera may go right up to it
    vis = solid | obst
    if dens is not None and a.cam_opacity < a.opacity:
        faint = (dens >= a.cam_opacity).cpu().numpy()
        lab_f, n_f = ndimage.label(faint)
        cnt_f = np.bincount(lab_f.ravel(), minlength=n_f + 1)
        cnt_f[0] = 0
        vis |= torch.from_numpy((cnt_f * vox3 >= a.min_blob)[lab_f] & (lab_f > 0)).to(dev)
        del lab_f, faint
    cam = free & ~dilate_ball(vis, a.cam_clear / a.voxel)
    del vis
    n_clear = int(cam.sum())
    foot_c = ndimage.binary_fill_holes(ndimage.binary_closing(foot2, structure=np.ones((3, 3)), iterations=3))
    if a.cam_margin > 0:
        foot_c = ndimage.distance_transform_edt(foot_c) * a.voxel > a.cam_margin
    cam &= torch.from_numpy(foot_c).to(dev)[:, :, None]
    n_foot = int(cam.sum())
    if a.cam_max > 0:
        khi = torch.where(reach, kk, torch.full_like(kk, -1)).amax(2).float().cpu().numpy()   # highest walkable foot
        khi = extrapolate(khi, foot2, (2.0, 8.0, 32.0))
        kcap = np.where(np.isnan(khi), -1, np.floor(khi + a.cam_max / a.voxel)).astype(np.int64)
        cam &= kk[None, None, :] <= torch.from_numpy(kcap).to(dev)[:, :, None]
    n_cap = int(cam.sum())
    # character eyes: feet in the free space with room for the body, --cam-eye up
    okf = reach & free
    okf[:, :, :nz - Hc + 1] &= free[:, :, Hc - 1:]
    eye = okf.nonzero().cpu().numpy()
    del okf
    eye[:, 2] = np.minimum(eye[:, 2] + int(round(a.cam_eye / a.voxel)), nz - 1)
    camn = cam.cpu().numpy()
    lab_c, n_c = ndimage.label(camn, structure=np.ones((3, 3, 3)))
    le = lab_c[eye[:, 0], eye[:, 1], eye[:, 2]]
    keep_c = np.zeros(n_c + 1, bool)
    keep_c[np.unique(le[le > 0])] = True
    camn = keep_c[lab_c]
    del lab_c
    cam = torch.from_numpy(camn).to(dev)
    eye_in = float((le > 0).mean()) if len(le) else 0.0
    cam_stats = dict(volume_m3=float(camn.sum()) * vox3, eye_inside=eye_in, components_kept=int(keep_c.sum()),
                     components=int(n_c), clear_m=a.cam_clear, max_m=a.cam_max, margin_m=a.cam_margin, eye_m=a.cam_eye)
    log(f"camera volume: {n_clear * vox3:.0f} m^3 with {a.cam_clear} m clearance -> {n_foot * vox3:.0f} m^3 inside the "
        f"footprint (margin {a.cam_margin} m) -> {n_cap * vox3:.0f} m^3 up to {a.cam_max or 'any'} m above the floor -> "
        f"{camn.sum() * vox3:.0f} m^3 in {int(keep_c.sum())} of {n_c} pieces reached from an eye; "
        f"{eye_in * 100:.1f}% of the eyes ({a.cam_eye:.2f} m up) are inside")
    if eye_in < 0.9:
        log("WARNING: many character eyes are outside the camera volume; a camera pivoting there starts blocked "
            "(lower --cam-clear or --cam-eye)")

# ---- meshes --------------------------------------------------------------------------------------
def gblur(x, sig=0.7):
    r_ = int(np.ceil(2.5 * sig))
    t = torch.arange(-r_, r_ + 1, device=dev).float()
    k1 = torch.exp(-0.5 * (t / sig) ** 2)
    k1 = (k1 / k1.sum()).half()
    y = x[None, None].half()
    y = F.conv3d(F.pad(y, (0, 0, 0, 0, r_, r_), mode="replicate"), k1.view(1, 1, -1, 1, 1))
    y = F.conv3d(F.pad(y, (0, 0, r_, r_, 0, 0), mode="replicate"), k1.view(1, 1, 1, -1, 1))
    y = F.conv3d(F.pad(y, (r_, r_, 0, 0, 0, 0), mode="replicate"), k1.view(1, 1, 1, 1, -1))
    return y[0, 0].float()


def mesh_of(inside, path, label):
    sdf = (0.5 - gblur(inside.float())).cpu()
    V, Fc = surface_nets(sdf, torch.ones_like(sdf, dtype=torch.bool))
    n0 = len(Fc)
    if a.decimate > 0:
        V, Fc = cluster_decimate(V, Fc, a.decimate)
    Vw = c + (lo + V * v) @ R                                                # index -> level -> splat frame
    trimesh.Trimesh(Vw, Fc, process=False).export(path)
    log(f"[{label}] {n0} -> {len(Fc)} triangles -> {path}")
    return Vw, Fc


def faces_of(S, path, label):
    V, Fc, per = merged_faces(S)
    Vw = c + (lo + V * v) @ R                                                # index -> level -> splat frame
    trimesh.Trimesh(Vw, Fc, process=False).export(path)
    log(f"[{label}] {sum(per.values())} merged faces ({per}) -> {len(Fc)} triangles, {len(V)} vertices -> {path}")
    return Vw, Fc


QEM_STEPS = (0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.96, 0.95, 0.9, 0.8)


def qem_of(inside, path, label, tol, steps=None):
    """Surface nets on the smoothed indicator, then quadric decimation (fast-simplification) as far as
    the 99.9th percentile distance of the result from the smoothed iso-surface stays within tol voxels.
    The distance is |sdf| / |grad sdf| sampled at 1M points on the decimated surface."""
    sdf = 0.5 - gblur(inside.float())
    V, Fc = surface_nets(sdf.cpu(), torch.ones((nx, ny, nz), dtype=torch.bool))
    gm = torch.sqrt(sum(g_ ** 2 for g_ in torch.gradient(sdf))).clamp(min=1e-3)
    vol = torch.stack([sdf, gm])[None]
    del gm, sdf

    def deviation(Vd, Fd):
        p, _ = trimesh.sample.sample_surface(trimesh.Trimesh(Vd, Fd, process=False), 1_000_000)
        q = torch.tensor(p, dtype=torch.float32, device=dev)
        gg = torch.stack([q[:, 2] / (nz - 1), q[:, 1] / (ny - 1), q[:, 0] / (nx - 1)], 1) * 2 - 1
        s = F.grid_sample(vol, gg.view(1, -1, 1, 1, 3), align_corners=True)[0, :, :, 0, 0]
        return np.percentile((s[0].abs() / s[1]).cpu().numpy(), [99, 99.9])

    Fi = Fc.astype(np.int32)
    for red in (steps or QEM_STEPS):                                         # most aggressive first
        Vd, Fd = fast_simplification.simplify(V, Fi, target_reduction=red, agg=7, preserve_border=True)
        d = deviation(Vd, Fd)
        if d[1] <= tol:
            break
    del vol
    Vw = c + (lo + Vd * v) @ R                                               # index -> level -> splat frame
    trimesh.Trimesh(Vw, Fd, process=False).export(path)
    log(f"[{label}] {len(Fc)} -> {len(Fd)} triangles (QEM {red}); distance from the iso-surface p99 "
        f"{d[0] * a.voxel * 100:.1f} cm, p99.9 {d[1] * a.voxel * 100:.1f} cm -> {path}")
    return Vw, Fd


mesh_mode = a.mesh
if mesh_mode == "qem" and fast_simplification is None:
    log("fast-simplification is not installed: falling back to --mesh faces")
    mesh_mode = "faces"
sh = smooth_binary(shell, a.smooth_shell) if a.smooth_shell > 0 and mesh_mode != "smooth" else shell.clone()
sh[:2], sh[-2:], sh[:, :2], sh[:, -2:] = True, True, True, True               # keep the border walls
if mesh_mode == "faces":
    faces_of(sh, f"{a.out}/collision.ply", "collision")
elif mesh_mode == "qem":
    qem_of(sh, f"{a.out}/collision.ply", "collision", a.qem_tol / a.voxel)
else:
    mesh_of(sh, f"{a.out}/collision.ply", "collision")
del sh
if cam is not None and bool(cam.any()):
    # smooth, not voxel faces: a camera slides along this surface and has no steps to climb
    if fast_simplification is not None:
        qem_of(~cam, f"{a.out}/camera.ply", "camera", a.qem_tol / a.voxel)
    else:
        mesh_of(~cam, f"{a.out}/camera.ply", "camera")
if fast_simplification is not None:                                          # display only: one fixed reduction
    qem_of(solid, f"{a.out}/visual.ply", "visual", np.inf, steps=(a.visual_reduction,))
else:
    mesh_of(solid, f"{a.out}/visual.ply", "visual")

# ---- outputs -------------------------------------------------------------------------------------
# exported feet (spawn points etc.): only those that ended up in the free space with room for the body
# (with --free air, a walkable pocket the capsule flood never reached is solid in the shell)
ok_feet = reach & free
ok_feet[:, :, :nz - Hc + 1] &= free[:, :, Hc - 1:]
fi = ok_feet.nonzero().cpu().numpy()
log(f"{int(ok_feet.sum())} of {int(reach.sum())} walkable feet are inside the free space")
del ok_feet
feet = lo + fi * v                                                           # foot voxel centre, level frame
feet[:, 2] -= 0.5 * v                                                        # top of the floor voxel
np.savez_compressed(f"{a.out}/reach.npz", feet=feet.astype(np.float32), seed=np.array(seed))
if a.save_stages:                                                            # every stage, for ue-less videos
    pk = lambda x: np.packbits(x.cpu().numpy() if torch.is_tensor(x) else x)
    np.savez_compressed(f"{a.out}/stages.npz", dims=np.array([nx, ny, nz]),
                        solid_raw=pk(solid_raw if solid_raw is not None else solid), solid_clean=pk(solid_clean),
                        filled=pk(filled_vox if a.floor_fill > 0 and 'filled_vox' in globals() else np.zeros((nx, ny, nz), bool)),
                        obstacle=pk(obst), standable=pk(st), reach=pk(reach), free=pk(free), shell=pk(shell),
                        camera=pk(cam if cam is not None else np.zeros((nx, ny, nz), bool)), seed=np.array(seed))
    log(f"stages saved to {a.out}/stages.npz")
extra = dict(camera=np.packbits(cam.cpu().numpy())) if cam is not None else {}
np.savez_compressed(f"{a.out}/voxels.npz", dims=np.array([nx, ny, nz]), solid=np.packbits(solid.cpu().numpy()),
                    standable=np.packbits(st.cpu().numpy()), reach=np.packbits(reach.cpu().numpy()),
                    shell=np.packbits(shell.cpu().numpy()), **extra)
if cam is not None and bool(cam.any()):
    # signed distance to the camera volume boundary in metres (> 0 inside: room left; < 0 outside: how far
    # back in), at voxel centres of a crop around the volume, clipped to +-CLIP m; level frame, see
    # camera_volume.py for sampling and clamping in the splat frame
    CLIP = 2.0
    camn = cam.cpu().numpy()
    pad = int(np.ceil(CLIP / a.voxel)) + 2
    b0, b1 = [], []
    for ax_ in range(3):
        nzi = np.nonzero(camn.any(tuple(i for i in range(3) if i != ax_)))[0]
        b0.append(max(0, int(nzi[0]) - pad))
        b1.append(min(camn.shape[ax_], int(nzi[-1]) + pad + 1))
    sub = camn[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]]
    sdf_c = np.where(sub, ndimage.distance_transform_edt(sub) - 0.5, 0.5 - ndimage.distance_transform_edt(~sub))
    sdf_c = ndimage.gaussian_filter((sdf_c * a.voxel).astype(np.float32), 0.7)
    np.savez_compressed(f"{a.out}/camera_sdf.npz", sdf=np.clip(sdf_c, -CLIP, CLIP).astype(np.float16),
                        lo=(lo + np.array(b0) * v).astype(np.float64), voxel_units=v, voxel_m=a.voxel, mpu=a.mpu,
                        c=c, R=R, clip=CLIP)
    log(f"camera sdf {sub.shape} (+-{CLIP} m) -> {a.out}/camera_sdf.npz")
    del sub, sdf_c
area = int(col.sum()) * a.voxel ** 2
meta = dict(kind="voxel", ply=a.ply, mpu=a.mpu, voxel_units=v, c=c.tolist(), R=R.tolist(), lo=lo.tolist(),
            dims=[nx, ny, nz], center=[float(cx), float(cy)], args=vars(a), seed_index=seed, reachable_m2=area,
            floor_filled_m2=filled_m2, camera=cam_stats,
            time_s=time.time() - T0)
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)

# debug: reachable floor height (top view) and a cross-section
kk = torch.arange(nz, device=dev)
hmap = torch.where(reach, kk, torch.full_like(kk, -1)).amax(2).cpu().numpy().astype(float)
img = np.full((nx, ny, 3), 20, np.uint8)
has = hmap >= 0
if has.any():
    hn = (hmap - hmap[has].min()) / max(1.0, np.ptp(hmap[has]))
    img[has] = cv2.applyColorMap((hn[has] * 255).astype(np.uint8)[:, None], cv2.COLORMAP_VIRIDIS)[:, 0]
img[~has & solid.any(2).cpu().numpy()] = (70, 70, 70)
sd = seed
cv2.circle(img, (sd[1], sd[0]), 4, (0, 0, 255), -1)
cv2.imwrite(f"{a.out}/debug/reach_top.png", np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]))
j = seed[1]
xs = np.zeros((nz, nx, 3), np.uint8)
xs[solid[:, j, :].T.cpu().numpy()] = (150, 150, 150)
xs[free[:, j, :].T.cpu().numpy()] = (60, 40, 20)
xs[reach[:, j, :].T.cpu().numpy()] = (0, 220, 255)
xs[shell[:, j, :].T.cpu().numpy() & ~solid[:, j, :].T.cpu().numpy()] = (40, 40, 160)
cv2.imwrite(f"{a.out}/debug/xsection.png", cv2.resize(xs[::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
if cam is not None:                                                          # camera volume: x-z and y-z through the seed
    rows = []
    for sl_ in ((slice(None), seed[1]), (seed[0], slice(None))):
        so, fr, ca, re = (t_[sl_].T.cpu().numpy() for t_ in (solid, free, cam, reach))
        im = np.zeros(so.shape + (3,), np.uint8)
        im[fr] = (60, 40, 20)
        im[ca] = (70, 160, 70)
        im[so] = (150, 150, 150)
        im[re] = (0, 220, 255)
        rows.append(cv2.resize(im[::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    wmax = max(r_.shape[1] for r_ in rows)
    rows = [np.pad(r_, ((0, 4), (0, wmax - r_.shape[1]), (0, 0))) for r_ in rows]
    cv2.imwrite(f"{a.out}/debug/xsection_camera.png", np.concatenate(rows, 0))
# density slices (log scale, threshold in white) at the seed row and at 1/4 and 3/4 of the chunk
rows = []
for jj_ in ((seed[1], ny // 4, 3 * ny // 4) if dens is not None else ()):
    dsl = dens[:, jj_, :].T.cpu().numpy()[::-1]
    im = cv2.applyColorMap(np.clip((np.log10(dsl + 1e-3) + 3) / 3.5 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    im[np.abs(dsl - a.opacity) < 0.04 * max(a.opacity, 0.1)] = (255, 255, 255)
    cv2.putText(im, f"y index {jj_}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(im)
if rows:
    cv2.imwrite(f"{a.out}/debug/density_slices.png", np.concatenate(rows, 0))
    np.savez_compressed(f"{a.out}/density_slices.npz", **{f"y{jj_}": dens[:, jj_, :].cpu().numpy().astype(np.float16)
                                                         for jj_ in (seed[1], ny // 4, 3 * ny // 4)})
log(f"done: reachable {area:.0f} m^2; outputs in {a.out}")
