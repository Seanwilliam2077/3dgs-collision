"""Export a run's collision meshes for Unreal Engine.

    py.bat export_ue.py out/chunk_a        (TSDF run of splat_to_collision.py)
    py.bat export_ue.py out/vox_carve      (voxel run of voxel_collision.py)

Coordinates: level frame (gravity along -Z), origin at the chunk centre in XY and at the median
height of the main ground in Z, centimetres, Y flipped (UE is left-handed):
    UE = 100 * mpu * ( x - x0, -(y - y0), z - z0 ),   level (x, y, z) = R @ (p_splat - c)
The GLBs are written so that UE 5.8's Interchange glTF importer, which maps glTF (x, y, z) m to
UE (x, z, y) cm, lands exactly on those coordinates. Check this again on other engine versions
(collision_proto/ue/probe_import.py).

Vertex colours of the visible layer are the splat's own colour at each vertex: the opacity-weighted
Gaussian density of the nearest Gaussians times their DC colour (view-independent part only). They are
a visual stand-in for the splat in an engine without a splat renderer, not part of the collision.

Writes <run>/ue/: one GLB per layer and scene_ue.json, which tells ue/setup_level.py and
ue/walk_test.py everything: level name, layers (file, visible, collides), walk-test phases, spawn
points on the ground, chunk bounds.
  TSDF run:  SplatCollision (coloured lite mesh), GroundNet, and if ground_patch.py ran GroundPatch +
             CollisionObjects; phases mesh / patch / all / split.
  voxel run: SplatVisual (coloured surface, no collision), VoxelShell (the collision), and if the run has
             camera.ply, CameraVolume (blocks the Camera channel only: spring arms stop at it, characters
             walk through it); phases voxel (shell only) and camera (shell + camera volume).
"""
import argparse
import json
import os

import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--k", type=int, default=24, help="Gaussians considered per vertex for the colour")
ap.add_argument("--n-spawn", type=int, default=400)
ap.add_argument("--ground-band", type=float, default=2.5,
                help="voxel run: spawn on feet within this many metres of the seed height (the main ground)")
a = ap.parse_args()

meta = json.load(open(f"{a.run}/meta.json"))
voxel_run = meta.get("kind") == "voxel"
mpu = meta["mpu"]
c = np.array(meta["c"])
R = np.array(meta["R"])
assert np.linalg.det(R) > 0
os.makedirs(f"{a.run}/ue", exist_ok=True)
name = os.path.basename(os.path.normpath(a.run))
to_level = lambda p: (np.asarray(p, np.float64) - c) @ R.T

if voxel_run:
    x0, y0 = meta["center"]
    feet = np.load(f"{a.run}/reach.npz")["feet"].astype(np.float64)       # level frame
    seed_z = meta["lo"][2] + meta["seed_index"][2] * meta["voxel_units"]
    ground = feet[np.abs(feet[:, 2] - seed_z) * mpu < a.ground_band]
    visual = trimesh.load(f"{a.run}/visual.ply", process=False)
    layers = [dict(name="SplatVisual", src=f"{a.run}/visual.ply", visible=True, collide=False, colour=True),
              dict(name="VoxelShell", src=f"{a.run}/collision.ply", visible=False, collide=True)]
    phases = [["voxel", ["VoxelShell"]]]
    if os.path.exists(f"{a.run}/camera.ply"):                         # blocks the Camera channel only
        layers.append(dict(name="CameraVolume", src=f"{a.run}/camera.ply", visible=False, collide=True, channel="camera"))
        phases.append(["camera", ["VoxelShell", "CameraVolume"]])
else:
    x0, y0 = meta["args"]["center"]
    if os.path.exists(f"{a.run}/ground_cells.npy"):                  # cleaned by ground_patch.py
        ground = np.load(f"{a.run}/ground_cells.npy")
    else:
        cells = np.load(f"{a.run}/walk_cells.npz")
        ground = cells["xyz"][cells["set"] == 0]
    has_patch = os.path.exists(f"{a.run}/ground_patch.ply")
    net_src = f"{a.run}/ground_net2.ply" if os.path.exists(f"{a.run}/ground_net2.ply") else f"{a.run}/ground_net.glb"
    visual = trimesh.load(f"{a.run}/collision_lite.ply", process=False)
    layers = [dict(name="SplatCollision", src=f"{a.run}/collision_lite.ply", visible=True, collide=not has_patch, colour=True),
              dict(name="GroundNet", src=net_src, visible=False, collide=True)]
    phases = [["mesh", ["SplatCollision"]]]
    if has_patch:
        layers += [dict(name="GroundPatch", src=f"{a.run}/ground_patch.ply", visible=False, collide=True),
                   dict(name="CollisionObjects", src=f"{a.run}/collision_objects.ply", visible=False, collide=True)]
        phases += [["patch", ["SplatCollision", "GroundPatch"]], ["all", ["SplatCollision", "GroundPatch", "GroundNet"]],
                   ["split", ["CollisionObjects", "GroundPatch", "GroundNet"]]]
    else:
        phases += [["all", ["SplatCollision", "GroundNet"]]]
z0 = float(np.median(ground[:, 2]))


def to_ue(pl):                       # level frame (scene units) -> UE cm
    return np.stack([pl[:, 0] - x0, -(pl[:, 1] - y0), pl[:, 2] - z0], 1) * mpu * 100.0


def ue_to_gltf(u):                    # UE cm -> glTF m such that the importer maps it back to u
    return np.stack([u[:, 0], u[:, 2], u[:, 1]], 1) / 100.0


# ---- vertex colours from the splat --------------------------------------------------------------
def splat_colours(V):
    v = PlyData.read(meta["ply"])["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    lo, hi = V.min(0) - 2.0 / mpu, V.max(0) + 2.0 / mpu
    keep = np.all((xyz > lo) & (xyz < hi), 1)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
    keep &= opac > 0.05
    xyz, opac = xyz[keep], opac[keep]
    rgb = np.clip(0.5 + 0.28209479 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[keep], 0, 1)
    sc = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)[keep].astype(np.float64))
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1)[keep].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w_, x_, y_, z_ = q.T
    Rg = np.stack([1 - 2 * (y_ ** 2 + z_ ** 2), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_),
                   2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ ** 2 + z_ ** 2), 2 * (y_ * z_ - w_ * x_),
                   2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ ** 2 + y_ ** 2)], 1).reshape(-1, 3, 3)
    print(f"[colour] {len(xyz)} Gaussians around the chunk, {len(V)} vertices")
    tree = cKDTree(xyz)
    col = np.zeros((len(V), 3))
    for s in range(0, len(V), 50_000):
        p = V[s:s + 50_000]
        d, idx = tree.query(p, k=a.k)
        loc = np.einsum("nkji,nkj->nki", Rg[idx], p[:, None, :] - xyz[idx]) / sc[idx]   # Gaussian's own axes
        dens = opac[idx] * np.exp(-0.5 * (loc ** 2).sum(-1))
        idw = opac[idx] / (d * mpu + 0.2) ** 2 * 1e-3                                    # fallback far from any splat
        w = dens + idw
        col[s:s + 50_000] = (w[..., None] * rgb[idx]).sum(1) / w.sum(1, keepdims=True)
    return np.concatenate([np.round(col * 255), np.full((len(V), 1), 255)], 1).astype(np.uint8)


# ---- meshes --------------------------------------------------------------------------------------
lo_all, hi_all = np.full(3, np.inf), np.full(3, -np.inf)
for L in layers:
    msh = trimesh.load(L.pop("src"), force="mesh", process=False)
    msh.remove_unreferenced_vertices()                  # UE drops them too; bounds must match for the axis check
    vu = to_ue(to_level(msh.vertices))
    m = trimesh.Trimesh(ue_to_gltf(vu), msh.faces, process=False)
    if L.pop("colour", False):
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=splat_colours(np.asarray(msh.vertices)))
    L["file"] = f"{L['name']}.glb"
    m.export(f"{a.run}/ue/{L['file']}")
    L["bounds_cm"] = [vu.min(0).tolist(), vu.max(0).tolist()]
    L["tris"] = int(len(msh.faces))
    lo_all, hi_all = np.minimum(lo_all, vu.min(0)), np.maximum(hi_all, vu.max(0))
    print(f"[ue] {L['name']}: {len(msh.faces)} tris, bounds {vu.min(0).round(0)} .. {vu.max(0).round(0)} cm")

# ---- spawn points and bounds ---------------------------------------------------------------------
sx, sy = meta["args"]["size"]
half = np.array([sx, sy]) / 2 * 100.0                         # cm
g_ue = to_ue(ground)
inner = np.all(np.abs(g_ue[:, :2]) < half - 300.0, 1)         # 3 m away from the chunk border
rng = np.random.default_rng(0)
pick = rng.choice(np.nonzero(inner)[0], size=min(a.n_spawn, int(inner.sum())), replace=False)
start = g_ue[np.argmin(np.linalg.norm(g_ue[:, :2], axis=1))]
info = dict(
    run=os.path.abspath(a.run).replace("\\", "/"), kind="voxel" if voxel_run else "tsdf", mpu=mpu, c=c.tolist(),
    R=R.tolist(), x0=x0, y0=y0, z0=z0,
    note="UE_cm = 100*mpu*(x-x0, -(y-y0), z-z0) with level (x,y,z) = R @ (p_splat - c)",
    level=f"Lvl_{name}", asset_dir=f"/Game/Splat/{name}",
    layers=layers, phases=phases,
    chunk_half_cm=half.tolist(),
    void_z_cm=float(lo_all[2] - 1500.0),
    player_start_cm=start.tolist(),
    spawns_cm=g_ue[pick].round(1).tolist(),
)
json.dump(info, open(f"{a.run}/ue/scene_ue.json", "w"), indent=1)
print(f"[ue] level {info['level']}; ground ref z0 = {z0 * mpu:.2f} m level; {len(pick)} spawns; player start {start.round(0)}")
