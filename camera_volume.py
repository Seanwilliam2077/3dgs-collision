"""Camera volume of a voxel_collision.py run: keep a camera where the splat looks right.

Library (numpy + scipy, no engine, no GPU), splat-frame positions in scene units:

    from camera_volume import CameraVolume
    vol = CameraVolume("out/final_vox")
    vol.sdf(p)              # [N,3] -> metres to the boundary (> 0 inside, < 0 outside)
    vol.clamp(p)            # free camera: positions outside move to the nearest allowed point
    vol.arm(pivot, want)    # third person: pivot -> wanted camera, cut where it would leave the volume

The volume already keeps --cam-clear (0.3 m) from every surface, so a point is enough for the camera
(no probe sphere). Engines can use camera.ply instead (collision for the camera channel only).

Check: random third-person arms and free cameras, with renders of what the volume prevents:

    py.bat camera_volume.py out/final_vox [--n 2000] [--arm 3]
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree


class CameraVolume:
    def __init__(self, run):
        z = np.load(f"{run}/camera_sdf.npz")
        self.grid = z["sdf"].astype(np.float32)
        self.lo, self.v, self.mpu = z["lo"], float(z["voxel_units"]), float(z["mpu"])
        self.c, self.R, self.clip = z["c"], z["R"], float(z["clip"])
        self.up = self.R[2]                                              # level z in the splat frame
        inner = np.argwhere((self.grid > 0) & (self.grid <= 1.5 * self.v * self.mpu))
        self._inner = self.lo + inner * self.v                           # just inside the boundary, level frame
        self._tree = cKDTree(self._inner)

    # frames: level = (splat - c) @ R.T, scene units both
    def to_level(self, p):
        return (np.asarray(p, np.float64) - self.c) @ self.R.T

    def to_splat(self, q):
        return self.c + np.asarray(q, np.float64) @ self.R

    def _sdf_level(self, q):
        """Trilinear sample at level-frame points [N,3]; -clip outside the stored crop."""
        f = (np.atleast_2d(q) - self.lo) / self.v
        i0 = np.floor(f).astype(np.int64)
        t = f - i0
        out = np.full(len(f), -self.clip, np.float64)
        ok = np.all((i0 >= 0) & (i0 + 1 < np.array(self.grid.shape)), axis=1)
        i, t = i0[ok], t[ok]
        acc = np.zeros(len(i))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                    acc += w * self.grid[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        out[ok] = acc
        return out

    def sdf(self, p):
        return self._sdf_level(self.to_level(p))

    def _grad_level(self, q):
        h = 0.5 * self.v
        g = np.stack([self._sdf_level(q + h * e) - self._sdf_level(q - h * e) for e in np.eye(3)], 1)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)

    def clamp(self, p, margin=0.05, iters=10):
        """Move splat-frame positions that are outside (or within margin m of the boundary) inside."""
        q = np.atleast_2d(self.to_level(p)).copy()
        s = self._sdf_level(q)
        bad = s < margin
        far = bad & (s <= -self.clip + 1e-3)                             # beyond the stored band: no gradient
        if far.any():
            q[far] = self._inner[self._tree.query(q[far])[1]]
        for _ in range(iters):
            s = self._sdf_level(q)
            m = s < margin
            if not m.any():
                break
            q[m] += self._grad_level(q[m]) * ((margin - s[m]) / self.mpu)[:, None]   # metres -> scene units
        stuck = self._sdf_level(q) < 0                                   # rare: corners where the gradient stalls
        if stuck.any():
            q[stuck] = self._inner[self._tree.query(q[stuck])[1]]
        return self.to_splat(q)

    def arm(self, pivot, want, margin=0.05):
        """Spring arm: for each pivot -> wanted camera segment, the last point before it leaves the volume
        (keeping margin m). Returns cameras and whether each pivot itself was inside."""
        P, W = np.atleast_2d(self.to_level(pivot)), np.atleast_2d(self.to_level(want))
        L = np.linalg.norm(W - P, axis=1)
        n = int(np.ceil(L.max() / (0.5 * self.v))) + 1
        ts = np.linspace(0, 1, n)
        pts = P[:, None] + (W - P)[:, None] * ts[None, :, None]
        s = self._sdf_level(pts.reshape(-1, 3)).reshape(len(P), n)
        bad = s < margin
        first = np.where(bad.any(1), bad.argmax(1), n)
        k = np.maximum(first - 1, 0)
        cam = pts[np.arange(len(P)), np.minimum(k, n - 1)]
        return self.to_splat(cam), ~bad[:, 0]


def near_alpha(g, P, up, mpu, far_m=0.6, res=32, batch=16):
    """How enclosed in splat each camera position P [N,3] (splat frame) is: alpha of the 6 cube faces
    rendered with only Gaussians closer than far_m, averaged over the whole sphere. A camera 0.35 m above
    open ground scores ~0.2 (the face below); inside foliage or fog it goes towards 1."""
    import torch
    from gs_core import look_at, intrinsics, rasterization
    side = np.cross(up, [1.0, 0, 0])
    side = side if np.linalg.norm(side) > 0.5 else np.cross(up, [0, 1.0, 0])
    side /= np.linalg.norm(side)
    dirs = [up, -up, side, -side, np.cross(up, side), -np.cross(up, side)]
    hints = [side, side, up, up, up, up]
    dev = g.means.device
    Kt = torch.tensor(intrinsics(res, res, 90.0), dtype=torch.float32, device=dev)
    out = []
    for s0 in range(0, len(P), batch):
        vms = [look_at(p, p + d, h) for p in P[s0:s0 + batch] for d, h in zip(dirs, hints)]
        vm = torch.tensor(np.stack(vms), dtype=torch.float32, device=dev)
        _, al, _ = rasterization(g.means, g.quats, g.scales, g.opac, g.sh, vm, Kt[None].expand(len(vms), 3, 3),
                                 res, res, sh_degree=g.sh_degree, near_plane=0.02 / mpu, far_plane=far_m / mpu,
                                 packed=True, render_mode="RGB")
        out.append(al[..., 0].mean((1, 2)).view(-1, 6).mean(1).cpu().numpy())
    return np.concatenate(out)


def _check(run, n=2000, arm_m=3.0, seed=0):
    sys.path.insert(0, "D:/3DGS_Research/collision_proto")
    import cv2
    from gs_core import load_ply, look_at, intrinsics, render

    meta = json.load(open(f"{run}/meta.json"))
    mpu, U = meta["mpu"], 1.0 / meta["mpu"]
    eye_m = meta["args"].get("cam_eye") or 0.9 * meta["args"]["height"]
    vol = CameraVolume(run)
    rng = np.random.default_rng(seed)
    feet = np.load(f"{run}/reach.npz")["feet"].astype(np.float64)         # level frame
    lo, dims, vu = np.array(meta["lo"]), np.array(meta["dims"]), meta["voxel_units"]

    # third person: random feet, random yaw, pitch -15 .. 60 deg (camera above the pivot), arm_m metres
    f = feet[rng.choice(len(feet), n)]
    piv = f + np.array([0, 0, eye_m * U])
    yaw, pit = rng.uniform(0, 2 * np.pi, n), np.radians(rng.uniform(-15, 60, n))
    d = np.stack([np.cos(pit) * np.cos(yaw), np.cos(pit) * np.sin(yaw), np.sin(pit)], 1)
    want = piv + d * arm_m * U
    got, piv_in = vol.arm(vol.to_splat(piv), vol.to_splat(want))
    got_l = vol.to_level(got)
    Lgot = np.linalg.norm(got_l - piv, axis=1) * mpu
    s_want = vol.sdf(vol.to_splat(want))
    print(f"third person ({n} arms of {arm_m} m): pivot inside {piv_in.mean() * 100:.1f}%; wanted camera outside "
          f"{(s_want < 0).mean() * 100:.1f}%; arm shortened {(Lgot < arm_m - 0.05).mean() * 100:.1f}%, "
          f"mean length {Lgot.mean():.2f} m; min sdf of the results (pivot inside) {vol.sdf(got[piv_in]).min():+.3f} m")

    # free camera: uniform in the grid box
    q = lo + rng.uniform(0, 1, (n, 3)) * (dims - 1) * vu
    s0 = vol._sdf_level(q)
    cl = vol.clamp(vol.to_splat(q))
    s1 = vol.sdf(cl)
    mv = np.linalg.norm(vol.to_level(cl) - q, axis=1) * mpu
    print(f"free camera ({n} uniform in the grid): outside {(s0 < 0).mean() * 100:.1f}%; after clamp min sdf "
          f"{s1.min():+.3f} m, outside {(s1 < 0).mean() * 100:.2f}%; moved median {np.median(mv[s0 < 0]):.2f} m")

    # splat within 0.6 m of the camera (what blurs the view up close): random points of the volume, and
    # the third-person cameras
    g = load_ply(meta["ply"])
    inside = np.argwhere(vol.grid > 0)
    pv = vol.to_splat(vol.lo + inside[rng.choice(len(inside), min(n, 600))] * vol.v)
    for name, P in (("random points in the volume", pv), ("third-person cameras", got[piv_in][:600])):
        na = near_alpha(g, P, vol.up, mpu)
        print(f"enclosure (alpha over the sphere, splat within 0.6 m), {name}: median {np.median(na):.3f}, "
              f"p90 {np.percentile(na, 90):.3f}, p99 {np.percentile(na, 99):.3f}; > 0.3 for {(na > 0.3).mean() * 100:.1f}%, "
              f"> 0.5 for {(na > 0.5).mean() * 100:.1f}%")

    # renders: wanted vs got for the worst arms, and outside vs clamped for free cameras
    Wr, Hr, fov = 640, 360, 70
    K = intrinsics(Wr, Hr, fov)

    def shot(eye_l, tgt_l, label):
        M = look_at(vol.to_splat(eye_l), vol.to_splat(tgt_l), vol.up)
        rgb, _, _ = render(g, M, K, Wr, Hr, near=0.1 * U, far=1e4)
        im = np.ascontiguousarray((np.clip(rgb.cpu().numpy(), 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.rectangle(im, (0, 0), (Wr, 26), (0, 0, 0), -1)
        cv2.putText(im, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return im

    rows = []
    cand = np.nonzero((s_want < -0.3) & piv_in)[0]
    for i in rng.choice(cand, min(4, len(cand)), replace=False) if len(cand) else []:
        tgt = piv[i]
        rows.append(np.concatenate([
            shot(want[i], tgt, f"3rd person: wanted ({arm_m:.0f} m arm), sdf {s_want[i]:+.2f} m"),
            shot(got_l[i], tgt, f"camera volume: arm {Lgot[i]:.2f} m")], 1))
    look = feet[np.argmin(np.linalg.norm(feet[:, :2] - feet[:, :2].mean(0), axis=1))] + np.array([0, 0, eye_m * U])
    cand = np.nonzero(s0 < -0.3)[0]
    cl_l = vol.to_level(cl)
    for i in rng.choice(cand, min(4, len(cand)), replace=False) if len(cand) else []:
        rows.append(np.concatenate([
            shot(q[i], look, f"free camera: wanted, sdf {s0[i]:+.2f} m"),
            shot(cl_l[i], look, f"clamped: moved {mv[i]:.2f} m")], 1))
    if rows:
        cv2.imwrite(f"{run}/debug/camera_check.jpg", np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"renders -> {run}/debug/camera_check.jpg")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--arm", type=float, default=3.0, help="third-person arm length (m)")
    a = ap.parse_args()
    _check(a.run, a.n, a.arm)
