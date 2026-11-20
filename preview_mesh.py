"""Side-by-side previews: splat render | collision mesh (flat shaded + wireframe) from the same camera.

Dependency-free point-sampling rasteriser (torch): triangles are sampled densely enough to be
hole-free at the output resolution, resolved with a z-buffer, then edges are drawn on top.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import trimesh

from gs_core import load_ply, look_at, intrinsics, render


@torch.no_grad()
def rasterize_mesh(V, Fc, M, K, W, H, near, light_w, wire=True, max_samples=40_000_000):
    dev = "cuda"
    V = torch.as_tensor(V, dtype=torch.float32, device=dev)
    Fc = torch.as_tensor(Fc, dtype=torch.long, device=dev)
    Mt = torch.as_tensor(M, dtype=torch.float32, device=dev)
    Vc = V @ Mt[:3, :3].T + Mt[:3, 3]
    tri = Vc[Fc]                                              # [T,3,3] camera space
    vis = (tri[..., 2] > near).all(1)
    tri, Fv = tri[vis], Fc[vis]
    n_w = torch.cross(V[Fv[:, 1]] - V[Fv[:, 0]], V[Fv[:, 2]] - V[Fv[:, 0]], dim=1)
    n_w = torch.nn.functional.normalize(n_w, dim=1)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    def project(p):
        return torch.stack([fx * p[..., 0] / p[..., 2] + cx, fy * p[..., 1] / p[..., 2] + cy], -1)

    uv = project(tri)
    e = torch.stack([(uv[:, 1] - uv[:, 0]).norm(dim=1), (uv[:, 2] - uv[:, 1]).norm(dim=1), (uv[:, 0] - uv[:, 2]).norm(dim=1)], 1)
    on = ((uv[..., 0].amax(1) > 0) & (uv[..., 0].amin(1) < W) & (uv[..., 1].amax(1) > 0) & (uv[..., 1].amin(1) < H))
    tri, uv, e, n_w = tri[on], uv[on], e[on], n_w[on]
    # samples per triangle edge so that neighbouring samples are < ~0.7 px apart
    k = torch.clamp(torch.ceil(e.amax(1) / 0.7), 1, 400).long()
    zbuf = torch.full((H * W,), 1e9, device=dev)
    shade_buf = torch.zeros((H * W, 3), device=dev)
    lw = torch.as_tensor(light_w, dtype=torch.float32, device=dev)
    lam = 0.35 + 0.65 * (n_w @ lw).abs()
    col = lam[:, None] * torch.tensor([0.78, 0.80, 0.84], device=dev)
    passes = []
    for kk in torch.unique(k).tolist():
        sel = (k == kk).nonzero()[:, 0]
        i, j = torch.meshgrid(torch.arange(kk + 1, device=dev), torch.arange(kk + 1, device=dev), indexing="ij")
        m = (i + j) <= kk
        b1, b2 = (i[m] / kk).float(), (j[m] / kk).float()
        bary = torch.stack([1 - b1 - b2, b1, b2], 1)          # [S,3]
        edge = (bary.amin(1) < 1e-6)
        step = max(1, max_samples // (bary.shape[0] * 8))
        for s0 in range(0, len(sel), step):
            s = sel[s0:s0 + step]
            p = torch.einsum("sb,tbc->tsc", bary, tri[s])    # [t,S,3]
            q = project(p)
            ok = (q[..., 0] >= 0) & (q[..., 0] < W) & (q[..., 1] >= 0) & (q[..., 1] < H)
            lin = (q[..., 1].long().clamp(0, H - 1) * W + q[..., 0].long().clamp(0, W - 1))[ok]
            z = p[..., 2][ok]
            zbuf.scatter_reduce_(0, lin, z, reduce="amin")
            passes.append((lin, z, col[s][:, None, :].expand(-1, bary.shape[0], -1)[ok], edge[None].expand(len(s), -1)[ok]))
    for lin, z, c, _ in passes:
        win = z <= zbuf[lin] * 1.0005
        shade_buf[lin[win]] = c[win]
    if wire:
        for lin, z, _, edge in passes:
            win = edge & (z <= zbuf[lin] * 1.002)
            shade_buf[lin[win]] *= 0.45
    img = shade_buf.view(H, W, 3)
    depth = torch.where(zbuf < 1e8, zbuf, torch.zeros_like(zbuf)).view(H, W)
    return img, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="output folder of splat_to_collision.py")
    ap.add_argument("--mesh", default="collision_lite.ply")
    ap.add_argument("--views", type=float, nargs="+", default=None,
                    help="repeated groups of 6: eye x y z_above_ground(m) yaw pitch_down fov  (x y in scene units, level frame)")
    ap.add_argument("--size", type=int, nargs=2, default=[800, 600])
    ap.add_argument("--tag", default="lite")
    a = ap.parse_args()
    meta = json.load(open(f"{a.run}/meta.json"))
    c, R, mpu = np.array(meta["c"]), np.array(meta["R"]), meta["mpu"]
    U = 1 / mpu
    up = R[2]
    mesh = trimesh.load(f"{a.run}/{a.mesh}", process=False)
    V, Fc = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    Vl = (V - c) @ R.T
    g = load_ply(meta["ply"])
    W, H = a.size
    light = R.T @ (np.array([0.35, 0.25, 0.9]) / np.linalg.norm([0.35, 0.25, 0.9]))
    os.makedirs(f"{a.run}/preview", exist_ok=True)
    views = np.array(a.views).reshape(-1, 6)
    for n, (x, y, h, yaw, pitch, fov) in enumerate(views):
        # ground height under the eye = lowest mesh vertex within 2 m horizontally
        d2 = (Vl[:, 0] - x) ** 2 + (Vl[:, 1] - y) ** 2
        near_v = Vl[d2 < (2 * U) ** 2]
        z0 = np.percentile(near_v[:, 2], 2) if len(near_v) else np.percentile(Vl[:, 2], 2)
        eye_l = np.array([x, y, z0 + h * U])
        yr, pr = np.radians(yaw), np.radians(pitch)
        dir_l = np.array([np.cos(pr) * np.cos(yr), np.cos(pr) * np.sin(yr), -np.sin(pr)])
        M = look_at(c + eye_l @ R, c + (eye_l + dir_l) @ R, up)
        K = intrinsics(W, H, fov)
        rgb, _, _ = render(g, M, K, W, H, near=0.05 * U)
        img, _ = rasterize_mesh(V, Fc, M, K, W, H, 0.05 * U, light)
        a_img = (rgb.cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
        b_img = (img.cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
        blend = cv2.addWeighted(a_img, 0.55, b_img, 0.45, 0)
        cv2.imwrite(f"{a.run}/preview/{a.tag}_{n}.jpg", np.hstack([a_img, b_img, blend]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"view {n}: eye level=({x:.3f},{y:.3f}) {h:.1f} m above ground, yaw {yaw:.0f} pitch {pitch:.0f}")


if __name__ == "__main__":
    main()
