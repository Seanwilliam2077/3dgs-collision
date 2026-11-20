"""Top-down overview render of a splat: verify the up direction and pick a chunk.

Writes top/bottom RGB + height images with a grid in level-frame coordinates (scene units).
"""
import argparse
import os

import cv2
import numpy as np

from gs_core import load_ply, estimate_frame, render

ap = argparse.ArgumentParser()
ap.add_argument("ply")
ap.add_argument("--out", default="out/overview")
ap.add_argument("--up", type=float, nargs=3, default=None)
ap.add_argument("--pct", type=float, default=3.0, help="percentile used for the framing box")
ap.add_argument("--size", type=int, default=1400)
ap.add_argument("--center", type=float, nargs=2, default=None, help="level-frame x y of view center")
ap.add_argument("--half", type=float, default=None, help="half extent of the view (scene units)")
ap.add_argument("--grid", type=float, default=None, help="grid spacing (scene units)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

g = load_ply(args.ply)
xyz = g.means.cpu().numpy().astype(np.float64)
c, R = estimate_frame(xyz, args.pct, args.up)
e1, e2, up = R
pl = (xyz - c) @ R.T
lo, hi = np.percentile(pl, [args.pct, 100 - args.pct], axis=0)
print("up =", up.round(4), " center =", c.round(4))
print("level-frame core box:", lo.round(3), hi.round(3))

cx, cy = args.center if args.center else (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]))
half = args.half if args.half else 0.5 * max(hi[0] - lo[0], hi[1] - lo[1]) * 1.05
S = args.size
s = S / (2 * half)                                   # pixels per unit
K = np.array([[s, 0, S / 2], [0, s, S / 2], [0, 0, 1]])
cam_h = 50.0 * max(hi[2] - lo[2], 1e-3) + hi[2]
tag = "" if args.center is None else f"_{cx:.2f}_{cy:.2f}"

for name, sign in [("top", 1.0), ("bottom", -1.0)]:
    eye = c + R.T @ np.array([cx, cy, sign * cam_h])
    f = -sign * up
    r = e1 * sign                                    # keep image x = +e1 for the top view
    d = np.cross(f, r)
    M = np.eye(4)
    M[:3, :3] = np.stack([r, d, f], 0)
    M[:3, 3] = -M[:3, :3] @ eye
    rgb, depth, alpha = render(g, M, K, S, S, near=1e-4, ortho=True)
    a = alpha.cpu().numpy()
    hgt = sign * (cam_h - depth.cpu().numpy())
    img = np.ascontiguousarray((rgb.cpu().numpy()[..., ::-1] * 255).astype(np.uint8))
    hv = hgt[a > 0.5]
    lo_h, hi_h = np.percentile(hv, [2, 98])
    vis = np.clip((hgt - lo_h) / (hi_h - lo_h + 1e-9), 0, 1)
    col = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[a < 0.5] = 0
    if name == "top":
        step = args.grid or 10 ** np.floor(np.log10(half))
        for im in (img, col):
            for v in np.arange(np.ceil((cx - half) / step) * step, cx + half, step):
                px = int((v - cx) * s + S / 2)
                cv2.line(im, (px, 0), (px, S - 1), (255, 255, 255), 1)
                cv2.putText(im, f"x={v:.2f}", (px + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for v in np.arange(np.ceil((cy - half) / step) * step, cy + half, step):
                py = int(S / 2 - (v - cy) * s)
                cv2.line(im, (0, py), (S - 1, py), (255, 255, 255), 1)
                cv2.putText(im, f"y={v:.2f}", (3, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(f"{args.out}/{name}{tag}_rgb.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(f"{args.out}/{name}{tag}_height.jpg", col, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{name}: coverage(alpha>0.5) {np.mean(a > 0.5):.3f}  height 2-98% {lo_h:.4f}..{hi_h:.4f}")
print(f"view center ({cx:.3f},{cy:.3f}) half {half:.3f} units, {1 / s:.5f} units/pixel")
