"""Perspective peeks into a splat in its level frame: py.bat video/peek.py <ply> <out.jpg> ex ey ez tx ty tz [fov]"""
import sys
sys.path.insert(0, "D:/3DGS_Research/collision_proto")
import numpy as np
import torch
import cv2
from gs_core import load_ply, estimate_frame, look_at, intrinsics, render

ply, out = sys.argv[1], sys.argv[2]
e = np.array([float(x) for x in sys.argv[3:6]])
t = np.array([float(x) for x in sys.argv[6:9]])
fov = float(sys.argv[9]) if len(sys.argv) > 9 else 70.0
g = load_ply(ply)
c, R = estimate_frame(g.means.cpu().numpy())
to_w = lambda p: c + p @ R
W, H = 1280, 720
M = look_at(to_w(e), to_w(t), R[2])
rgb, depth, alpha = render(g, M, intrinsics(W, H, fov), W, H, near=0.05, far=1e4)
img = (np.clip(rgb.cpu().numpy(), 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
cv2.imwrite(out, img)
print("frame up =", R[2].round(4), "saved", out)
