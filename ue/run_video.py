"""Composite the UE run (ue/run_capture.py) over the splat: py.bat ue/run_video.py out/final_vox [--still 3,8]

For every captured frame: the splat is rendered with gsplat from UE's player camera (same position, rotation
and 90 deg horizontal FOV, at the capture size), the character comes from the UE capture (SceneColor HDR +
scene depth, show-only list; lit here from its GBuffer base colour and normal with the level's sun and a sky term,
since lit scene colour comes out black from scene captures on this machine), tone-mapped with one fixed exposure for the whole clip, and is hidden where the
splat is opaque and more than 25 cm nearer. A soft contact shadow is drawn on the floor under the capsule.
Writes <run>/showcase/ue_run.mp4 (1920x1080, 30 fps).
"""
import argparse
import json
import math
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import OpenEXR
import torch
from PIL import Image, ImageDraw, ImageFont

from gs_core import load_ply, look_at, intrinsics, render

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--still", default=None)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--crf", type=int, default=19)
a = ap.parse_args()
run = a.run.rstrip("/\\")
info = json.load(open(f"{run}/ue/scene_ue.json"))
lg = json.load(open(f"{run}/ue/run/run_log.json"))
mpu, c, R = info["mpu"], np.array(info["c"]), np.array(info["R"])
x0, y0, z0 = info["x0"], info["y0"], info["z0"]
CW, CH = lg["cap_size"]
OW, OH = 1920, 1080
FOV = lg["fov"]
frames = lg["frames"]
half = lg["capsule_half_cm"]
g = load_ply(json.load(open(f"{run}/meta.json"))["ply"])
K = intrinsics(CW, CH, FOV)


def ue2splat(u):
    u = np.asarray(u, np.float64).reshape(-1, 3)
    lv = np.stack([u[:, 0] / (100 * mpu) + x0, -u[:, 1] / (100 * mpu) + y0, u[:, 2] / (100 * mpu) + z0], 1)
    return c + lv @ R


def view(fr):
    eye = np.array(fr["cam"])
    p, y = math.radians(fr["rot"][0]), math.radians(fr["rot"][1])
    fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    e_s, t_s = ue2splat(eye)[0], ue2splat(eye + 100.0 * fwd)[0]
    return look_at(e_s, t_s, R[2]), e_s


SUN = np.array([-math.cos(math.radians(-50)) * math.cos(math.radians(30)), -math.cos(math.radians(-50)) * math.sin(math.radians(30)),
                -math.sin(math.radians(-50))])                  # towards the level's Sun (pitch -50, yaw 30), UE axes


def read_exr(fn):
    with OpenEXR.File(fn) as e:
        return np.asarray(next(iter(e.channels().values())).pixels, np.float32)


def load_cap(i):
    """Character colour (FinalColor HDR, linear), depth in cm (inf off the character), mask."""
    d = f"{run}/ue/run/d{i:04d}.exr"
    dep = read_exr(d if os.path.exists(d) else f"{run}/ue/run/f{i:04d}.exr")[..., 3]
    mask = np.isfinite(dep) & (dep > 1.0) & (dep < 6e4)
    base = np.nan_to_num(read_exr(f"{run}/ue/run/b{i:04d}.exr")[..., :3])
    nrm = np.nan_to_num(read_exr(f"{run}/ue/run/n{i:04d}.exr")[..., :3])
    sun = np.clip(nrm @ SUN, 0, None)[..., None]
    sky = (0.55 + 0.45 * nrm[..., 2:3])
    rgb = base * (2.2 * sun * np.array([1.0, 0.96, 0.88]) + 0.75 * sky * np.array([0.80, 0.88, 1.0]))
    return rgb.astype(np.float32), np.where(mask, dep, np.inf), mask


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.clip(x, 1e-8, None), 1 / 2.4) - 0.055)


# one exposure for the clip: the character's median luminance -> mid grey after the curve
lum = []
for i in range(0, len(frames), max(1, len(frames) // 12)):
    rgb, _, m = load_cap(i)
    if m.sum() > 500:
        lum.append(np.median((rgb[m] @ [0.2126, 0.7152, 0.0722])))
EXPO = 0.22 / max(float(np.median(lum)), 1e-6)
print(f"{len(frames)} frames, character median luminance {np.median(lum):.4g} -> exposure {EXPO:.4g}", flush=True)

FONT_T = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 40)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
WALL = lg["wall_index"]
ground_z = [None]


def shadow_mask(fr, M):
    """Soft contact shadow: a 45 cm disc on the floor under the capsule, fading while airborne."""
    px, py, pz = fr["pawn"]
    foot = pz - half
    if not fr["falling"] or ground_z[0] is None:
        ground_z[0] = foot
    hgt = max(0.0, foot - ground_z[0])
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rad = 45.0 + 0.15 * hgt
    pts = np.stack([px + rad * np.cos(ang), py + rad * np.sin(ang), np.full_like(ang, ground_z[0] + 2.0)], 1)
    P = ue2splat(pts)
    q = (M[:3, :3] @ P.T).T + M[:3, 3]
    if (q[:, 2] < 0.05).any():
        return None
    uv = (K[:2, :2] @ (q[:, :2] / q[:, 2:3]).T).T + K[:2, 2]
    m = np.zeros((CH, CW), np.float32)
    cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 1.0, lineType=cv2.LINE_AA)
    r_px = max(3.0, 0.35 * float(np.ptp(uv[:, 0])) / 2)
    return cv2.GaussianBlur(m, (0, 0), r_px) * float(np.exp(-hgt / 120.0))


def frame(i):
    fr = frames[i]
    M, _ = view(fr)
    rgb_s, dep_s, al_s = render(g, M, K, CW, CH, near=0.1 / mpu, far=1e4)
    sp = rgb_s.cpu().numpy()
    al = al_s.cpu().numpy()
    dep_cm = dep_s.cpu().numpy() * mpu * 100.0
    sp = sp + (1 - al[..., None]) * np.array([0.55, 0.62, 0.70], np.float32) * np.linspace(1.0, 0.8, CH)[:, None, None]
    sh = shadow_mask(fr, M)
    if sh is not None:
        sp = sp * (1 - 0.5 * sh[..., None])
    ch, cdep, cm = load_cap(fr["i"])
    vis = cm & ~((al > 0.6) & (dep_cm < cdep - 25.0))
    col = srgb(aces(ch * EXPO))
    k = cv2.GaussianBlur(vis.astype(np.float32), (0, 0), 0.6)[..., None]
    img = sp * (1 - k) + col * k
    img = cv2.resize(np.clip(img, 0, 1).astype(np.float32), (OW, OH), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray((img * 255 + 0.5).astype(np.uint8))
    dr = ImageDraw.Draw(pil, "RGBA")
    st = dict(stroke_width=2, stroke_fill=(0, 0, 0, 200))
    dr.text((44, 40), "UE 5.8 · 单人在 3DGS 场景里跑", font=FONT_T, fill=(255, 255, 255, 255), **st)
    dr.text((44, 100), "角色移动、跳跃和弹簧臂相机由 UE 实时计算（体素碰撞外壳 + 相机体积）；3DGS 由 gsplat 按同一机位渲染后合成",
            font=FONT_S, fill=(228, 235, 244, 255), **st)
    if fr["wp"] == WALL and i > 0 and frames[i - 1]["wp"] == WALL:
        p0 = frames[max(0, i - 25)]["pawn"]
        if math.hypot(fr["pawn"][0] - p0[0], fr["pawn"][1] - p0[1]) < 60:
            dr.text((44, OH - 80), "撞到碰撞外壳：场景边界走不出去", font=FONT_T, fill=(120, 230, 255, 255), **st)
    return np.asarray(pil)


# output frames on a fixed 30 fps clock (game time of each capture)
ts = np.array([f["t"] for f in frames])
n_out = int(ts[-1] * a.fps)
pick = [int(np.argmin(np.abs(ts - k / a.fps))) for k in range(n_out)]
os.makedirs(f"{run}/showcase", exist_ok=True)
if a.still:
    for s in a.still.split(","):
        i = pick[min(int(float(s) * a.fps), n_out - 1)]
        cv2.imwrite(f"{run}/showcase/ue_run_t{float(s):04.1f}.jpg", frame(i)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("still", s, flush=True)
else:
    import imageio_ffmpeg
    out = f"{run}/showcase/ue_run.mp4"
    wr = imageio_ffmpeg.write_frames(out, (OW, OH), fps=a.fps, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", str(a.crf), "-preset", "medium"], macro_block_size=2)
    wr.send(None)
    for k, i in enumerate(pick):
        wr.send(np.ascontiguousarray(frame(i)).tobytes())
        if k % 60 == 0:
            print(f"frame {k}/{n_out}", flush=True)
    wr.close()
    print("wrote", out, flush=True)
