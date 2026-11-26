"""Process video: how a 3DGS scene becomes (1) voxels, (2) the characters' collision and (3) the camera's volume.

    py.bat voxel_collision.py <ply> --out <run> ... --save-stages 1      (stages incl. cam_clear / cam_foot / cam_cap)
    py.bat video/showcase.py <run> bake                                  (voxel colours from the splat)
    py.bat video/process_video.py <run> --still 1,5,9,13,18,23,27,31,34 --scale 0.5     (previews)
    py.bat video/process_video.py <run>                                  -> <run>/showcase/process.mp4

Sections and steps (seconds):
  场景体素化     0 scene · 2.5 01 voxelisation sweep (voxels coloured from the splat) · 7.5 02 floaters removed
  角色碰撞区域   10 03 walkable floor flood from the seed · 15.5 04 collision shell grows out of the ground
  相机区域       21 05 free air · 24.5 06 0.3 m clearance from anything visible · 28.5 07 capped 5 m above the floor
                 · 32-34.5 08 the camera volume (footprint and eye-connected pieces only)
The camera section is a cutaway: everything (splat and voxels) between the camera and a vertical plane through the
statue is removed, so the volume's cross-section shows its height and the gap kept around objects; at the end the
camera rises a little to show the top of the volume.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from gs_core import load_ply, look_at, intrinsics, render
from vox_render import VoxRender, PAL

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--still", default=None)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--crf", type=int, default=19)
ap.add_argument("--ss", type=int, default=2)
a = ap.parse_args()
dev = "cuda"
run = a.run.rstrip("/\\")
W, H = int(round(1920 * a.scale / 2) * 2), int(round(1080 * a.scale / 2) * 2)
T_END = 34.5

meta = json.load(open(f"{run}/meta.json"))
c, R = np.array(meta["c"]), np.array(meta["R"])
lo, v = np.array(meta["lo"]), float(meta["voxel_units"])
nx, ny, nz = meta["dims"]
N = nx * ny * nz
mpu = meta["mpu"]
st = np.load(f"{run}/stages.npz")
un = lambda k: torch.from_numpy(np.unpackbits(st[k])[:N].reshape(nx, ny, nz).astype(bool)).to(dev)
solid_raw, solid, reach, free, shell = un("solid_raw"), un("solid_clean"), un("reach"), un("free"), un("shell")
camv, cam_clear, cam_foot, cam_cap = un("camera"), un("cam_clear"), un("cam_foot"), un("cam_cap")
floaters = solid_raw & ~solid
seed = [int(x) for x in st["seed"]]
z_floor = lo[2] + seed[2] * v
STATUE = np.array([-2.0, -5.0, z_floor])
S = int(meta["args"]["step"] / meta["args"]["voxel"] + 1e-6)
kk = torch.arange(nz, device=dev)
to_w = lambda p: c + np.asarray(p, dtype=np.float64) @ R
os.makedirs(f"{run}/showcase", exist_ok=True)

# ---- voxel colours (class 1) and a dimmed grey version for the collision steps ----------------------
COL = torch.zeros((N, 3), dtype=torch.float16, device=dev)
COL[torch.from_numpy(np.flatnonzero(solid.cpu().numpy())).to(dev)] = \
    torch.from_numpy(np.load(f"{run}/voxel_rgb.npz")["rgb"]).to(dev).half() / 255
lum = (COL.float() @ torch.tensor([0.3, 0.59, 0.11], device=dev))[:, None]
COL_DIM = (0.18 + 0.35 * lum).expand(-1, 3).half().contiguous()
PAL[3] = ((0.55, 0.55, 0.58), (0.85, 0.85, 0.9))               # floaters before they are flagged
vr = VoxRender(lo, v, (nx, ny, nz), COL, ss=a.ss)

# ---- walkable flood: arrival step of each reachable foot, painted on the floor voxel below it -------
dist = torch.full((nx, ny, nz), -1, dtype=torch.int16, device=dev)
r_ = torch.zeros((1, 1, nx, ny, nz), dtype=torch.float16, device=dev)
r_[0, 0, seed[0], seed[1], seed[2]] = 1
dist[seed[0], seed[1], seed[2]] = 0
rf = reach[None, None].half()
it = 0
while True:
    it += 1
    rn = F.max_pool3d(r_, (3, 3, 2 * S + 1), stride=1, padding=(1, 1, S)) * rf
    new = (rn > 0) & (r_ == 0)
    if not new.any():
        break
    dist[new[0, 0]] = it
    r_ = rn
N_WAVE = it
wave = torch.full((nx, ny, nz), 1e9, dtype=torch.float32, device=dev)
fi = (dist >= 0).nonzero()
fi = fi[fi[:, 2] > 0]
wave[fi[:, 0], fi[:, 1], fi[:, 2] - 1] = dist[fi[:, 0], fi[:, 1], fi[:, 2]].float()
wave_floor = (wave < 1e9) & solid

# floor height per column (extrapolated walkable floor) for the shell growing out of the ground / the cap
has = reach.any(2).cpu().numpy()
kf = torch.where(reach, kk, torch.full_like(kk, nz)).amin(2).float().cpu().numpy()
num, den = np.zeros(kf.shape), np.zeros(kf.shape)
for sig in (2.0, 8.0, 32.0):
    n_ = ndimage.gaussian_filter(np.where(has, kf, 0.0), sig)
    d_ = ndimage.gaussian_filter(has.astype(float), sig)
    t_ = (den < 0.05) & (d_ > 1e-9)
    num[t_], den[t_] = n_[t_], d_[t_]
kfloor = torch.from_numpy(np.where(den > 0, num / np.maximum(den, 1e-12), 0)).float().to(dev)
h_m = (kk[None, None, :].float() - kfloor[:, :, None]) * v * mpu        # metres above the floor
near_free = F.max_pool3d(free[None, None].half(), 3, stride=1, padding=1)[0, 0] > 0
shell_surf = shell & near_free & (h_m <= 2.5)                             # walls shown up to 2.5 m
cut_clear = free & ~cam_clear                                             # removed: too close to something visible
cut_cap = cam_foot & ~cam_cap                                             # removed: above the cap
cut_rest = cam_clear & ~camv & ~cut_cap                                   # removed: off the footprint, pockets
H_TOP = float(h_m[free].max())
print(f"grid {nx}x{ny}x{nz}; flood {N_WAVE} steps; free {int(free.sum()) * (v * mpu) ** 3:.0f} m^3, "
      f"camera {int(camv.sum()) * (v * mpu) ** 3:.0f} m^3, free air up to {H_TOP:.1f} m", flush=True)

# sweep order for the voxelisation: along level x with per-voxel jitter
gen = torch.Generator(device=dev).manual_seed(0)
jit = torch.rand((nx, ny, nz), generator=gen, device=dev) * 0.16 - 0.08
reveal = (torch.linspace(0, 1, nx, device=dev)[:, None, None] + jit + 0.08) / 1.16
XY = torch.stack(torch.meshgrid(torch.tensor(lo[0] + (np.arange(nx) + 0.5) * v, device=dev, dtype=torch.float32),
                                torch.tensor(lo[1] + (np.arange(ny) + 0.5) * v, device=dev, dtype=torch.float32),
                                indexing="ij"), -1)                       # [nx,ny,2] voxel centres

# ---- splat ---------------------------------------------------------------------------------------
g = load_ply(meta["ply"], dev)
P_l = ((g.means.double() - torch.tensor(c, device=dev)) @ torch.tensor(R, device=dev).T).float()
n_gauss = len(g.means)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, t0, t1):
    return smooth((t - t0) / (t1 - t0))


def camera(t):
    """Inside the clearing for the voxels, higher for the collision, then out to a low cutaway view."""
    an = math.radians(-50.0 + 2.2 * t)                     # starts where no tree is in the way at 5 m
    k1, k2, k3 = ramp(t, 9.0, 11.0), ramp(t, 20.5, 22.5), ramp(t, 32.0, 34.5)
    rad = 32.0 * (1 - k2) + 62.0 * k2 - 6.0 * k3
    hgt = (26.0 * (1 - k1) + 50.0 * k1) * (1 - k2) + 22.0 * k2 + 14.0 * k3
    tz = 5.0 * (1 - k2) + 12.0 * k2
    return STATUE + [rad * math.cos(an), rad * math.sin(an), hgt], STATUE + [0, 0, tz]


FOV = 62.0
K = intrinsics(W, H, FOV)
_bg = None


def background():
    global _bg
    if _bg is None:
        y = torch.linspace(0, 1, H, device=dev)[:, None, None]
        _bg = (torch.tensor([0.10, 0.12, 0.16], device=dev) * (1 - y) + torch.tensor([0.035, 0.04, 0.05], device=dev) * y).expand(H, W, 3)
    return _bg


# ---- captions --------------------------------------------------------------------------------------
sc = a.scale
FONT_NUM = ImageFont.truetype("C:/Windows/Fonts/bahnschrift.ttf", int(64 * sc))
FONT_T = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", int(52 * sc))
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", int(30 * sc))
FONT_C = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", int(30 * sc))
shell_tris = None
if os.path.exists(f"{run}/collision.ply"):
    from plyfile import PlyData
    shell_tris = len(PlyData.read(f"{run}/collision.ply")["face"])
STEPS = [  # start, end, number, title, subtitle
    (0.0, 2.5, "", "3D 高斯泼溅场景", f"{n_gauss / 1e4:.0f} 万个高斯，只有颜色没有表面：角色会穿过去，相机会钻进物体里"),
    (2.5, 7.5, "01", "体素化", "把空间切成 10 cm 方格，高斯密度超过阈值的格子记为实心，颜色取自 3DGS"),
    (7.5, 10.0, "02", "清理", "删除悬空的小碎块（浮游物）"),
    (10.0, 15.5, "03", "可走区域", "胶囊体（身高 1.8 m、半径 0.35 m）从起点出发，按台阶和坡度找出能站、能走到的地面"),
    (15.5, 21.0, "04", "角色碰撞外壳", "角色到不了的地方一律填实：地面的洞被封住，场景边界变成墙"
                                   + (f"；合并成 {shell_tris / 1e4:.1f} 万三角形" if shell_tris else "")),
    (21.0, 24.5, "05", "相机可用空间", "相机只能在空气里：角色能到的地方往上的整片空间（剖面）"),
    (24.5, 28.5, "06", "离物体至少 0.3 m", "雕像、树冠、草尖（含半透明的高斯）外扩 0.3 m 挖掉：镜头不贴脸、不钻进模糊里"),
    (28.5, 32.0, "07", "限高 5 m", "最高到可走地面以上 5 m：再高会拍到场景外缘的空洞"),
    (32.0, T_END, "08", "相机体积", f"去掉场景范围外和角色视线到不了的碎块：{meta['camera']['volume_m3']:.0f} m³，"
                                   "导出为只挡相机的碰撞体"),
]
SECTIONS = [(0.0, 10.0, "一 · 场景体素化"), (10.0, 21.0, "二 · 角色碰撞区域"), (21.0, T_END, "三 · 相机区域")]
BAND = None


def caption(img, t, seed_px=None):
    global BAND
    if BAND is None:
        hb = int(210 * sc)
        BAND = np.ones((H, W, 1), np.float32)
        BAND[:hb, :, 0] = 0.45 + 0.55 * (np.arange(hb, dtype=np.float32) / hb)[:, None] ** 1.5
    img = (img.astype(np.float32) * BAND).astype(np.uint8)
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil, "RGBA")
    sw = max(1, int(3 * sc))
    for s0, s1, num, title, sub in STEPS:
        if s0 <= t < s1 + (1.0 if s1 >= T_END else 0.0):
            fade = min(1.0, (t - s0) / 0.4 + (1.0 if s0 == 0 else 0.0), max(0.0, (s1 - t) / 0.3) if s1 < T_END else 1.0)
            al = int(255 * fade)
            x, y = int(60 * sc), int(48 * sc)
            if num:
                dr.text((x, y - int(4 * sc)), num, font=FONT_NUM, fill=(40, 150, 255, al), stroke_width=sw,
                        stroke_fill=(0, 0, 0, int(al * 0.8)))
                x += int(FONT_NUM.getlength(num) + 22 * sc)
            dr.text((x, y), title, font=FONT_T, fill=(255, 255, 255, al), stroke_width=sw, stroke_fill=(0, 0, 0, int(al * 0.8)))
            dr.text((int(62 * sc), y + int(84 * sc)), sub, font=FONT_S, fill=(235, 242, 250, al),
                    stroke_width=max(1, int(2 * sc)), stroke_fill=(0, 0, 0, int(al * 0.85)))
    for s0, s1, name in SECTIONS:
        if s0 <= t < s1 + 1.0:
            w_ = FONT_C.getlength(name) + 36 * sc
            x1, y0 = W - int(40 * sc), int(40 * sc)
            dr.rounded_rectangle((x1 - w_, y0, x1, y0 + int(52 * sc)), int(12 * sc), fill=(0, 0, 0, 150))
            dr.text((x1 - w_ / 2, y0 + int(26 * sc)), name, font=FONT_C, fill=(255, 255, 255, 235), anchor="mm")
            break
    if seed_px is not None:
        (sx, sy), pulse = seed_px
        rr = int((10 + 8 * pulse) * sc)
        dr.ellipse((sx - rr, sy - rr, sx + rr, sy + rr), outline=(255, 240, 120, 230), width=max(2, int(3 * sc)))
        dr.text((sx + rr + 6, sy - int(18 * sc)), "起点", font=FONT_S, fill=(255, 240, 120, 230))
    return np.asarray(pil)


def seed_project(eye, tgt, t):
    from vox_render import basis
    p = lo + (np.array(seed) + 0.5) * v
    r_, d_, f = basis(eye, tgt)
    q = p - eye
    z = q @ f
    if z <= 0:
        return None
    fx = 0.5 * W / math.tan(math.radians(FOV) / 2)
    return (W / 2 + fx * (q @ r_) / z, H / 2 + fx * (q @ d_) / z), 0.5 + 0.5 * math.sin(t * 6)


# ---- one frame --------------------------------------------------------------------------------------
def frame(t):
    eye, tgt = camera(t)
    M = look_at(to_w(eye), to_w(tgt), R[2])
    bg = background()
    cutaway = t >= 21.0
    if cutaway:                                   # remove the half of the world between the camera and the statue
        fh = np.array([tgt[0] - eye[0], tgt[1] - eye[1]])
        fh /= np.linalg.norm(fh)
        off = 0.0
        fh_t = torch.tensor(fh, dtype=torch.float32, device=dev)
        keep_g = ((P_l[:, :2] - torch.tensor(STATUE[:2], dtype=torch.float32, device=dev)) @ fh_t) >= off
        gs = g.subset(keep_g)
        keep_v = (((XY - torch.tensor(STATUE[:2], dtype=torch.float32, device=dev)) @ fh_t) >= off)[:, :, None]
    else:
        gs = g
    sp, sz, sa = render(gs, M, K, W, H, near=0.2, far=1e4)
    cls = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=dev)
    alpha, layers, seed_px, sweep = {}, 1, None, None
    vr.colors = COL
    dim = 1.0
    if t < 2.5:                                                           # the splat alone
        pass
    elif t < 7.5:                                                         # 01 voxelisation sweep
        sweep = (t - 2.9) / 4.2
        cls[solid & (reveal <= sweep)] = 1
        cls[floaters & (reveal <= sweep)] = 3
    elif 7.5 <= t < 10.0:                                                 # 02 floaters
        cls[solid] = 1
        cls[floaters] = 2 if (t >= 8.6 or math.sin((t - 7.5) * 12) > -0.3) else 3
        alpha[2] = 1.0 - ramp(t, 8.6, 9.6)
        layers, dim = 2, 0.35
    elif 10.0 <= t < 15.5:                                                # 03 walkable flood
        k = ramp(t, 10.0, 10.8)
        vr.colors = (COL.float() * (1 - k) + COL_DIM.float() * k).half()
        cls[solid] = 1
        cls[wave_floor & (wave <= N_WAVE * ramp(t, 10.4, 14.8))] = 4
        dim = 0.3
        seed_px = seed_project(eye, tgt, t)
    elif 15.5 <= t < 21.0:                                                # 04 collision shell
        vr.colors = COL_DIM
        fade = 1.0 - ramp(t, 15.8, 19.0)
        if fade > 0:
            cls[solid] = 1
            cls[wave_floor] = 4
        cls[shell_surf & (h_m <= -1.0 + 3.6 * ramp(t, 15.8, 19.5))] = 5
        alpha = {1: fade, 4: fade, 5: 0.55 + 0.45 * fade}
        layers, dim = 3, 0.35 + 0.3 * ramp(t, 19.0, 20.5)
    elif 21.0 <= t < 24.5:                                                # 05 free air
        cls[free & keep_v] = 6
        alpha = {6: 0.5 * ramp(t, 21.0, 22.0)}
    elif 24.5 <= t < 28.5:                                                # 06 clearance
        cls[cam_clear & keep_v] = 6
        if t < 27.5:
            cls[cut_clear & keep_v] = 7
        alpha = {6: 0.5, 7: 0.85 * (1.0 - ramp(t, 26.3, 27.5)) * (0.75 + 0.25 * math.sin((t - 24.5) * 10))}
    elif 28.5 <= t < 32.0:                                                # 07 cap (the red top sinks)
        cls[cam_cap & keep_v] = 6
        top = H_TOP - (H_TOP - 5.0) * ramp(t, 28.8, 31.0)
        cls[cut_cap & keep_v & (h_m <= top)] = 7
        alpha = {6: 0.5, 7: 0.75}
    else:                                                                 # 08 the camera volume
        cls[camv & keep_v] = 6
        if t < 33.2:
            cls[cut_rest & keep_v] = 7
        alpha = {6: 0.5 + 0.1 * ramp(t, 32.0, 33.0), 7: 0.85 * (1 - ramp(t, 32.4, 33.2))}
    if t >= 2.5:
        vo, va, vz = vr.raycast(cls, eye, tgt, W, H, FOV, layers=layers, alpha=alpha)
    else:
        vo, va, vz = torch.zeros((H, W, 3), device=dev), torch.zeros((H, W), device=dev), torch.full((H, W), float("inf"), device=dev)
    splat_img = sp + (1 - sa[..., None]) * bg
    if sweep is not None:                                                 # dim the splat behind the sweep front
        from vox_render import pixel_dirs
        D, _ = pixel_dirs(eye, tgt, W, H, FOV)
        px = eye[0] + sz * D[..., 0]
        u = ((px - lo[0]) / v + 0.5) / nx
        kd = ((sweep - u) / 0.04).clamp(0, 1)
        splat_img = splat_img * (1 - 0.7 * kd[..., None])
    else:
        splat_img = splat_img * dim
    if cutaway:                                                           # splat in front of the volume wins
        sp_front = (sa > 0.5) & (sz < vz - 0.3)
        img = torch.where(sp_front[..., None], splat_img, vo + (1 - va[..., None]) * splat_img)
    else:
        img = vo + (1 - va[..., None]) * splat_img
    img8 = (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return caption(img8, t, seed_px)


if a.still:
    for ts in a.still.split(","):
        t0 = time.time()
        im = frame(float(ts))
        fn = f"{run}/showcase/process_t{float(ts):04.1f}.jpg"
        cv2.imwrite(fn, im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{fn} in {time.time() - t0:.1f}s", flush=True)
else:
    import imageio_ffmpeg
    out = f"{run}/showcase/process.mp4"
    wr = imageio_ffmpeg.write_frames(out, (W, H), fps=a.fps, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", str(a.crf), "-preset", "medium"], macro_block_size=2)
    wr.send(None)
    n = int(round(T_END * a.fps))
    t0 = time.time()
    for i in range(n):
        wr.send(np.ascontiguousarray(frame(i / a.fps)).tobytes())
        if i % 90 == 0:
            el = time.time() - t0
            print(f"frame {i}/{n}  {el:.0f}s elapsed, eta {el / max(i, 1) * (n - i):.0f}s", flush=True)
    wr.close()
    print("wrote", out, flush=True)
