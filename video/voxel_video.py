"""Explainer video: how a 3DGS scene becomes a voxel collision body, step by step.

    py.bat voxel_collision.py <ply> --out out/final_vox ... --save-stages 1      (first)
    py.bat video/voxel_video.py out/final_vox --still 3,10,18,26,36,44 --scale 0.5   (preview frames)
    py.bat video/voxel_video.py out/final_vox                                      (the video)

The splat is rendered with gsplat; the voxels are ray-cast through the voxel grid (3D DDA, up to
--layers occupied cells per ray, translucent fill + bright cube edges) and composited on top.
Stages (seconds): 0 scene, 4 voxelisation sweep, 16 clean-up, 21 walkable flood, 31 collision
shell, 41 final. Writes <run>/voxelization.mp4 (H.264 through imageio-ffmpeg).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from gs_core import load_ply, look_at, intrinsics, render

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--out", default=None)
ap.add_argument("--still", default=None, help="comma-separated times (s): write PNG previews instead of the video")
ap.add_argument("--scale", type=float, default=1.0, help="resolution scale of 1920x1080")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--fov", type=float, default=62.0)
ap.add_argument("--orbit", type=float, nargs=4, default=[-2.0, -5.0, 28.0, 36.0],
                help="orbit centre x y, radius, height above the seed floor (scene units)")
ap.add_argument("--arc", type=float, nargs=2, default=[-45.0, 150.0], help="orbit start angle and span (degrees)")
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--crf", type=int, default=18)
a = ap.parse_args()
dev = "cuda"
W, H = int(round(1920 * a.scale / 2) * 2), int(round(1080 * a.scale / 2) * 2)
T_END = 47.0

meta = json.load(open(f"{a.run}/meta.json"))
c, R = np.array(meta["c"]), np.array(meta["R"])
lo, v = np.array(meta["lo"]), float(meta["voxel_units"])
nx, ny, nz = meta["dims"]
mpu = meta["mpu"]
st = np.load(f"{a.run}/stages.npz")
un = lambda k: torch.from_numpy(np.unpackbits(st[k])[: nx * ny * nz].reshape(nx, ny, nz).astype(bool)).to(dev)
solid_raw, solid_clean, reach, free, shell = un("solid_raw"), un("solid_clean"), un("reach"), un("free"), un("shell")
seed = [int(x) for x in st["seed"]]
floaters = solid_raw & ~solid_clean
S = int(meta["args"]["step"] / meta["args"]["voxel"] + 1e-6)
kk = torch.arange(nz, device=dev)

# ---- walkable flood: arrival iteration of each reachable foot, shown on the floor voxel below it -----
dist = torch.full((nx, ny, nz), -1, dtype=torch.int16, device=dev)
r = torch.zeros((1, 1, nx, ny, nz), dtype=torch.float16, device=dev)
r[0, 0, seed[0], seed[1], seed[2]] = 1
dist[seed[0], seed[1], seed[2]] = 0
rf = reach[None, None].half()
it = 0
while True:
    it += 1
    rn = F.max_pool3d(r, (3, 3, 2 * S + 1), stride=1, padding=(1, 1, S)) * rf
    new = (rn > 0) & (r == 0)
    if not new.any():
        break
    dist[new[0, 0]] = it
    r = rn
N_WAVE = it
wave = torch.full((nx, ny, nz), 1e9, dtype=torch.float32, device=dev)
fi = (dist >= 0).nonzero()
fi = fi[fi[:, 2] > 0]
wave[fi[:, 0], fi[:, 1], fi[:, 2] - 1] = dist[fi[:, 0], fi[:, 1], fi[:, 2]].float()
wave_floor = wave < 1e9

# floor height per column (lowest reachable foot), extrapolated, for the shell growing out of the ground
has = reach.any(2).cpu().numpy()
kf = torch.where(reach, kk, torch.full_like(kk, nz)).amin(2).float().cpu().numpy()
num, den = np.zeros(kf.shape), np.zeros(kf.shape)
for sig in (2.0, 8.0, 32.0):
    n_ = ndimage.gaussian_filter(np.where(has, kf, 0.0), sig)
    d_ = ndimage.gaussian_filter(has.astype(float), sig)
    t_ = (den < 0.05) & (d_ > 1e-9)
    num[t_], den[t_] = n_[t_], d_[t_]
kfloor = torch.from_numpy(np.where(den > 0, num / np.maximum(den, 1e-12), 0)).float().to(dev)
fr = free[None, None].half()
near_free = (F.max_pool3d(fr, 3, stride=1, padding=1)[0, 0] > 0)
shell_h = (kk[None, None, :].float() - kfloor[:, :, None])               # voxels above the floor
# shown: the shell's surface layer, walls cut at 8 m above the floor (they go up to the grid top)
SHELL_CAP = 8.0 / (v * mpu)
shell_surf = shell & near_free & (shell_h <= SHELL_CAP)

# sweep: along level x, with per-voxel jitter so the front is ragged, like cubes popping in
gen = torch.Generator(device=dev).manual_seed(0)
jit = torch.rand((nx, ny, nz), generator=gen, device=dev) * 0.16 - 0.08
xs = torch.linspace(0, 1, nx, device=dev)[:, None, None]
reveal = (xs + jit + 0.08) / 1.16                                         # 0..1

print(f"grid {nx}x{ny}x{nz} @ {v * mpu:.2f} m; raw {int(solid_raw.sum())}, floaters {int(floaters.sum())}, "
      f"walkable flood {N_WAVE} steps, shell surface {int(shell_surf.sum())} voxels", flush=True)

# ---- splat ------------------------------------------------------------------------------------------
g = load_ply(meta["ply"], dev)
to_w = lambda p: c + np.asarray(p, dtype=np.float64) @ R
z_floor = lo[2] + seed[2] * v
fx = 0.5 * W / math.tan(math.radians(a.fov) / 2)
K = intrinsics(W, H, a.fov)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def camera(t):
    """Level-frame eye and target: a slow orbit around the scene centre."""
    ox, oy, rad, hgt = a.orbit
    ang = math.radians(a.arc[0] + a.arc[1] * smooth(t / T_END))
    rad_t = rad * (1.0 - 0.15 * smooth((t - 36) / 10))                      # slight push-in at the end
    eye = np.array([ox + rad_t * math.cos(ang), oy + rad_t * math.sin(ang), z_floor + hgt * (1 - 0.2 * smooth((t - 36) / 10))])
    tgt = np.array([ox, oy, z_floor + 2.0])
    return eye, tgt


def basis(eye, tgt):
    f = tgt - eye
    f /= np.linalg.norm(f)
    r_ = np.cross(f, [0.0, 0.0, 1.0])
    r_ /= np.linalg.norm(r_)
    d_ = np.cross(f, r_)
    return r_, d_, f


# ---- voxel ray casting ----------------------------------------------------------------------------
PAL = {  # class: fill rgb, edge rgb
    1: ((0.05, 0.22, 0.50), (0.62, 0.86, 1.00)),     # solid
    2: ((0.85, 0.12, 0.08), (1.00, 0.60, 0.50)),     # floaters
    4: ((0.06, 0.50, 0.25), (0.70, 1.00, 0.75)),     # walkable floor
    5: ((0.03, 0.38, 0.48), (0.50, 0.98, 1.00)),     # collision shell
}


@torch.no_grad()
def raycast(cls, alpha_mul, eye, tgt):
    """cls uint8 [nx,ny,nz] (0 empty); alpha_mul [6] per class. Returns premultiplied rgb [H,W,3], alpha [H,W]."""
    r_, d_, f = basis(eye, tgt)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    xn, yn = (xx + 0.5 - W / 2) / fx, (yy + 0.5 - H / 2) / fx
    rt, dt_, ft = (torch.tensor(q, dtype=torch.float32, device=dev) for q in (r_, d_, f))
    dirs = (xn[..., None] * rt + yn[..., None] * dt_ + ft).view(-1, 3)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    og = torch.tensor((eye - lo) / v + 0.5, dtype=torch.float32, device=dev)
    dg = dirs / v
    dg = torch.where(dg.abs() < 1e-9, torch.full_like(dg, 1e-9), dg)
    inv = 1.0 / dg
    nvec = torch.tensor([nx, ny, nz], dtype=torch.float32, device=dev)
    t0, t1 = (0 - og) * inv, (nvec - og) * inv
    tmin, tmax = torch.minimum(t0, t1), torch.maximum(t0, t1)
    tnear, axis0 = tmin.max(1)
    tnear = tnear.clamp(min=0)
    tfar = tmax.min(1).values
    act = (tfar > tnear).nonzero()[:, 0]
    out = torch.zeros((H * W, 3), device=dev)
    T = torch.ones(H * W, device=dev)
    if len(act) == 0:
        return out.view(H, W, 3), (1 - T).view(H, W)
    pid = act
    dgA, invA = dg[act], inv[act]
    te = tnear[act]
    p0 = og + dgA * (te + 1e-4)[:, None]
    cell = p0.floor().long()
    cell = torch.minimum(torch.maximum(cell, torch.zeros_like(cell)), (nvec.long() - 1)[None])
    step = torch.sign(dgA).long()
    tmax_ = ((cell + (step > 0).long()).float() - og) * invA
    tdel = invA.abs()
    last = axis0[act]
    tfA = tfar[act]
    TA = torch.ones(len(act), device=dev)
    acc = torch.zeros((len(act), 3), device=dev)
    hits = torch.zeros(len(act), dtype=torch.int8, device=dev)
    flat = cls.view(-1)
    fill_c = torch.zeros((6, 3), device=dev)
    edge_c = torch.zeros((6, 3), device=dev)
    for k_, (fc, ec) in PAL.items():
        fill_c[k_], edge_c[k_] = torch.tensor(fc), torch.tensor(ec)
    am = torch.tensor(alpha_mul, dtype=torch.float32, device=dev)
    pix_ang = 1.0 / fx
    for itr in range(nx + ny + nz + 2):
        idx = (cell[:, 0] * ny + cell[:, 1]) * nz + cell[:, 2]
        cc = flat[idx].long()
        hit = cc > 0
        if hit.any():
            h = hit.nonzero()[:, 0]
            ax = last[h]
            p = og + dgA[h] * te[h, None]
            fr_ = p - p.floor()
            # in-face coordinates: the two axes other than the entry axis
            oth = torch.stack([(ax + 1) % 3, (ax + 2) % 3], 1)
            uw = torch.gather(fr_, 1, oth)
            ed = torch.minimum(uw, 1 - uw).min(1).values
            pxv = te[h] * pix_ang / v                                     # pixel size in voxels
            lw = (1.1 * pxv).clamp(0.035, 0.22)
            e = (1 - (ed - 0.5 * lw) / lw).clamp(0, 1) * ((1 / pxv - 1.5) / 3).clamp(0, 1)
            sgn = torch.gather(step[h], 1, ax[:, None])[:, 0]
            shade = torch.where(ax == 2, torch.where(sgn < 0, torch.full_like(e, 1.0), torch.full_like(e, 0.45)),
                                torch.where(ax == 0, torch.full_like(e, 0.78), torch.full_like(e, 0.64)))
            ch = cc[h]
            first = hits[h] == 0
            fa = torch.where(first, torch.full_like(e, 0.42), torch.full_like(e, 0.16))
            col = fill_c[ch] * shade[:, None] * (1 - e[:, None]) + edge_c[ch] * e[:, None]
            al = (fa * (1 - e) + 0.95 * e) * am[ch]
            acc[h] += (TA[h] * al)[:, None] * col
            TA[h] = TA[h] * (1 - al)
            hits[h] += 1
        # advance one cell
        axn = tmax_.argmin(1)
        te = torch.gather(tmax_, 1, axn[:, None])[:, 0]
        cell.scatter_add_(1, axn[:, None], torch.gather(step, 1, axn[:, None]))
        tmax_.scatter_add_(1, axn[:, None], torch.gather(tdel, 1, axn[:, None]))
        last = axn
        alive = (te < tfA) & (hits < a.layers) & (TA > 0.02) & \
                (cell >= 0).all(1) & (cell < nvec.long()[None]).all(1)
        if itr % 8 == 7 or not alive.all():
            done = ~alive
            if done.any():
                d = done.nonzero()[:, 0]
                out[pid[d]] = acc[d]
                T[pid[d]] = TA[d]
                k = alive.nonzero()[:, 0]
                if len(k) == 0:
                    break
                pid, dgA, te, cell, step, tmax_, tdel, last, tfA, TA, acc, hits = (
                    x[k] for x in (pid, dgA, te, cell, step, tmax_, tdel, last, tfA, TA, acc, hits))
    else:
        out[pid] = acc
        T[pid] = TA
    return out.view(H, W, 3), (1 - T).view(H, W)


# ---- captions ---------------------------------------------------------------------------------------
FONT_NUM = ImageFont.truetype("C:/Windows/Fonts/bahnschrift.ttf", int(64 * a.scale))
FONT_T = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", int(52 * a.scale))
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", int(30 * a.scale))
STAGES = [  # start, end, number, title, subtitle
    (0.0, 4.0, "", "3D 高斯泼溅场景", "625 万个高斯，看起来是实体，其实没有表面：相机和角色会直接穿过去"),
    (4.0, 16.0, "01", "体素化", "把空间切成 10 cm 的方格，高斯密度超过阈值的格子记为实心"),
    (16.0, 21.0, "02", "清理", "删除悬空的小碎块（浮游物）"),
    (21.0, 31.0, "03", "可走区域", "胶囊体（身高 1.8 m、半径 0.35 m）从起点出发，找出能站、能走到的地面"),
    (31.0, 41.0, "04", "碰撞外壳", "角色到不了的地方一律填实：地面的洞被封住，边界变成墙"),
    (41.0, T_END, "05", "碰撞体", "合并成 9.4 万个三角形，作为隐形碰撞体：掉不进洞，也走不出范围"),
]


BAND = None


def caption(img, t, seed_px=None):
    global BAND
    if BAND is None:                                              # dark gradient behind the titles
        hb = int(210 * a.scale)
        BAND = np.ones((H, W, 1), np.float32)
        BAND[:hb, :, 0] = 0.45 + 0.55 * (np.arange(hb, dtype=np.float32) / hb)[:, None] ** 1.5
    img = (img.astype(np.float32) * BAND).astype(np.uint8)
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil, "RGBA")
    for s0, s1, num, title, sub in STAGES:
        if s0 <= t < s1 + (0.0 if s1 < T_END else 1.0):
            fade = min(1.0, (t - s0) / 0.6, max(0.0, (s1 - t) / 0.4) if s1 < T_END else 1.0)
            al = int(255 * fade)
            x, y = int(60 * a.scale), int(48 * a.scale)
            sw = max(1, int(3 * a.scale))
            if num:
                dr.text((x, y - int(4 * a.scale)), num, font=FONT_NUM, fill=(40, 150, 255, al),
                        stroke_width=sw, stroke_fill=(0, 0, 0, int(al * 0.8)))
                x += int(FONT_NUM.getlength(num) + 22 * a.scale)
            dr.text((x, y), title, font=FONT_T, fill=(255, 255, 255, al), stroke_width=sw, stroke_fill=(0, 0, 0, int(al * 0.8)))
            dr.text((int(62 * a.scale), y + int(84 * a.scale)), sub, font=FONT_S, fill=(235, 242, 250, al),
                    stroke_width=max(1, int(2 * a.scale)), stroke_fill=(0, 0, 0, int(al * 0.85)))
    if seed_px is not None:
        (sx, sy), pulse = seed_px
        rr = int((10 + 8 * pulse) * a.scale)
        dr.ellipse((sx - rr, sy - rr, sx + rr, sy + rr), outline=(255, 240, 120, 230), width=max(2, int(3 * a.scale)))
        dr.text((sx + rr + 6, sy - int(18 * a.scale)), "起点", font=FONT_S, fill=(255, 240, 120, 230))
    return np.asarray(pil)


# ---- one frame --------------------------------------------------------------------------------------
def frame(t):
    eye, tgt = camera(t)
    M = look_at(to_w(eye), to_w(tgt), R[2])
    rgb, depth, alpha = render(g, M, K, W, H, near=0.2, far=1e4)
    cls = torch.zeros((nx, ny, nz), dtype=torch.uint8, device=dev)
    am = [0, 1, 1, 1, 1, 1]
    dim = 1.0
    sweep = None
    seed_px = None
    if 4.0 <= t < 16.0:
        s = (t - 4.5) / 10.0
        sweep = s
        cls[solid_raw & (reveal <= s)] = 1
    elif 16.0 <= t < 21.0:
        cls[solid_clean] = 1
        on = (math.sin((t - 16.0) * 10) > -0.3) if t < 18.0 else True
        cls[floaters] = 2 if on else 1
        am[2] = 1.0 - smooth((t - 18.0) / 2.0)
        dim = 0.3
    elif 21.0 <= t < 31.0:
        cls[solid_clean] = 1
        w = N_WAVE * smooth((t - 21.5) / 8.0)
        cls[wave_floor & (wave <= w)] = 4
        dim = 0.3
        seed_px = seed_project(eye, tgt, t)
    elif 31.0 <= t < 41.0:
        # the old voxels fade while the shell grows out of the ground over the same seconds
        fade_old = 1.0 - smooth((t - 31.5) / 5.5)
        if fade_old > 0:
            cls[solid_clean] = 1
            cls[wave_floor] = 4
        grow = (t - 31.5) / 5.0 * SHELL_CAP
        cls[shell_surf & (shell_h <= grow)] = 5
        am[1] = am[4] = fade_old
        dim = 0.45
    elif t >= 41.0:
        cls[shell_surf] = 5
        am[5] = 1.0 - 0.6 * smooth((t - 41.0) / 2.0)
        dim = 0.45 + 0.55 * smooth((t - 41.0) / 2.0)
    vox, va = raycast(cls, am, eye, tgt) if t >= 4.0 else (torch.zeros((H, W, 3), device=dev), torch.zeros((H, W), device=dev))
    base = rgb
    if sweep is not None:
        # dim the splat where the sweep has passed (by the splat's own surface position)
        r_, d_, f = basis(eye, tgt)
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        xn, yn = (xx + 0.5 - W / 2) / fx, (yy + 0.5 - H / 2) / fx
        px = eye[0] + depth * (xn * r_[0] + yn * d_[0] + f[0])
        u = ((px - lo[0]) / v + 0.5) / nx
        k_ = ((sweep - u) / 0.04).clamp(0, 1)
        base = rgb * (1 - 0.7 * k_[..., None])
    else:
        base = rgb * dim
    img = base * (1 - va[..., None]) + vox
    # glow: blurred voxel layer added on top
    vn = (vox.clamp(0, 1) * 255).byte().cpu().numpy()
    glow = cv2.GaussianBlur(vn, (0, 0), 6 * a.scale).astype(np.float32) / 255.0
    img = (img.clamp(0, 1).cpu().numpy() + 0.22 * glow).clip(0, 1)
    img = (img * 255).astype(np.uint8)
    return caption(img, t, seed_px)


def seed_project(eye, tgt, t):
    p = lo + np.array(seed) * v
    r_, d_, f = basis(eye, tgt)
    q = p - eye
    z = q @ f
    if z <= 0:
        return None
    return ((W / 2 + fx * (q @ r_) / z, H / 2 + fx * (q @ d_) / z), 0.5 + 0.5 * math.sin(t * 6))


# camera sanity: the orbit should stay in free air
for t_ in np.linspace(0, T_END, 24):
    e_, _ = camera(t_)
    i_ = np.round((e_ - lo) / v).astype(int)
    if (i_ >= 0).all() and (i_ < [nx, ny, nz]).all() and bool(solid_raw[i_[0], i_[1], i_[2]]):
        print(f"warning: camera inside solid at t={t_:.1f}s, {e_}")

if a.still:
    os.makedirs(f"{a.run}/video_stills", exist_ok=True)
    for ts in a.still.split(","):
        t0 = time.time()
        im = frame(float(ts))
        cv2.imwrite(f"{a.run}/video_stills/t{float(ts):05.1f}.jpg", im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"still t={ts}s in {time.time() - t0:.1f}s", flush=True)
else:
    import imageio_ffmpeg
    out = a.out or f"{a.run}/voxelization.mp4"
    wr = imageio_ffmpeg.write_frames(out, (W, H), fps=a.fps, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", str(a.crf), "-preset", "slow"], macro_block_size=2)
    wr.send(None)
    n = int(T_END * a.fps)
    t0 = time.time()
    for i in range(n):
        wr.send(np.ascontiguousarray(frame(i / a.fps)).tobytes())
        if i % 30 == 0:
            el = time.time() - t0
            print(f"frame {i}/{n}  {el:.0f}s elapsed, eta {el / max(i, 1) * (n - i):.0f}s", flush=True)
    wr.close()
    print("wrote", out)
