"""One short cut of all the demo videos of a voxel run: py.bat video/montage.py out/final_vox [--still 10,30]

Sources (all 1080p30): <run>/voxelization.mp4 (video/voxel_video.py, the step-by-step process) and
<run>/showcase/{orbit,wide,split,compare,ue_run}.mp4 (video/showcase.py, ue/run_video.py).
Each clip is cut and sped up (frame skipping, so motion stays at 30 fps); sped-up parts carry an "N× 速" tag.
Clips are joined with short cross-fades, a thin progress bar with chapter ticks runs along the bottom, and a
summary card closes the video. Writes <run>/showcase/montage.mp4.
"""
import argparse
import json
import math
import os
from collections import deque

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--still", default=None, help="output times (s): JPEG previews instead of the video")
ap.add_argument("--crf", type=int, default=20)
a = ap.parse_args()
run = a.run.rstrip("/\\")
W, H, FPS = 1920, 1080, 30
XF = 12                                                     # cross-fade frames (0.4 s)
meta = json.load(open(f"{run}/meta.json"))
info = json.load(open(f"{run}/ue/scene_ue.json"))
tris = {L["name"]: L["tris"] for L in info["layers"]}

# chapter: (name, file, [(t0, t1, speed), ...]) -- parts of one chapter are joined without a fade
CHAPTERS = [
    ("体素化过程", f"{run}/voxelization.mp4", [(0.5, 4.0, 1.0), (4.0, 16.0, 2.0), (16.0, 21.0, 1.5), (21.0, 31.0, 2.0),
                                              (31.0, 41.0, 2.0), (41.0, 47.0, 1.5)]),
    ("全视角", f"{run}/showcase/orbit.mp4", [(0.0, 24.0, 3.0)]),
    ("大视角 · 一半一半", f"{run}/showcase/wide.mp4", [(0.0, 16.0, 2.0)]),
    ("分屏", f"{run}/showcase/split.mp4", [(2.5, 10.0, 1.5)]),
    ("对比", f"{run}/showcase/compare.mp4", [(0.0, 14.0, 2.0)]),
    ("UE 实跑", f"{run}/showcase/ue_run.mp4", [(0.0, 11.0, 2.0), (11.0, 17.2, 1.0)]),
]
END_S = 4.0

FONT_TAG = ImageFont.truetype("C:/Windows/Fonts/bahnschrift.ttf", 30)
FONT_CH = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
FONT_T = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 60)
FONT_B = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 34)
FONT_BB = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 34)


def n_out(parts):
    return sum(int(round((t1 - t0) * FPS / sp)) for t0, t1, sp in parts)


lens = [n_out(p) for _, _, p in CHAPTERS]
n_end = int(END_S * FPS)
starts = []                                                 # output frame where each chapter starts
pos = 0
for k, L in enumerate(lens):
    starts.append(pos)
    pos += L - (XF if k + 1 < len(lens) else 0)
starts.append(pos - XF)                                     # end card fades in over the last chapter
TOTAL = pos - XF + n_end
print("chapters", [f"{c[0]} {L / FPS:.1f}s" for c, L in zip(CHAPTERS, lens)], f"-> total {TOTAL / FPS:.1f}s", flush=True)


def tag(img, speed):
    if speed == 1.0:
        return img
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil, "RGBA")
    s = f"{speed:g}×"
    w = FONT_TAG.getlength(s) + 28
    x1, y0 = W - 36, 36
    dr.rounded_rectangle((x1 - w, y0, x1, y0 + 46), 12, fill=(0, 0, 0, 140))
    dr.text((x1 - w / 2, y0 + 23), s, font=FONT_TAG, fill=(255, 255, 255, 235), anchor="mm")
    return np.asarray(pil)


def chapter_frames(path, parts):
    """Output frames of one chapter: source frame floor(j * speed) of each part."""
    for t0, t1, sp in parts:
        n = int(round((t1 - t0) * FPS / sp))
        rd = imageio_ffmpeg.read_frames(path, input_params=["-ss", f"{t0:.3f}"])
        meta_ = next(rd)
        assert tuple(meta_["size"]) == (W, H), (path, meta_["size"])
        i_src = -1
        cur = None
        for j in range(n):
            want = int(math.floor(j * sp + 1e-6))
            while i_src < want:
                cur = next(rd)
                i_src += 1
            yield tag(np.frombuffer(cur, np.uint8).reshape(H, W, 3), sp)
        rd.close()


def end_card():
    img = Image.new("RGB", (W, H), (14, 17, 23))
    dr = ImageDraw.Draw(img)
    for y in range(H):                                       # soft vertical gradient
        c_ = int(14 + 12 * (1 - y / H))
        dr.line([(0, y), (W, y)], fill=(c_, c_ + 3, c_ + 9))
    x, y = 180, 200
    dr.text((x, y), "3DGS → 体素碰撞", font=FONT_T, fill=(255, 255, 255))
    lines = [
        ("输入", f"一个 3DGS PLY（本例 625 万高斯），单卡 {meta['time_s']:.0f} s 体素化成 10 cm 方格，不依赖引擎"),
        ("角色碰撞外壳", f"{tris['VoxelShell'] / 1e4:.1f} 万三角形：角色掉不进洞、走不出场景"),
        ("相机体积", f"{meta['camera']['volume_m3']:.0f} m³：相机不进物体、不出场景；UE 实测相机出界 30.6% → 4.2%"),
        ("UE 5.8", "导入即用：外壳挡角色，相机体积只挡相机；64 个角色实测 0 次掉落"),
    ]
    y += 130
    for k_, v_ in lines:
        dr.text((x, y), k_, font=FONT_BB, fill=(90, 200, 255))
        dr.text((x + 260, y), v_, font=FONT_B, fill=(228, 235, 244))
        y += 76
    return np.asarray(img)


def progress(img, j):
    """Thin bar along the bottom with chapter ticks and the current chapter's name."""
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil, "RGBA")
    y0, h_ = H - 8, 8
    dr.rectangle((0, y0, W, H), fill=(0, 0, 0, 110))
    dr.rectangle((0, y0, int(W * j / TOTAL), H), fill=(90, 200, 255, 220))
    for s in starts[1:-1]:
        x = int(W * s / TOTAL)
        dr.rectangle((x - 1, y0 - 4, x + 1, H), fill=(255, 255, 255, 200))
    k = max(i for i, s in enumerate(starts[:-1]) if s <= j)
    if j < starts[-1]:
        x = int(W * (starts[k] + (starts[k + 1] - starts[k]) / 2) / TOTAL)
        dr.text((x, y0 - 8), CHAPTERS[k][0], font=FONT_CH, fill=(255, 255, 255, 200), anchor="mb",
                stroke_width=2, stroke_fill=(0, 0, 0, 160))
    return np.asarray(pil)


def all_frames():
    tail = deque()
    for k, (name, path, parts) in enumerate(CHAPTERS):
        gen = chapter_frames(path, parts)
        for i, fr in enumerate(gen):
            if tail and i < XF:                               # cross-fade from the previous chapter
                w_ = (i + 1) / (XF + 1)
                yield (tail.popleft().astype(np.float32) * (1 - w_) + fr.astype(np.float32) * w_).astype(np.uint8)
                continue
            tail.append(fr)
            if len(tail) > XF:
                yield tail.popleft()
    card = end_card()
    for i in range(n_end):
        if tail:
            w_ = (i + 1) / (XF + 1)
            yield (tail.popleft().astype(np.float32) * (1 - w_) + card.astype(np.float32) * w_).astype(np.uint8)
        else:
            yield card


out = f"{run}/showcase/montage.mp4"
want = None
if a.still:
    want = {int(float(s) * FPS) for s in a.still.split(",")}
else:
    wr = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", str(a.crf), "-preset", "medium"], macro_block_size=2)
    wr.send(None)
j = 0
for fr in all_frames():
    if want is None:
        wr.send(np.ascontiguousarray(progress(fr, j)).tobytes())
    elif j in want:
        Image.fromarray(progress(fr, j)).save(f"{run}/showcase/montage_t{j / FPS:04.1f}.jpg", quality=90)
        print("still", j / FPS, flush=True)
        if j >= max(want):
            break
    if j % 300 == 0:
        print(f"frame {j}/{TOTAL}", flush=True)
    j += 1
if want is None:
    wr.close()
    print("wrote", out, j, "frames", flush=True)
