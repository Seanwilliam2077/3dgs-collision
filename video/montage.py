"""The demo cut: the voxelisation / collision / camera-volume process, then the character running in UE.

    py.bat video/montage.py out/final_vox_stages/showcase/process.mp4 out/final_vox/showcase/ue_run.mp4 \
        out/final_vox/showcase/montage.mp4 [--still 10,40]

process.mp4 comes from video/process_video.py (sections 场景体素化 / 角色碰撞区域 / 相机区域, kept at 1x), ue_run.mp4
from ue/run_video.py (the run at 2x until shortly before the wall, then 1x). Sped-up parts skip frames (still 30 fps)
and carry a speed tag; the UE part gets the same section chip as the process sections. A cross-fade joins the two
videos, and a thin progress bar with section ticks runs along the bottom.
"""
import argparse
import math
from collections import deque

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("process")
ap.add_argument("ue")
ap.add_argument("out")
ap.add_argument("--still", default=None, help="output times (s): JPEG previews next to the output instead of the video")
ap.add_argument("--crf", type=int, default=20)
a = ap.parse_args()
W, H, FPS = 1920, 1080, 30

# chapter: (name, file, [(t0, t1, speed), ...], cross-fade frames into it, section chip drawn here or None)
CHAPTERS = [
    ("场景体素化", a.process, [(0.0, 10.0, 1.0)], 0, None),
    ("角色碰撞区域", a.process, [(10.0, 21.0, 1.0)], 0, None),
    ("相机区域", a.process, [(21.0, 34.5, 1.0)], 0, None),
    ("UE 实机", a.ue, [(0.0, 11.0, 2.0), (11.0, 17.2, 1.0)], 12, "四 · UE 实机"),
]

FONT_TAG = ImageFont.truetype("C:/Windows/Fonts/bahnschrift.ttf", 30)
FONT_CH = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
FONT_C = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 30)


def n_out(parts):
    return sum(int(round((t1 - t0) * FPS / sp)) for t0, t1, sp in parts)


starts, pos = [], 0
for name, _, parts, xf, _ in CHAPTERS:
    pos -= xf
    starts.append(pos)
    pos += n_out(parts)
TOTAL = pos
starts.append(TOTAL)
print("chapters", [f"{c[0]} {n_out(c[2]) / FPS:.1f}s" for c in CHAPTERS], f"-> total {TOTAL / FPS:.1f}s", flush=True)


def overlay(img, speed, chip):
    if speed == 1.0 and chip is None:
        return img
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil, "RGBA")
    x1, y0 = W - 40, 40
    if chip:                                          # same look as process_video.py's section chips
        w_ = FONT_C.getlength(chip) + 36
        dr.rounded_rectangle((x1 - w_, y0, x1, y0 + 52), 12, fill=(0, 0, 0, 150))
        dr.text((x1 - w_ / 2, y0 + 26), chip, font=FONT_C, fill=(255, 255, 255, 235), anchor="mm")
        y0 += 64
    if speed != 1.0:
        s = f"{speed:g}×"
        w_ = FONT_TAG.getlength(s) + 28
        dr.rounded_rectangle((x1 - w_, y0, x1, y0 + 46), 12, fill=(0, 0, 0, 140))
        dr.text((x1 - w_ / 2, y0 + 23), s, font=FONT_TAG, fill=(255, 255, 255, 235), anchor="mm")
    return np.asarray(pil)


def chapter_frames(path, parts, chip):
    """Output frames of one chapter: source frame floor(j * speed) of each part."""
    for t0, t1, sp in parts:
        n = int(round((t1 - t0) * FPS / sp))
        rd = imageio_ffmpeg.read_frames(path, input_params=["-ss", f"{t0:.3f}"])
        m_ = next(rd)
        assert tuple(m_["size"]) == (W, H), (path, m_["size"])
        i_src, cur = -1, None
        for j in range(n):
            want = int(math.floor(j * sp + 1e-6))
            while i_src < want:
                try:
                    cur = next(rd)
                except StopIteration:                 # source shorter than asked: hold the last frame
                    break
                i_src += 1
            yield overlay(np.frombuffer(cur, np.uint8).reshape(H, W, 3), sp, chip)
        rd.close()


def progress(img, j):
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil, "RGBA")
    y0 = H - 8
    dr.rectangle((0, y0, W, H), fill=(0, 0, 0, 110))
    dr.rectangle((0, y0, int(W * j / TOTAL), H), fill=(90, 200, 255, 220))
    for s in starts[1:-1]:
        x = int(W * s / TOTAL)
        dr.rectangle((x - 1, y0 - 4, x + 1, H), fill=(255, 255, 255, 200))
    k = max(i for i, s in enumerate(starts[:-1]) if s <= j)
    x = int(W * (starts[k] + starts[k + 1]) / 2 / TOTAL)
    dr.text((x, y0 - 8), CHAPTERS[k][0], font=FONT_CH, fill=(255, 255, 255, 200), anchor="mb",
            stroke_width=2, stroke_fill=(0, 0, 0, 160))
    return np.asarray(pil)


def all_frames():
    tail = deque()
    for name, path, parts, xf, chip in CHAPTERS:
        for i, fr in enumerate(chapter_frames(path, parts, chip)):
            if i < xf and tail:                       # cross-fade from the previous chapter
                w_ = (i + 1) / (xf + 1)
                yield (tail.popleft().astype(np.float32) * (1 - w_) + fr.astype(np.float32) * w_).astype(np.uint8)
                continue
            tail.append(fr)
            if len(tail) > 12:
                yield tail.popleft()
    while tail:
        yield tail.popleft()


want = {int(float(s) * FPS) for s in a.still.split(",")} if a.still else None
if want is None:
    wr = imageio_ffmpeg.write_frames(a.out, (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", str(a.crf), "-preset", "medium"], macro_block_size=2)
    wr.send(None)
j = 0
for fr in all_frames():
    if want is None:
        wr.send(np.ascontiguousarray(progress(fr, j)).tobytes())
    elif j in want:
        fn = a.out.rsplit(".", 1)[0] + f"_t{j / FPS:04.1f}.jpg"
        Image.fromarray(progress(fr, j)).save(fn, quality=90)
        print("still", fn, flush=True)
        if j >= max(want):
            break
    j += 1
if want is None:
    wr.close()
    print("wrote", a.out, j, "frames", flush=True)
