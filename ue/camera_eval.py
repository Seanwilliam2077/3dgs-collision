"""Check the UE walk test's spring-arm cameras against the camera volume, and render what they saw.

    py.bat ue/camera_eval.py out/final_vox

Per phase of walk_result.json (voxel = shell only, camera = shell + CameraVolume on the Camera channel):
  origin inside   the arm origin (character's head) is inside the camera volume
  outside         camera position outside the camera volume (sdf < 0) / by more than 0.2 m
  arm             camera distance from the arm origin; shortened = more than 5 cm under the arm length
  enclosure       alpha of the splat within 0.6 m over the whole sphere around the camera
                  (> 0.3: splat right at the camera blurs the view)
  vs python       camera phase: UE's camera against camera_volume.arm() from the same origin and
                  rotation (UE probes with a 12 cm sphere, the python arm with a point and 5 cm margin)
Renders (debug/): ue_camera_pairs.jpg = worst shell-only cameras (UE) next to the same arm cut by the
camera volume; ue_camera_views.jpg = random UE cameras of the camera phase whose arm was shortened.
"""
import json
import sys

sys.path.insert(0, "D:/3DGS_Research/collision_proto")
import cv2
import numpy as np

from camera_volume import CameraVolume, near_alpha
from gs_core import load_ply, look_at, intrinsics, render

run = sys.argv[1].rstrip("/\\")
info = json.load(open(f"{run}/ue/scene_ue.json"))
res = json.load(open(f"{run}/ue/walk_result.json"))
mpu, c, R = info["mpu"], np.array(info["c"]), np.array(info["R"])
x0, y0, z0 = info["x0"], info["y0"], info["z0"]
vol = CameraVolume(run)
boom = res["camera_boom"]
L_cm = boom["arm_cm"]
print("camera boom:", boom)


def ue2splat(u):
    u = np.asarray(u, np.float64)
    lv = np.stack([u[:, 0] / (100 * mpu) + x0, -u[:, 1] / (100 * mpu) + y0, u[:, 2] / (100 * mpu) + z0], 1)
    return c + lv @ R


def uedir2splat(d):
    d = np.asarray(d, np.float64)
    lv = np.stack([d[:, 0], -d[:, 1], d[:, 2]], 1)
    return (lv / np.linalg.norm(lv, axis=1, keepdims=True)) @ R


def unpack(ph):
    a = np.array([row for ag in ph["cams"] for row in ag], np.float64)
    org = a[:, 0:3] + np.array(boom["target_offset"])
    p, y = np.radians(a[:, 6]), np.radians(a[:, 7])
    fwd = np.stack([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)], 1)
    right = np.stack([-np.sin(y), np.cos(y), np.zeros_like(y)], 1)
    up = np.stack([-np.cos(y) * np.sin(p), -np.sin(y) * np.sin(p), np.cos(p)], 1)
    so = np.array(boom["socket_offset"])
    want = org - fwd * L_cm + so[0] * fwd + so[1] * right + so[2] * up
    return dict(org=org, cam=a[:, 3:6], want=want, fwd=a[:, 8:11])


g = load_ply(info.get("ply") or json.load(open(f"{run}/meta.json"))["ply"])
rng = np.random.default_rng(0)
phases = {}
for ph in res["phases"]:
    d = unpack(ph)
    org_s, cam_s, want_s = ue2splat(d["org"]), ue2splat(d["cam"]), ue2splat(d["want"])
    s_org, s_cam, s_want = vol.sdf(org_s), vol.sdf(cam_s), vol.sdf(want_s)
    arm = np.linalg.norm(d["cam"] - d["org"], axis=1)
    sub = rng.choice(len(cam_s), min(800, len(cam_s)), replace=False)
    enc = near_alpha(g, cam_s[sub], vol.up, mpu)
    print(f"[{ph['name']}] {len(arm)} camera samples; origin inside {np.mean(s_org >= 0) * 100:.1f}%; wanted camera "
          f"outside {np.mean(s_want < 0) * 100:.1f}%; camera outside {np.mean(s_cam < 0) * 100:.1f}% "
          f"(> 0.2 m: {np.mean(s_cam < -0.2) * 100:.1f}%, of those with the origin inside: "
          f"{np.mean(s_cam[s_org >= 0] < 0) * 100:.1f}%); arm shortened {np.mean(arm < L_cm - 5) * 100:.1f}%, "
          f"mean {arm.mean() / 100:.2f} m")
    print(f"[{ph['name']}] enclosure (800 cameras): median {np.median(enc):.3f}, p90 {np.percentile(enc, 90):.3f}, "
          f"> 0.3 for {np.mean(enc > 0.3) * 100:.1f}%, > 0.5 for {np.mean(enc > 0.5) * 100:.1f}%")
    py_cam, _ = vol.arm(org_s, want_s)
    py_arm = np.linalg.norm(vol.to_level(py_cam) - vol.to_level(org_s), axis=1) * mpu * 100
    if ph["name"] == "camera":
        ok = s_org >= 0.05
        dif = arm[ok] - py_arm[ok]
        print(f"[camera] UE arm - python arm (origin inside): median {np.median(dif):+.1f} cm, p10 {np.percentile(dif, 10):+.1f}, "
              f"p90 {np.percentile(dif, 90):+.1f}; |diff| < 25 cm for {np.mean(np.abs(dif) < 25) * 100:.1f}%")
    phases[ph["name"]] = dict(cam=cam_s, fwd=uedir2splat(d["fwd"]), s_cam=s_cam, s_org=s_org, arm=arm,
                              py_cam=py_cam, py_arm=py_arm, org=org_s)

Wr, Hr = 640, 360
K = intrinsics(Wr, Hr, 90.0)                                   # UE camera default: 90 deg horizontal


def shot(pos, fwd, label):
    M = look_at(pos, pos + fwd, vol.up)
    rgb, _, _ = render(g, M, K, Wr, Hr, near=0.1 / mpu, far=1e4)
    im = np.ascontiguousarray((np.clip(rgb.cpu().numpy(), 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
    cv2.rectangle(im, (0, 0), (Wr, 26), (0, 0, 0), -1)
    cv2.putText(im, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


if "voxel" in phases:
    P = phases["voxel"]
    cand = np.nonzero((P["s_cam"] < -0.3) & (P["s_org"] >= 0.05))[0]
    if len(cand):                                                # the most enclosed (blurred) of them, one per agent area
        enc_c = near_alpha(g, P["cam"][cand], vol.up, mpu)
        order, picked = cand[np.argsort(-enc_c)], []
        for i in order:
            if all(np.linalg.norm(P["cam"][i] - P["cam"][j]) * mpu > 3.0 for j in picked):
                picked.append(i)
            if len(picked) == 5:
                break
        cand = np.array(picked)
    rows = []
    for i in cand:
        rows.append(np.concatenate([
            shot(P["cam"][i], P["fwd"][i], f"UE, shell only: arm {P['arm'][i] / 100:.2f} m, {-P['s_cam'][i]:.2f} m outside"),
            shot(P["py_cam"][i], P["fwd"][i], f"same arm cut by the camera volume: {P['py_arm'][i] / 100:.2f} m")], 1))
    if rows:
        cv2.imwrite(f"{run}/debug/ue_camera_pairs.jpg", np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"renders -> {run}/debug/ue_camera_pairs.jpg ({len(cand)} candidates)")
if "camera" in phases:
    P = phases["camera"]
    cand = np.nonzero(P["arm"] < L_cm - 100)[0]
    pick = rng.choice(cand, min(8, len(cand)), replace=False) if len(cand) else []
    tiles = [shot(P["cam"][i], P["fwd"][i], f"UE, camera volume: arm {P['arm'][i] / 100:.2f} m, sdf {P['s_cam'][i]:+.2f} m")
             for i in pick]
    if len(tiles) >= 2:
        tiles = tiles[:len(tiles) // 2 * 2]
        rows = [np.concatenate(tiles[k:k + 2], 1) for k in range(0, len(tiles), 2)]
        cv2.imwrite(f"{run}/debug/ue_camera_views.jpg", np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"renders -> {run}/debug/ue_camera_views.jpg")
