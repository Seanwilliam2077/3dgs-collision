"""One player character runs a scripted route in PIE; its third-person camera and a render of the character
are captured every frame, for compositing over the splat (ue/run_video.py).

UnrealEditor.exe CollisionWalk.uproject -ExecutePythonScript="run_capture.py <run>/ue" -unattended -UseFixedTimeStep -FPS=30

UE 5.8 here has no Gaussian-splat renderer, so the splat itself is not in the UE frame: UE supplies the
character (BP_ThirdPersonCharacter: CharacterMovement against the voxel shell, spring-arm camera against the
camera volume), run_video.py renders the splat from the same camera with gsplat and composites the two.
Route: from the PlayerStart onto a 5.5 m ring around the statue, two laps (one jump), then straight out into
the bushes until the collision shell stops it, a jump against the wall, and back.
Capture: a SceneCapture2D (placed in the editor level; Python cannot spawn into PIE) copies the player
camera's transform and FOV each frame and renders only the character (show-only list) as SceneColor (HDR)
+ scene depth (cm), plus the GBuffer base colour and world normal (lit scene colour comes out black from scene
captures on this machine, so run_video.py shades the character itself), RGBA16F at 1.5x the video size, EXR.
Writes <run>/ue/run/d%04d.exr, b%04d.exr, n%04d.exr and <run>/ue/run/run_log.json (per frame: game time, camera, pawn).
"""
import json
import math
import os
import sys
import traceback

import unreal

UE_DIR = (sys.argv[-1] if len(sys.argv) > 1 else "D:/3DGS_Research/collision_proto/out/final_vox/ue").replace("\\", "/").rstrip("/")
OUT = f"{UE_DIR}/run"
CAP_W, CAP_H = 2880, 1620
STATUE = (-40.0, 20.0)
RING = 550.0
PITCH = -14.0
info = json.load(open(f"{UE_DIR}/scene_ue.json"))
LEVEL = f"{info['asset_dir']}/{info['level']}"
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
GS = unreal.GameplayStatics
V = unreal.Vector
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith(".exr"):
        os.remove(f"{OUT}/{f}")


def log(*a):
    print("RUN", *a)


ARGV_RUN = "ExecutePythonScript" in unreal.SystemLibrary.get_command_line()
if ARGV_RUN:
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)


def finish_editor():
    if ARGV_RUN:
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)


def ring(deg):
    return (STATUE[0] + RING * math.cos(math.radians(deg)), STATUE[1] + RING * math.sin(math.radians(deg)))


# waypoints (x, y, action): two laps counter-clockwise, then out through 90 deg into the bushes and back
ROUTE = [ring(0)] + [ring(d) for d in range(30, 720 + 31, 30)]
JUMP_AT = 14                                                    # jump when leaving this waypoint
OUT_DIR = (math.cos(math.radians(90)), math.sin(math.radians(90)))
ROUTE += [(STATUE[0] + 1300 * OUT_DIR[0], STATUE[1] + 1300 * OUT_DIR[1])]   # beyond the edge (~9 m): blocked
WALL_I = len(ROUTE) - 1
ROUTE += [ring(90), ring(150), ring(200)]

les.load_level(LEVEL)
S = dict(stage="begin", frames=0, wp=0, log=[], n=0, hist=[], blocked_t=None, jumped=set(), yaw=None)


def place_capture():
    out = []
    for lab in ("RunCaptureDepth", "RunCaptureBase", "RunCaptureNormal"):
        cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, V(0, 0, 200))
        cap.set_actor_label(lab)
        out.append(cap)
    return out


def setup_capture(world, cap, pawn, cam, source):
    comp = cap.capture_component2d
    rt = unreal.RenderingLibrary.create_render_target2d(world, CAP_W, CAP_H, unreal.TextureRenderTargetFormat.RTF_RGBA16F,
                                                        unreal.LinearColor(0, 0, 0, 0))
    comp.set_editor_property("texture_target", rt)
    comp.set_editor_property("capture_source", source)
    if source == unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR:      # fixed exposure: no eye adaptation
        pp = comp.get_editor_property("post_process_settings")
        pp.set_editor_property("override_auto_exposure_method", True)
        pp.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
        pp.set_editor_property("override_auto_exposure_bias", True)
        pp.set_editor_property("auto_exposure_bias", 0.0)
        comp.set_editor_property("post_process_settings", pp)
        comp.set_editor_property("post_process_blend_weight", 1.0)
    comp.set_editor_property("primitive_render_mode", unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST)
    comp.set_editor_property("show_only_actors", [pawn])
    comp.set_editor_property("capture_every_frame", False)
    comp.set_editor_property("capture_on_movement", False)
    comp.set_editor_property("fov_angle", cam.get_editor_property("field_of_view"))
    return comp, rt


def tick(dt):
    try:
        S["frames"] += 1
        st = S["stage"]
        if st == "begin":
            if S["frames"] > 30:
                S["editor_cap"] = place_capture()
                les.editor_request_begin_play()
                S["stage"] = "wait_pie"
            return
        world = ues.get_game_world()
        if st == "wait_pie":
            if world is None or GS.get_player_pawn(world, 0) is None:
                return
            S["stage"], S["wait_until"] = "setup", GS.get_time_seconds(world) + 1.0
            return
        now = GS.get_time_seconds(world)
        if st == "setup":
            if now < S["wait_until"]:
                return
            pawn = GS.get_player_pawn(world, 0)
            S["pawn"], S["pc"] = pawn, GS.get_player_controller(world, 0)
            S["cam"] = pawn.get_components_by_class(unreal.CameraComponent)[0]
            S["boom"] = pawn.get_components_by_class(unreal.SpringArmComponent)[0]
            caps = {a.get_actor_label(): a for a in GS.get_all_actors_of_class(world, unreal.SceneCapture2D)}
            S["caps"] = []
            for lab, src, tag in (("RunCaptureDepth", unreal.SceneCaptureSource.SCS_SCENE_COLOR_SCENE_DEPTH, "d"),
                                  ("RunCaptureBase", unreal.SceneCaptureSource.SCS_BASE_COLOR, "b"),
                                  ("RunCaptureNormal", unreal.SceneCaptureSource.SCS_NORMAL, "n")):
                comp, rt = setup_capture(world, caps[lab], pawn, S["cam"], src)
                S["caps"].append((caps[lab], comp, rt, tag))
            S["t0"] = now
            p = pawn.get_actor_location()
            S["yaw"] = math.degrees(math.atan2(ROUTE[0][1] - p.y, ROUTE[0][0] - p.x))
            for a in GS.get_all_actors_of_class(world, unreal.StaticMeshActor):
                lab = a.get_actor_label()
                if lab in ("VoxelShell", "CameraVolume"):
                    a.set_actor_enable_collision(True)
            log("setup: pawn", pawn.get_name(), "fov", S["cam"].get_editor_property("field_of_view"),
                "arm", S["boom"].get_editor_property("target_arm_length"), "speed",
                pawn.character_movement.get_editor_property("max_walk_speed"), "route", len(ROUTE))
            S["stage"] = "run"
            return
        if st == "run":
            pawn, pc = S["pawn"], S["pc"]
            p = pawn.get_actor_location()
            el = now - S["t0"]
            # steering
            wx, wy = ROUTE[S["wp"]]
            dx, dy = wx - p.x, wy - p.y
            d = math.hypot(dx, dy)
            reach_r = 120.0 if S["wp"] != WALL_I else 60.0
            S["hist"].append((now, p.x, p.y))
            while len(S["hist"]) > 2 and S["hist"][1][0] <= now - 1.0:
                S["hist"].pop(0)
            h0 = S["hist"][0]
            stuck = now - h0[0] >= 0.9 and math.hypot(p.x - h0[1], p.y - h0[2]) < 40.0
            if S["wp"] == WALL_I and stuck:
                if S["blocked_t"] is None:
                    S["blocked_t"] = now
                    log("blocked by the wall at", [round(p.x), round(p.y), round(p.z)], "t", round(el, 2))
                if now - S["blocked_t"] > 0.6 and "wall" not in S["jumped"]:
                    pawn.jump()
                    S["jumped"].add("wall")
                    S["jump_end"] = now + 0.3
                if now - S["blocked_t"] > 2.0:
                    S["wp"] += 1
            elif d < reach_r or (stuck and S["wp"] != WALL_I):
                if S["wp"] == JUMP_AT and "lap" not in S["jumped"]:
                    pawn.jump()
                    S["jumped"].add("lap")
                    S["jump_end"] = now + 0.3
                S["wp"] += 1
                if S["wp"] >= len(ROUTE):
                    S["stage"] = "done"
                    return
            if S.get("jump_end") and now > S["jump_end"]:
                pawn.stop_jumping()
                S["jump_end"] = None
            pawn.add_movement_input(V(dx / max(d, 1e-3), dy / max(d, 1e-3), 0.0), 1.0, False)
            # camera: control yaw eases toward the running direction
            vel = pawn.get_velocity()
            if math.hypot(vel.x, vel.y) > 100.0:
                want = math.degrees(math.atan2(vel.y, vel.x))
                dyaw = (want - S["yaw"] + 540.0) % 360.0 - 180.0
                S["yaw"] += dyaw * min(1.0, 2.2 * dt)
            pc.set_control_rotation(unreal.Rotator(roll=0.0, pitch=PITCH, yaw=S["yaw"]))
            # capture: the player camera as it is after this frame's tick
            cam = S["cam"]
            cl, cr = cam.get_world_location(), cam.get_world_rotation()
            for cap, comp, rt, tag in S["caps"]:
                cap.set_actor_location_and_rotation(cl, cr, False, True)
                comp.capture_scene()
                unreal.RenderingLibrary.export_render_target(world, rt, OUT, f"{tag}{S['n']:04d}.exr")
            fw = cam.get_forward_vector()
            S["log"].append(dict(i=S["n"], t=round(el, 4), dt=round(dt, 4), cam=[cl.x, cl.y, cl.z],
                                 rot=[cr.pitch, cr.yaw, cr.roll], fwd=[fw.x, fw.y, fw.z], pawn=[p.x, p.y, p.z],
                                 falling=bool(pawn.character_movement.is_falling()), wp=S["wp"]))
            S["n"] += 1
            if el > 40.0:
                log("time limit")
                S["stage"] = "done"
            return
        if st == "done":
            pawn = S["pawn"]
            res = dict(frames=S["log"], cap_size=[CAP_W, CAP_H], fov=S["cam"].get_editor_property("field_of_view"),
                       capsule_half_cm=pawn.capsule_component.get_unscaled_capsule_half_height(),
                       capsule_r_cm=pawn.capsule_component.get_unscaled_capsule_radius(),
                       route=ROUTE, wall_index=WALL_I)
            json.dump(res, open(f"{OUT}/run_log.json", "w"))
            dts = [f["dt"] for f in S["log"]]
            log("wrote", len(S["log"]), "frames; dt min/max", min(dts), max(dts))
            S["stage"] = "quit"
            S["quit_frames"] = S["frames"] + 30
            les.editor_request_end_play()
            return
        if st == "quit" and S["frames"] > S["quit_frames"]:
            unreal.unregister_slate_post_tick_callback(HANDLE)
            eas.destroy_actors(S["editor_cap"])
            log("DONE")
            finish_editor()
    except Exception:
        log("ERROR", traceback.format_exc())
        unreal.unregister_slate_post_tick_callback(HANDLE)
        try:
            les.editor_request_end_play()
        finally:
            finish_editor()


HANDLE = unreal.register_slate_post_tick_callback(tick)
log("registered; route", len(ROUTE), "waypoints")
