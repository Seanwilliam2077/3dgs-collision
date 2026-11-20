"""Automated walk test in PIE: many third-person characters random-walk on the collision mesh.

UnrealEditor.exe CollisionWalk.uproject -ExecutePythonScript="walk_test.py <run>/ue" -unattended

Uses the template's BP_ThirdPersonCharacter (UE 5.8: capsule r 35 cm / half-height 90 cm, step 45 cm,
walkable slope 44.8 deg, 600 cm/s) driven by an AI controller and AddMovementInput, i.e. the real
CharacterMovementComponent against the real Chaos collision of the imported meshes.
Agents are placed in the editor level before PIE (Python cannot spawn into the PIE world).

Phases come from scene_ue.json (export_ue.py), each with fresh spawns on the ground, e.g. for a TSDF run:
  mesh    collision mesh only
  patch   mesh + ground patch (ground_patch.py)
  all     mesh + ground patch + ground net
  split   objects mesh (mesh minus its ground) + ground patch + ground net: the shipped setup
and for a voxel run (voxel_collision.py):
  voxel   the voxel shell only
Events per agent:
  drop   fell >= 1 m and landed; 'surface' says on what (SplatCollision = real geometry below a
         ledge, GroundPatch / GroundNet = fell through a hole in the mesh)
  void   fell below everything; agent is respawned
  stuck  less than 60 cm of progress in 2 s while walking; 'blocked' if something is in front at knee
         (50 cm, just above step height) or waist height, else 'snag' (caught on a bump, a crack or a
         too-steep slope); 'on' = the surface underneath
  trapped  3 stuck events within 4 m and 12 s (e.g. in a pit); 'on' as above; agent is respawned
  border within 1.5 m of the chunk border (the invisible walls)
mobility = distance walked on the ground / (agents * phase time * max speed).
Cameras: each agent's own CameraBoom (the template's SpringArm: Camera channel probe) runs as it would
for a player, with its rotation set in world space: yaw = walking direction +- 60 deg, pitch -50 .. +10
deg, redrawn every 2-5 s. Logged 4x a second per agent: arm origin, camera position, rotation, camera
forward (UE cm). ue/camera_eval.py checks them against the camera volume and renders their views.
Writes <run>/ue/walk_result.json and screenshots to the project's Saved/Screenshots.
"""
import json
import math
import random
import sys
import traceback

import unreal

UE_DIR = (sys.argv[-1] if len(sys.argv) > 1 else "D:/3DGS_Research/collision_proto/out/chunk_a/ue").replace("\\", "/").rstrip("/")
N_AGENTS = 64
PHASE_S = 45.0
TURN_EVERY = (3.0, 7.0)
SAMPLE_DT = 0.25
STUCK_S, STUCK_CM = 2.0, 60.0     # stuck: less than 60 cm of progress in 2 s while trying to walk
TRAP_S = 12.0                     # trapped: 3 stuck events within 4 m and 12 s; the agent is respawned
CAM_PITCH, CAM_YAW, CAM_EVERY = (-50.0, 10.0), 60.0, (2.0, 5.0)   # spring arm rotation, redrawn every 2-5 s

info = json.load(open(f"{UE_DIR}/scene_ue.json"))
PHASES = [(p[0], set(p[1])) for p in info["phases"]]          # name, layers with collision on (the others off)
COLLIDERS = tuple(L["name"] for L in info["layers"])
SHOT_PHASE = PHASES[-1][0]
LEVEL = f"{info['asset_dir']}/{info['level']}"
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
GS = unreal.GameplayStatics
V = unreal.Vector
HX, HY = info["chunk_half_cm"]
Z_VOID = info["void_z_cm"]
rng = random.Random(1)
spawns = info["spawns_cm"][:]
rng.shuffle(spawns)


def log(*a):
    print("WALK", *a)


ARGV_RUN = "ExecutePythonScript" in unreal.SystemLibrary.get_command_line()
if ARGV_RUN:     # otherwise -ExecutePythonScript closes the editor right after this file returns
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)


def finish_editor():
    if ARGV_RUN:
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)   # the executer then quits


# the editor must not throttle when in the background: the project sets
# [/Script/UnrealEd.EditorPerformanceSettings] bThrottleCPUWhenNotForeground=False
les.load_level(LEVEL)
char_cls = unreal.EditorAssetLibrary.load_blueprint_class("/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter")

S = dict(stage="begin", frames=0, t0=None, phase=None, agents=[], phases=[], shots=0, spawn_i=0, extra={}, ignore=[])


def actor_of(hit):
    if hit is None:
        return None
    for v in hit.to_tuple():
        if isinstance(v, unreal.Actor):
            return v
    return None


def trace(world, a, b, ignore):
    hit = unreal.SystemLibrary.line_trace_single(world, a, b, unreal.TraceTypeQuery.ECC_VISIBILITY, True, ignore,
                                                 unreal.DrawDebugTrace.NONE, True)
    act = actor_of(hit)
    return act.get_actor_label() if act else None


def next_spawn():
    p = spawns[S["spawn_i"] % len(spawns)]
    S["spawn_i"] += 1
    return V(p[0], p[1], p[2] + 110.0)


def place_agents_in_editor():
    """Python cannot spawn into the PIE world, so the agents are placed in the editor level (not saved);
    PIE copies them and gives each an AI controller (AutoPossessAI = PlacedInWorld)."""
    out = []
    for i in range(N_AGENTS):
        a = eas.spawn_actor_from_class(char_cls, next_spawn(), unreal.Rotator(roll=0, pitch=0, yaw=rng.uniform(-180, 180)))
        a.set_actor_label(f"WalkAgent_{i:02d}")
        out.append(a)
    return out


def adopt_agent(a, idx):
    a.capsule_component.set_collision_response_to_channel(unreal.CollisionChannel.ECC_PAWN, unreal.CollisionResponseType.ECR_IGNORE)
    boom = a.get_components_by_class(unreal.SpringArmComponent)[0]
    cam = a.get_components_by_class(unreal.CameraComponent)[0]
    # an AI controller forces the control pitch to 0, so drive the boom's own rotation (world space).
    # NEVER notify: a property change notification reruns the Blueprint construction script, which
    # replaces the components (the references here would then point at dead ones)
    never = unreal.PropertyAccessChangeNotifyMode.NEVER
    boom.set_editor_property("use_pawn_control_rotation", False, never)
    boom.set_editor_property("enable_camera_lag", False, never)
    boom.set_editor_property("enable_camera_rotation_lag", False, never)
    boom.set_absolute(False, True, False)
    return dict(actor=a, idx=idx, yaw=math.radians(a.get_actor_rotation().yaw), next_turn=0.0, falling=False, fall_z=0.0,
                slow_t=0.0, last=None, dist=0.0, track=[], next_sample=0.0, boom=boom, cam=cam, cams=[],
                cam_pitch=0.0, cam_yaw=0.0, next_cam=0.0)


def boom_settings(boom):
    vec = lambda v: [round(v.x, 1), round(v.y, 1), round(v.z, 1)]
    return dict(arm_cm=boom.get_editor_property("target_arm_length"), probe_cm=boom.get_editor_property("probe_size"),
                probe_channel=str(boom.get_editor_property("probe_channel")),
                collision_test=boom.get_editor_property("do_collision_test"),
                socket_offset=vec(boom.get_editor_property("socket_offset")),
                target_offset=vec(boom.get_editor_property("target_offset")))


def respawn(ag):
    ag["actor"].set_actor_location(next_spawn(), False, True)
    ag["actor"].character_movement.stop_movement_immediately()
    ag["falling"] = False
    ag["last"] = None
    ag["hist"] = []


def start_phase(world, k, now):
    name, on = PHASES[k]
    S["phase_k"] = k
    S["phase"] = dict(name=name, colliders=sorted(on), t0=now, events=[], dist_cm=0.0, agents=N_AGENTS,
                      max_abs_xy=[0.0, 0.0])
    for label, act in S["extra"].items():
        act.set_actor_enable_collision(label in on)
    for ag in S["agents"]:
        respawn(ag)
        ag["next_turn"] = now
        ag["track"] = []
        ag["cams"] = []
        ag["next_cam"] = now
        ag["slow_t"] = 0.0
    log("phase", name, "colliding:", sorted(on))


def step_agent(world, ag, now, dt):
    a = ag["actor"]
    p = a.get_actor_location()
    ph = S["phase"]
    falling = a.character_movement.is_falling()
    if ag["last"] is not None and not falling:
        ph["dist_cm"] += math.hypot(p.x - ag["last"].x, p.y - ag["last"].y)
    ag["last"] = p
    ph["max_abs_xy"][0] = max(ph["max_abs_xy"][0], abs(p.x))
    ph["max_abs_xy"][1] = max(ph["max_abs_xy"][1], abs(p.y))
    # the camera: boom rotation in world space, follows the walking direction
    if not unreal.SystemLibrary.is_valid(ag["boom"]):
        raise RuntimeError(f"agent {ag['idx']}: camera boom was replaced (construction script rerun?)")
    if now >= ag["next_cam"]:
        ag["cam_pitch"] = rng.uniform(*CAM_PITCH)
        ag["cam_yaw"] = rng.uniform(-CAM_YAW, CAM_YAW)
        ag["next_cam"] = now + rng.uniform(*CAM_EVERY)
    ag["boom"].set_world_rotation(unreal.Rotator(roll=0.0, pitch=ag["cam_pitch"],
                                                 yaw=math.degrees(ag["yaw"]) + ag["cam_yaw"]), False, True)
    if now >= ag["next_sample"]:
        ag["track"].append([round(p.x), round(p.y), round(p.z), int(falling)])
        b, cl, cf = ag["boom"].get_world_location(), ag["cam"].get_world_location(), ag["cam"].get_forward_vector()
        br = ag["boom"].get_world_rotation()
        ag["cams"].append([round(b.x, 1), round(b.y, 1), round(b.z, 1), round(cl.x, 1), round(cl.y, 1), round(cl.z, 1),
                           round(br.pitch, 2), round(br.yaw, 2), round(cf.x, 4), round(cf.y, 4), round(cf.z, 4)])
        ag["next_sample"] = now + SAMPLE_DT
    ev = lambda kind, **kw: ph["events"].append(dict(kind=kind, agent=ag["idx"], t=round(now - ph["t0"], 2),
                                                     at=[round(p.x), round(p.y), round(p.z)], **kw))
    # falls
    if falling and not ag["falling"]:
        ag["fall_z"] = p.z
        ag["fall_xy"] = [round(p.x), round(p.y)]
    if not falling and ag["falling"]:
        drop = ag["fall_z"] - p.z
        if drop >= 100.0:
            surf = trace(world, V(p.x, p.y, p.z), V(p.x, p.y, p.z - 300.0), S["ignore"])
            ev("drop", drop_cm=round(drop), surface=surf, from_xy=ag["fall_xy"])
    ag["falling"] = falling
    if p.z < Z_VOID:
        ev("void", from_xy=ag.get("fall_xy"))
        respawn(ag)
        return
    if abs(p.x) > HX - 150 or abs(p.y) > HY - 150:
        if not ag.get("at_border"):
            ev("border")
        ag["at_border"] = True
    else:
        ag["at_border"] = False
    # steering: random heading, re-drawn every few seconds or when stuck
    if now >= ag["next_turn"]:
        ag["yaw"] += rng.uniform(-math.pi / 2, math.pi / 2)
        ag["next_turn"] = now + rng.uniform(*TURN_EVERY)
    # stuck = less than STUCK_CM of horizontal progress over the last STUCK_S seconds (velocity is no use:
    # a character pushing up a too-steep slope jitters with non-zero velocity while going nowhere)
    hist = ag.setdefault("hist", [])
    if falling:
        hist.clear()
    else:
        hist.append((now, p.x, p.y))
        while len(hist) > 2 and hist[1][0] <= now - STUCK_S:
            hist.pop(0)
    if hist and now - hist[0][0] >= STUCK_S and math.hypot(p.x - hist[0][1], p.y - hist[0][2]) < STUCK_CM:
        fwd = V(math.cos(ag["yaw"]), math.sin(ag["yaw"]), 0.0)
        front = None
        for dz in (0.0, -40.0):                       # waist, knee (50 cm above the floor)
            o = V(p.x, p.y, p.z + dz)
            front = front or trace(world, o, o + fwd * 90.0, S["ignore"])
        on = trace(world, V(p.x, p.y, p.z), V(p.x, p.y, p.z - 250.0), S["ignore"])
        ev("stuck", what="blocked" if front else "snag", by=front, on=on)
        hist.clear()
        recent = [q for q in ag.setdefault("stucks", []) if now - q[0] < TRAP_S and math.hypot(p.x - q[1], p.y - q[2]) < 400.0]
        ag["stucks"] = recent + [(now, p.x, p.y)]
        if len(ag["stucks"]) >= 3:                   # keeps getting stuck in the same place: trapped
            ev("trapped", on=on)
            ag["stucks"] = []
            respawn(ag)
            return
        ag["yaw"] += math.pi + rng.uniform(-1.0, 1.0)
        ag["next_turn"] = now + rng.uniform(*TURN_EVERY)
    a.add_movement_input(V(math.cos(ag["yaw"]), math.sin(ag["yaw"]), 0.0), 1.0, False)


def finish_phase():
    ph = S["phase"]
    ph["tracks"] = [ag["track"] for ag in S["agents"]]
    ph["cams"] = [ag["cams"] for ag in S["agents"]]     # [arm origin xyz, camera xyz, pitch, yaw, forward xyz]
    kinds = {}
    for e in ph["events"]:
        k = e["kind"] + {"drop": lambda: ":" + str(e.get("surface")), "stuck": lambda: ":" + str(e.get("what")),
                         "trapped": lambda: ":" + str(e.get("on"))}.get(e["kind"], lambda: "")()
        kinds[k] = kinds.get(k, 0) + 1
    ph["summary"] = kinds
    speed = S["agents"][0]["actor"].character_movement.get_editor_property("max_walk_speed")
    ph["mobility"] = ph["dist_cm"] / (N_AGENTS * PHASE_S * speed)
    log("phase", ph["name"], "walked %.1f km, mobility %.2f" % (ph["dist_cm"] / 1e5, ph["mobility"]), kinds,
        "max |x|,|y| cm", [round(v) for v in ph["max_abs_xy"]])
    S["phases"].append(ph)


def shot(world, ag):
    pc = GS.get_player_controller(world, 0)
    pc.set_view_target_with_blend(ag["actor"], 0.0)
    unreal.SystemLibrary.execute_console_command(world, "HighResShot 1600x900")
    S["shots"] += 1


def tick(dt):
    try:
        S["frames"] += 1
        st = S["stage"]
        if st == "begin":
            if S["frames"] > 30:
                S["editor_agents"] = place_agents_in_editor()
                les.editor_request_begin_play()
                S["stage"] = "wait_pie"
            return
        world = ues.get_game_world()
        if st == "wait_pie":
            if world is None or GS.get_player_pawn(world, 0) is None:
                return
            S["stage"], S["wait_until"] = "spawn", GS.get_time_seconds(world) + 2.0
            return
        now = GS.get_time_seconds(world)
        if st == "spawn":
            if now < S["wait_until"]:
                return
            S["extra"] = {act.get_actor_label(): act for act in GS.get_all_actors_of_class(world, unreal.StaticMeshActor)
                          if act.get_actor_label() in COLLIDERS}
            found = sorted((a for a in GS.get_all_actors_of_class(world, char_cls) if a.get_actor_label().startswith("WalkAgent_")),
                           key=lambda a: a.get_actor_label())
            S["agents"] = [adopt_agent(a, i) for i, a in enumerate(found)]
            assert len(S["agents"]) == N_AGENTS and all(a.get_controller() for a in found), "agents missing or not possessed"
            S["ignore"] = [ag["actor"] for ag in S["agents"]] + [GS.get_player_pawn(world, 0)]
            log("spawned", len(S["agents"]), "agents; extra collision actors", sorted(S["extra"]))
            for label, act in S["extra"].items():
                comp = act.static_mesh_component
                log(" ", label, "responses: camera", comp.get_collision_response_to_channel(unreal.CollisionChannel.ECC_CAMERA),
                    "pawn", comp.get_collision_response_to_channel(unreal.CollisionChannel.ECC_PAWN))
            b0 = S["agents"][0]["boom"]
            log("camera boom", boom_settings(b0), "pawn control rotation", b0.get_editor_property("use_pawn_control_rotation"),
                "same component as the actor's:", b0 == S["agents"][0]["actor"].get_components_by_class(unreal.SpringArmComponent)[0])
            start_phase(world, 0, now)
            S["stage"] = "run"
            return
        if st == "run":
            for ag in S["agents"]:
                step_agent(world, ag, now, dt)
            el = now - S["phase"]["t0"]
            if S["phase"]["name"] == SHOT_PHASE and S["shots"] < 4 and el > 10.0 + 5.0 * S["shots"]:
                shot(world, S["agents"][S["shots"] * 7])
            if el >= PHASE_S:
                finish_phase()
                if S["phase_k"] + 1 < len(PHASES):
                    start_phase(world, S["phase_k"] + 1, now)
                else:
                    S["stage"] = "done"
            return
        if st == "done":
            res = dict(agents=N_AGENTS, phase_s=PHASE_S, phases=S["phases"], camera_boom=boom_settings(S["agents"][0]["boom"]),
                       character=dict(radius_cm=S["agents"][0]["actor"].capsule_component.get_unscaled_capsule_radius(),
                                      half_height_cm=S["agents"][0]["actor"].capsule_component.get_unscaled_capsule_half_height(),
                                      step_cm=S["agents"][0]["actor"].character_movement.get_editor_property("max_step_height"),
                                      walkable_deg=S["agents"][0]["actor"].character_movement.get_walkable_floor_angle(),
                                      speed_cms=S["agents"][0]["actor"].character_movement.get_editor_property("max_walk_speed")))
            json.dump(res, open(f"{UE_DIR}/walk_result.json", "w"))
            log("wrote", f"{UE_DIR}/walk_result.json", res["character"])
            S["stage"] = "quit"
            S["quit_frames"] = S["frames"] + 30
            les.editor_request_end_play()
            return
        if st == "quit" and S["frames"] > S["quit_frames"]:
            unreal.unregister_slate_post_tick_callback(HANDLE)
            eas.destroy_actors(S["editor_agents"])
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
log("registered; agents", N_AGENTS, "phase", PHASE_S, "s")
