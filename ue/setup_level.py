"""Build a walk-test level in the CollisionWalk project from <run>/ue (written by export_ue.py).

UnrealEditor.exe CollisionWalk.uproject -ExecutePythonScript="setup_level.py <run>/ue" -unattended
(full editor, quits when done: the -run=pythonscript commandlet cannot spawn actors)

Everything comes from <run>/ue/scene_ue.json:
  <asset_dir>/SM_<layer>   one static mesh per layer: complex-as-simple collision, Nanite off
                           (with Nanite on, complex collision uses the reduced fallback mesh)
  <asset_dir>/M_SplatVertexColor   unlit, two-sided, emissive = vertex colour (for visible layers)
  <level>                  the layers (hidden in game unless visible; actor collision per layer),
                           invisible walls on the chunk border, lights, PlayerStart
Other runs' assets and levels are left alone.
"""
import json
import sys

import unreal

ue_dir = (sys.argv[-1] if len(sys.argv) > 1 else "D:/3DGS_Research/collision_proto/out/chunk_a/ue").replace("\\", "/").rstrip("/")
info = json.load(open(f"{ue_dir}/scene_ue.json"))
at = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
DEST = info["asset_dir"]
LEVEL = f"{DEST}/{info['level']}"


def log(*a):
    print("SETUP", *a)


for p in ["/Game/Probe", DEST]:
    if eal.does_directory_exist(p):
        eal.delete_directory(p)


def import_mesh(fn, name, expect):
    t = unreal.AssetImportTask()
    t.filename = fn
    t.destination_path = f"{DEST}/{name}"
    t.automated = True
    t.replace_existing = True
    t.save = False
    at.import_asset_tasks([t])
    meshes = [unreal.load_asset(p) for p in t.imported_object_paths]
    meshes = [m for m in meshes if isinstance(m, unreal.StaticMesh)]
    assert len(meshes) == 1, t.imported_object_paths
    m = meshes[0]
    ns = m.get_editor_property("nanite_settings")
    ns.set_editor_property("enabled", False)
    m.set_editor_property("nanite_settings", ns)
    bs = m.get_editor_property("body_setup")
    bs.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
    b = m.get_bounding_box()
    lo, hi = expect
    err = max(abs(b.min.x - lo[0]), abs(b.min.y - lo[1]), abs(b.min.z - lo[2]),
              abs(b.max.x - hi[0]), abs(b.max.y - hi[1]), abs(b.max.z - hi[2]))
    log(name, "tris", m.get_num_triangles(0), "bbox mismatch vs export (cm):", round(err, 2))
    assert err < 5.0, "glTF axis convention differs from probe_import.py; fix ue_to_gltf in export_ue.py"
    return m


mat = None
meshes = {}
for L in info["layers"]:
    meshes[L["name"]] = import_mesh(f"{ue_dir}/{L['file']}", f"SM_{L['name']}", L["bounds_cm"])
    if L["visible"]:
        if mat is None:
            mat = at.create_asset("M_SplatVertexColor", DEST, unreal.Material, unreal.MaterialFactoryNew())
            mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
            mat.set_editor_property("two_sided", True)
            vc = mel.create_material_expression(mat, unreal.MaterialExpressionVertexColor, -400, 0)
            mel.connect_material_property(vc, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            mel.recompile_material(mat)
        meshes[L["name"]].set_material(0, mat)
eal.save_directory(DEST, only_if_is_dirty=False, recursive=True)

# ---- level -----------------------------------------------------------------------------------------
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
log("new_level", les.new_level(LEVEL, False))
V = unreal.Vector
Rt = unreal.Rotator


def spawn_mesh(sm, label, loc=V(0, 0, 0), scale=V(1, 1, 1), hidden=False, collide=True, channel=None):
    act = eas.spawn_actor_from_object(sm, loc)
    act.set_actor_label(label)
    act.set_actor_scale3d(scale)
    comp = act.static_mesh_component
    comp.set_collision_profile_name("BlockAll")
    if channel == "camera":                    # camera volume: only camera traces (spring arms) stop at it
        comp.set_collision_enabled(unreal.CollisionEnabled.QUERY_ONLY)
        comp.set_collision_response_to_all_channels(unreal.CollisionResponseType.ECR_IGNORE)
        comp.set_collision_response_to_channel(unreal.CollisionChannel.ECC_CAMERA, unreal.CollisionResponseType.ECR_BLOCK)
        log(label, "camera only: camera", comp.get_collision_response_to_channel(unreal.CollisionChannel.ECC_CAMERA),
            "pawn", comp.get_collision_response_to_channel(unreal.CollisionChannel.ECC_PAWN),
            "visibility", comp.get_collision_response_to_channel(unreal.CollisionChannel.ECC_VISIBILITY))
    if hidden:
        act.set_actor_hidden_in_game(True)
    act.set_actor_enable_collision(collide)
    return act


for L in info["layers"]:
    spawn_mesh(meshes[L["name"]], L["name"], hidden=not L["visible"], collide=L["collide"], channel=L.get("channel"))

# invisible walls on the chunk border
hx, hy = info["chunk_half_cm"]
z_lo = min(L["bounds_cm"][0][2] for L in info["layers"]) - 3000.0
z_hi = max(L["bounds_cm"][1][2] for L in info["layers"]) + 3000.0
cube = unreal.load_asset("/Engine/BasicShapes/Cube")
zc, zs = (z_lo + z_hi) / 2, (z_hi - z_lo) / 100.0
for label, loc, sc in [("Wall_PX", V(hx + 50, 0, zc), V(1, (2 * hy + 200) / 100, zs)),
                       ("Wall_NX", V(-hx - 50, 0, zc), V(1, (2 * hy + 200) / 100, zs)),
                       ("Wall_PY", V(0, hy + 50, zc), V((2 * hx + 200) / 100, 1, zs)),
                       ("Wall_NY", V(0, -hy - 50, zc), V((2 * hx + 200) / 100, 1, zs))]:
    spawn_mesh(cube, label, loc, sc, hidden=True)

sun = eas.spawn_actor_from_class(unreal.DirectionalLight, V(0, 0, 3000), Rt(-50, 30, 0))
sun.set_actor_label("Sun")
sun.light_component.set_mobility(unreal.ComponentMobility.MOVABLE)
sun.light_component.set_editor_property("atmosphere_sun_light", True)
sky = eas.spawn_actor_from_class(unreal.SkyLight, V(0, 0, 3000))
sky.set_actor_label("SkyLight")
sky.light_component.set_mobility(unreal.ComponentMobility.MOVABLE)
sky.light_component.set_editor_property("real_time_capture", True)
eas.spawn_actor_from_class(unreal.SkyAtmosphere, V(0, 0, 0)).set_actor_label("SkyAtmosphere")

s = info["player_start_cm"]
eas.spawn_actor_from_class(unreal.PlayerStart, V(s[0], s[1], s[2] + 120.0)).set_actor_label("PlayerStart")

log("saved level", les.save_current_level(), LEVEL)
log("actors", len(eas.get_all_level_actors()))
log("DONE")
if "ExecutePythonScript" in unreal.SystemLibrary.get_command_line():
    unreal.SystemLibrary.quit_editor()
