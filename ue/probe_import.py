"""Probe how this UE version imports glTF axes, and which Python APIs exist.
Run: UnrealEditor-Cmd.exe CollisionWalk.uproject -run=pythonscript -script=<this file>"""
import unreal

print("PROBE engine", unreal.SystemLibrary.get_engine_version())
task = unreal.AssetImportTask()
task.filename = "D:/3DGS_Research/collision_proto/ue/probe_axes.glb"
task.destination_path = "/Game/Probe"
task.automated = True
task.replace_existing = True
task.save = True
unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
print("PROBE imported", list(task.imported_object_paths))
for p in task.imported_object_paths:
    a = unreal.load_asset(p)
    print("PROBE asset", p, type(a).__name__)
    if isinstance(a, unreal.StaticMesh):
        b = a.get_bounding_box()
        print("PROBE bbox min", b.min, "max", b.max)
        md = a.get_static_mesh_description(0)
        for i in range(md.get_vertex_count()):
            print("PROBE vertex", i, md.get_vertex_position(unreal.VertexID(i)))
for name in ["StaticMeshEditorSubsystem", "EditorActorSubsystem", "LevelEditorSubsystem", "UnrealEditorSubsystem",
             "MaterialEditingLibrary", "CollisionTraceFlag", "GameplayStatics", "InterchangeManager",
             "InterchangeGenericAssetsPipeline"]:
    print("PROBE api", name, hasattr(unreal, name))
