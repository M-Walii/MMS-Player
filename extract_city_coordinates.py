import bpy

print("World-Space Coordinates of City Points:")
for obj in bpy.data.objects:
    # Consider objects that end with "_Point"
    if obj.name.endswith("_Point"):
        world_co = obj.matrix_world.translation
        print(f"{obj.name}: X={world_co.x:.3f}, Y={world_co.y:.3f}, Z={world_co.z:.3f}")
