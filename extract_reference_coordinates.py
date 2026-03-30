import bpy

print("World-Space Coordinates of Reference Points:")
print("=" * 50)

found_objects = []

for obj in bpy.data.objects:
    if obj.name.startswith("RefH-gb-"):
        world_co = obj.matrix_world.translation
        found_objects.append({
            'name': obj.name,
            'x': world_co.x,
            'y': world_co.y,
            'z': world_co.z
        })
        print(f"{obj.name}: X={world_co.x:.3f}, Y={world_co.y:.3f}, Z={world_co.z:.3f}")

if not found_objects:
    print("No objects found starting with 'RefH-gb-'")
else:
    print(f"\nTotal Reference Points Found: {len(found_objects)}")
    
    if found_objects:
        x_coords = [obj['x'] for obj in found_objects]
        y_coords = [obj['y'] for obj in found_objects]
        z_coords = [obj['z'] for obj in found_objects]
        
        print(f"\nCoordinate Ranges:")
        print(f"X: {min(x_coords):.3f} to {max(x_coords):.3f}")
        print(f"Y: {min(y_coords):.3f} to {max(y_coords):.3f}")
        print(f"Z: {min(z_coords):.3f} to {max(z_coords):.3f}")
