import bpy
mat = bpy.data.objects['Map'].matrix_world
print("Map 4x4 world matrix:")
for row in mat:
    print(list(row))
