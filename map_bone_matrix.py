import bpy

# 1) Get world matrix for armature and bone
arm        = bpy.data.objects['skeleton #5']
pose_bone  = arm.pose.bones['Bone Spine2']
bone_world = arm.matrix_world @ pose_bone.matrix

# 2) Create inverse of Bone→World matrix (World→Bone)
bone_inv   = bone_world.inverted()

# 3) Get world matrix for Map object (Map→World)
map_obj    = bpy.data.objects['Map']
map_world  = map_obj.matrix_world

# 4) Combine to create Map→Bone matrix
map_to_bone = bone_inv @ map_world

# 5) Print 4×4 matrix values
print("Map→Bone 4×4 matrix:")
for row in map_to_bone:
    print(list(row))
