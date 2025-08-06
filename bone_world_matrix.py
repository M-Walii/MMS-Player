import bpy
import json

# 1) Select the armature and bone
arm = bpy.data.objects['skeleton #5']
pose_bone = arm.pose.bones['Bone Spine2']

# 2) Create bone's world matrix (arm.matrix_world @ pose_bone.matrix)
bone_world = arm.matrix_world @ pose_bone.matrix

# 3) Convert matrix to list-of-lists and print it
mat = [[bone_world[i][j] for j in range(4)] for i in range(4)]
print("Bone 4x4 world matrix:")
for row in mat:
    print(row)
