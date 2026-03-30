import bpy
import json

arm = bpy.data.objects['skeleton #5']
pose_bone = arm.pose.bones['Bone Spine2']

bone_world = arm.matrix_world @ pose_bone.matrix

mat = [[bone_world[i][j] for j in range(4)] for i in range(4)]
print("Bone 4x4 world matrix:")
for row in mat:
    print(row)
