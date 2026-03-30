import argparse
import bpy
import json
import sys
import os
from pathlib import Path
from sklearn.decomposition import PCA
import numpy as np

def load_bone_dict(bonedict_path):
    path = Path(bonedict_path)
    assert path.exists(), f"Making sure {bonedict_path} exists."
    with open(path, "r") as stream:
        return json.load(stream)

def import_assets(blend_file):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    with bpy.data.libraries.load(blend_file) as (data_from, data_to):
        data_to.objects = data_from.objects
        data_to.worlds = data_from.worlds
    for obj in data_to.objects:
        bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.world = data_to.worlds[0]

def extract_trajectory(armature, bonelist, start, stop):
    dom_bone = []
    ndom_bone = []
    hand_data = {"dom": dom_bone, "ndom": ndom_bone}
    for frame in range(start, stop + 1):
        bpy.context.scene.frame_set(frame)
        for (hand_preference, bone_name, parent_bone_name) in bonelist:
            bone = armature.pose.bones[bone_name]
            parent_bone = armature.pose.bones[parent_bone_name]
            rel_head = parent_bone.matrix.inverted() @ bone.head
            data = hand_data[hand_preference]
            data.append([rel_head.x, rel_head.y, rel_head.z])
    return (dom_bone, ndom_bone)

def perform_pca(data):
    pca = PCA(n_components=3)
    pca.fit(data)
    pc1_pc2_ratio = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1] if pca.explained_variance_ratio_[1] != 0 else 0
    return {
        "components": {
            "PC1": list(pca.components_[0]),
            "PC2": list(pca.components_[1]),
            "PC3": list(pca.components_[2])
        },
        "explained_variance_ratio": list(pca.explained_variance_ratio_),
        "explained_variance": list(pca.explained_variance_),
        "pc1_pc2_ratio": pc1_pc2_ratio
    }

def extract_single_gloss_trajectory(gloss_path, bone_dict, output_path):
    bonelist = bone_dict.get("trajectory", [])
    
    if not bonelist:
        return
    
    import_assets(str(gloss_path))
    
    armature = bpy.data.objects.get(gloss_path.stem)
    if not armature:
        return
    
    action = armature.animation_data.action
    start = int(action.frame_range[0])
    stop = int(action.frame_range[1])

    dom_trajectory, ndom_trajectory = extract_trajectory(armature, bonelist, start, stop)

    dom_pca = perform_pca(np.array(dom_trajectory))
    ndom_pca = perform_pca(np.array(ndom_trajectory))

    trajectory_data = {
        "gloss": gloss_path.stem,
        "dominant_hand": {
            "trajectory": dom_trajectory,
            "components": dom_pca['components'],
            "explained_variance_ratio": dom_pca['explained_variance_ratio'],
            "explained_variance": dom_pca['explained_variance'],
            "pc1_pc2_ratio": dom_pca['pc1_pc2_ratio']
        },
        "non_dominant_hand": {
            "trajectory": ndom_trajectory,
            "components": ndom_pca['components'],
            "explained_variance_ratio": ndom_pca['explained_variance_ratio'],
            "explained_variance": ndom_pca['explained_variance'],
            "pc1_pc2_ratio": ndom_pca['pc1_pc2_ratio']
        }
    }

    output_file = output_path.joinpath(f"{gloss_path.stem}.trajectoryInfo.json")
    write_to_file(output_file, trajectory_data)

def write_to_file(filename, data):
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=2)

if __name__ == "__main__":
    argv = sys.argv

    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1 :]  # get all args after "--"

    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-path", type=str, help="Path to the generated signs/trimmed folder.")
    parser.add_argument("--bonedict-path", type=str, help="Path to extract bones.")
    args = parser.parse_args(argv)

    bone_dict = load_bone_dict(args.bonedict_path)
    gen_path = Path(args.generated_path)

    blend_files = list(gen_path.glob("*.blend"))
    
    for gloss_file in blend_files:
        extract_single_gloss_trajectory(gloss_file, bone_dict, gen_path)
