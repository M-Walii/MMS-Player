
import sys, os, site, traceback
sys.path.insert(0, site.getusersitepackages())

try:
    import bpy
    _RUNNING_IN_BLENDER = True
    _BLENDER_BG = bpy.app.background
except Exception:
    _RUNNING_IN_BLENDER = False
    _BLENDER_BG = False

# Your default absolute paths (edit if needed)
_DEFAULT_INPUT_MMS  = r"C:/Users/walee/Documents/Thesis/MMS-Player/MMS-examples/ApplyInflectionsH50R50T50AngleSigned.mms.csv"
_DEFAULT_CORPUS_DIR = r"C:/Users/walee/Documents/Thesis/AVASAGcorpus-MLsubset91/AVASAGcorpus-MLsubset91"
_DEFAULT_POINT_DICT = r"C:/Users/walee/Documents/Thesis/MMS-Player/pointing_dict.json"
_DEFAULT_OUTPUT_DIR = r"C:/Users/walee/Documents/Thesis/MMS-Player/inflected_mms"

if _RUNNING_IN_BLENDER and not _BLENDER_BG:
    if len(sys.argv) <= 1 or sys.argv[0].endswith("blender"):
        sys.argv = [
            "generate_mms_prime_h50_r50_t50_angle_signed.py",
            "--input-mms",  _DEFAULT_INPUT_MMS,
            "--corpus-dir", _DEFAULT_CORPUS_DIR,
            "--pointing-dict", _DEFAULT_POINT_DICT,
            "--output-dir", _DEFAULT_OUTPUT_DIR,
        ]

import json
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import mathutils

BONE_WORLD = np.array([
   [1.623146658857877e-06, -9.530092626164333e-08, 0.9999999403953552, -4.030154286738252e-06],
   [0.9999996423721313, -0.0007954105967655778, -1.631790041756176e-06, 0.011902960017323494],
   [0.0007954107131808996, 0.9999995827674866, 2.53407730355093e-08, 126.90105438232422],
   [0.0,                   0.0,                0.0,                        1.0]
], dtype=float)
BONE_INV = np.linalg.inv(BONE_WORLD)


def parse_arguments() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description='Generate MMS Prime from MMS data with inflections (angle_signed 2D) - H50-R50-T50')
    parser.add_argument('--input-mms', required=True, help='Input MMS CSV (e.g., HandReloc-INDEX-X9.mms.csv)')
    parser.add_argument('--corpus-dir', required=True, help='Corpus directory (contains generated/signs/trimmed)')
    parser.add_argument('--pointing-dict', required=True, help='JSON: { line_index: [x,y,z] (world) }')
    parser.add_argument('--output-dir', required=True, help='Where to write *_mms_prime.csv')
    return parser.parse_args()


def validate_paths(args: Any) -> None:
    if not os.path.isfile(args.input_mms):
        raise FileNotFoundError(f"Input MMS file not found: {args.input_mms}")
    if not os.path.isdir(args.corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {args.corpus_dir}")
    if not os.path.isfile(args.pointing_dict):
        raise FileNotFoundError(f"Pointing dictionary not found: {args.pointing_dict}")
    os.makedirs(args.output_dir, exist_ok=True)


def load_pointing_dictionary(pointing_dict_path: str) -> Dict[int, List[float]]:
    with open(pointing_dict_path) as f:
        raw_dict = json.load(f)

    pointing_dict: Dict[int, List[float]] = {}
    for k, v in raw_dict.items():
        key = int(k)
        if key < 0:
            raise ValueError(f"Line number {k} cannot be negative")
        if not isinstance(v, list) or len(v) != 3:
            raise ValueError(f"Invalid coordinates for line {k}: {v} (need [x,y,z])")
        coords = [float(v[0]), float(v[1]), float(v[2])]
        pointing_dict[key] = coords

    if not pointing_dict:
        raise ValueError("Pointing dictionary is empty")
    return pointing_dict


def load_trajectory_data(trajectory_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(trajectory_file_path) as f:
        trajectory_data = json.load(f)

    traj = np.array(trajectory_data['dominant_hand']['trajectory'], dtype=float)
    if len(traj) < 2:
        raise ValueError("Trajectory must have ≥ 2 frames")

    starting_point = traj[0]
    initial_vector = traj[1] - traj[0]
    return starting_point, initial_vector


def compute_inflection_angles_angle_signed(
    starting_point: np.ndarray,
    initial_vector: np.ndarray,
    target_point: np.ndarray
) -> Tuple[float, float]:
    target_vector = (target_point - starting_point).astype(float)

    initial_unit1 = initial_vector / np.linalg.norm(initial_vector)
    target_unit1  = target_vector  / np.linalg.norm(target_vector)

    initial_xz = mathutils.Vector((float(initial_unit1[0]), float(initial_unit1[2])))
    target_xz  = mathutils.Vector((float(target_unit1[0]),  float(target_unit1[2])))
    alpha1 = initial_xz.angle_signed(target_xz, 0.0)

    initial_yx = mathutils.Vector((float(initial_unit1[1]), float(initial_unit1[0])))
    target_yx  = mathutils.Vector((float(target_unit1[1]),  float(target_unit1[0])))
    beta1 = initial_yx.angle_signed(target_yx, 0.0)

    return float(alpha1), float(beta1)


def apply_inflections(
    mms_df: pd.DataFrame,
    pointing_dict: Dict[int, List[float]],
    corpus_dir: str
) -> pd.DataFrame:
    mms_prime = mms_df.copy()

    for col in ("domhandrotx", "domhandroty", "domhandrotz", "domhandrelocay", "domhandrelocaz", "torsorelocay", "torsorelocaz"):
        if col not in mms_prime.columns:
            mms_prime[col] = 0.0
        mms_prime[col] = mms_prime[col].astype(float)

    index_trajectory_file = os.path.join(
        corpus_dir, 'generated', 'signs', 'trimmed', 'INDEX.trajectoryInfo.json'
    )
    if not os.path.isfile(index_trajectory_file):
        raise ValueError(f"INDEX trajectory file not found: {index_trajectory_file}")

    starting_point, initial_vector = load_trajectory_data(index_trajectory_file)

    target_point = None
    for line_num, row in mms_df.iterrows():
        if line_num in pointing_dict:
            world_xyz = np.array(pointing_dict[line_num], dtype=float)  # [x, y, z]
            world_h = np.append(world_xyz, 1.0)                         # [x, y, z, 1]
            local_h = BONE_INV.dot(world_h)                             # bone-space homogeneous
            target_point = local_h[:3]                                   # strip w

        maingloss = row['maingloss'] if 'maingloss' in row else None
        if maingloss == 'INDEX' and target_point is not None:
            alpha1, beta1 = compute_inflection_angles_angle_signed(
                starting_point,
                initial_vector,
                target_point
            )
            
            hand_alpha1 = alpha1 * 0.5
            hand_beta1 = beta1 * 0.5
            reloc_alpha1 = alpha1 * 0.5
            reloc_beta1 = beta1 * 0.5
            torso_alpha1 = alpha1 * 0.5
            torso_beta1 = beta1 * 0.5

            mms_prime.at[line_num, 'domhandrotx'] = 0.0
            mms_prime.at[line_num, 'domhandroty'] = hand_alpha1
            mms_prime.at[line_num, 'domhandrotz'] = hand_beta1
            
            mms_prime.at[line_num, 'domhandrelocay'] = reloc_alpha1
            mms_prime.at[line_num, 'domhandrelocaz'] = reloc_beta1
            
            mms_prime.at[line_num, 'torsorelocay'] = torso_alpha1
            mms_prime.at[line_num, 'torsorelocaz'] = torso_beta1

            target_point = None  # reset for next target

    return mms_prime


def main() -> None:
    try:
        args = parse_arguments()
        validate_paths(args)

        pointing_dict = load_pointing_dictionary(args.pointing_dict)
        mms_df = pd.read_csv(args.input_mms)
        if mms_df.empty:
            raise ValueError("MMS file is empty")

        mms_prime = apply_inflections(mms_df, pointing_dict, args.corpus_dir)

        output_file_path = os.path.join(
            args.output_dir,
            os.path.basename(args.input_mms).replace('.mms.csv', '_H50R50T50_angle_signed.mms_prime.csv')
        )
        mms_prime.to_csv(output_file_path, index=False)

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        if _RUNNING_IN_BLENDER and not _BLENDER_BG:
            return
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
