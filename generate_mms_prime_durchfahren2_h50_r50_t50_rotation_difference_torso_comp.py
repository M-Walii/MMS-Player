
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
_DEFAULT_INPUT_MMS  = r"C:/Users/walee/Documents/Thesis/MMS-Player/MMS-examples/0011_mms.csv"
_DEFAULT_CORPUS_DIR = r"C:/Users/walee/Documents/Thesis/AVASAGcorpus-MLsubset91/AVASAGcorpus-MLsubset91"
_DEFAULT_POINT_DICT = r"C:/Users/walee/Documents/Thesis/MMS-Player/pointing_dict.json"
_DEFAULT_OUTPUT_DIR = r"C:/Users/walee/Documents/Thesis/MMS-Player/inflected_mms"

if _RUNNING_IN_BLENDER and not _BLENDER_BG:
    if len(sys.argv) <= 1 or sys.argv[0].endswith("blender"):
        sys.argv = [
            "generate_mms_prime_durchfahren2_h50_r50_t50_rotation_difference_torso_comp.py",
            "--input-mms",  _DEFAULT_INPUT_MMS,
            "--corpus-dir", _DEFAULT_CORPUS_DIR,
            "--pointing-dict", _DEFAULT_POINT_DICT,
            "--output-dir", _DEFAULT_OUTPUT_DIR,
        ]

import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import mathutils
from mathutils import Vector

BONE_WORLD = np.array([
   [1.623146658857877e-06, -9.530092626164333e-08, 0.9999999403953552, -4.030154286738252e-06],
   [0.9999996423721313, -0.0007954105967655778, -1.631790041756176e-06, 0.011902960017323494],
   [0.0007954107131808996, 0.9999995827674866, 2.53407730355093e-08, 126.90105438232422],
   [0.0,                   0.0,                0.0,                        1.0]
], dtype=float)
BONE_INV = np.linalg.inv(BONE_WORLD)

HAMBURG_COORDS = np.array([-94.7089, 1.3793, 156.156], dtype=float)

TORSO_BASE = np.array([0.0, 0.0, 0.0], dtype=float)


def parse_arguments() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description='Generate MMS Prime with quaternion inflections for DURCHFAHREN2 (H50-R50-T50, torso compensated)')
    parser.add_argument('--input-mms', required=True, help='Input MMS CSV (e.g., 0011_mms.csv)')
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


def compute_inflection_angles_quaternion_method(
    starting_point: np.ndarray,
    initial_vector: np.ndarray,
    target_point: np.ndarray
) -> Tuple[float, float, float]:
    initial_unit = initial_vector / np.linalg.norm(initial_vector)
    target_vector = target_point - starting_point
    target_unit = target_vector / np.linalg.norm(target_vector)
    
    initial_vec = Vector((float(initial_unit[0]), float(initial_unit[1]), float(initial_unit[2])))
    target_vec = Vector((float(target_unit[0]), float(target_unit[1]), float(target_unit[2])))
    
    Q = initial_vec.rotation_difference(target_vec)
    
    euler_angles = Q.to_euler("ZXY")
    
    alpha = float(euler_angles.x)
    beta = float(euler_angles.y)
    gamma = float(euler_angles.z)
    
    return alpha, beta, gamma


def compute_torso_compensated_target(
    target_point: np.ndarray,
    torso_base: np.ndarray,
    alpha_t: float,
    beta_t: float,
    gamma_t: float
) -> np.ndarray:
    rot_t = mathutils.Euler((alpha_t, beta_t, gamma_t), "ZXY").to_matrix()

    d_vec = Vector((
        float(target_point[0] - torso_base[0]),
        float(target_point[1] - torso_base[1]),
        float(target_point[2] - torso_base[2]),
    ))

    d_prime = rot_t.inverted() @ d_vec

    t_prime = np.array([
        float(d_prime.x + torso_base[0]),
        float(d_prime.y + torso_base[1]),
        float(d_prime.z + torso_base[2]),
    ], dtype=float)
    return t_prime


def find_previous_city(mms_df: pd.DataFrame, start_line: int, pointing_dict: Dict[int, List[float]]) -> Optional[Tuple[int, np.ndarray]]:
    for line_num in range(start_line - 1, -1, -1):
        if line_num in pointing_dict:
            coords = np.array(pointing_dict[line_num], dtype=float)
            return (line_num, coords)
    
    return None


def find_next_city(mms_df: pd.DataFrame, start_line: int, pointing_dict: Dict[int, List[float]]) -> Optional[Tuple[int, np.ndarray]]:
    for line_num in range(start_line + 1, len(mms_df)):
        if line_num in pointing_dict:
            coords = np.array(pointing_dict[line_num], dtype=float)
            return (line_num, coords)
    
    return None


def apply_inflections(
    mms_df: pd.DataFrame,
    pointing_dict: Dict[int, List[float]],
    corpus_dir: str
) -> pd.DataFrame:
    mms_prime = mms_df.copy()

    for col in ("domhandrotx", "domhandroty", "domhandrotz", 
                "domhandrelocax", "domhandrelocay", "domhandrelocaz",
                "torsorelocax", "torsorelocay", "torsorelocaz"):
        if col not in mms_prime.columns:
            mms_prime[col] = 0.0
        mms_prime[col] = mms_prime[col].astype(float)

    durchfahren2_trajectory_file = os.path.join(
        corpus_dir, 'generated', 'signs', 'trimmed', 'DURCHFAHREN2.trajectoryInfo.json'
    )
    if not os.path.isfile(durchfahren2_trajectory_file):
        raise ValueError(f"DURCHFAHREN2 trajectory file not found: {durchfahren2_trajectory_file}")

    starting_point, initial_vector = load_trajectory_data(durchfahren2_trajectory_file)

    last_city_coords = HAMBURG_COORDS.copy()
    last_city_line = None
    first_durchfahren2_found = False

    for line_num, row in mms_df.iterrows():
        maingloss = row['maingloss'] if 'maingloss' in row else None
        
        if line_num in pointing_dict:
            last_city_coords = np.array(pointing_dict[line_num], dtype=float)
            last_city_line = line_num

        if maingloss == 'DURCHFAHREN2':
            if not first_durchfahren2_found:
                previous_city_coords = HAMBURG_COORDS.copy()
                previous_city_line = None
                first_durchfahren2_found = True
            else:
                if last_city_line is not None:
                    previous_city_coords = last_city_coords.copy()
                    previous_city_line = last_city_line
                else:
                    previous_city_coords = HAMBURG_COORDS.copy()
                    previous_city_line = None
            
            next_city_info = find_next_city(mms_df, line_num, pointing_dict)
            
            if next_city_info is None:
                continue
            
            next_line_num, next_city_coords = next_city_info
            
            target_vector_world = next_city_coords - previous_city_coords
            
            world_prev_h = np.append(previous_city_coords, 1.0)
            local_prev_h = BONE_INV.dot(world_prev_h)
            previous_city_bone = local_prev_h[:3]
            
            world_next_h = np.append(next_city_coords, 1.0)
            local_next_h = BONE_INV.dot(world_next_h)
            next_city_bone = local_next_h[:3]
            
            target_vector_bone = next_city_bone - previous_city_bone
            
            target_point_bone = starting_point + target_vector_bone
            
            alpha, beta, gamma = compute_inflection_angles_quaternion_method(
                starting_point,
                initial_vector,
                target_point_bone
            )
            
            torso_alpha = alpha * 0.5
            torso_beta = beta * 0.5
            torso_gamma = gamma * 0.5

            target_prime = compute_torso_compensated_target(
                target_point_bone,
                TORSO_BASE,
                torso_alpha,
                torso_beta,
                torso_gamma
            )
            hand_alpha, hand_beta, hand_gamma = compute_inflection_angles_quaternion_method(
                starting_point,
                initial_vector,
                target_prime
            )

            mms_prime.at[line_num, 'domhandrotx'] = hand_alpha
            mms_prime.at[line_num, 'domhandroty'] = hand_beta
            mms_prime.at[line_num, 'domhandrotz'] = hand_gamma
            
            mms_prime.at[line_num, 'domhandrelocax'] = hand_alpha
            mms_prime.at[line_num, 'domhandrelocay'] = hand_beta
            mms_prime.at[line_num, 'domhandrelocaz'] = hand_gamma

            mms_prime.at[line_num, 'torsorelocax'] = torso_alpha
            mms_prime.at[line_num, 'torsorelocay'] = torso_beta
            mms_prime.at[line_num, 'torsorelocaz'] = torso_gamma

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
            os.path.basename(args.input_mms).replace('_mms.csv', '_H50R50T50_mms_prime.csv')
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

