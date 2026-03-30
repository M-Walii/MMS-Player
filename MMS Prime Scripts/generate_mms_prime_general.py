import json
import os
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

BONE_WORLD = np.array([
   [1.623146658857877e-06, -9.530092626164333e-08, 0.9999999403953552, -4.030154286738252e-06],
   [0.9999996423721313, -0.0007954105967655778, -1.631790041756176e-06, 0.011902960017323494],
   [0.0007954107131808996, 0.9999995827674866, 2.53407730355093e-08, 126.90105438232422],
   [0.0,                   0.0,                0.0,                        1.0]
], dtype=float)

BONE_INV = np.linalg.inv(BONE_WORLD)


def parse_condition(condition_str: str) -> Tuple[float, float, float]:
    condition_str = condition_str.upper().strip()
    
    if condition_str == "H100":
        return 1.0, 0.0, 0.0
    elif condition_str == "H100-R100":
        return 1.0, 1.0, 0.0
    elif condition_str == "H50-T50":
        return 0.5, 0.0, 0.5
    elif condition_str == "H50-R50-T50":
        return 0.5, 0.5, 0.5
    else:
        raise ValueError(f"Unknown condition: {condition_str}. Valid options: H100, H100-R100, H50-T50, H50-R50-T50")


def parse_arguments() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description='Generate MMS Prime from MMS data with inflections - General Script')
    
    parser.add_argument('--condition', required=True, 
                       choices=['H100', 'H100-R100', 'H50-T50', 'H50-R50-T50'],
                       help='Test condition: H100, H100-R100, H50-T50, or H50-R50-T50')
    parser.add_argument('--solver', required=True,
                       choices=['arctan', 'multi'],
                       help='Angle solver: arctan (simple) or multi (adaptive dual-order)')
    parser.add_argument('--input-mms', required=True, 
                       help='Input MMS CSV filename (e.g., HandReloc-INDEX-X9.mms.csv)')
    parser.add_argument('--corpus-dir', required=True,
                       help='Directory containing corpus data')
    parser.add_argument('--pointing-dict', required=True,
                       help='Path to JSON file containing pointing dictionary')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for MMS Prime files')
    
    return parser.parse_args()


def validate_paths(args: Any) -> None:
    if not os.path.isfile(args.input_mms):
        raise FileNotFoundError(f"Input MMS file not found: {args.input_mms}")
    
    if not os.path.isdir(args.corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {args.corpus_dir}")
    
    if not os.path.isfile(args.pointing_dict):
        raise FileNotFoundError(f"Pointing dictionary not found: {args.pointing_dict}")
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory: {args.output_dir}")


def load_pointing_dictionary(pointing_dict_path: str) -> Dict[int, List[float]]:
    try:
        with open(pointing_dict_path) as f:
            raw_dict = json.load(f)
            
        pointing_dict = {}
        for k, v in raw_dict.items():
            try:
                key = int(k)
                if key < 0:
                    raise ValueError(f"Line number {k} cannot be negative")
                    
                if not isinstance(v, list) or len(v) != 3:
                    raise ValueError(
                        f"Invalid coordinates for line {k}: {v}\n"
                        f"Expected format: [x, y, z] (3 numbers)"
                    )
                    
                try:
                    coords = [float(coord) for coord in v]
                    pointing_dict[key] = coords
                except ValueError:
                    raise ValueError(
                        f"Invalid coordinates for line {k}: {v}\n"
                        f"All coordinates must be numbers"
                    )
                    
            except ValueError as e:
                raise ValueError(f"Invalid pointing dictionary format: {str(e)}")
                
        if not pointing_dict:
            raise ValueError("Pointing dictionary is empty")
        return pointing_dict
        
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON format in pointing dictionary: {pointing_dict_path}\n"
            f"Please ensure the file is a valid JSON file"
        )

def load_trajectory_data(trajectory_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        with open(trajectory_file_path) as f:
            trajectory_data = json.load(f)
        
        traj = np.array(trajectory_data['dominant_hand']['trajectory'])
        
        starting_point = traj[0]
        
        initial_vector = traj[1] - traj[0]
        
        return starting_point, initial_vector
        
    except Exception as e:
        raise ValueError(f"Error loading trajectory data: {str(e)}")


def compute_inflection_angles_arctan(
    starting_point: np.ndarray,
    initial_vector: np.ndarray,
    target_point: np.ndarray
) -> Tuple[float, float]:
    target_vector = target_point - starting_point

    initial_unit = initial_vector / np.linalg.norm(initial_vector)
    target_unit = target_vector / np.linalg.norm(target_vector)

    initial_alpha = np.arctan2(initial_unit[0], initial_unit[2])  # X/Z
    target_alpha = np.arctan2(target_unit[0], target_unit[2])
    alpha = target_alpha - initial_alpha
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    initial_beta = np.arctan2(initial_unit[1], initial_unit[0])  # Y/X
    target_beta = np.arctan2(target_unit[1], target_unit[0])
    beta = target_beta - initial_beta
    beta = (beta + np.pi) % (2 * np.pi) - np.pi

    return alpha, beta


def compute_inflection_angles_multi_order(
    starting_point: np.ndarray,
    initial_vector: np.ndarray,
    target_point: np.ndarray
) -> Tuple[float, float]:
    def normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n != 0 else v

    def rotate_y(vec: np.ndarray, ang: float) -> np.ndarray:
        c, s = np.cos(ang), np.sin(ang)
        x, y, z = vec
        return np.array([c * x + s * z, y, -s * x + c * z])

    def rotate_z(vec: np.ndarray, ang: float) -> np.ndarray:
        c, s = np.cos(ang), np.sin(ang)
        x, y, z = vec
        return np.array([c * x - s * y, s * x + c * y, z])

    def yaw_then_roll(u: np.ndarray, v: np.ndarray) -> Tuple[float, float, float]:
        # Solve Y first by matching XZ azimuths
        az_u = np.arctan2(u[0], u[2])
        az_v = np.arctan2(v[0], v[2])
        a = (az_v - az_u + np.pi) % (2 * np.pi) - np.pi
        u1 = rotate_y(u, a)
        el_u = np.arctan2(u1[1], u1[0])
        el_v = np.arctan2(v[1], v[0])
        b = (el_v - el_u + np.pi) % (2 * np.pi) - np.pi
        u2 = rotate_z(u1, b)
        # angular error
        err = np.clip(np.dot(normalize(u2), normalize(v)), -1.0, 1.0)
        ang_err = np.arccos(err)
        return a, b, ang_err

    def roll_then_yaw(u: np.ndarray, v: np.ndarray) -> Tuple[float, float, float]:
        # Solve Z first by matching XY elevations
        el_u = np.arctan2(u[1], u[0])
        el_v = np.arctan2(v[1], v[0])
        b = (el_v - el_u + np.pi) % (2 * np.pi) - np.pi
        u1 = rotate_z(u, b)
        az_u = np.arctan2(u1[0], u1[2])
        az_v = np.arctan2(v[0], v[2])
        a = (az_v - az_u + np.pi) % (2 * np.pi) - np.pi
        u2 = rotate_y(u1, a)
        err = np.clip(np.dot(normalize(u2), normalize(v)), -1.0, 1.0)
        ang_err = np.arccos(err)
        return a, b, ang_err

    # Build unit vectors
    u = normalize(initial_vector)
    v = normalize(target_point - starting_point)

    a1, b1, e1 = yaw_then_roll(u, v)
    a2, b2, e2 = roll_then_yaw(u, v)

    if e1 <= e2:
        return a1, b1
    return a2, b2


def apply_inflections(
    mms_df: pd.DataFrame,            # Source MMS data
    pointing_dict: Dict[int, List[float]],  # Target points dictionary
    corpus_dir: str,                 # Corpus directory path
    condition: str,                  # Test condition
    solver: str                      # Angle solver type
) -> pd.DataFrame:
    mms_prime = mms_df.copy()
    
    domhandrot_factor, domhandreloca_factor, torsoreloca_factor = parse_condition(condition)

    index_trajectory_file = os.path.join(
        corpus_dir, 'generated', 'signs', 'trimmed', 'INDEX.trajectoryInfo.json'
    )
    if not os.path.isfile(index_trajectory_file):
        raise ValueError(f"INDEX trajectory file not found: {index_trajectory_file}")
    
    starting_point, initial_vector = load_trajectory_data(index_trajectory_file)
    
    if solver == "arctan":
        compute_angles = compute_inflection_angles_arctan
    else:  # multi
        compute_angles = compute_inflection_angles_multi_order
    
    last_target = None
    for line_num, row in mms_df.iterrows():
        maingloss = row['maingloss']

        if line_num in pointing_dict:
            world_xyz = np.array(pointing_dict[line_num], dtype=float)  # [x, y, z]

            world_h = np.append(world_xyz, 1.0)  # [x, y, z, 1]

            local_h = BONE_INV.dot(world_h)      # [x', y', z', w']

            last_target = local_h[:3]            # strip w component

        if maingloss == 'INDEX' and last_target is not None:
            try:
                alpha, beta = compute_angles(
                    starting_point,
                    initial_vector,
                    last_target
                )
                
                hand_alpha = alpha * domhandrot_factor
                hand_beta = beta * domhandrot_factor
                reloc_alpha = alpha * domhandreloca_factor
                reloc_beta = beta * domhandreloca_factor
                torso_alpha = alpha * torsoreloca_factor
                torso_beta = beta * torsoreloca_factor

                mms_prime.at[line_num, 'domhandrotx'] = 0.0
                mms_prime.at[line_num, 'domhandroty'] = float(hand_alpha)
                mms_prime.at[line_num, 'domhandrotz'] = float(hand_beta)
                
                if domhandreloca_factor > 0:
                    mms_prime.at[line_num, 'domhandrelocay'] = float(reloc_alpha)
                    mms_prime.at[line_num, 'domhandrelocaz'] = float(reloc_beta)
                
                if torsoreloca_factor > 0:
                    mms_prime.at[line_num, 'torsorelocay'] = float(torso_alpha)
                    mms_prime.at[line_num, 'torsorelocaz'] = float(torso_beta)
                
                last_target = None
            except Exception as e:
                raise
    
    return mms_prime


def main() -> None:
    try:
        args = parse_arguments()
        validate_paths(args)
        
        pointing_dict = load_pointing_dictionary(args.pointing_dict)
        
        try:
            mms_df = pd.read_csv(args.input_mms)
            if mms_df.empty:
                raise ValueError("MMS file is empty")
        except Exception as e:
            raise ValueError(f"Error loading MMS data: {str(e)}")
        
        mms_prime = apply_inflections(mms_df, pointing_dict, args.corpus_dir, args.condition, args.solver)
        
        base_name = os.path.basename(args.input_mms).replace('.mms.csv', '')
        output_filename = f"{base_name}_{args.condition}_{args.solver}.mms_prime.csv"
        output_file_path = os.path.join(args.output_dir, output_filename)
        
        mms_prime.to_csv(output_file_path, index=False)
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
