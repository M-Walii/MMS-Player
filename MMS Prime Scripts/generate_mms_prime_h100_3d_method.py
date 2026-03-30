"""
Generate MMS Prime from MMS data with inflections using 3d method.

This script implements the 3d solution for calculating elevation angles correctly.
The key difference is in BETA calculation: instead of projecting on YX plane, we calculate
the angle between the 3D target vector and its horizontal projection.

Example:
    python generate_mms_prime_h100_3d_method.py --input-mms "path/to/mms.csv" 
                                --corpus-dir "path/to/corpus" 
                                --output-dir "path/to/output" 
                                --pointing-dict "path/to/dict.json"
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

BONE_WORLD = np.array([
   [1.623146658857877e-06, -9.530092626164333e-08, 0.9999999403953552, -4.030154286738252e-06],
   [0.9999996423721313, -0.0007954105967655778, -1.631790041756176e-06, 0.011902960017323494],
   [0.0007954107131808996, 0.9999995827674866, 2.53407730355093e-08, 126.90105438232422],
   [0.0,                   0.0,                0.0,                        1.0]
], dtype=float)

BONE_INV = np.linalg.inv(BONE_WORLD)


def parse_arguments() -> Any:
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate MMS Prime using 3d method')
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
    """Validate that all required paths exist and are accessible."""
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
    """Load and validate the pointing dictionary from JSON file."""
    try:
        with open(pointing_dict_path) as f:
            raw_dict = json.load(f)
            print(f"\nLoaded pointing dictionary: {raw_dict}")
            
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
            
        print(f"Processed pointing dictionary: {pointing_dict}")
        return pointing_dict
        
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON format in pointing dictionary: {pointing_dict_path}\n"
            f"Please ensure the file is a valid JSON file"
        )


def load_trajectory_data(trajectory_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load trajectory data from JSON file."""
    try:
        with open(trajectory_file_path) as f:
            trajectory_data = json.load(f)
        
        traj = np.array(trajectory_data['dominant_hand']['trajectory'])
        starting_point = traj[0]
        initial_vector = traj[1] - traj[0]
        
        return starting_point, initial_vector
        
    except Exception as e:
        raise ValueError(f"Error loading trajectory data: {str(e)}")


def compute_inflection_angles_3d_method(
    starting_point: np.ndarray,
    initial_vector: np.ndarray,
    target_point: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate ALPHA and BETA using 3d method.
    
    ALPHA: Horizontal rotation angle (same as before - correct)
    BETA: Elevation angle = 3D_angle - Horizontal_angle
    
    Args:
        starting_point: Origin point (avatar finger base)
        initial_vector: Current pointing direction (3D)
        target_point: Target location in 3D space
    
    Returns:
        Tuple[float, float]: (alpha, beta) angles in radians
    """
    
    target_vector = target_point - starting_point
    
    initial_unit = initial_vector / np.linalg.norm(initial_vector)
    target_unit = target_vector / np.linalg.norm(target_vector)
    
    # ALPHA: Horizontal angle in XZ plane (this was already correct)
    initial_alpha = np.arctan2(initial_unit[0], initial_unit[2])
    target_alpha = np.arctan2(target_unit[0], target_unit[2])
    alpha = target_alpha - initial_alpha
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    
    # For INITIAL vector:
    # Step 1: Full 3D angle (initial to target in 3D space)
    # Using arctan2 on unit vectors
    # Calculate angle using X and Z components (magnitude in XZ plane)
    initial_horiz_distance = np.sqrt(initial_unit[0]**2 + initial_unit[2]**2)
    initial_3d_angle = np.arctan2(initial_unit[1], initial_horiz_distance)
    
    # Step 2: Horizontal angle (project to Y=0, then calculate angle)
    # For horizontal projection, set Y=0 and calculate angle
    initial_horiz_proj = initial_unit.copy()
    initial_horiz_proj[1] = 0.0
    initial_horiz_proj = initial_horiz_proj / np.linalg.norm(initial_horiz_proj)
    # Horizontal angle is angle in XZ plane
    initial_horizontal_angle = np.arctan2(initial_horiz_proj[0], initial_horiz_proj[2])
    
    # Initial elevation = 3D_angle - horizontal_angle
    initial_elevation = initial_3d_angle - initial_horizontal_angle
    
    # For TARGET vector:
    # Step 1: Full 3D angle (target in 3D space)
    target_horiz_distance = np.sqrt(target_unit[0]**2 + target_unit[2]**2)
    target_3d_angle = np.arctan2(target_unit[1], target_horiz_distance)
    
    # Step 2: Horizontal angle (project to Y=0, then calculate angle)
    target_horiz_proj = target_unit.copy()
    target_horiz_proj[1] = 0.0
    target_horiz_proj = target_horiz_proj / np.linalg.norm(target_horiz_proj)
    # Horizontal angle is angle in XZ plane
    target_horizontal_angle = np.arctan2(target_horiz_proj[0], target_horiz_proj[2])
    
    # Target elevation = 3D_angle - horizontal_angle
    target_elevation = target_3d_angle - target_horizontal_angle
    
    # BETA = difference in elevations
    beta = target_elevation - initial_elevation
    return alpha, beta


def apply_inflections(
    mms_df: pd.DataFrame,
    pointing_dict: Dict[int, List[float]],
    corpus_dir: str
) -> pd.DataFrame:
    """Apply inflection angles using 3d method."""
    mms_prime = mms_df.copy()
    
    print("\nApplying inflections using 3d METHOD...")
    print(f"Pointing dictionary: {pointing_dict}")
    
    index_trajectory_file = os.path.join(
        corpus_dir, 'generated', 'signs', 'trimmed', 'INDEX.trajectoryInfo.json'
    )
    if not os.path.isfile(index_trajectory_file):
        raise ValueError(f"INDEX trajectory file not found: {index_trajectory_file}")
    
    starting_point, initial_vector = load_trajectory_data(index_trajectory_file)
    print(f"\nUsing INDEX trajectory data:")
    print(f"  Starting point: {starting_point}")
    print(f"  Initial vector: {initial_vector}")
    
    last_target = None
    for line_num, row in mms_df.iterrows():
        maingloss = row['maingloss']
        print(f"\nProcessing row {line_num}: {maingloss}")

        if line_num in pointing_dict:
            world_xyz = np.array(pointing_dict[line_num], dtype=float)
            print(f"  World target: {world_xyz}")

            world_h = np.append(world_xyz, 1.0)
            local_h = BONE_INV.dot(world_h)
            last_target = local_h[:3]
            print(f"  Bone-space target: {last_target}")

        if maingloss == 'INDEX' and last_target is not None:
            try:
                print(f"  Calculating angles for target: {last_target}")
                alpha, beta = compute_inflection_angles_3d_method(
                    starting_point,
                    initial_vector,
                    last_target
                )
                print(f"  Final angles: alpha={alpha:.6f}, beta={beta:.6f}")
                
                hand_alpha = alpha
                hand_beta = beta

                print(f"  Hand angles (100%): alpha={hand_alpha:.6f}, beta={hand_beta:.6f}")

                mms_prime.at[line_num, 'domhandrotx'] = 0.0
                mms_prime.at[line_num, 'domhandroty'] = float(hand_alpha)
                mms_prime.at[line_num, 'domhandrotz'] = float(hand_beta)
                
                print(f"  Updated INDEX row {line_num}")
                last_target = None
            except Exception as e:
                print(f"Error processing INDEX at row {line_num}: {str(e)}")
        else:
            if maingloss != 'INDEX':
                print(f"  Skipping row - not INDEX")
            elif last_target is None:
                print(f"  Skipping INDEX row - last_target is None")
    
    return mms_prime


def main() -> None:
    """Main function to generate MMS Prime data."""
    try:
        args = parse_arguments()
        validate_paths(args)
        
        pointing_dict = load_pointing_dictionary(args.pointing_dict)
        print(f"Loaded pointing dictionary with {len(pointing_dict)} entries")
        
        print(f"Loading MMS data from: {args.input_mms}")
        try:
            mms_df = pd.read_csv(args.input_mms)
            if mms_df.empty:
                raise ValueError("MMS file is empty")
        except Exception as e:
            raise ValueError(f"Error loading MMS data: {str(e)}")
        
        mms_prime = apply_inflections(mms_df, pointing_dict, args.corpus_dir)
        
        base_name = os.path.basename(args.input_mms).replace('.mms.csv', '')
        output_file_path = os.path.join(args.output_dir, f"{base_name}_3d_method.mms_prime.csv")
        mms_prime.to_csv(output_file_path, index=False)
        print(f"\nSaved MMS Prime to: {output_file_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

