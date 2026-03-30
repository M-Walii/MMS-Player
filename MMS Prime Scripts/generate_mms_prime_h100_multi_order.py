"""
Generate MMS Prime from MMS data with inflections.

This script takes an MMS (Motion Measurement System) file and generates an MMS Prime file
by computing inflection angles (ALPHA and BETA) for hand rotations based on target points
specified in a pointing dictionary.

Example:
    python generate_mms_prime.py --input-mms "path/to/mms.csv" --corpus-dir "path/to/corpus" 
                                --output-dir "path/to/output" --pointing-dict "path/to/dict.json"
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Bone_Spine2-world matrix
# Generated with bone_world_matrix.py
BONE_WORLD = np.array([
   [1.623146658857877e-06, -9.530092626164333e-08, 0.9999999403953552, -4.030154286738252e-06],
   [0.9999996423721313, -0.0007954105967655778, -1.631790041756176e-06, 0.011902960017323494],
   [0.0007954107131808996, 0.9999995827674866, 2.53407730355093e-08, 126.90105438232422],
   [0.0,                   0.0,                0.0,                        1.0]
], dtype=float)

# Precompute inverse for world→bone conversion
BONE_INV = np.linalg.inv(BONE_WORLD)

# # Map-world matrix
# MAP_WORLD = np.array([
#     [ 0.9984923601150513,  -0.0019954282324761152,   0.054854705929756165,  -151.26473999023438 ],
#     [ 0.054884374141693115,  0.051805611699819565,   -0.9971478581428528,   -79.25841522216797 ],
#     [-0.0008520446135662496,  0.9986552000045776,      0.051837027072906494,  105.77490997314453 ],
#     [ 0.0,                    0.0,                     0.0,                     1.0               ]
# ], dtype=float)

# # Precompute inverse for Map→World→Bone chain 
# MAP_INV = np.linalg.inv(MAP_WORLD)

# # Direct Map→Bone matrix (Blender se calculate ki)
# MAP_TO_BONE = np.array([
#     [ 0.05488530918955803,  0.05259993299841881, -0.9971062541007996,  -79.28734588623047  ],
#     [-0.000895726669114083, 0.9986137747764587,   0.05263015627861023,  -21.063081741333008],
#     [ 0.998492419719696,   -0.0019954186864197254, 0.05485633760690689, -151.26461791992188],
#     [ 0.0,                  0.0,                   0.0,                   1.0              ]
# ], dtype=float)


def parse_arguments() -> Any:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - input_mms: Source MMS CSV file path
            - corpus_dir: Source corpus directory path
            - pointing_dict: Source pointing dictionary path
            - output_dir: Target output directory path
    
    Raises:
        SystemExit: If required arguments are missing or invalid
    """
    import argparse
    parser = argparse.ArgumentParser(description='Generate MMS Prime from MMS data with inflections')
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
    """
    Validate that all required paths exist and are accessible.
    
    Args:
        args: Source command line arguments containing all paths
    
    Raises:
        FileNotFoundError: If any required path does not exist
        PermissionError: If any path is not accessible
    """
    # Check if input MMS file exists
    if not os.path.isfile(args.input_mms):
        raise FileNotFoundError(f"Input MMS file not found: {args.input_mms}")
    
    # Check if corpus directory exists
    if not os.path.isdir(args.corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {args.corpus_dir}")
    
    # Check if pointing dictionary exists
    if not os.path.isfile(args.pointing_dict):
        raise FileNotFoundError(f"Pointing dictionary not found: {args.pointing_dict}")
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory: {args.output_dir}")


def load_pointing_dictionary(pointing_dict_path: str) -> Dict[int, List[float]]:
    """
    Load and validate the pointing dictionary from JSON file.
    
    Args:
        pointing_dict_path: Source pointing dictionary JSON file path
    
    Returns:
        Dict[int, List[float]]: Dictionary mapping line numbers to target coordinates
    
    Raises:
        ValueError: If dictionary format is invalid
        json.JSONDecodeError: If JSON file is malformed
    """
    try:
        with open(pointing_dict_path) as f:
            raw_dict = json.load(f)
            print(f"\nLoaded pointing dictionary: {raw_dict}")  # Debug print
            
        # Convert string keys to integers and validate values
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
                    # Validate that all coordinates are numbers
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
            
        print(f"Processed pointing dictionary: {pointing_dict}")  # Debug print
        return pointing_dict
        
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON format in pointing dictionary: {pointing_dict_path}\n"
            f"Please ensure the file is a valid JSON file"
        )

def load_trajectory_data(trajectory_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trajectory data from JSON file.
    
    Args:
        trajectory_file_path: Path to trajectory JSON file
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Starting point and initial direction vector
    """
    try:
        with open(trajectory_file_path) as f:
            trajectory_data = json.load(f)
        
        # Load full trajectory as an (N×3) array
        traj = np.array(trajectory_data['dominant_hand']['trajectory'])
        
        # Starting point is the first trajectory frame
        starting_point = traj[0]
        
        # Initial pointing direction: vector from frame 0 to frame 1
        initial_vector = traj[1] - traj[0]
        
        return starting_point, initial_vector
        
    except Exception as e:
        raise ValueError(f"Error loading trajectory data: {str(e)}")


def compute_inflection_angles(
    starting_point: np.ndarray,
    initial_vector: np.ndarray,
    target_point: np.ndarray
) -> Tuple[float, float]:
    """
    Adaptive solver: try both orders (Y→Z and Z→Y) and pick the one that
    minimizes angular error between rotated initial_vector and target direction.
    """

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
    corpus_dir: str                  # Corpus directory path
) -> pd.DataFrame:
    """
    Apply inflection angles to the MMS data to generate MMS Prime.
    Uses INDEX row's trajectory data for all cities, and pointing_dict for target points.
    Angles are in radians and placed in the current INDEX row based on last_target logic.
    """
    mms_prime = mms_df.copy()
    
    print("\nApplying inflections to generate MMS Prime (using INDEX trajectory)...")
    print(f"Pointing dictionary: {pointing_dict}")
    
    # First, get the INDEX row's trajectory data
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
            # 1) Get target point coordinates from pointing dictionary in world-space.
            #    These are the raw XYZ coordinates from the JSON file.
            world_xyz = np.array(pointing_dict[line_num], dtype=float)  # [x, y, z]

            # 2) Convert to homogeneous coordinates by appending w=1.
            #    This is needed for matrix multiplication with 4x4 transform matrices.
            world_h = np.append(world_xyz, 1.0)  # [x, y, z, 1]

            # 3) Transform from world coordinates to bone-space using inverse bone matrix.
            #    BONE_INV converts from world-space to bone-space coordinates.
            #    This gives us the target point relative to the bone's local coordinate system.
            local_h = BONE_INV.dot(world_h)      # [x', y', z', w']

            last_target = local_h[:3]            # strip w component
            print(f"  Converted world→bone local target: {last_target}")

        if maingloss == 'INDEX' and last_target is not None:
            try:
                print(f"  Using last_target for inflection: {last_target}")
                alpha, beta = compute_inflection_angles(
                    starting_point,
                    initial_vector,
                    last_target
                )
                print(f"  Calculated total angles: alpha={alpha}, beta={beta}")
                
                # Apply full angles to hand only
                hand_alpha = alpha
                hand_beta = beta

                print(f"  Hand angles (100%): alpha={hand_alpha}, beta={hand_beta}")

                # Update hand rotation (100% of total rotation)
                mms_prime.at[line_num, 'domhandrotx'] = 0.0
                mms_prime.at[line_num, 'domhandroty'] = float(hand_alpha)
                mms_prime.at[line_num, 'domhandrotz'] = float(hand_beta)
                
                print(f"  Updated hand and torso angles in INDEX row {line_num}")
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
    """
    Main function to generate MMS Prime data.
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_paths(args)
        
        # Load pointing dictionary
        pointing_dict = load_pointing_dictionary(args.pointing_dict)
        print(f"Loaded pointing dictionary with {len(pointing_dict)} entries")
        
        # Load MMS data
        print(f"Loading MMS data from: {args.input_mms}")
        try:
            mms_df = pd.read_csv(args.input_mms)
            if mms_df.empty:
                raise ValueError("MMS file is empty")
        except Exception as e:
            raise ValueError(f"Error loading MMS data: {str(e)}")
        
        # Generate MMS Prime
        mms_prime = apply_inflections(mms_df, pointing_dict, args.corpus_dir)
        
        # Save output
        output_file_path = os.path.join(args.output_dir, 
                                       os.path.basename(args.input_mms).replace('.mms.csv', 
                                                                              '.mms_prime.csv'))
        mms_prime.to_csv(output_file_path, index=False)
        print(f"\nSaved inflected MMS Prime to: {output_file_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
