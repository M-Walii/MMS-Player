"""
Generate MMS Prime from MMS data with inflections - H50-R50-T50 condition.
Sequential Calculation Solution: Physics-based approach that applies torso movement first, then recalculates hand angles.

This script implements test condition #4: H50-R50-T50 with Sequential Calculation
- Step 1: Apply torso relocation angles
- Step 2: Calculate new hand position after torso movement
- Step 3: Recalculate hand angles from new position
- Step 4: Apply hand rotation and relocation angles

The solution follows real-world physics: torso moves first, then hand adjusts from new position.

Example:
    python generate_mms_prime_h50_r50_t50_sequential.py --input-mms "path/to/mms.csv" --corpus-dir "path/to/corpus" 
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
    parser = argparse.ArgumentParser(description='Generate MMS Prime from MMS data with inflections - H50-R50-T50 Sequential Calculation')
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
    Calculate ALPHA (horizontal angle in XZ plane) and BETA (vertical angle in XY plane)
    using atan2 to rotate from initial direction to target direction.

    Args:
        starting_point (np.ndarray): Origin (avatar finger base)
        initial_vector (np.ndarray): Current pointing direction vector (3D)
        target_point (np.ndarray): Desired pointing location in 3D space

    Returns:
        Tuple[float, float]: (alpha, beta) angles in radians
    """

    # Step 1: Target vector = direction from start to target
    target_vector = target_point - starting_point

    # Step 2: Normalize both vectors
    initial_unit = initial_vector / np.linalg.norm(initial_vector)
    target_unit = target_vector / np.linalg.norm(target_vector)

    # Step 3: ALPHA – Horizontal angle in XZ plane
    initial_alpha = np.arctan2(initial_unit[0], initial_unit[2])  # X/Z
    target_alpha = np.arctan2(target_unit[0], target_unit[2])
    alpha = target_alpha - initial_alpha
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    # Step 4: BETA – Vertical angle in XY plane
    initial_beta = np.arctan2(initial_unit[1], initial_unit[0])  # Y/X
    target_beta = np.arctan2(target_unit[1], target_unit[0])
    beta = target_beta - initial_beta
    beta = (beta + np.pi) % (2 * np.pi) - np.pi

    return alpha, beta


def apply_torso_movement(
    starting_point: np.ndarray,
    torso_alpha: float,
    torso_beta: float
) -> np.ndarray:
    """
    Apply torso movement to the starting point.
    This simulates the effect of torso relocation on hand position.
    
    Args:
        starting_point: Original hand position
        torso_alpha: Torso rotation angle in Y axis
        torso_beta: Torso rotation angle in Z axis
    
    Returns:
        np.ndarray: New hand position after torso movement
    """
    # Create rotation matrices for torso movement
    rotation_y = R.from_euler('y', torso_alpha, degrees=False)
    rotation_z = R.from_euler('z', torso_beta, degrees=False)
    
    # Combine rotations (Y first, then Z)
    combined_rotation = rotation_z * rotation_y
    
    # Apply rotation to starting point
    new_hand_position = combined_rotation.apply(starting_point)
    
    return new_hand_position


def apply_inflections(
    mms_df: pd.DataFrame,            # Source MMS data
    pointing_dict: Dict[int, List[float]],  # Target points dictionary
    corpus_dir: str                  # Corpus directory path
) -> pd.DataFrame:
    """
    Apply inflection angles to the MMS data to generate MMS Prime - H50-R50-T50 condition.
    Uses Sequential Calculation: torso first, then hand from new position.
    """
    mms_prime = mms_df.copy()
    
    print("\nApplying inflections to generate MMS Prime (H50-R50-T50 Sequential Calculation)...")
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
                
                # Calculate total angles first
                alpha, beta = compute_inflection_angles(
                    starting_point,
                    initial_vector,
                    last_target
                )
                print(f"  Calculated total angles: alpha={alpha}, beta={beta}")
                
                # H50-R50-T50 condition: Split angles equally (50%) among all three
                reloc_alpha = alpha * 0.5
                reloc_beta = beta * 0.5
                torso_alpha = alpha * 0.5
                torso_beta = beta * 0.5

                print(f"  SEQUENTIAL CALCULATION:")
                print(f"  Step 1: Apply torso movement (50% angles)")
                
                # Step 1: Apply torso movement to get new hand position
                new_hand_position = apply_torso_movement(
                    starting_point, torso_alpha, torso_beta
                )
                print(f"  Original hand position: {starting_point}")
                print(f"  New hand position after torso: {new_hand_position}")

                print(f"  Step 2: Recalculate hand angles from new position")
                
                # Step 2: Recalculate hand angles from new hand position
                hand_alpha, hand_beta = compute_inflection_angles(
                    new_hand_position,  # Use new position after torso movement
                    initial_vector,      # Same initial direction
                    last_target          # Same target
                )
                print(f"  Recalculated hand angles: alpha={hand_alpha}, beta={hand_beta}")

                print(f"  Step 3: Apply all angles")
                print(f"  Torso angles (50%): alpha={torso_alpha}, beta={torso_beta}")
                print(f"  Hand rotation angles (recalculated): alpha={hand_alpha}, beta={hand_beta}")
                print(f"  Hand relocation angles (50%): alpha={reloc_alpha}, beta={reloc_beta}")

                # Step 3: Apply all calculated angles
                # Update hand rotation (recalculated from new position)
                mms_prime.at[line_num, 'domhandrotx'] = 0.0
                mms_prime.at[line_num, 'domhandroty'] = float(hand_alpha)
                mms_prime.at[line_num, 'domhandrotz'] = float(hand_beta)
                
                # Update hand relocation (50% of total rotation)
                mms_prime.at[line_num, 'domhandrelocay'] = float(reloc_alpha)
                mms_prime.at[line_num, 'domhandrelocaz'] = float(reloc_beta)
                
                # Update torso relocation (50% of total rotation)
                mms_prime.at[line_num, 'torsorelocay'] = float(torso_alpha)
                mms_prime.at[line_num, 'torsorelocaz'] = float(torso_beta)
                
                print(f"  Step 4: All angles applied successfully")
                print(f"  Updated hand rotation (recalculated), hand relocation, and torso relocation in INDEX row {line_num}")
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
    Main function to generate MMS Prime data with Sequential Calculation.
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
                                                                              '_sequential.mms_prime.csv'))
        mms_prime.to_csv(output_file_path, index=False)
        print(f"\nSaved inflected MMS Prime (H50-R50-T50 Sequential Calculation) to: {output_file_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
