"""Generate MMS Prime using quaternion-based inflections for DURCHFAHREN2 (100% to dominant hand)."""

# -------- Blender GUI-safe header (no CLI needed) --------
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
            "generate_mms_prime_durchfahren2_rotation_difference.py",
            "--input-mms",  _DEFAULT_INPUT_MMS,
            "--corpus-dir", _DEFAULT_CORPUS_DIR,
            "--pointing-dict", _DEFAULT_POINT_DICT,
            "--output-dir", _DEFAULT_OUTPUT_DIR,
        ]
# -------- end header --------

import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import mathutils
from mathutils import Vector

# Bone_Spine2-world matrix (same as your original)
BONE_WORLD = np.array([
   [1.623146658857877e-06, -9.530092626164333e-08, 0.9999999403953552, -4.030154286738252e-06],
   [0.9999996423721313, -0.0007954105967655778, -1.631790041756176e-06, 0.011902960017323494],
   [0.0007954107131808996, 0.9999995827674866, 2.53407730355093e-08, 126.90105438232422],
   [0.0,                   0.0,                0.0,                        1.0]
], dtype=float)
BONE_INV = np.linalg.inv(BONE_WORLD)

# Hardcoded starting city (Hamburg)
HAMBURG_COORDS = np.array([-94.7089, 1.3793, 156.156], dtype=float)


def parse_arguments() -> Any:
    """Parse CLI arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate MMS Prime with quaternion inflections for DURCHFAHREN2')
    parser.add_argument('--input-mms', required=True, help='Input MMS CSV (e.g., 0011_combined.mms.csv)')
    parser.add_argument('--corpus-dir', required=True, help='Corpus directory (contains generated/signs/trimmed)')
    parser.add_argument('--pointing-dict', required=True, help='JSON: { line_index: [x,y,z] (world) }')
    parser.add_argument('--output-dir', required=True, help='Where to write *_mms_prime.csv')
    return parser.parse_args()


def validate_paths(args: Any) -> None:
    """Validate existence of input/output paths."""
    if not os.path.isfile(args.input_mms):
        raise FileNotFoundError(f"Input MMS file not found: {args.input_mms}")
    if not os.path.isdir(args.corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {args.corpus_dir}")
    if not os.path.isfile(args.pointing_dict):
        raise FileNotFoundError(f"Pointing dictionary not found: {args.pointing_dict}")
    os.makedirs(args.output_dir, exist_ok=True)


def load_pointing_dictionary(pointing_dict_path: str) -> Dict[int, List[float]]:
    """Load pointing targets from JSON into {line_index: [x, y, z] }."""
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
    """Load DURCHFAHREN2 trajectory and return (starting_point, initial_vector)."""
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
    """Compute (alpha, beta, gamma) using quaternion rotation_difference."""
    # Normalize vectors
    initial_unit = initial_vector / np.linalg.norm(initial_vector)
    target_vector = target_point - starting_point
    target_unit = target_vector / np.linalg.norm(target_vector)
    
    # Convert to mathutils.Vector for rotation_difference method
    initial_vec = Vector((float(initial_unit[0]), float(initial_unit[1]), float(initial_unit[2])))
    target_vec = Vector((float(target_unit[0]), float(target_unit[1]), float(target_unit[2])))
    
    # Step 1: Calculate rotation quaternion Q
    # Q represents the rotation needed to go from initial_vector to target_vector
    Q = initial_vec.rotation_difference(target_vec)
    
    # Step 2: Convert quaternion to Euler angles using ZXY order
    # This gives us (α, β, γ) = (alpha, beta, gamma)
    euler_angles = Q.to_euler("ZXY")
    
    alpha = float(euler_angles.x)   # α (rotation around X-axis)
    beta = float(euler_angles.y)    # β (rotation around Y-axis)
    gamma = float(euler_angles.z)   # γ (rotation around Z-axis)
    
    return alpha, beta, gamma


def find_previous_city(mms_df: pd.DataFrame, start_line: int, pointing_dict: Dict[int, List[float]]) -> Optional[Tuple[int, np.ndarray]]:
    """Find the previous city before start_line that has coordinates in pointing_dict. Returns (line_num, coords) or None."""
    for line_num in range(start_line - 1, -1, -1):  # Go backwards from start_line
        if line_num in pointing_dict:
            coords = np.array(pointing_dict[line_num], dtype=float)
            return (line_num, coords)
    
    return None


def find_next_city(mms_df: pd.DataFrame, start_line: int, pointing_dict: Dict[int, List[float]]) -> Optional[Tuple[int, np.ndarray]]:
    """Find the next city after start_line that has coordinates in pointing_dict. Returns (line_num, coords) or None."""
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
    """Apply quaternion inflections for DURCHFAHREN2, 100% to dominant hand rotation and 100% to hand relocation."""
    mms_prime = mms_df.copy()

    # Ensure float dtypes for output columns (avoid pandas warnings)
    for col in ("domhandrotx", "domhandroty", "domhandrotz", 
                "domhandrelocax", "domhandrelocay", "domhandrelocaz"):
        if col not in mms_prime.columns:
            mms_prime[col] = 0.0
        mms_prime[col] = mms_prime[col].astype(float)

    durchfahren2_trajectory_file = os.path.join(
        corpus_dir, 'generated', 'signs', 'trimmed', 'DURCHFAHREN2.trajectoryInfo.json'
    )
    if not os.path.isfile(durchfahren2_trajectory_file):
        raise ValueError(f"DURCHFAHREN2 trajectory file not found: {durchfahren2_trajectory_file}")

    starting_point, initial_vector = load_trajectory_data(durchfahren2_trajectory_file)
    
    # Debug: Print DURCHFAHREN2 trajectory data
    print(f"\n{'='*60}")
    print(f"DURCHFAHREN2 TRAJECTORY DATA (from JSON):")
    print(f"{'='*60}")
    with open(durchfahren2_trajectory_file) as f:
        trajectory_data = json.load(f)
    traj = np.array(trajectory_data['dominant_hand']['trajectory'], dtype=float)
    print(f"  Trajectory file: {durchfahren2_trajectory_file}")
    print(f"  Total trajectory frames: {len(traj)}")
    print(f"  First frame (starting_point): [{traj[0][0]:.6f}, {traj[0][1]:.6f}, {traj[0][2]:.6f}]")
    if len(traj) > 1:
        print(f"  Second frame: [{traj[1][0]:.6f}, {traj[1][1]:.6f}, {traj[1][2]:.6f}]")
        print(f"  Initial vector (traj[1] - traj[0]): [{initial_vector[0]:.6f}, {initial_vector[1]:.6f}, {initial_vector[2]:.6f}]")
        print(f"  Initial vector magnitude: {np.linalg.norm(initial_vector):.6f}")
    print(f"{'='*60}\n")

    # Track the last city encountered (starts with Hamburg for first DURCHFAHREN2)
    last_city_coords = HAMBURG_COORDS.copy()
    last_city_line = None
    first_durchfahren2_found = False

    for line_num, row in mms_df.iterrows():
        maingloss = row['maingloss'] if 'maingloss' in row else None
        
        # If this line has coordinates in pointing_dict, update last_city
        if line_num in pointing_dict:
            last_city_coords = np.array(pointing_dict[line_num], dtype=float)
            last_city_line = line_num

        # If this is a DURCHFAHREN2 row, compute inflection
        if maingloss == 'DURCHFAHREN2':
            # For FIRST DURCHFAHREN2, always use HAMBURG as previous city (hardcoded)
            if not first_durchfahren2_found:
                previous_city_coords = HAMBURG_COORDS.copy()
                previous_city_line = None
                first_durchfahren2_found = True
                print(f"  Note: First DURCHFAHREN2 at line {line_num}, using HAMBURG (hardcoded) as previous city")
            else:
                # For subsequent DURCHFAHREN2, use the last city we encountered
                if last_city_line is not None:
                    previous_city_coords = last_city_coords.copy()
                    previous_city_line = last_city_line
                    print(f"  Note: Using last encountered city (line {last_city_line}) as previous city")
                else:
                    # Fallback: if no city found yet, use Hamburg
                    previous_city_coords = HAMBURG_COORDS.copy()
                    previous_city_line = None
                    print(f"  Note: No city found before DURCHFAHREN2 at line {line_num}, using HAMBURG (hardcoded)")
            
            # Find next city (city that comes AFTER this DURCHFAHREN2)
            next_city_info = find_next_city(mms_df, line_num, pointing_dict)
            
            if next_city_info is None:
                print(f"\nWarning: Line {line_num}: DURCHFAHREN2 found but no next city detected. Skipping.")
                continue
            
            next_line_num, next_city_coords = next_city_info
            
            # Compute target vector: from previous city to next city (world space)
            target_vector_world = next_city_coords - previous_city_coords
            
            # Convert both cities to bone-space
            world_prev_h = np.append(previous_city_coords, 1.0)
            local_prev_h = BONE_INV.dot(world_prev_h)
            previous_city_bone = local_prev_h[:3]
            
            world_next_h = np.append(next_city_coords, 1.0)
            local_next_h = BONE_INV.dot(world_next_h)
            next_city_bone = local_next_h[:3]
            
            # Target vector in bone-space
            target_vector_bone = next_city_bone - previous_city_bone
            
            # For the computation, we use:
            # - starting_point: from DURCHFAHREN2 trajectory (already in bone-space)
            # - initial_vector: from DURCHFAHREN2 trajectory
            # - target_point: starting_point + target_vector_bone (to get direction)
            target_point_bone = starting_point + target_vector_bone
            
            # Compute inflection angles
            alpha, beta, gamma = compute_inflection_angles_quaternion_method(
                starting_point,
                initial_vector,
                target_point_bone
            )
            
            # Print calculated values to console
            print(f"\n{'='*60}")
            print(f"Line {line_num}: DURCHFAHREN2 inflection calculated")
            print(f"{'='*60}")
            if previous_city_line is not None:
                print(f"  Previous city (before DURCHFAHREN2): line {previous_city_line} (world: [{previous_city_coords[0]:.3f}, {previous_city_coords[1]:.3f}, {previous_city_coords[2]:.3f}])")
            else:
                print(f"  Previous city (before DURCHFAHREN2): HAMBURG (hardcoded) (world: [{previous_city_coords[0]:.3f}, {previous_city_coords[1]:.3f}, {previous_city_coords[2]:.3f}])")
            print(f"  Next city (after DURCHFAHREN2): line {next_line_num} (world: [{next_city_coords[0]:.3f}, {next_city_coords[1]:.3f}, {next_city_coords[2]:.3f}])")
            print(f"  Target vector (world): [{target_vector_world[0]:.3f}, {target_vector_world[1]:.3f}, {target_vector_world[2]:.3f}]")
            print(f"  Target vector (bone-space): [{target_vector_bone[0]:.6f}, {target_vector_bone[1]:.6f}, {target_vector_bone[2]:.6f}]")
            print(f"  Starting point (trajectory, bone-space): [{starting_point[0]:.6f}, {starting_point[1]:.6f}, {starting_point[2]:.6f}]")
            print(f"  Initial vector (normalized): [{initial_vector[0]/np.linalg.norm(initial_vector):.6f}, "
                  f"{initial_vector[1]/np.linalg.norm(initial_vector):.6f}, "
                  f"{initial_vector[2]/np.linalg.norm(initial_vector):.6f}]")
            print(f"  Target vector (normalized): [{target_vector_bone[0]/np.linalg.norm(target_vector_bone):.6f}, "
                  f"{target_vector_bone[1]/np.linalg.norm(target_vector_bone):.6f}, "
                  f"{target_vector_bone[2]/np.linalg.norm(target_vector_bone):.6f}]")
            print(f"\n  QUATERNION METHOD RESULTS:")
            print(f"    Alpha (α):  {alpha:.6f} radians ({np.degrees(alpha):.2f}°)")
            print(f"    Beta (β):   {beta:.6f} radians ({np.degrees(beta):.2f}°)")
            print(f"    Gamma (γ):  {gamma:.6f} radians ({np.degrees(gamma):.2f}°)")
            
            # H100-R100 distribution: 100% to hand rotation, 100% to hand relocation
            hand_alpha = alpha
            hand_beta = beta
            hand_gamma = gamma            
            reloc_alpha = alpha
            reloc_beta = beta
            reloc_gamma = gamma            
            print(f"\n  DISTRIBUTION (H100-R100):")
            print(f"    Hand rotation (100%):")
            print(f"      domhandrotx = {hand_alpha:.6f}")
            print(f"      domhandroty = {hand_beta:.6f}")
            print(f"      domhandrotz = {hand_gamma:.6f}")
            print(f"    Hand relocation (100%):")
            print(f"      domhandrelocax = {reloc_alpha:.6f}")
            print(f"      domhandrelocay = {reloc_beta:.6f}")
            print(f"      domhandrelocaz = {reloc_gamma:.6f}")
            print(f"{'='*60}\n")
            
            # Apply 100% to dominant hand rotation
            mms_prime.at[line_num, 'domhandrotx'] = hand_alpha
            mms_prime.at[line_num, 'domhandroty'] = hand_beta
            mms_prime.at[line_num, 'domhandrotz'] = hand_gamma
            
            # Apply 100% to hand relocation
            mms_prime.at[line_num, 'domhandrelocax'] = reloc_alpha
            mms_prime.at[line_num, 'domhandrelocay'] = reloc_beta
            mms_prime.at[line_num, 'domhandrelocaz'] = reloc_gamma

    return mms_prime


def main() -> None:
    try:
        args = parse_arguments()
        validate_paths(args)

        pointing_dict = load_pointing_dictionary(args.pointing_dict)
        mms_df = pd.read_csv(args.input_mms)
        if mms_df.empty:
            raise ValueError("MMS file is empty")

        print(f"\n{'='*60}")
        print(f"DURCHFAHREN2 QUATERNION METHOD - MMS Prime Generation")
        print(f"{'='*60}")
        print(f"Input MMS: {args.input_mms}")
        print(f"Pointing Dict: {args.pointing_dict}")
        print(f"Output Dir: {args.output_dir}")
        print(f"Starting city: HAMBURG (hardcoded)")
        print(f"{'='*60}\n")

        mms_prime = apply_inflections(mms_df, pointing_dict, args.corpus_dir)

        output_file_path = os.path.join(
            args.output_dir,
            os.path.basename(args.input_mms).replace('_mms.csv', '_H100R100_mms_prime.csv')
        )
        mms_prime.to_csv(output_file_path, index=False)
        print(f"\n{'='*60}")
        print(f"SUCCESS: Saved inflected MMS Prime to:")
        print(f"  {output_file_path}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        if _RUNNING_IN_BLENDER and not _BLENDER_BG:
            return
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
