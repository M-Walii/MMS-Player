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
            
        # Get starting point from first trajectory point
        starting_point = np.array(trajectory_data['dominant_hand']['trajectory'][0])
        
        # Get initial direction from components
        initial_vector = np.array(trajectory_data['dominant_hand']['components']['PC1'])
        
        return starting_point, initial_vector
        
    except Exception as e:
        raise ValueError(f"Error loading trajectory data: {str(e)}")


def compute_inflection_angles(
    starting_point: np.ndarray,       # Origin/source point
    initial_vector: np.ndarray,       # Starting direction
    target_point: np.ndarray         # Target/destination point
) -> Tuple[float, float]:
    """
    Calculate ALPHA and BETA angles to align initial_vector to a target_point vector.
    
    Args:
        starting_point: Starting point coordinates (source)
        initial_vector: Initial direction vector (source)
        target_point: Target point coordinates (destination)
    
    Returns:
        Tuple[float, float]: Computed ALPHA and BETA angles in radians
    
    Raises:
        ValueError: If vectors are invalid or cannot be normalized
    """
    try:
        # Calculate target vector
        target_vector = target_point - starting_point
        norm = np.linalg.norm(target_vector)
        if norm == 0:
            raise ValueError("Target vector has zero length")
        target_vector /= norm
        
        # Calculate rotation quaternion
        rotation_quaternion = R.from_rotvec(
            np.cross(initial_vector, target_vector)
        ).as_quat()
        
        # Convert to Euler angles (in radians)
        rotation = R.from_quat(rotation_quaternion)
        alpha, beta, _ = rotation.as_euler('xyz', degrees=False)
        
        return alpha, beta
        
    except Exception as e:
        raise ValueError(f"Error computing inflection angles: {str(e)}")


def apply_inflections(
    mms_df: pd.DataFrame,            # Source MMS data
    pointing_dict: Dict[int, List[float]],  # Target points dictionary
    corpus_dir: str                  # Corpus directory path
) -> pd.DataFrame:
    """
    Apply inflection angles to the MMS data to generate MMS Prime.
    Uses INDEX row's trajectory data for all cities, and pointing_dict for target points.
    Angles are in radians and placed in the next INDEX row.
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
    
    # Now process each city row
    for line_num, row in mms_df.iterrows():
        city_name = row['maingloss']
        print(f"\nProcessing row {line_num}: {city_name}")
        
        if city_name != 'INDEX' and line_num in pointing_dict:
            try:
                # Get target point from pointing_dict
                target_point = np.array(pointing_dict[line_num])
                print(f"  Target point: {target_point}")
                
                # Calculate angles using INDEX trajectory data
                alpha, beta = compute_inflection_angles(
                    starting_point,    # INDEX's trajectory starting point
                    initial_vector,    # INDEX's trajectory initial direction
                    target_point       # City's coordinates
                )
                print(f"  Calculated angles: alpha={alpha}, beta={beta}")
                
                # Update the next INDEX row
                index_row = line_num + 1
                if index_row < len(mms_prime) and mms_prime.at[index_row, 'maingloss'] == 'INDEX':
                    mms_prime.at[index_row, 'domhandrotx'] = float(alpha)
                    mms_prime.at[index_row, 'domhandroty'] = float(beta)
                    mms_prime.at[index_row, 'domhandrotz'] = 0.0
                    print(f"  Updated angles in INDEX row {index_row}")
                else:
                    print(f"  Could not update INDEX row {index_row}")
                    
            except Exception as e:
                print(f"Error processing {city_name}: {str(e)}")
        else:
            print(f"  Skipping row - not a city or no coordinates in pointing dict")
    
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
