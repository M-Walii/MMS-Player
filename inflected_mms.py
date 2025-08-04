import pandas as pd
import os
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inflect MMS data with predefined angles')
    parser.add_argument('--input-mms', required=True, help='Input MMS CSV filename (e.g., HandReloc-INDEX-X9.mms.csv)')
    parser.add_argument('--pointing-dict', help='Path to JSON file containing pointing dictionary (optional, will use POINTING_DICT env var if not specified)')
    parser.add_argument('--output-dir', help='Output directory (optional, will use MMS_OUTPUT_DIR env var if not specified)')
    
    return parser.parse_args()

def inflect_mms(mms_file_path, pointing_dict, output_dir=None):
    """
    Inflect MMS data with predefined angles from pointing dictionary.
    
    Args:
        mms_file_path (str): Path to input MMS CSV file
        pointing_dict (dict): Dictionary containing line numbers and target angles
        output_dir (str, optional): Directory to save output file
    """
    if not os.path.isfile(mms_file_path):
        raise FileNotFoundError(f"The file '{mms_file_path}' does not exist.")

    # Load MMS data
    mms_df = pd.read_csv(mms_file_path)
    
    # Create a copy for inflection
    inflected_mms = mms_df.copy()

    # Iterate through the pointing dictionary to apply inflections
    for line_number, target_angles in pointing_dict.items():
        if 0 <= line_number < len(inflected_mms):
            inflected_mms.at[line_number, 'domhandrotx'] = target_angles[0]
            inflected_mms.at[line_number, 'domhandroty'] = target_angles[1]
            inflected_mms.at[line_number, 'domhandrotz'] = target_angles[2]
    
    # Determine output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, os.path.basename(mms_file_path))
    else:
        output_file_path = mms_file_path

    # Save the inflected MMS
    inflected_mms.to_csv(output_file_path, index=False)
    
    print(f"Inflected MMS saved to: {output_file_path}")
    return inflected_mms

def main():
    args = parse_arguments()
    
    # Get paths from environment variables
    mms_input_dir = os.getenv('MMS_INPUT_DIR')
    output_dir = args.output_dir or os.getenv('MMS_OUTPUT_DIR')
    pointing_dict_path = args.pointing_dict or os.getenv('POINTING_DICT')
    
    if not all([mms_input_dir, output_dir, pointing_dict_path]):
        raise EnvironmentError("Please set the following environment variables:\n"
                            "MMS_INPUT_DIR - Directory containing MMS CSV files\n"
                            "MMS_OUTPUT_DIR - Directory for output files\n"
                            "POINTING_DICT - Path to pointing dictionary JSON file")
    
    # Construct file paths
    mms_file_path = os.path.join(mms_input_dir, args.input_mms)
    
    # Load pointing dictionary and convert string keys to integers
    with open(pointing_dict_path) as f:
        raw_dict = json.load(f)
        pointing_dict = {int(k): v for k, v in raw_dict.items()}
    
    # Inflect the MMS based on the pointing dictionary
    try:
        inflected_mms = inflect_mms(mms_file_path, pointing_dict, output_dir)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
