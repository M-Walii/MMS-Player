import json
import numpy as np
from typing import List, Dict, Any
import argparse
import os
import pandas as pd

class EntityLocator:
    def __init__(self):
        """Initialize EntityLocator."""
        pass
        
    def create_pointing_dict_from_mms(
        self, 
        mms_file_path: str, 
        location_dict: Dict[str, List[float]]
    ) -> Dict[int, List[float]]:
        """Create pointing dictionary from MMS data and location dictionary.
        
        Args:
            mms_file_path: Path to MMS CSV file
            location_dict: Dictionary mapping city names to their coordinates
            
        Returns:
            Dictionary mapping line numbers (0-based) to city coordinates
        """
        pointing_dict = {}
        
        # Read MMS file using pandas
        mms_df = pd.read_csv(mms_file_path)
        
        # Process each row
        for line_num, row in mms_df.iterrows():
            city_name = row['maingloss']
            
            # If city name exists in location dictionary, copy its coordinates
            if city_name in location_dict:
                pointing_dict[line_num] = location_dict[city_name]
                print(f"Found city {city_name} at line {line_num}")
                
        return pointing_dict

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create pointing dictionary from MMS data')
    parser.add_argument('--input-mms', required=True, 
                       help='Input MMS CSV file path')
    parser.add_argument('--location-dict', required=True,
                       help='Location dictionary JSON file path')
    parser.add_argument('--output-dict', required=True,
                       help='Output pointing dictionary JSON file path')
    return parser.parse_args()

def main():
    """Main function to execute the pointing dictionary creation."""
    # Parse arguments
    args = parse_arguments()
    
    # Load location dictionary
    with open(args.location_dict, 'r') as f:
        location_dict = json.load(f)
    
    # Initialize EntityLocator and create pointing dictionary
    locator = EntityLocator()
    pointing_dict = locator.create_pointing_dict_from_mms(args.input_mms, location_dict)
    
    # Save pointing dictionary
    with open(args.output_dict, 'w') as f:
        json.dump(pointing_dict, f, indent=4)
    
    print(f"Pointing dictionary created successfully and saved to: {args.output_dict}")
    print(f"Found {len(pointing_dict)} cities in MMS file")

if __name__ == "__main__":
    main() 