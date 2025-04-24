#!/usr/bin/env python3
"""
Script to update all AFL data at once
"""

import os
import sys
import argparse
from fetch_afl_data import main as fetch_data
from load_afl_data import AFLDataLoader

def update_data(verbose=True):
    """
    Update all AFL data, fetching the latest from the source repository
    """
    if verbose:
        print("Updating AFL data...")
    
    # Run the data fetching script
    fetch_data()
    
    if verbose:
        print("\nVerifying data integrity...")
    
    # Verify that the data was downloaded correctly
    loader = AFLDataLoader()
    
    # Check that all directories exist
    all_dirs_exist = (
        os.path.exists(loader.data_dir) and
        os.path.exists(loader.match_dir) and
        os.path.exists(loader.player_dir) and
        os.path.exists(loader.odds_dir)
    )
    
    if not all_dirs_exist:
        print("ERROR: Not all data directories were created. Please check the logs above.")
        return False
    
    # List available data
    if verbose:
        loader.list_available_data()
        print("\nData update completed successfully!")
    
    return True

def main():
    """
    Main function for data updating
    """
    parser = argparse.ArgumentParser(description='Update AFL data')
    parser.add_argument('--quiet', action='store_true', help='Run quietly with minimal output')
    args = parser.parse_args()
    
    update_data(verbose=not args.quiet)

if __name__ == "__main__":
    main() 