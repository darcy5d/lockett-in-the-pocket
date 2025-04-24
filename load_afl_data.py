#!/usr/bin/env python3
"""
Script to load and explore AFL data from the downloaded akareen/AFL-Data-Analysis repository
"""

import os
import pandas as pd
import glob
from pathlib import Path

class AFLDataLoader:
    """
    Class to load and process AFL data from the downloaded repository
    """
    def __init__(self, data_dir="afl_data"):
        """
        Initialize the AFLDataLoader with the data directory
        """
        self.data_dir = data_dir
        self.match_dir = os.path.join(data_dir, "data", "matches")
        self.player_dir = os.path.join(data_dir, "data", "players")
        self.odds_dir = os.path.join(data_dir, "odds_data")
        
        # Verify the data directories exist
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist. Please run fetch_afl_data.py first.")
    
    def list_available_data(self):
        """
        List the available data files in the data directory
        """
        print("Available data files:")
        
        # Check if directories exist before listing files
        if os.path.exists(self.match_dir):
            print("\nMatch Data Files:")
            for file in sorted(glob.glob(os.path.join(self.match_dir, "*.csv"))):
                print(f"  - {os.path.basename(file)}")
        
        if os.path.exists(self.player_dir):
            print("\nPlayer Data Files:")
            # Count player files
            performance_files = glob.glob(os.path.join(self.player_dir, "*performance_details.csv"))
            personal_files = glob.glob(os.path.join(self.player_dir, "*personal_details.csv"))
            print(f"  - {len(performance_files)} player performance files")
            print(f"  - {len(personal_files)} player personal files")
            
            # Show a few examples
            if performance_files:
                print("  Examples (performance files):")
                for file in sorted(performance_files)[:5]:
                    print(f"  - {os.path.basename(file)}")
        
        if os.path.exists(self.odds_dir):
            print("\nOdds Data Files:")
            for file in sorted(glob.glob(os.path.join(self.odds_dir, "*.csv"))):
                print(f"  - {os.path.basename(file)}")
    
    def load_matches(self, year=None):
        """
        Load match data from CSV files
        
        Args:
            year (int, optional): Specific year to load. If None, loads all years.
            
        Returns:
            pandas.DataFrame: DataFrame containing match data
        """
        if not os.path.exists(self.match_dir):
            print(f"Match directory {self.match_dir} does not exist. Please run fetch_afl_data.py first.")
            return None
        
        if year:
            file_pattern = os.path.join(self.match_dir, f"matches_{year}.csv")
            files = glob.glob(file_pattern)
            if not files:
                print(f"No match data found for year {year}")
                return None
        else:
            files = glob.glob(os.path.join(self.match_dir, "matches_*.csv"))
            if not files:
                print("No match data files found")
                return None
        
        # Load and concatenate all match data files
        dfs = []
        for file in sorted(files):
            try:
                df = pd.read_csv(file)
                year_from_file = os.path.basename(file).replace("matches_", "").replace(".csv", "")
                df['year_from_file'] = year_from_file
                dfs.append(df)
                print(f"Loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not dfs:
            return None
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_player_data(self, player_name=None, data_type="performance"):
        """
        Load player data (performance or personal)
        
        Args:
            player_name (str, optional): If provided, loads data for a specific player.
                                        Otherwise returns a list of available players.
            data_type (str): Type of data to load - "performance" or "personal"
            
        Returns:
            pandas.DataFrame or list: Player data or list of available players
        """
        if not os.path.exists(self.player_dir):
            print(f"Player directory {self.player_dir} does not exist. Please run fetch_afl_data.py first.")
            return None
        
        # Validate data_type
        if data_type not in ["performance", "personal"]:
            print(f"Invalid data_type: {data_type}. Must be 'performance' or 'personal'.")
            return None
        
        file_suffix = f"{data_type}_details.csv"
        
        if player_name:
            # Find player files that match the name (case insensitive)
            player_files = glob.glob(os.path.join(self.player_dir, f"*{player_name.lower()}*_{file_suffix}"), 
                                    recursive=True)
            
            if not player_files:
                print(f"No player {data_type} data found for {player_name}")
                return None
            
            # If multiple matches, show options
            if len(player_files) > 1:
                print(f"Multiple players found matching '{player_name}':")
                for i, file_path in enumerate(player_files):
                    print(f"  {i+1}. {os.path.basename(file_path)}")
                print("\nPlease refine your search.")
                return [os.path.basename(file).replace(f"_{file_suffix}", "") for file in player_files]
            
            # Load the player's data
            try:
                player_df = pd.read_csv(player_files[0])
                print(f"Loaded {os.path.basename(player_files[0])}")
                return player_df
            except Exception as e:
                print(f"Error loading player data: {e}")
                return None
        else:
            # Return list of available players
            all_files = glob.glob(os.path.join(self.player_dir, f"*{file_suffix}"))
            return [os.path.basename(file).replace(f"_{file_suffix}", "") for file in all_files]
    
    def load_odds_data(self, year=None):
        """
        Load odds data from CSV files
        
        Args:
            year (int, optional): Specific year to load. If None, loads all years.
            
        Returns:
            pandas.DataFrame: DataFrame containing odds data
        """
        if not os.path.exists(self.odds_dir):
            print(f"Odds directory {self.odds_dir} does not exist. Please run fetch_afl_data.py first.")
            return None
        
        # Check for excel files
        excel_files = glob.glob(os.path.join(self.odds_dir, "*.xlsx"))
        csv_files = glob.glob(os.path.join(self.odds_dir, "*.csv"))
        
        if year and csv_files:
            file_pattern = os.path.join(self.odds_dir, f"*{year}*.csv")
            files = glob.glob(file_pattern)
            if not files:
                print(f"No odds data found for year {year}")
                return None
        elif csv_files:
            files = csv_files
        elif excel_files:
            print("Found Excel odds data files. Loading...")
            dfs = []
            for file in sorted(excel_files):
                try:
                    df = pd.read_excel(file)
                    dfs.append(df)
                    print(f"Loaded {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            if not dfs:
                return None
            
            return pd.concat(dfs, ignore_index=True)
        else:
            print("No odds data files found")
            return None
        
        # Load and concatenate all odds data files
        dfs = []
        for file in sorted(files):
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"Loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not dfs:
            return None
        
        return pd.concat(dfs, ignore_index=True)

def main():
    """
    Main function to demonstrate usage of the AFLDataLoader class
    """
    loader = AFLDataLoader()
    
    # Check if data has been downloaded
    if not os.path.exists(loader.data_dir):
        print(f"Data directory '{loader.data_dir}' not found. Please run fetch_afl_data.py first.")
        return
    
    # List available data
    loader.list_available_data()
    
    print("\nExample Usage:")
    print("from load_afl_data import AFLDataLoader")
    print("loader = AFLDataLoader()")
    print("")
    print("# Load match data for 2023")
    print("matches_2023 = loader.load_matches(2023)")
    print("")
    print("# Load player performance data for 'Bontempelli'")
    print("bontempelli_performance = loader.load_player_data('bontempelli', data_type='performance')")
    print("")
    print("# Load player personal data for 'Bontempelli'")
    print("bontempelli_personal = loader.load_player_data('bontempelli', data_type='personal')")
    print("")
    print("# Load odds data")
    print("odds_data = loader.load_odds_data()")

if __name__ == "__main__":
    main() 