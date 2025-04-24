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
            print("\nPlayer Data Directories:")
            # List player subdirectories
            players = [d for d in os.listdir(self.player_dir) 
                       if os.path.isdir(os.path.join(self.player_dir, d))]
            print(f"  - {len(players)} player directories available")
            
            # Show a few examples
            if players:
                print("  Examples:")
                for player in sorted(players)[:5]:
                    print(f"  - {player}")
        
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
    
    def load_player_data(self, player_name=None):
        """
        Load player performance data
        
        Args:
            player_name (str, optional): If provided, loads data for a specific player.
                                        Otherwise returns a list of available players.
        
        Returns:
            pandas.DataFrame or list: Player data or list of available players
        """
        if not os.path.exists(self.player_dir):
            print(f"Player directory {self.player_dir} does not exist. Please run fetch_afl_data.py first.")
            return None
        
        if player_name:
            # Find player directories that match the name (case insensitive)
            player_dirs = [d for d in os.listdir(self.player_dir) 
                          if os.path.isdir(os.path.join(self.player_dir, d)) and
                          player_name.lower() in d.lower()]
            
            if not player_dirs:
                print(f"No player data found for {player_name}")
                return None
            
            # If multiple matches, show options
            if len(player_dirs) > 1:
                print(f"Multiple players found matching '{player_name}':")
                for i, name in enumerate(player_dirs):
                    print(f"  {i+1}. {name}")
                print("\nPlease refine your search.")
                return player_dirs
            
            # Load the player's performance data
            player_dir = os.path.join(self.player_dir, player_dirs[0])
            stats_files = glob.glob(os.path.join(player_dir, "*_STATS.csv"))
            
            if not stats_files:
                print(f"No statistics file found for {player_dirs[0]}")
                return None
            
            try:
                player_df = pd.read_csv(stats_files[0])
                return player_df
            except Exception as e:
                print(f"Error loading player data: {e}")
                return None
        else:
            # Return list of available players
            return [d for d in os.listdir(self.player_dir) 
                    if os.path.isdir(os.path.join(self.player_dir, d))]
    
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
        
        if year:
            file_pattern = os.path.join(self.odds_dir, f"*{year}*.csv")
        else:
            file_pattern = os.path.join(self.odds_dir, "*.csv")
        
        files = glob.glob(file_pattern)
        if not files:
            print("No odds data files found" + (f" for year {year}" if year else ""))
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
    print("# Load player data for 'Bontempelli'")
    print("bontempelli_data = loader.load_player_data('Bontempelli')")
    print("")
    print("# Load odds data")
    print("odds_data = loader.load_odds_data()")

if __name__ == "__main__":
    main() 