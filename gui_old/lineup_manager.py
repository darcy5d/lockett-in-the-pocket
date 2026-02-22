#!/usr/bin/env python3
import pandas as pd
import os
import glob
from pathlib import Path

class LineupManager:
    def __init__(self, lineups_dir="afl_data/data/lineups"):
        # Use absolute path to handle relative paths properly
        self.lineups_dir = Path(lineups_dir)
        if not os.path.isabs(lineups_dir):
            # Try current directory first
            if not self.lineups_dir.exists():
                # Then try from project root
                project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                self.lineups_dir = project_root / lineups_dir
                print(f"Looking for lineups in: {self.lineups_dir}")
        
        self.team_lineups = {}
        self.team_name_mapping = {
            # Modern team names to file name mapping
            'Adelaide': 'adelaide',
            'Brisbane Lions': 'brisbane_lions',
            'Carlton': 'carlton',
            'Collingwood': 'collingwood',
            'Essendon': 'essendon',
            'Fremantle': 'fremantle',
            'Geelong': 'geelong',
            'Gold Coast': 'gold_coast',
            'Greater Western Sydney': 'greater_western_sydney',
            'Hawthorn': 'hawthorn',
            'Melbourne': 'melbourne',
            'North Melbourne': 'north_melbourne',
            'Port Adelaide': 'port_adelaide',
            'Richmond': 'richmond',
            'St Kilda': 'st_kilda',
            'Sydney': 'sydney',
            'West Coast': 'west_coast',
            'Western Bulldogs': 'western_bulldogs',
            # Some teams might have historical names
            'Brisbane Bears': 'brisbane_bears',
            'Fitzroy': 'fitzroy',
            'Footscray': 'footscray',
            'Kangaroos': 'kangaroos',
            'South Melbourne': 'south_melbourne',
            'University': 'university'
        }
        
        # Always create the reverse mapping
        self.file_to_team_mapping = {}
        for team_name, filename in self.team_name_mapping.items():
            self.file_to_team_mapping[filename] = team_name
            
        self.initialize()
    
    def initialize(self):
        """Initialize the lineup manager by scanning available files."""
        # Verify the lineups directory exists
        if not self.lineups_dir.exists():
            print(f"Warning: Lineups directory {self.lineups_dir} does not exist.")
            return
        
        # Get all lineup files
        lineup_files = glob.glob(str(self.lineups_dir / "team_lineups_*.csv"))
        
        # Extract team names from filenames
        for file_path in lineup_files:
            filename = os.path.basename(file_path)
            team_key = filename.replace("team_lineups_", "").replace(".csv", "")
            
            # Store the file path for each team
            if team_key in self.file_to_team_mapping:
                team_name = self.file_to_team_mapping[team_key]
                self.team_lineups[team_name] = file_path
            else:
                # If no mapping exists, use the key as the team name
                self.team_lineups[team_key] = file_path
        
        print(f"Loaded {len(self.team_lineups)} team lineup files")
    
    def get_latest_lineup(self, team_name):
        """Get the latest lineup for the specified team."""
        if team_name not in self.team_lineups:
            # Try to find by alternate name using the mapping
            team_key = self.team_name_mapping.get(team_name)
            if not team_key or team_key not in self.file_to_team_mapping:
                print(f"No lineup data found for team: {team_name}")
                return [], []
            team_name = self.file_to_team_mapping[team_key]
        
        file_path = self.team_lineups[team_name]
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Get the latest entry
            latest_entry = df.iloc[-1]
            
            # Extract player names (semicolon-separated)
            latest_players = latest_entry["players"].split(";")
            
            # Get the year of the latest entry
            latest_year = latest_entry["year"]
            
            # Get all players who have played in the current season
            current_season_df = df[df["year"] == latest_year]
            
            # Collect all players who have played in the current season
            all_season_players = set()
            for player_list in current_season_df["players"]:
                all_season_players.update(player_list.split(";"))
            
            # Sort the player lists alphabetically
            latest_players = sorted(latest_players)
            all_season_players = sorted(list(all_season_players))
            
            # Return the latest lineup and all players from the current season
            return latest_players, all_season_players
        
        except Exception as e:
            print(f"Error loading lineup data for {team_name}: {e}")
            return [], []
    
    def get_team_names(self):
        """Get a list of all team names with lineup data."""
        return sorted(list(self.team_lineups.keys()))
    
    def get_team_year_round(self, team_name):
        """Get the year and round of the latest lineup for the specified team."""
        if team_name not in self.team_lineups:
            return None, None
        
        file_path = self.team_lineups[team_name]
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Get the latest entry
            latest_entry = df.iloc[-1]
            
            # Return year and round
            return latest_entry["year"], latest_entry["round_num"]
        
        except Exception as e:
            print(f"Error getting year/round for {team_name}: {e}")
            return None, None


# For testing
if __name__ == "__main__":
    lineup_manager = LineupManager()
    
    # Test with a team
    test_team = "Collingwood"
    latest_lineup, all_players = lineup_manager.get_latest_lineup(test_team)
    
    year, round_num = lineup_manager.get_team_year_round(test_team)
    
    print(f"\nLatest lineup for {test_team} (Year: {year}, Round: {round_num}):")
    for player in latest_lineup:
        print(f"  - {player}")
    
    print(f"\nAll players who have played for {test_team} in {year}:")
    for player in all_players:
        if player not in latest_lineup:
            print(f"  - {player}") 