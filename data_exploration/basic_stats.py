#!/usr/bin/env python3
"""
Basic statistics script for AFL data
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root directory to the path so we can import from datafetch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datafetch.load_afl_data import AFLDataLoader

def analyze_matches(year=None):
    """
    Analyze match data for a specific year or all years
    """
    loader = AFLDataLoader()
    
    # Load match data
    if year:
        print(f"Loading match data for {year}...")
        matches = loader.load_matches(year)
    else:
        print("Loading all match data...")
        matches = loader.load_matches()
    
    if matches is None or matches.empty:
        print("No match data found.")
        return
    
    print(f"\nAnalyzing {len(matches)} matches...")
    
    # Basic statistics
    print("\nBasic Match Statistics:")
    print(f"Total matches: {len(matches)}")
    
    # Convert team names to strings if they're not already
    if 'team_1' in matches.columns:
        matches['team_1'] = matches['team_1'].astype(str)
        matches['team_2'] = matches['team_2'].astype(str)
    
    # Calculate score statistics if available
    if 'team_1_final_score' in matches.columns and 'team_2_final_score' in matches.columns:
        total_scores = pd.concat([matches['team_1_final_score'], matches['team_2_final_score']])
        
        print(f"Average score: {total_scores.mean():.2f}")
        print(f"Median score: {total_scores.median():.2f}")
        print(f"Highest score: {total_scores.max()}")
        print(f"Lowest score: {total_scores.min()}")
    
    return matches

def analyze_player(player_name):
    """
    Analyze data for a specific player
    """
    loader = AFLDataLoader()
    
    # Load player performance data
    print(f"Loading performance data for {player_name}...")
    performance = loader.load_player_data(player_name, data_type='performance')
    
    if performance is None or isinstance(performance, list):
        if isinstance(performance, list):
            print(f"Multiple players found matching '{player_name}'. Please be more specific.")
            for player in performance:
                print(f"  - {player}")
        return
    
    # Load player personal data
    personal = loader.load_player_data(player_name, data_type='personal')
    
    print(f"\nAnalyzing data for {personal['first_name'].iloc[0]} {personal['last_name'].iloc[0]}...")
    
    print("\nPlayer Info:")
    print(f"Name: {personal['first_name'].iloc[0]} {personal['last_name'].iloc[0]}")
    print(f"Born: {personal['born_date'].iloc[0]}")
    print(f"Debut: {personal['debut_date'].iloc[0]}")
    print(f"Height: {personal['height'].iloc[0]} cm")
    print(f"Weight: {personal['weight'].iloc[0]} kg")
    
    print("\nCareer Statistics:")
    print(f"Games played: {len(performance)}")
    print(f"Teams: {', '.join(performance['team'].unique())}")
    
    # Calculate average statistics if available
    numerical_columns = performance.select_dtypes(include=[np.number]).columns
    if not numerical_columns.empty:
        print("\nAverage statistics per game:")
        for col in numerical_columns:
            if col not in ['year', 'games_played', 'percentage_of_game_played']:
                avg_val = performance[col].mean()
                if not pd.isna(avg_val):
                    print(f"Average {col.replace('_', ' ')}: {avg_val:.2f}")
    
    return performance, personal

def main():
    """
    Main function to demonstrate basic stats analysis
    """
    # Analyze match data for 2023
    matches_2023 = analyze_matches(2023)
    
    # Analyze data for a specific player (Bontempelli)
    player_data = analyze_player('bontempelli')

if __name__ == "__main__":
    main() 