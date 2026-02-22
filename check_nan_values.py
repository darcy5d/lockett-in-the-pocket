import pandas as pd
import numpy as np
import os

# Define file paths for data (same as in neural_network.py)
match_files = [
    'afl_data/data/matches/matches_2017.csv',
    'afl_data/data/matches/matches_2018.csv',
    'afl_data/data/matches/matches_2019.csv',
    'afl_data/data/matches/matches_2020.csv',
    'afl_data/data/matches/matches_2021.csv',
    'afl_data/data/matches/matches_2022.csv',
    'afl_data/data/matches/matches_2023.csv'
]

lineup_files = [
    'afl_data/data/lineups/team_lineups_sydney.csv',
    'afl_data/data/lineups/team_lineups_university.csv',
    'afl_data/data/lineups/team_lineups_west_coast.csv'
]

player_files = [
    'afl_data/data/players/zunneberg_noel_13081946_performance_details.csv',
    'afl_data/data/players/zunneberg_noel_13081946_personal_details.csv',
    'afl_data/data/players/zurhaar_cameron_22051998_performance_details.csv'
]

# Check if files exist
def check_files_exist(file_list):
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"WARNING: File not found: {file_path}")
        else:
            print(f"Found: {file_path}")

# Load match data
def load_data():
    print("\n1. Checking match files...")
    check_files_exist(match_files)
    match_data = pd.concat([pd.read_csv(f) for f in match_files if os.path.exists(f)])
    
    print("\n2. Checking lineup files...")
    check_files_exist(lineup_files)
    lineup_data = pd.concat([pd.read_csv(f) for f in lineup_files if os.path.exists(f)])
    
    print("\n3. Checking player files...")  
    check_files_exist(player_files)
    player_data = pd.concat([pd.read_csv(f) for f in player_files if os.path.exists(f)])
    
    # Add match_type column
    match_data['match_type'] = 'Regular'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Qualifying Final', na=False), 'match_type'] = 'Qualifying Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Elimination Final', na=False), 'match_type'] = 'Elimination Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Semi Final', na=False), 'match_type'] = 'Semi Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Preliminary Final', na=False), 'match_type'] = 'Preliminary Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Grand Final', na=False), 'match_type'] = 'Grand Final'
    
    # Create dummy variables for match types
    match_type_dummies = pd.get_dummies(match_data['match_type'], prefix='match_type')
    match_data = pd.concat([match_data, match_type_dummies], axis=1)
    
    # Create target columns from existing data
    # Convert goal and behind columns to numeric
    for col in match_data.columns:
        if '_goals' in col or '_behinds' in col:
            match_data[col] = pd.to_numeric(match_data[col], errors='coerce')
    
    # Calculate team1 and team2 scores
    match_data['team1_score'] = match_data['team_1_final_goals'] * 6 + match_data['team_1_final_behinds']
    match_data['team2_score'] = match_data['team_2_final_goals'] * 6 + match_data['team_2_final_behinds']
    
    # Create match_winner column (1 if team1 wins, 0 if team2 wins, 0.5 if draw)
    match_data['match_winner'] = (match_data['team1_score'] > match_data['team2_score']).astype(float)
    match_data.loc[match_data['team1_score'] == match_data['team2_score'], 'match_winner'] = 0.5
    
    # Create margin column (team1_score - team2_score)
    match_data['margin'] = match_data['team1_score'] - match_data['team2_score']
    
    # Rename columns for clarity
    match_data['team1_goals'] = match_data['team_1_final_goals']
    match_data['team1_behinds'] = match_data['team_1_final_behinds']
    match_data['team2_goals'] = match_data['team_2_final_goals']
    match_data['team2_behinds'] = match_data['team_2_final_behinds']
    
    # Merge data - prepare for merges
    match_data['round_num'] = match_data['round_num'].astype(str)
    lineup_data['round_num'] = lineup_data['round_num'].astype(str)
    
    # Merge match data with lineup data
    merged_data = pd.merge(match_data, lineup_data, 
                           left_on=['year', 'team_1_team_name', 'round_num'], 
                           right_on=['year', 'team_name', 'round_num'], 
                           how='left')
    
    # Prepare for merge with player data
    merged_data['round_num'] = merged_data['round_num'].astype(str)
    player_data['round'] = player_data['round'].astype(str)
    
    # Merge with player data
    merged_data = pd.merge(merged_data, player_data, 
                           left_on=['year', 'team_1_team_name', 'round_num'], 
                           right_on=['year', 'team', 'round'], 
                           how='left')
    
    return merged_data

def analyze_data(merged_data):
    # Get info about dataframe
    print("\n===== DATAFRAME INFO =====")
    print(f"Shape: {merged_data.shape}")
    print(f"Column types:\n{merged_data.dtypes.value_counts()}")
    
    # Count NaN values in each column
    nan_counts = merged_data.isna().sum()
    nan_percent = nan_counts / len(merged_data) * 100
    
    # Print columns with NaN values
    print("\n===== COLUMNS WITH NaN VALUES =====")
    nan_info = pd.DataFrame({
        'Count': nan_counts,
        'Percent': nan_percent
    })
    nan_info = nan_info.sort_values('Count', ascending=False)
    print(nan_info[nan_info['Count'] > 0])
    
    # Check for infinite values
    inf_counts = np.isinf(merged_data.select_dtypes(include=['number'])).sum()
    print("\n===== COLUMNS WITH INFINITE VALUES =====")
    inf_info = pd.DataFrame({
        'Count': inf_counts
    })
    print(inf_info[inf_info['Count'] > 0])
    
    # Statistics of numeric columns
    print("\n===== NUMERIC COLUMNS STATISTICS =====")
    numeric_cols = merged_data.select_dtypes(include=['number']).columns
    print(merged_data[numeric_cols].describe().T[['min', 'max', 'mean', 'std']])
    
    # Check target columns
    print("\n===== TARGET COLUMNS =====")
    target_cols = ['match_winner', 'margin', 'team1_goals', 'team1_behinds', 'team2_goals', 'team2_behinds']
    print(merged_data[target_cols].describe())
    
    # Check for NaN values in the target columns
    print("\nNaN values in target columns:")
    print(merged_data[target_cols].isna().sum())
    
    return merged_data

def main():
    print("===== CHECKING DATA FOR NaN VALUES =====")
    
    # Load and prepare data
    merged_data = load_data()
    
    # Analyze data for NaN values
    merged_data = analyze_data(merged_data)
    
    print("\n===== COMPLETED DATA ANALYSIS =====")

if __name__ == "__main__":
    main() 