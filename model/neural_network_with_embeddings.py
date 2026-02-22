import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Embedding, Lambda
from tensorflow.keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import glob
from tensorflow.keras.metrics import Precision, Recall
import csv
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical
import joblib  # Added for saving the scaler
import json    # Added for saving player_index and feature_cols
from bounded_output_layer import BoundedOutputLayer  # Import the custom layer
import sys

# Set the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root directory: {PROJECT_ROOT}")

# Add the model directory to the path to ensure BoundedOutputLayer is available
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))

# Register the BoundedOutputLayer to make it available for model loading
tf.keras.utils.get_custom_objects()['BoundedOutputLayer'] = BoundedOutputLayer

# Custom pooling layer with explicit output shape
class MeanPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, **kwargs):
        self.output_dim = output_dim
        super(MeanPoolingLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(MeanPoolingLayer, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config

# Register the custom layer
tf.keras.utils.get_custom_objects()['MeanPoolingLayer'] = MeanPoolingLayer

# Extend match_files to include data from 1897 to 2025
match_files = [
    os.path.join(PROJECT_ROOT, f'afl_data/data/matches/matches_{year}.csv') for year in range(1897, 2026)
]

# Get all lineup files dynamically
lineup_files = glob.glob(os.path.join(PROJECT_ROOT, 'afl_data/data/lineups/team_lineups_*.csv'))
print(f"Found {len(lineup_files)} lineup files")

# Player embedding parameters
NUM_PLAYERS = 2000  # Estimated maximum number of unique players
EMBEDDING_DIM = 32  # Size of embedding vector for each player
MAX_PLAYERS_PER_TEAM = 22  # Maximum number of players per team

# Function to process player data and create player index lookup
def create_player_index():
    print("\nCreating player index for embeddings...")
    
    # Get all player files
    player_performance_files = glob.glob(os.path.join(PROJECT_ROOT, 'afl_data/data/players/*_performance_details.csv'))
    
    # Create a set of all unique player IDs
    player_ids = set()
    for file in player_performance_files:
        # Extract player ID from filename 
        player_id = os.path.basename(file).split('_performance')[0]
        player_ids.add(player_id)
    
    # Create player index lookup
    player_index = {player_id: i+1 for i, player_id in enumerate(sorted(player_ids))}
    
    # Add 0 for unknown/missing players
    player_index['unknown'] = 0
    
    print(f"Created player index with {len(player_index)} players")
    return player_index

# Function to extract player performance statistics
def extract_player_stats():
    """
    Extract player performance statistics from player detail files, including all key performance indicators.
    
    Returns:
        dict: A dictionary mapping player IDs to their performance stats
    """
    print("\nExtracting player performance statistics...")
    
    # Get all player performance files
    player_performance_files = glob.glob(os.path.join(PROJECT_ROOT, 'afl_data/data/players/*_performance_details.csv'))
    
    # Dictionary to store player stats
    player_stats = {}
    
    # All metrics we care about
    metrics = [
        'kicks', 'marks', 'handballs', 'goals', 'behinds', 
        'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances',
        'clangers', 'free_kicks_for', 'free_kicks_against', 
        'contested_possessions', 'uncontested_possessions', 'contested_marks',
        'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist',
        'percentage_of_game_played'
    ]
    
    for file in player_performance_files:
        try:
            # Extract player ID from filename 
            player_id = os.path.basename(file).split('_performance')[0]
            
            # Read the performance data
            df = pd.read_csv(file)
            
            # Only process if it has enough required columns
            # We need at least the basic columns to be useful
            essential_columns = ['goals', 'behinds', 'kicks', 'marks', 'handballs']
            if not all(col in df.columns for col in essential_columns):
                continue
                
            # Get the recent games (last 5 years)
            recent_years = [2020, 2021, 2022, 2023, 2024, 2025]
            recent_games = df[df['year'].isin(recent_years)]
            
            # If no recent games, use all available data
            if len(recent_games) == 0:
                recent_games = df
            
            # Calculate averages if there's data
            if len(recent_games) > 0:
                # Create dictionary to store stats
                player_stat_dict = {'games_played': len(recent_games)}
                
                # Process each metric
                for metric in metrics:
                    if metric in recent_games.columns:
                        # Calculate average, handling NaN values
                        avg_value = recent_games[metric].dropna().mean() 
                        player_stat_dict[f'avg_{metric}'] = avg_value if not np.isnan(avg_value) else 0
                    else:
                        # Set to 0 if column doesn't exist
                        player_stat_dict[f'avg_{metric}'] = 0
                
                # Store the statistics
                player_stats[player_id] = player_stat_dict
                
        except Exception as e:
            # Just skip any problem files
            print(f"Error processing {file}: {e}")
            continue
    
    print(f"Extracted performance stats for {len(player_stats)} players")
    return player_stats

# Function to enhance player embeddings with performance data
def enhance_player_embeddings(player_index, team1_player_indices, team2_player_indices):
    """
    Enhance player embeddings with comprehensive performance data.
    
    Args:
        player_index (dict): Dictionary mapping player IDs to indices
        team1_player_indices (ndarray): Player indices for team 1
        team2_player_indices (ndarray): Player indices for team 2
        
    Returns:
        tuple: Enhanced player features for both teams
    """
    print("\nEnhancing player embeddings with performance data...")
    
    # Extract player statistics
    player_stats = extract_player_stats()
    
    # Create reverse mapping from indices to player IDs
    idx_to_player = {v: k for k, v in player_index.items()}
    
    # List of all stat fields we're using
    stat_fields = [
        'kicks', 'marks', 'handballs', 'goals', 'behinds', 
        'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances',
        'clangers', 'free_kicks_for', 'free_kicks_against', 
        'contested_possessions', 'uncontested_possessions', 'contested_marks',
        'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist',
        'percentage_of_game_played'
    ]
    
    # Number of features is the number of stat fields
    num_features = len(stat_fields)
    
    # Initialize enhanced features
    n_matches = team1_player_indices.shape[0]
    team1_enhanced = np.zeros((n_matches, num_features))
    team2_enhanced = np.zeros((n_matches, num_features))
    
    # Process each match
    for i in range(n_matches):
        # Process team 1
        team1_players = team1_player_indices[i]
        team1_stats = [player_stats.get(idx_to_player.get(idx, 'unknown'), {}) 
                      for idx in team1_players if idx > 0]  # Skip padding (0 indices)
        
        # Process team 2
        team2_players = team2_player_indices[i]
        team2_stats = [player_stats.get(idx_to_player.get(idx, 'unknown'), {}) 
                      for idx in team2_players if idx > 0]  # Skip padding (0 indices)
        
        # Calculate team averages - weighted by games played
        if team1_stats:
            # Get games played for weighting (minimum 1 game to avoid division by zero)
            team1_games = np.array([s.get('games_played', 1) for s in team1_stats])
            team1_games = np.where(team1_games > 0, team1_games, 1)
            team1_weights = team1_games / np.sum(team1_games) if np.sum(team1_games) > 0 else np.ones_like(team1_games) / len(team1_games)
            
            # Calculate weighted averages for each feature
            for j, field in enumerate(stat_fields):
                avg_field = f'avg_{field}'
                team1_enhanced[i, j] = np.sum([s.get(avg_field, 0) * w for s, w in zip(team1_stats, team1_weights)])
        
        if team2_stats:
            # Get games played for weighting (minimum 1 game to avoid division by zero)
            team2_games = np.array([s.get('games_played', 1) for s in team2_stats])
            team2_games = np.where(team2_games > 0, team2_games, 1)
            team2_weights = team2_games / np.sum(team2_games) if np.sum(team2_games) > 0 else np.ones_like(team2_games) / len(team2_games)
            
            # Calculate weighted averages for each feature
            for j, field in enumerate(stat_fields):
                avg_field = f'avg_{field}'
                team2_enhanced[i, j] = np.sum([s.get(avg_field, 0) * w for s, w in zip(team2_stats, team2_weights)])
    
    print(f"Created enhanced features for {n_matches} matches with {num_features} features per team")
    return team1_enhanced, team2_enhanced

# Function to load and prepare data with player embeddings
def load_and_prepare_data():
    print("\n----- LOADING AND PREPARING DATA WITH PLAYER EMBEDDINGS -----")
    
    # Create player index
    player_index = create_player_index()
    
    # Load match data
    print("\n1. Loading match data...")
    match_dfs = []
    for f in match_files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                match_dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
    
    match_data = pd.concat(match_dfs)
    print(f"Match data shape: {match_data.shape}")
    
    # Add match_type column
    print("\n2. Adding match_type column...")
    match_data['match_type'] = 'Regular'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Qualifying Final', na=False), 'match_type'] = 'Qualifying Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Elimination Final', na=False), 'match_type'] = 'Elimination Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Semi Final', na=False), 'match_type'] = 'Semi Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Preliminary Final', na=False), 'match_type'] = 'Preliminary Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Grand Final', na=False), 'match_type'] = 'Grand Final'
    
    # Create dummy variables for match types
    print("\n3. Creating dummy variables for match types...")
    match_type_dummies = pd.get_dummies(match_data['match_type'], prefix='match_type')
    match_data = pd.concat([match_data, match_type_dummies], axis=1)
    
    # Create target columns from existing data
    print("\n4. Creating target columns...")
    # Convert goal and behind columns to numeric
    for col in match_data.columns:
        if '_goals' in col or '_behinds' in col:
            match_data[col] = pd.to_numeric(match_data[col], errors='coerce')
    
    # Fill any NaN values with column means
    for col in match_data.columns:
        if '_goals' in col or '_behinds' in col:
            if match_data[col].isna().sum() > 0:
                col_mean = match_data[col].mean()
                match_data[col] = match_data[col].fillna(col_mean)
                print(f"Filled {col} NaN values with mean: {col_mean:.2f}")
    
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
    
    # Load lineup data
    print("\n5. Loading lineup data...")
    lineup_dfs = []
    for f in lineup_files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                lineup_dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
    
    lineup_data = pd.concat(lineup_dfs)
    print(f"Lineup data shape: {lineup_data.shape}")
    
    # Extract player IDs from lineup data
    print("\n6. Extracting player IDs from lineup data...")
    # Convert columns to string for merging
    match_data['round_num'] = match_data['round_num'].astype(str)
    lineup_data['round_num'] = lineup_data['round_num'].astype(str)
    
    # Process team lineups
    team1_lineups = {}
    team2_lineups = {}
    
    # Process each match
    for index, match in match_data.iterrows():
        match_id = f"{match['year']}_{match['round_num']}"
        team1_name = match['team_1_team_name']
        team2_name = match['team_2_team_name']
        
        # Get team1 lineup
        team1_lineup = lineup_data[(lineup_data['year'] == match['year']) & 
                                   (lineup_data['round_num'] == match['round_num']) & 
                                   (lineup_data['team_name'] == team1_name)]
        
        # Get team2 lineup
        team2_lineup = lineup_data[(lineup_data['year'] == match['year']) & 
                                   (lineup_data['round_num'] == match['round_num']) & 
                                   (lineup_data['team_name'] == team2_name)]
        
        # Extract player IDs and map to indices
        if not team1_lineup.empty:
            # Get player IDs from columns like 'player1', 'player2', etc.
            player_cols = [col for col in team1_lineup.columns if col.startswith('player') and col not in ['player_count']]
            team1_players = []
            
            for col in player_cols:
                if col in team1_lineup.columns:
                    player_id = team1_lineup[col].iloc[0] if not team1_lineup[col].iloc[0] is None else 'unknown'
                    player_idx = player_index.get(str(player_id), 0)  # Use 0 for unknown players
                    team1_players.append(player_idx)
            
            # Pad or truncate to MAX_PLAYERS_PER_TEAM
            if len(team1_players) < MAX_PLAYERS_PER_TEAM:
                team1_players.extend([0] * (MAX_PLAYERS_PER_TEAM - len(team1_players)))
            elif len(team1_players) > MAX_PLAYERS_PER_TEAM:
                team1_players = team1_players[:MAX_PLAYERS_PER_TEAM]
                
            team1_lineups[match_id] = team1_players
        else:
            team1_lineups[match_id] = [0] * MAX_PLAYERS_PER_TEAM
            
        # Same for team2
        if not team2_lineup.empty:
            player_cols = [col for col in team2_lineup.columns if col.startswith('player') and col not in ['player_count']]
            team2_players = []
            
            for col in player_cols:
                if col in team2_lineup.columns:
                    player_id = team2_lineup[col].iloc[0] if not team2_lineup[col].iloc[0] is None else 'unknown'
                    player_idx = player_index.get(str(player_id), 0)
                    team2_players.append(player_idx)
            
            if len(team2_players) < MAX_PLAYERS_PER_TEAM:
                team2_players.extend([0] * (MAX_PLAYERS_PER_TEAM - len(team2_players)))
            elif len(team2_players) > MAX_PLAYERS_PER_TEAM:
                team2_players = team2_players[:MAX_PLAYERS_PER_TEAM]
                
            team2_lineups[match_id] = team2_players
        else:
            team2_lineups[match_id] = [0] * MAX_PLAYERS_PER_TEAM
    
    # Create arrays for team1 and team2 player indices
    match_ids = [f"{match['year']}_{match['round_num']}" for _, match in match_data.iterrows()]
    
    team1_player_indices = np.array([team1_lineups.get(match_id, [0] * MAX_PLAYERS_PER_TEAM) for match_id in match_ids])
    team2_player_indices = np.array([team2_lineups.get(match_id, [0] * MAX_PLAYERS_PER_TEAM) for match_id in match_ids])
    
    print(f"Team 1 player indices shape: {team1_player_indices.shape}")
    print(f"Team 2 player indices shape: {team2_player_indices.shape}")
    
    # Select numeric features only to avoid conversion issues
    print("\n7. Selecting numeric features...")
    numeric_cols = match_data.select_dtypes(include=['number']).columns.tolist()
    
    # Remove target columns from features
    target_cols = ['match_winner', 'margin', 'team1_goals', 'team1_behinds', 'team2_goals', 'team2_behinds']
    feature_cols = [col for col in numeric_cols if col not in target_cols]
    
    # Include match_type dummy variables
    for col in match_data.columns:
        if col.startswith('match_type_') and col not in feature_cols:
            feature_cols.append(col)
    
    # Remove quarter-by-quarter goals and behinds from feature columns
    feature_cols = [col for col in feature_cols if not any(q in col for q in ['_q1_', '_q2_', '_q3_'])]
    
    print(f"Selected {len(feature_cols)} numeric features")
    
    # Extract match features and targets
    X_match = match_data[feature_cols].values
    y_winner = match_data['match_winner'].values
    y_margin = match_data['margin'].values

    # Ensure goal and behind values are reasonable (no negative values)
    y_team1_goals = np.maximum(0, match_data['team1_goals'].values)
    y_team1_behinds = np.maximum(0, match_data['team1_behinds'].values)
    y_team2_goals = np.maximum(0, match_data['team2_goals'].values)
    y_team2_behinds = np.maximum(0, match_data['team2_behinds'].values)

    # Add data statistics for monitoring
    print(f"Team 1 Goals stats: min={y_team1_goals.min()}, max={y_team1_goals.max()}, mean={y_team1_goals.mean()}")
    print(f"Team 1 Behinds stats: min={y_team1_behinds.min()}, max={y_team1_behinds.max()}, mean={y_team1_behinds.mean()}")
    print(f"Team 2 Goals stats: min={y_team2_goals.min()}, max={y_team2_goals.max()}, mean={y_team2_goals.mean()}")
    print(f"Team 2 Behinds stats: min={y_team2_behinds.min()}, max={y_team2_behinds.max()}, mean={y_team2_behinds.mean()}")

    print(f"X_match shape: {X_match.shape}")
    print(f"team1_player_indices shape: {team1_player_indices.shape}")
    print(f"team2_player_indices shape: {team2_player_indices.shape}")
    
    print("\n----- COMPLETED DATA PREPARATION WITH PLAYER EMBEDDINGS -----\n")
    
    return X_match, team1_player_indices, team2_player_indices, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds, feature_cols, len(player_index)

# Load and prepare data with player embeddings
X_match, team1_player_indices, team2_player_indices, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds, feature_cols, num_players = load_and_prepare_data()

# Get enhanced player features
player_index = create_player_index()
team1_enhanced, team2_enhanced = enhance_player_embeddings(player_index, team1_player_indices, team2_player_indices)

# Normalize match features
scaler = StandardScaler()
X_match_normalized = scaler.fit_transform(X_match)

# Normalize enhanced player features
team1_scaler = StandardScaler()
team1_enhanced_normalized = team1_scaler.fit_transform(team1_enhanced)
team2_scaler = StandardScaler()
team2_enhanced_normalized = team2_scaler.fit_transform(team2_enhanced)

# Split data into training and test sets (including enhanced features)
X_train_match, X_test_match, X_train_team1_players, X_test_team1_players, X_train_team2_players, X_test_team2_players, X_train_team1_enhanced, X_test_team1_enhanced, X_train_team2_enhanced, X_test_team2_enhanced, y_train_winner, y_test_winner, y_train_margin, y_test_margin, y_train_team1_goals, y_test_team1_goals, y_train_team1_behinds, y_test_team1_behinds, y_train_team2_goals, y_test_team2_goals, y_train_team2_behinds, y_test_team2_behinds = train_test_split(
    X_match_normalized, team1_player_indices, team2_player_indices, team1_enhanced_normalized, team2_enhanced_normalized, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds, test_size=0.2, random_state=42)

# One-hot encode y_train_winner and y_test_winner for three classes
# Convert 0, 0.5, 1 to proper indices for to_categorical
y_train_winner_classes = np.zeros_like(y_train_winner, dtype=int)
y_train_winner_classes[y_train_winner == 1] = 0  # team1 wins
y_train_winner_classes[y_train_winner == 0.5] = 1  # draw
y_train_winner_classes[y_train_winner == 0] = 2  # team2 wins

y_test_winner_classes = np.zeros_like(y_test_winner, dtype=int)
y_test_winner_classes[y_test_winner == 1] = 0  # team1 wins
y_test_winner_classes[y_test_winner == 0.5] = 1  # draw
y_test_winner_classes[y_test_winner == 0] = 2  # team2 wins

# Now one-hot encode with proper indices
y_train_winner = to_categorical(y_train_winner_classes, num_classes=3)
y_test_winner = to_categorical(y_test_winner_classes, num_classes=3)

# Define model with player embeddings
print("\nDefining model with player embeddings...")

# Input layers
match_input = Input(shape=(X_match.shape[1],), name="match_input")  # Match features
team1_players_input = Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team1_players_input")  # Team1 player IDs
team2_players_input = Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team2_players_input")  # Team2 player IDs
team1_enhanced_input = Input(shape=(team1_enhanced.shape[1],), name="team1_enhanced_input")  # Team1 enhanced features
team2_enhanced_input = Input(shape=(team2_enhanced.shape[1],), name="team2_enhanced_input")  # Team2 enhanced features

# Player embedding layer
player_embedding = Embedding(input_dim=num_players, output_dim=EMBEDDING_DIM, name='player_embedding')

# Apply embeddings to both teams
team1_embeddings = player_embedding(team1_players_input)
team2_embeddings = player_embedding(team2_players_input)

# Aggregate player embeddings for each team using custom layer with explicit output shape
team1_aggregated = MeanPoolingLayer(output_dim=EMBEDDING_DIM, name='team1_aggregated')(team1_embeddings)
team2_aggregated = MeanPoolingLayer(output_dim=EMBEDDING_DIM, name='team2_aggregated')(team2_embeddings)

# Concatenate match features with team embeddings and enhanced features
combined = Concatenate()([match_input, team1_aggregated, team2_aggregated, team1_enhanced_input, team2_enhanced_input])

# Shared layers
x = Dense(128, activation='relu')(combined)
x = Dense(64, activation='relu')(x)

# Define AFL realistic max values
MAX_GOALS = 25
MIN_GOALS = 5
MAX_BEHINDS = 15
MIN_BEHINDS = 3

# Team 1 goals prediction
team1_goals_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_input])
team1_goals_branch = Dense(64, activation='relu')(team1_goals_branch)
team1_goals_branch = Dense(32, activation='relu')(team1_goals_branch)
team1_goals_branch = Dense(1, activation='relu')(team1_goals_branch)
team1_goals = BoundedOutputLayer(min_value=MIN_GOALS, max_value=MAX_GOALS, name='team1_goals')(team1_goals_branch)

# Team 1 behinds prediction
team1_behinds_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_input])
team1_behinds_branch = Dense(64, activation='relu')(team1_behinds_branch)
team1_behinds_branch = Dense(32, activation='relu')(team1_behinds_branch)
team1_behinds_branch = Dense(1, activation='relu')(team1_behinds_branch)
team1_behinds = BoundedOutputLayer(min_value=MIN_BEHINDS, max_value=MAX_BEHINDS, name='team1_behinds')(team1_behinds_branch)

# Team 2 goals prediction
team2_goals_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_input])
team2_goals_branch = Dense(64, activation='relu')(team2_goals_branch)
team2_goals_branch = Dense(32, activation='relu')(team2_goals_branch)
team2_goals_branch = Dense(1, activation='relu')(team2_goals_branch)
team2_goals = BoundedOutputLayer(min_value=MIN_GOALS, max_value=MAX_GOALS, name='team2_goals')(team2_goals_branch)

# Team 2 behinds prediction
team2_behinds_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_input])
team2_behinds_branch = Dense(64, activation='relu')(team2_behinds_branch)
team2_behinds_branch = Dense(32, activation='relu')(team2_behinds_branch)
team2_behinds_branch = Dense(1, activation='relu')(team2_behinds_branch)
team2_behinds = BoundedOutputLayer(min_value=MIN_BEHINDS, max_value=MAX_BEHINDS, name='team2_behinds')(team2_behinds_branch)

# Calculate scores and margin
team1_score = Lambda(lambda x: x[0] * 6 + x[1], name='team1_score')([team1_goals, team1_behinds])
team2_score = Lambda(lambda x: x[0] * 6 + x[1], name='team2_score')([team2_goals, team2_behinds])
calculated_margin = Lambda(lambda x: x[0] - x[1], name='calculated_margin')([team1_score, team2_score])

# Winner prediction (3 classes: team1 win, draw, team2 win)
match_winner = Dense(3, activation='softmax', name='match_winner')(x)

# Create model with our bounded prediction outputs
model = Model(
    inputs=[match_input, team1_players_input, team2_players_input, team1_enhanced_input, team2_enhanced_input], 
    outputs=[match_winner, team1_goals, team1_behinds, team2_goals, team2_behinds, calculated_margin]
)

# Compile model with appropriate losses - we removed the separate margin output
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'match_winner': 'categorical_crossentropy', 
        'team1_goals': 'mse', 
        'team1_behinds': 'mse', 
        'team2_goals': 'mse', 
        'team2_behinds': 'mse',
        'calculated_margin': 'mse'  # This should match the target margin
    },
    loss_weights={
        'match_winner': 1.0,
        'team1_goals': 1.0,
        'team1_behinds': 1.0, 
        'team2_goals': 1.0,
        'team2_behinds': 1.0,
        'calculated_margin': 1.0
    },
    metrics={
        'match_winner': ['accuracy', Precision(name='precision'), Recall(name='recall')], 
        'team1_goals': tf.keras.metrics.MeanAbsoluteError(name='mae'), 
        'team1_behinds': tf.keras.metrics.MeanAbsoluteError(name='mae'), 
        'team2_goals': tf.keras.metrics.MeanAbsoluteError(name='mae'), 
        'team2_behinds': tf.keras.metrics.MeanAbsoluteError(name='mae'),
        'calculated_margin': tf.keras.metrics.MeanAbsoluteError(name='mae')
    }
)

# Summary of the model
model.summary()

# Create dictionary with targets - remove margin since we now use calculated_margin directly
targets_dict = {
    'match_winner': y_train_winner, 
    'team1_goals': y_train_team1_goals, 
    'team1_behinds': y_train_team1_behinds,
    'team2_goals': y_train_team2_goals, 
    'team2_behinds': y_train_team2_behinds,
    'calculated_margin': y_train_margin  # Use the true margin for the calculated margin
}

# Test dictionary with targets - remove margin since we now use calculated_margin directly
test_targets_dict = {
    'match_winner': y_test_winner, 
    'team1_goals': y_test_team1_goals, 
    'team1_behinds': y_test_team1_behinds,
    'team2_goals': y_test_team2_goals, 
    'team2_behinds': y_test_team2_behinds,
    'calculated_margin': y_test_margin  # Use the true margin for the calculated margin
}

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

# Train the model with more epochs
print("\nTraining the model...")
model.fit(
    [X_train_match, X_train_team1_players, X_train_team2_players, X_train_team1_enhanced, X_train_team2_enhanced], 
    targets_dict,
    epochs=20,  # Increased from 10 to 20 
    batch_size=32, 
    validation_split=0.2,
    callbacks=[
        # Add early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        # Add model checkpoint to save the best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(PROJECT_ROOT, 'model/output/best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
)

# Evaluate the model with our new architecture
print("\nEvaluating the model...")
predictions = model.predict([X_test_match, X_test_team1_players, X_test_team2_players, X_test_team1_enhanced, X_test_team2_enhanced])

# Update predictions indices since we removed margin
# predictions[0] = match_winner
# predictions[1] = team1_goals
# predictions[2] = team1_behinds
# predictions[3] = team2_goals
# predictions[4] = team2_behinds
# predictions[5] = calculated_margin

# Debugging information for F1 score calculation
print("Unique values in y_test_winner:", np.unique(y_test_winner))
print("Unique values in predictions[0]:", np.unique(np.round(predictions[0])))
print("Shape of y_test_winner:", y_test_winner.shape)
print("Shape of predictions[0]:", predictions[0].shape)

# Calculate F1 Score for match winner
f1 = f1_score(np.argmax(y_test_winner, axis=1), np.argmax(predictions[0], axis=1), average='macro')

# Calculate RMSE for regression tasks
rmse_team1_goals = rmse(y_test_team1_goals, predictions[1])
rmse_team1_behinds = rmse(y_test_team1_behinds, predictions[2])
rmse_team2_goals = rmse(y_test_team2_goals, predictions[3])
rmse_team2_behinds = rmse(y_test_team2_behinds, predictions[4])
rmse_calculated_margin = rmse(y_test_margin, predictions[5])

# Sample predictions for visual inspection
print("\nSample Predictions (first 5 matches):")
for i in range(min(5, len(predictions[0]))):
    true_winner_idx = np.argmax(y_test_winner[i])
    true_winner = "Team 1" if true_winner_idx == 0 else ("Draw" if true_winner_idx == 1 else "Team 2")
    
    pred_winner_idx = np.argmax(predictions[0][i])
    pred_winner = "Team 1" if pred_winner_idx == 0 else ("Draw" if pred_winner_idx == 1 else "Team 2")
    
    print(f"Match {i+1}:")
    print(f"  Predicted scores: Team 1 {predictions[1][i][0]:.1f}G {predictions[2][i][0]:.1f}B ({predictions[1][i][0]*6 + predictions[2][i][0]:.1f}) - Team 2 {predictions[3][i][0]:.1f}G {predictions[4][i][0]:.1f}B ({predictions[3][i][0]*6 + predictions[4][i][0]:.1f})")
    print(f"  Actual scores: Team 1 {y_test_team1_goals[i]:.1f}G {y_test_team1_behinds[i]:.1f}B ({y_test_team1_goals[i]*6 + y_test_team1_behinds[i]:.1f}) - Team 2 {y_test_team2_goals[i]:.1f}G {y_test_team2_behinds[i]:.1f}B ({y_test_team2_goals[i]*6 + y_test_team2_behinds[i]:.1f})")
    print(f"  Predicted margin: {predictions[5][i][0]:.1f}, Actual margin: {y_test_margin[i]:.1f}")
    print(f"  Predicted winner: {pred_winner}, True winner: {true_winner}")
    print(f"  Win probabilities: Team 1 {predictions[0][i][0]:.2f}, Draw {predictions[0][i][1]:.2f}, Team 2 {predictions[0][i][2]:.2f}")
    print()

# Compile evaluation results
evaluation_results = model.evaluate([X_test_match, X_test_team1_players, X_test_team2_players, X_test_team1_enhanced, X_test_team2_enhanced], test_targets_dict, return_dict=True)
evaluation_results['f1_score'] = f1
evaluation_results['rmse_team1_goals'] = rmse_team1_goals
evaluation_results['rmse_team1_behinds'] = rmse_team1_behinds
evaluation_results['rmse_team2_goals'] = rmse_team2_goals
evaluation_results['rmse_team2_behinds'] = rmse_team2_behinds
evaluation_results['rmse_calculated_margin'] = rmse_calculated_margin

# Calculate margin consistency (how well predicted margin matches calculated margin)
margin_consistency = rmse(predictions[5], predictions[5])
evaluation_results['margin_consistency'] = margin_consistency

# Print margin consistency metrics
print(f"\nMargin Consistency:")
print(f"Difference between predicted margin and calculated margin: {margin_consistency:.4f} (RMSE)")

# Function to categorize metrics based on thresholds
def categorize_metric(value, thresholds):
    if value > thresholds['amazing']:
        return 'Amazing'
    elif value > thresholds['great']:
        return 'Great'
    elif value > thresholds['good']:
        return 'Good'
    elif value > thresholds['neutral']:
        return 'Neutral'
    elif value > thresholds['bad']:
        return 'Bad'
    else:
        return 'Terrible'

# Define thresholds for each metric
classification_thresholds = {
    'amazing': 0.95,
    'great': 0.90,
    'good': 0.80,
    'neutral': 0.70,
    'bad': 0.60
}

f1_thresholds = classification_thresholds

regression_thresholds = {
    'amazing': 0.05,
    'great': 0.10,
    'good': 0.15,
    'neutral': 0.20,
    'bad': 0.25
}

# Categorize evaluation results
categorized_results = {
    'accuracy_category': categorize_metric(evaluation_results['match_winner_accuracy'], classification_thresholds),
    'f1_score_category': categorize_metric(f1, f1_thresholds),
    'rmse_team1_goals_category': categorize_metric(rmse_team1_goals, regression_thresholds),
    'rmse_team1_behinds_category': categorize_metric(rmse_team1_behinds, regression_thresholds),
    'rmse_team2_goals_category': categorize_metric(rmse_team2_goals, regression_thresholds),
    'rmse_team2_behinds_category': categorize_metric(rmse_team2_behinds, regression_thresholds),
    'rmse_calculated_margin_category': categorize_metric(rmse_calculated_margin, regression_thresholds)
}

# Write categorized evaluation metrics to a .txt file
with open(os.path.join(PROJECT_ROOT, 'model/output/evaluation_metrics.txt'), 'a') as f:
    f.write("\nCategorized Results:\n")
    for key, value in categorized_results.items():
        f.write(f"{key}: {value}\n")

print("Categorized evaluation metrics saved to model/output/evaluation_metrics.txt")

# --- Added Saving Logic ---
# Ensure output directory exists
output_dir = os.path.join(PROJECT_ROOT, 'model/output')
os.makedirs(output_dir, exist_ok=True)

# Save the scaler
scaler_path = os.path.join(output_dir, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# Save the player index (convert keys to string for JSON compatibility if needed)
player_index_path = os.path.join(output_dir, 'player_index.json')
# Ensure player_index is loaded before saving
if 'player_index' not in locals():
     player_index = create_player_index() # Re-create if not available globally

# Convert integer keys to strings if necessary for JSON dump, handle potential non-string keys robustly
try:
    player_index_serializable = {str(k): v for k, v in player_index.items()}
    with open(player_index_path, 'w') as f:
        json.dump(player_index_serializable, f, indent=4)
    print(f"Player index saved to {player_index_path}")
except Exception as e:
    print(f"Error saving player index: {e}. Player Index Keys: {list(player_index.keys())[:10]}") # Print first 10 keys for debugging

# Save the feature columns
feature_cols_path = os.path.join(output_dir, 'feature_cols.json')
with open(feature_cols_path, 'w') as f:
    json.dump(feature_cols, f, indent=4)
print(f"Feature columns saved to {feature_cols_path}")

# Save the model - include saving a GUI-compatible version
model_path = os.path.join(output_dir, 'model.h5')
model.save(model_path, save_format='tf')
print(f"Model saved to {model_path}")

# Also save model weights separately for more compatibility options
weights_path = os.path.join(output_dir, 'model.weights.h5')
model.save_weights(weights_path)
print(f"Model weights saved to {weights_path}")

# Save a GUI-compatible version of the model
gui_resources_dir = os.path.join(PROJECT_ROOT, 'gui/resources')
os.makedirs(gui_resources_dir, exist_ok=True)
gui_model_path = os.path.join(gui_resources_dir, 'gui_model.h5')
model.save(gui_model_path, save_format='tf')
print(f"GUI-compatible model saved to {gui_model_path}")

# Also save in another location for SimplePredictor
simple_predictor_model_path = os.path.join(gui_resources_dir, 'simple_predictor_model.h5')
model.save(simple_predictor_model_path, save_format='tf')
print(f"SimplePredictor-compatible model saved to {simple_predictor_model_path}")

print("--- Model and preprocessing objects saved successfully. ---")

# --- End of Added Saving Logic ---
