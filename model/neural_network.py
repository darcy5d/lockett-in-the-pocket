import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Define file paths for data
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
    # Add more lineup files as needed
]

player_files = [
    'afl_data/data/players/zunneberg_noel_13081946_performance_details.csv',
    'afl_data/data/players/zunneberg_noel_13081946_personal_details.csv',
    'afl_data/data/players/zurhaar_cameron_22051998_performance_details.csv'
    # Add more player files as needed
]

# Function to load and prepare data with match type dummy variables
def load_and_prepare_data():
    print("\n----- DEBUGGING DATA PROCESSING PIPELINE -----")
    
    # Load match data
    print("\n1. Loading match data...")
    match_data = pd.concat([pd.read_csv(f) for f in match_files])
    print(f"Match data shape: {match_data.shape}")
    print("Match data columns:")
    print(match_data.columns.tolist())
    print("\nSample match data:")
    print(match_data.head(2))
    
    # Add match_type column
    print("\n2. Adding match_type column...")
    match_data['match_type'] = 'Regular'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Qualifying Final', na=False), 'match_type'] = 'Qualifying Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Elimination Final', na=False), 'match_type'] = 'Elimination Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Semi Final', na=False), 'match_type'] = 'Semi Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Preliminary Final', na=False), 'match_type'] = 'Preliminary Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Grand Final', na=False), 'match_type'] = 'Grand Final'
    print("Match types distribution:")
    print(match_data['match_type'].value_counts())

    # Create dummy variables for match types
    print("\n3. Creating dummy variables for match types...")
    match_type_dummies = pd.get_dummies(match_data['match_type'], prefix='match_type')
    match_data = pd.concat([match_data, match_type_dummies], axis=1)
    print("Match data with dummies shape:", match_data.shape)
    print("Match data with dummies columns:")
    print(match_data.columns.tolist())
    
    # Create target columns from existing data
    print("\n4. Creating target columns...")
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
    
    print("Added target columns: match_winner, margin, team1_goals, team1_behinds, team2_goals, team2_behinds")
    print("Sample target data:")
    print(match_data[['match_winner', 'margin', 'team1_goals', 'team1_behinds', 'team2_goals', 'team2_behinds']].head(5))
    
    # Load lineup and player data
    print("\n5. Loading lineup and player data...")
    lineup_data = pd.concat([pd.read_csv(f) for f in lineup_files])
    player_data = pd.concat([pd.read_csv(f) for f in player_files])
    
    print(f"Lineup data shape: {lineup_data.shape}")
    print("Lineup data columns:")
    print(lineup_data.columns.tolist())
    
    print(f"\nPlayer data shape: {player_data.shape}")
    print("Player data columns:")
    print(player_data.columns.tolist())
    
    # Merge data
    print("\n6. Merging match data with lineup data...")
    # Convert columns to string before merging
    match_data['round_num'] = match_data['round_num'].astype(str)
    lineup_data['round_num'] = lineup_data['round_num'].astype(str)
    
    merged_data = pd.merge(match_data, lineup_data, 
                           left_on=['year', 'team_1_team_name', 'round_num'], 
                           right_on=['year', 'team_name', 'round_num'], 
                           how='left')
    
    print(f"Merged with lineup data shape: {merged_data.shape}")
    print("Merged with lineup data columns:")
    print(merged_data.columns.tolist())
    
    # Prepare for merge with player data
    print("\n7. Preparing for merge with player data...")
    # Convert to string before merging
    merged_data['round_num'] = merged_data['round_num'].astype(str)
    player_data['round'] = player_data['round'].astype(str)

    merged_data = pd.merge(merged_data, player_data, 
                         left_on=['year', 'team_1_team_name', 'round_num'], 
                         right_on=['year', 'team', 'round'], 
                         how='left')
    
    print(f"\nFinal merged data shape: {merged_data.shape}")
    print("Final merged data columns:")
    print(merged_data.columns.tolist())
    
    # Check for missing values in important columns
    print("\n8. Checking for missing values in important columns...")
    missing_values = merged_data[['match_winner', 'margin', 'team1_goals', 'team1_behinds', 'team2_goals', 'team2_behinds']].isnull().sum()
    print("Missing values in target columns:")
    print(missing_values)
    
    # Fill missing values if needed
    if missing_values.sum() > 0:
        print("Filling missing values in target columns...")
        merged_data[['match_winner', 'margin', 'team1_goals', 'team1_behinds', 'team2_goals', 'team2_behinds']] = (
            merged_data[['match_winner', 'margin', 'team1_goals', 'team1_behinds', 'team2_goals', 'team2_behinds']].fillna(0)
        )
    
    # Extract features and targets
    print("\n9. Extracting features and targets...")
    
    # Select numeric features only to avoid conversion issues
    print("Selecting only numeric features...")
    numeric_cols = merged_data.select_dtypes(include=['number']).columns.tolist()
    
    # Remove target columns from features
    target_cols = ['match_winner', 'margin', 'team1_goals', 'team1_behinds', 'team2_goals', 'team2_behinds']
    feature_cols = [col for col in numeric_cols if col not in target_cols]
    
    # Include match_type dummy variables
    for col in merged_data.columns:
        if col.startswith('match_type_') and col not in feature_cols:
            feature_cols.append(col)
    
    print(f"Selected {len(feature_cols)} numeric features")
    print("First 10 feature columns:", feature_cols[:10], "...")
    
    X = merged_data[feature_cols]
    y_winner = merged_data['match_winner']
    y_margin = merged_data['margin']
    y_team1_goals = merged_data['team1_goals']
    y_team1_behinds = merged_data['team1_behinds']
    y_team2_goals = merged_data['team2_goals']
    y_team2_behinds = merged_data['team2_behinds']
    
    print(f"X shape: {X.shape}")
    print(f"y_winner shape: {y_winner.shape}")
    
    print("\n----- END OF DEBUGGING DATA PROCESSING PIPELINE -----\n")
    
    return X, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds, feature_cols

# Load and prepare data
X, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds, feature_cols = load_and_prepare_data()

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train_winner, y_test_winner, y_train_margin, y_test_margin, y_train_team1_goals, y_test_team1_goals, y_train_team1_behinds, y_test_team1_behinds, y_train_team2_goals, y_test_team2_goals, y_train_team2_behinds, y_test_team2_behinds = train_test_split(
    X_normalized, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds, test_size=0.2, random_state=42)

# Now define the model with the correct input shape based on actual data
input_shape = (X.shape[1],)  # Use the actual number of features
print(f"Defining model with input shape: {input_shape}")

# Input layer
inputs = Input(shape=input_shape, name="input_layer")

# Shared layers
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

# Branch for match winner prediction
match_winner = Dense(32, activation='relu')(x)
match_winner = Dense(1, activation='sigmoid', name='match_winner')(match_winner)

# Branch for margin prediction
margin = Dense(32, activation='relu')(x)
margin = Dense(1, activation='linear', name='margin')(margin)

# Branch for Team 1 goals prediction
team1_goals = Dense(32, activation='relu')(x)
team1_goals = Dense(1, activation='linear', name='team1_goals')(team1_goals)

# Branch for Team 1 behinds prediction
team1_behinds = Dense(32, activation='relu')(x)
team1_behinds = Dense(1, activation='linear', name='team1_behinds')(team1_behinds)

# Branch for Team 2 goals prediction
team2_goals = Dense(32, activation='relu')(x)
team2_goals = Dense(1, activation='linear', name='team2_goals')(team2_goals)

# Branch for Team 2 behinds prediction
team2_behinds = Dense(32, activation='relu')(x)
team2_behinds = Dense(1, activation='linear', name='team2_behinds')(team2_behinds)

# Create model with outputs
model = Model(inputs=inputs, outputs=[match_winner, margin, team1_goals, team1_behinds, team2_goals, team2_behinds])

# Compile model
model.compile(optimizer='adam',
              loss={'match_winner': 'binary_crossentropy', 'margin': 'mse',
                    'team1_goals': 'mse', 'team1_behinds': 'mse', 'team2_goals': 'mse', 'team2_behinds': 'mse'},
              metrics={'match_winner': 'accuracy', 'margin': 'mae',
                       'team1_goals': 'mae', 'team1_behinds': 'mae', 'team2_goals': 'mae', 'team2_behinds': 'mae'})

# Summary of the model
model.summary()

# Create dictionary with targets
targets_dict = {
    'match_winner': y_train_winner, 
    'margin': y_train_margin,
    'team1_goals': y_train_team1_goals, 
    'team1_behinds': y_train_team1_behinds,
    'team2_goals': y_train_team2_goals, 
    'team2_behinds': y_train_team2_behinds
}

# Test dictionary with targets
test_targets_dict = {
    'match_winner': y_test_winner, 
    'margin': y_test_margin,
    'team1_goals': y_test_team1_goals, 
    'team1_behinds': y_test_team1_behinds,
    'team2_goals': y_test_team2_goals, 
    'team2_behinds': y_test_team2_behinds
}

# Train the model
print("\nTraining the model...")
model.fit(X_train, targets_dict,
          epochs=40, batch_size=32, validation_split=0.2)

# Evaluate the model
print("\nEvaluating the model...")
model.evaluate(X_test, test_targets_dict) 