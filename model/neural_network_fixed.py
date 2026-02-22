import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import os

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

# Function to load and prepare match data only (avoid problematic player data)
def load_and_prepare_data():
    print("\n----- LOADING AND PREPARING DATA -----")
    
    # Load match data from all existing files
    print("Loading match data...")
    match_dfs = []
    for f in match_files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                match_dfs.append(df)
                print(f"Loaded {f}: {df.shape[0]} rows")
            except Exception as e:
                print(f"Error loading {f}: {e}")
    
    match_data = pd.concat(match_dfs)
    print(f"Combined match data shape: {match_data.shape}")
    
    # Add match_type column for finals
    print("\nAdding match type indicators...")
    match_data['match_type'] = 'Regular'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Qualifying Final', na=False), 'match_type'] = 'Qualifying Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Elimination Final', na=False), 'match_type'] = 'Elimination Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Semi Final', na=False), 'match_type'] = 'Semi Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Preliminary Final', na=False), 'match_type'] = 'Preliminary Final'
    match_data.loc[match_data['round_num'].astype(str).str.contains('Grand Final', na=False), 'match_type'] = 'Grand Final'
    
    # Convert goal and behind columns to numeric, properly handling NaN values
    print("\nConverting goals and behinds to numeric values...")
    for col in match_data.columns:
        if '_goals' in col or '_behinds' in col:
            match_data[col] = pd.to_numeric(match_data[col], errors='coerce')
    
    # Fill any missing data in the core columns
    numeric_cols = match_data.select_dtypes(include=['number']).columns
    match_data[numeric_cols] = match_data[numeric_cols].fillna(match_data[numeric_cols].mean())
    
    # Create dummy variables for match types
    print("Creating match type dummy variables...")
    match_type_dummies = pd.get_dummies(match_data['match_type'], prefix='match_type')
    match_data = pd.concat([match_data, match_type_dummies], axis=1)
    
    # Calculate team1 and team2 scores
    print("Calculating scores and margins...")
    match_data['team1_score'] = match_data['team_1_final_goals'] * 6 + match_data['team_1_final_behinds']
    match_data['team2_score'] = match_data['team_2_final_goals'] * 6 + match_data['team_2_final_behinds']
    
    # Create match_winner column (1 if team1 wins, 0 if team2 wins, 0.5 if draw)
    match_data['match_winner'] = (match_data['team1_score'] > match_data['team2_score']).astype(float)
    match_data.loc[match_data['team1_score'] == match_data['team2_score'], 'match_winner'] = 0.5
    
    # Create margin column (team1_score - team2_score)
    match_data['margin'] = match_data['team1_score'] - match_data['team2_score']
    
    # Rename some columns for clarity
    match_data['team1_goals'] = match_data['team_1_final_goals']
    match_data['team1_behinds'] = match_data['team_1_final_behinds']
    match_data['team2_goals'] = match_data['team_2_final_goals']
    match_data['team2_behinds'] = match_data['team_2_final_behinds']
    
    # Create home/away feature
    print("Adding derived features...")
    # Use venue as a categorical feature
    venue_dummies = pd.get_dummies(match_data['venue'], prefix='venue')
    match_data = pd.concat([match_data, venue_dummies], axis=1)
    
    # Check for remaining NaN values
    na_count = match_data.isna().sum()
    if na_count.sum() > 0:
        print(f"\nWARNING: {na_count.sum()} NaN values remain in the data")
        cols_with_na = na_count[na_count > 0]
        print(f"Columns with NaN values:\n{cols_with_na}")
        
        # Final cleaning - fill any remaining NaNs
        match_data = match_data.fillna(0)
    
    # Extract features and targets
    print("\nPreparing features and targets...")
    # Use these features - numeric values and dummy variables
    feature_cols = [
        # Year
        'year',
        
        # Quarter scores for team 1
        'team_1_q1_goals', 'team_1_q1_behinds',
        'team_1_q2_goals', 'team_1_q2_behinds',
        'team_1_q3_goals', 'team_1_q3_behinds',
        
        # Quarter scores for team 2
        'team_2_q1_goals', 'team_2_q1_behinds',
        'team_2_q2_goals', 'team_2_q2_behinds',
        'team_2_q3_goals', 'team_2_q3_behinds',
        
        # Cumulative scores at quarter 3
        'team1_q3_score', 'team2_q3_score',
        
        # Match type indicators
        'match_type_Regular', 'match_type_Elimination Final', 
        'match_type_Qualifying Final', 'match_type_Semi Final',
        'match_type_Preliminary Final', 'match_type_Grand Final'
    ]
    
    # Calculate quarter 3 scores
    match_data['team1_q3_score'] = match_data['team_1_q3_goals'] * 6 + match_data['team_1_q3_behinds']
    match_data['team2_q3_score'] = match_data['team_2_q3_goals'] * 6 + match_data['team_2_q3_behinds']
    
    # Use only columns that exist in the data
    feature_cols = [col for col in feature_cols if col in match_data.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Target variables
    target_cols = ['match_winner', 'margin', 'team1_goals', 'team1_behinds', 'team2_goals', 'team2_behinds']
    
    # Final feature set and targets
    X = match_data[feature_cols]
    y_winner = match_data['match_winner']
    y_margin = match_data['margin']
    y_team1_goals = match_data['team1_goals']
    y_team1_behinds = match_data['team1_behinds']
    y_team2_goals = match_data['team2_goals']
    y_team2_behinds = match_data['team2_behinds']
    
    print(f"X shape: {X.shape}")
    print(f"y_winner shape: {y_winner.shape}")
    
    # Check for any remaining NaN values
    X_na_count = X.isna().sum().sum()
    y_na_count = sum([y.isna().sum() for y in [y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds]])
    
    if X_na_count > 0 or y_na_count > 0:
        print(f"WARNING: {X_na_count} NaN values in features, {y_na_count} NaN values in targets")
        # Clean up any remaining NaNs
        X = X.fillna(0)
        y_winner = y_winner.fillna(0)
        y_margin = y_margin.fillna(0)
        y_team1_goals = y_team1_goals.fillna(0)
        y_team1_behinds = y_team1_behinds.fillna(0)
        y_team2_goals = y_team2_goals.fillna(0)
        y_team2_behinds = y_team2_behinds.fillna(0)
    
    print("----- DATA PREPARATION COMPLETE -----\n")
    
    return X, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds

# Load and prepare data
X, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds = load_and_prepare_data()

# Use RobustScaler instead of StandardScaler to be less affected by outliers
print("Normalizing features...")
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_normalized = scaler.fit_transform(X)

# Split data into training and test sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train_winner, y_test_winner, y_train_margin, y_test_margin, y_train_team1_goals, y_test_team1_goals, y_train_team1_behinds, y_test_team1_behinds, y_train_team2_goals, y_test_team2_goals, y_train_team2_behinds, y_test_team2_behinds = train_test_split(
    X_normalized, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds, test_size=0.2, random_state=42)

# Now define the model with the correct input shape based on actual data
input_shape = (X.shape[1],)  # Use the actual number of features
print(f"Defining model with input shape: {input_shape}")

# Input layer
inputs = Input(shape=input_shape, name="input_layer")

# Shared layers with dropout for regularization
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.3)(x)  # Add dropout to reduce overfitting
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)  # Add dropout to reduce overfitting

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

# Compile model - use a smaller learning rate for better stability
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
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

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10,
    restore_best_weights=True
)

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, 
    targets_dict,
    epochs=100,  # Use more epochs with early stopping
    batch_size=32, 
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
print("\nEvaluating the model...")
evaluation = model.evaluate(X_test, test_targets_dict)

# Print results
print("\nFinal Results:")
metric_names = ['loss'] + [f"{name}_{metric}" for name in model.output_names for metric in ['loss', 'mae' if name != 'match_winner' else 'accuracy']]
for name, value in zip(metric_names[:len(evaluation)], evaluation):
    print(f"{name}: {value:.4f}")

# Make predictions on test data
predictions = model.predict(X_test)
match_winner_preds = predictions[0]
margin_preds = predictions[1]

# Calculate winner prediction accuracy
winner_accuracy = np.mean((match_winner_preds > 0.5) == y_test_winner)
print(f"\nMatch winner prediction accuracy: {winner_accuracy:.4f}")

# Calculate margin prediction error
margin_mae = np.mean(np.abs(margin_preds - y_test_margin.values.reshape(-1, 1)))
print(f"Margin prediction mean absolute error: {margin_mae:.4f}")

print("\nTraining complete! Model is ready for use.") 