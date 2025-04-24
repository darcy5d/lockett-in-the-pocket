import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Define input shape
input_shape = (61,)  # Adjust based on the number of features

# Input layer
inputs = Input(shape=input_shape)

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

# Branches for Team 1 quarter scores prediction
team1_q1_goals = Dense(32, activation='relu')(x)
team1_q1_goals = Dense(1, activation='linear', name='team1_q1_goals')(team1_q1_goals)
team1_q1_behinds = Dense(32, activation='relu')(x)
team1_q1_behinds = Dense(1, activation='linear', name='team1_q1_behinds')(team1_q1_behinds)

team1_q2_goals = Dense(32, activation='relu')(x)
team1_q2_goals = Dense(1, activation='linear', name='team1_q2_goals')(team1_q2_goals)
team1_q2_behinds = Dense(32, activation='relu')(x)
team1_q2_behinds = Dense(1, activation='linear', name='team1_q2_behinds')(team1_q2_behinds)

team1_q3_goals = Dense(32, activation='relu')(x)
team1_q3_goals = Dense(1, activation='linear', name='team1_q3_goals')(team1_q3_goals)
team1_q3_behinds = Dense(32, activation='relu')(x)
team1_q3_behinds = Dense(1, activation='linear', name='team1_q3_behinds')(team1_q3_behinds)

team1_q4_goals = Dense(32, activation='relu')(x)
team1_q4_goals = Dense(1, activation='linear', name='team1_q4_goals')(team1_q4_goals)
team1_q4_behinds = Dense(32, activation='relu')(x)
team1_q4_behinds = Dense(1, activation='linear', name='team1_q4_behinds')(team1_q4_behinds)

# Branches for Team 2 quarter scores prediction
team2_q1_goals = Dense(32, activation='relu')(x)
team2_q1_goals = Dense(1, activation='linear', name='team2_q1_goals')(team2_q1_goals)
team2_q1_behinds = Dense(32, activation='relu')(x)
team2_q1_behinds = Dense(1, activation='linear', name='team2_q1_behinds')(team2_q1_behinds)

team2_q2_goals = Dense(32, activation='relu')(x)
team2_q2_goals = Dense(1, activation='linear', name='team2_q2_goals')(team2_q2_goals)
team2_q2_behinds = Dense(32, activation='relu')(x)
team2_q2_behinds = Dense(1, activation='linear', name='team2_q2_behinds')(team2_q2_behinds)

team2_q3_goals = Dense(32, activation='relu')(x)
team2_q3_goals = Dense(1, activation='linear', name='team2_q3_goals')(team2_q3_goals)
team2_q3_behinds = Dense(32, activation='relu')(x)
team2_q3_behinds = Dense(1, activation='linear', name='team2_q3_behinds')(team2_q3_behinds)

team2_q4_goals = Dense(32, activation='relu')(x)
team2_q4_goals = Dense(1, activation='linear', name='team2_q4_goals')(team2_q4_goals)
team2_q4_behinds = Dense(32, activation='relu')(x)
team2_q4_behinds = Dense(1, activation='linear', name='team2_q4_behinds')(team2_q4_behinds)

# Create model with new outputs
model = Model(inputs=inputs, outputs=[match_winner, margin, team1_goals, team1_behinds, team2_goals, team2_behinds,
                                      team1_q1_goals, team1_q1_behinds, team1_q2_goals, team1_q2_behinds,
                                      team1_q3_goals, team1_q3_behinds, team1_q4_goals, team1_q4_behinds,
                                      team2_q1_goals, team2_q1_behinds, team2_q2_goals, team2_q2_behinds,
                                      team2_q3_goals, team2_q3_behinds, team2_q4_goals, team2_q4_behinds])

# Compile model with new loss functions
model.compile(optimizer='adam',
              loss={'match_winner': 'binary_crossentropy', 'margin': 'mse',
                    'team1_goals': 'mse', 'team1_behinds': 'mse', 'team2_goals': 'mse', 'team2_behinds': 'mse',
                    'team1_q1_goals': 'mse', 'team1_q1_behinds': 'mse', 'team1_q2_goals': 'mse', 'team1_q2_behinds': 'mse',
                    'team1_q3_goals': 'mse', 'team1_q3_behinds': 'mse', 'team1_q4_goals': 'mse', 'team1_q4_behinds': 'mse',
                    'team2_q1_goals': 'mse', 'team2_q1_behinds': 'mse', 'team2_q2_goals': 'mse', 'team2_q2_behinds': 'mse',
                    'team2_q3_goals': 'mse', 'team2_q3_behinds': 'mse', 'team2_q4_goals': 'mse', 'team2_q4_behinds': 'mse'},
              metrics={'match_winner': 'accuracy', 'margin': 'mae',
                       'team1_goals': 'mae', 'team1_behinds': 'mae', 'team2_goals': 'mae', 'team2_behinds': 'mae',
                       'team1_q1_goals': 'mae', 'team1_q1_behinds': 'mae', 'team1_q2_goals': 'mae', 'team1_q2_behinds': 'mae',
                       'team1_q3_goals': 'mae', 'team1_q3_behinds': 'mae', 'team1_q4_goals': 'mae', 'team1_q4_behinds': 'mae',
                       'team2_q1_goals': 'mae', 'team2_q1_behinds': 'mae', 'team2_q2_goals': 'mae', 'team2_q2_behinds': 'mae',
                       'team2_q3_goals': 'mae', 'team2_q3_behinds': 'mae', 'team2_q4_goals': 'mae', 'team2_q4_behinds': 'mae'})

# Summary of the model
model.summary()

# Function to load and prepare data
def load_and_prepare_data():
    # Load match data
    match_data = pd.concat([pd.read_csv(f) for f in match_files])
    
    # Load lineup and player data
    lineup_data = pd.concat([pd.read_csv(f) for f in lineup_files])
    player_data = pd.concat([pd.read_csv(f) for f in player_files])
    
    # Merge data
    merged_data = pd.merge(match_data, lineup_data, left_on=['year', 'team_1_team_name', 'round_num'], right_on=['year', 'team_name', 'round_num'], how='left')
    merged_data = pd.merge(merged_data, player_data, left_on=['year', 'team_1_team_name', 'round_num'], right_on=['year', 'team', 'round'], how='left')
    
    # Extract features and targets
    X = merged_data.drop(columns=['match_winner', 'margin', 'team1_goals', 'team1_behinds', 'team2_goals', 'team2_behinds'])  # Adjust column names as needed
    y_winner = merged_data['match_winner']  # Adjust column name as needed
    y_margin = merged_data['margin']  # Adjust column name as needed
    y_team1_goals = merged_data['team1_goals']  # Adjust column name as needed
    y_team1_behinds = merged_data['team1_behinds']  # Adjust column name as needed
    y_team2_goals = merged_data['team2_goals']  # Adjust column name as needed
    y_team2_behinds = merged_data['team2_behinds']  # Adjust column name as needed
    
    return X, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds

# Load and prepare data
X, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds = load_and_prepare_data()

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train_winner, y_test_winner, y_train_margin, y_test_margin, y_train_team1_goals, y_test_team1_goals, y_train_team1_behinds, y_test_team1_behinds, y_train_team2_goals, y_test_team2_goals, y_train_team2_behinds, y_test_team2_behinds = train_test_split(
    X_normalized, y_winner, y_margin, y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, {'match_winner': y_train_winner, 'margin': y_train_margin,
                    'team1_goals': y_train_team1_goals, 'team1_behinds': y_train_team1_behinds,
                    'team2_goals': y_train_team2_goals, 'team2_behinds': y_train_team2_behinds,
                    'team1_q1_goals': y_train_team1_q1_goals, 'team1_q1_behinds': y_train_team1_q1_behinds,
                    'team1_q2_goals': y_train_team1_q2_goals, 'team1_q2_behinds': y_train_team1_q2_behinds,
                    'team1_q3_goals': y_train_team1_q3_goals, 'team1_q3_behinds': y_train_team1_q3_behinds,
                    'team1_q4_goals': y_train_team1_q4_goals, 'team1_q4_behinds': y_train_team1_q4_behinds,
                    'team2_q1_goals': y_train_team2_q1_goals, 'team2_q1_behinds': y_train_team2_q1_behinds,
                    'team2_q2_goals': y_train_team2_q2_goals, 'team2_q2_behinds': y_train_team2_q2_behinds,
                    'team2_q3_goals': y_train_team2_q3_goals, 'team2_q3_behinds': y_train_team2_q3_behinds,
                    'team2_q4_goals': y_train_team2_q4_goals, 'team2_q4_behinds': y_train_team2_q4_behinds},
          epochs=40, batch_size=32, validation_split=0.2)

# Evaluate the model
model.evaluate(X_test, {'match_winner': y_test_winner, 'margin': y_test_margin,
                        'team1_goals': y_test_team1_goals, 'team1_behinds': y_test_team1_behinds,
                        'team2_goals': y_test_team2_goals, 'team2_behinds': y_test_team2_behinds,
                        'team1_q1_goals': y_test_team1_q1_goals, 'team1_q1_behinds': y_test_team1_q1_behinds,
                        'team1_q2_goals': y_test_team1_q2_goals, 'team1_q2_behinds': y_test_team1_q2_behinds,
                        'team1_q3_goals': y_test_team1_q3_goals, 'team1_q3_behinds': y_test_team1_q3_behinds,
                        'team1_q4_goals': y_test_team1_q4_goals, 'team1_q4_behinds': y_test_team1_q4_behinds,
                        'team2_q1_goals': y_test_team2_q1_goals, 'team2_q1_behinds': y_test_team2_q1_behinds,
                        'team2_q2_goals': y_test_team2_q2_goals, 'team2_q2_behinds': y_test_team2_q2_behinds,
                        'team2_q3_goals': y_test_team2_q3_goals, 'team2_q3_behinds': y_test_team2_q3_behinds,
                        'team2_q4_goals': y_test_team2_q4_goals, 'team2_q4_behinds': y_test_team2_q4_behinds}) 