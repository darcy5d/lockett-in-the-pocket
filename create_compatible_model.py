#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import joblib
import json
from pathlib import Path

# Import our BoundedOutputLayer
from model.bounded_output_layer import BoundedOutputLayer

# Constants
EMBEDDING_DIM = 32
MAX_PLAYERS_PER_TEAM = 22
MAX_GOALS = 25.0
MIN_GOALS = 5.0
MAX_BEHINDS = 15.0
MIN_BEHINDS = 3.0

# Create a custom MeanPoolingLayer to match the one used in both models
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

def build_compatible_model():
    """Build a model architecture compatible with SimplePredictor"""
    
    # Register the custom layers
    tf.keras.utils.get_custom_objects()['MeanPoolingLayer'] = MeanPoolingLayer
    tf.keras.utils.get_custom_objects()['BoundedOutputLayer'] = BoundedOutputLayer
    
    # Define dimensions
    feature_size = 13  # Match what's in feature_cols.json
    num_players = 13207  # Match what's in player_index.json
    enhanced_features_size = 21  # Standard metrics size
    
    # Input layers
    match_input = tf.keras.layers.Input(shape=(feature_size,), name="match_input")
    team1_players_input = tf.keras.layers.Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team1_players_input", dtype=tf.int32)
    team2_players_input = tf.keras.layers.Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team2_players_input", dtype=tf.int32)
    team1_enhanced_input = tf.keras.layers.Input(shape=(enhanced_features_size,), name="team1_enhanced_input")
    team2_enhanced_input = tf.keras.layers.Input(shape=(enhanced_features_size,), name="team2_enhanced_input")
    
    # Player embedding layer
    player_embedding = tf.keras.layers.Embedding(
        input_dim=num_players + 1,  # +1 for unknown players (0 index)
        output_dim=EMBEDDING_DIM, 
        name='player_embedding'
    )
    
    # Apply embeddings to both teams
    team1_embeddings = player_embedding(team1_players_input)
    team2_embeddings = player_embedding(team2_players_input)
    
    # Aggregate player embeddings using our custom layer with defined output shape
    team1_aggregated = MeanPoolingLayer(output_dim=EMBEDDING_DIM, name='team1_aggregated')(team1_embeddings)
    team2_aggregated = MeanPoolingLayer(output_dim=EMBEDDING_DIM, name='team2_aggregated')(team2_embeddings)
    
    # Normalize enhanced features to prevent extreme values
    team1_enhanced_norm = tf.keras.layers.BatchNormalization(name='team1_enhanced_norm')(team1_enhanced_input)
    team2_enhanced_norm = tf.keras.layers.BatchNormalization(name='team2_enhanced_norm')(team2_enhanced_input)
    
    # Concatenate inputs
    combined = tf.keras.layers.Concatenate()([
        match_input, 
        team1_aggregated, team2_aggregated,
        team1_enhanced_norm, team2_enhanced_norm
    ])
    
    # Shared layers
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Branch for Team 1 goals prediction
    team1_goals_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_norm])
    team1_goals_branch = tf.keras.layers.Dense(64, activation='relu')(team1_goals_branch)
    team1_goals_branch = tf.keras.layers.Dense(32, activation='relu')(team1_goals_branch)
    team1_goals = BoundedOutputLayer(MAX_GOALS, MIN_GOALS, name='team1_goals')(team1_goals_branch)
    
    # Branch for Team 1 behinds prediction
    team1_behinds_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_norm])
    team1_behinds_branch = tf.keras.layers.Dense(64, activation='relu')(team1_behinds_branch)
    team1_behinds_branch = tf.keras.layers.Dense(32, activation='relu')(team1_behinds_branch)
    team1_behinds = BoundedOutputLayer(MAX_BEHINDS, MIN_BEHINDS, name='team1_behinds')(team1_behinds_branch)
    
    # Branch for Team 2 goals prediction
    team2_goals_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_norm])
    team2_goals_branch = tf.keras.layers.Dense(64, activation='relu')(team2_goals_branch)
    team2_goals_branch = tf.keras.layers.Dense(32, activation='relu')(team2_goals_branch)
    team2_goals = BoundedOutputLayer(MAX_GOALS, MIN_GOALS, name='team2_goals')(team2_goals_branch)
    
    # Branch for Team 2 behinds prediction
    team2_behinds_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_norm])
    team2_behinds_branch = tf.keras.layers.Dense(64, activation='relu')(team2_behinds_branch)
    team2_behinds_branch = tf.keras.layers.Dense(32, activation='relu')(team2_behinds_branch)
    team2_behinds = BoundedOutputLayer(MAX_BEHINDS, MIN_BEHINDS, name='team2_behinds')(team2_behinds_branch)
    
    # Calculate team scores from goals and behinds
    team1_score = tf.keras.layers.Lambda(lambda x: x[0]*6 + x[1], name='team1_score')([team1_goals, team1_behinds])
    team2_score = tf.keras.layers.Lambda(lambda x: x[0]*6 + x[1], name='team2_score')([team2_goals, team2_behinds])
    
    # Calculate derived margin from scores (team1_score - team2_score)
    calculated_margin = tf.keras.layers.Lambda(lambda x: x[0] - x[1], name='calculated_margin')([team1_score, team2_score])
    
    # Convert margin to win probability using a sigmoid function with temperature scaling
    winner_logit = tf.keras.layers.Lambda(lambda x: x * 0.1, name='margin_scaled')(calculated_margin)
    team1_win_prob = tf.keras.layers.Lambda(lambda x: tf.sigmoid(x), name='team1_win_prob')(winner_logit)
    team2_win_prob = tf.keras.layers.Lambda(lambda x: 1 - tf.sigmoid(x), name='team2_win_prob')(winner_logit)
    
    # Draw probability is highest when margin is close to 0
    draw_prob = tf.keras.layers.Lambda(lambda x: tf.exp(-tf.square(x) * 5.0), name='draw_prob')(winner_logit)
    
    # Normalize probabilities to sum to 1
    total_prob = tf.keras.layers.Lambda(lambda x: x[0] + x[1] + x[2], name='total_prob')([team1_win_prob, draw_prob, team2_win_prob])
    team1_win_prob_norm = tf.keras.layers.Lambda(lambda x: x[0] / x[1], name='team1_win_prob_norm')([team1_win_prob, total_prob])
    draw_prob_norm = tf.keras.layers.Lambda(lambda x: x[0] / x[1], name='draw_prob_norm')([draw_prob, total_prob])
    team2_win_prob_norm = tf.keras.layers.Lambda(lambda x: x[0] / x[1], name='team2_win_prob_norm')([team2_win_prob, total_prob])
    
    # Reshape tensors
    team1_win_reshaped = tf.keras.layers.Reshape((1,))(team1_win_prob_norm)
    draw_prob_reshaped = tf.keras.layers.Reshape((1,))(draw_prob_norm)
    team2_win_reshaped = tf.keras.layers.Reshape((1,))(team2_win_prob_norm)
    
    # Concatenate to form match winner output
    match_winner = tf.keras.layers.Concatenate(name='match_winner')([
        team1_win_reshaped, 
        draw_prob_reshaped,
        team2_win_reshaped
    ])
    
    # Create model
    model = tf.keras.models.Model(
        inputs=[match_input, team1_players_input, team2_players_input, team1_enhanced_input, team2_enhanced_input],
        outputs=[match_winner, team1_goals, team1_behinds, team2_goals, team2_behinds, calculated_margin]
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss={
            'match_winner': 'categorical_crossentropy',
            'team1_goals': 'mse',
            'team1_behinds': 'mse',
            'team2_goals': 'mse',
            'team2_behinds': 'mse',
            'calculated_margin': 'mse'
        },
        metrics={
            'match_winner': ['accuracy'],
            'team1_goals': ['mae'],
            'team1_behinds': ['mae'],
            'team2_goals': ['mae'],
            'team2_behinds': ['mae'],
            'calculated_margin': ['mae']
        }
    )
    
    return model

def train_and_save_model():
    """Train a simple model and save it in the format expected by SimplePredictor"""
    print("Building a compatible model architecture...")
    model = build_compatible_model()
    
    model.summary()
    
    # Create dummy training data
    num_samples = 1000
    feature_size = 13
    
    # Generate random input data
    X_match = np.random.randn(num_samples, feature_size)
    team1_players = np.random.randint(0, 13207, size=(num_samples, MAX_PLAYERS_PER_TEAM))
    team2_players = np.random.randint(0, 13207, size=(num_samples, MAX_PLAYERS_PER_TEAM))
    team1_enhanced = np.random.randn(num_samples, 21)
    team2_enhanced = np.random.randn(num_samples, 21)
    
    # Generate random target data
    match_winner = np.zeros((num_samples, 3))
    match_winner[:, 0] = np.random.uniform(0, 1, num_samples)
    match_winner[:, 1] = np.random.uniform(0, 0.1, num_samples)
    match_winner[:, 2] = 1 - match_winner[:, 0] - match_winner[:, 1]
    # Normalize to make sure they sum to 1
    match_winner = match_winner / np.sum(match_winner, axis=1, keepdims=True)
    
    team1_goals = np.random.uniform(MIN_GOALS, MAX_GOALS, size=(num_samples, 1))
    team1_behinds = np.random.uniform(MIN_BEHINDS, MAX_BEHINDS, size=(num_samples, 1))
    team2_goals = np.random.uniform(MIN_GOALS, MAX_GOALS, size=(num_samples, 1))
    team2_behinds = np.random.uniform(MIN_BEHINDS, MAX_BEHINDS, size=(num_samples, 1))
    
    # Calculate margins
    team1_score = team1_goals * 6 + team1_behinds
    team2_score = team2_goals * 6 + team2_behinds
    calculated_margin = team1_score - team2_score
    
    # Create target dictionary
    targets = {
        'match_winner': match_winner,
        'team1_goals': team1_goals,
        'team1_behinds': team1_behinds,
        'team2_goals': team2_goals,
        'team2_behinds': team2_behinds,
        'calculated_margin': calculated_margin
    }
    
    # Train the model for just a few epochs
    print("Training model with dummy data...")
    model.fit(
        [X_match, team1_players, team2_players, team1_enhanced, team2_enhanced],
        targets,
        epochs=2,
        batch_size=32
    )
    
    # Save the model
    output_dir = Path("model/output_compatible")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as .h5 format
    model_path = output_dir / "model.h5"
    model.save(str(model_path), save_format="h5")
    print(f"Model saved to {model_path}")
    
    # Save weights separately
    weights_path = output_dir / "model.weights.h5"
    model.save_weights(str(weights_path))
    print(f"Weights saved to {weights_path}")
    
    # Create and save a dummy scaler
    scaler = joblib.load("model/output/scaler.joblib")
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Copy the player index from the trained model
    with open("model/output/player_index.json", "r") as f:
        player_index = json.load(f)
    
    player_index_path = output_dir / "player_index.json"
    with open(player_index_path, "w") as f:
        json.dump(player_index, f)
    print(f"Player index saved to {player_index_path}")
    
    # Copy the feature columns from the trained model
    with open("model/output/feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    
    feature_cols_path = output_dir / "feature_cols.json"
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"Feature columns saved to {feature_cols_path}")
    
    print("Done! The model is now ready to be used with SimplePredictor.")

if __name__ == "__main__":
    train_and_save_model() 