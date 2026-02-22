import os
import sys
import tensorflow as tf
import numpy as np
import json
import joblib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import custom layers
from model.custom_layers import (
    MeanPoolingLayer, BoundedOutputLayer, ScoreCalculationLayer,
    MarginCalculationLayer, WinProbabilityLayer, DrawProbabilityLayer,
    ProbabilityNormalizationLayer
)

# Import custom loss functions
from model.custom_loss import mse, mae

# Constants
EMBEDDING_DIM = 32
MAX_PLAYERS_PER_TEAM = 22
MAX_GOALS = 25
MIN_GOALS = 5
MAX_BEHINDS = 15
MIN_BEHINDS = 3

# Register these loss functions with keras
tf.keras.utils.get_custom_objects().update({
    'mse': mse,
    'mae': mae
})

def load_model_resources():
    """Load necessary resources for rebuilding the model"""
    output_dir = os.path.join(PROJECT_ROOT, 'model/output')
    
    # Load scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    else:
        print(f"Warning: Scaler not found at {scaler_path}")
        scaler = None
    
    # Load player index
    player_index_path = os.path.join(output_dir, 'player_index.json')
    if os.path.exists(player_index_path):
        with open(player_index_path, 'r') as f:
            player_index = json.load(f)
        print(f"Loaded player index with {len(player_index)} entries")
        # Make sure we have the maximum player index
        num_players = max(int(v) for v in player_index.values()) + 1
    else:
        print(f"Warning: Player index not found at {player_index_path}")
        num_players = 15000  # Use a safe default
    
    # Load feature columns
    feature_cols_path = os.path.join(output_dir, 'feature_cols.json')
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, 'r') as f:
            feature_cols = json.load(f)
        print(f"Loaded feature columns with {len(feature_cols)} features")
        feature_size = len(feature_cols)
    else:
        print(f"Warning: Feature columns not found at {feature_cols_path}")
        feature_size = 119  # Use a safe default
    
    # Load weights
    weights_path = os.path.join(output_dir, 'model.weights.h5')
    if os.path.exists(weights_path):
        print(f"Found weights at {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}")
        weights_path = None
    
    return scaler, player_index, feature_size, num_players, weights_path

def build_custom_model(feature_size, num_players):
    """Build a new model with custom layers instead of Lambda layers"""
    print(f"Building model with feature_size={feature_size} and num_players={num_players}")
    
    # Input layers
    match_input = tf.keras.layers.Input(shape=(feature_size,), name="match_input")
    team1_players_input = tf.keras.layers.Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team1_players_input", dtype=tf.int32)
    team2_players_input = tf.keras.layers.Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team2_players_input", dtype=tf.int32)
    
    # Enhanced features inputs
    metrics_size = 21  # Number of player statistics
    team1_enhanced_input = tf.keras.layers.Input(shape=(metrics_size,), name="team1_enhanced_input")
    team2_enhanced_input = tf.keras.layers.Input(shape=(metrics_size,), name="team2_enhanced_input")
    
    # Player embedding layer
    player_embedding = tf.keras.layers.Embedding(
        input_dim=num_players + 1,  # +1 for unknown players (0 index)
        output_dim=EMBEDDING_DIM, 
        name='player_embedding'
    )
    
    # Apply embeddings to both teams
    team1_embeddings = player_embedding(team1_players_input)
    team2_embeddings = player_embedding(team2_players_input)
    
    # Aggregate player embeddings
    team1_aggregated = MeanPoolingLayer(output_dim=EMBEDDING_DIM, name='team1_aggregated')(team1_embeddings)
    team2_aggregated = MeanPoolingLayer(output_dim=EMBEDDING_DIM, name='team2_aggregated')(team2_embeddings)
    
    # Normalize enhanced features
    team1_enhanced_norm = tf.keras.layers.BatchNormalization(name='team1_enhanced_norm')(team1_enhanced_input)
    team2_enhanced_norm = tf.keras.layers.BatchNormalization(name='team2_enhanced_norm')(team2_enhanced_input)
    
    # Concatenate match features with team embeddings and enhanced features
    combined = tf.keras.layers.Concatenate()([
        match_input, 
        team1_aggregated, team2_aggregated,
        team1_enhanced_norm, team2_enhanced_norm
    ])
    
    # Shared layers
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Team 1 goals prediction
    team1_goals_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_norm])
    team1_goals_branch = tf.keras.layers.Dense(64, activation='relu')(team1_goals_branch)
    team1_goals_branch = tf.keras.layers.Dense(32, activation='relu')(team1_goals_branch)
    team1_goals = BoundedOutputLayer(MIN_GOALS, MAX_GOALS, name='team1_goals')(team1_goals_branch)
    
    # Team 1 behinds prediction
    team1_behinds_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_norm])
    team1_behinds_branch = tf.keras.layers.Dense(64, activation='relu')(team1_behinds_branch)
    team1_behinds_branch = tf.keras.layers.Dense(32, activation='relu')(team1_behinds_branch)
    team1_behinds = BoundedOutputLayer(MIN_BEHINDS, MAX_BEHINDS, name='team1_behinds')(team1_behinds_branch)
    
    # Team 2 goals prediction
    team2_goals_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_norm])
    team2_goals_branch = tf.keras.layers.Dense(64, activation='relu')(team2_goals_branch)
    team2_goals_branch = tf.keras.layers.Dense(32, activation='relu')(team2_goals_branch)
    team2_goals = BoundedOutputLayer(MIN_GOALS, MAX_GOALS, name='team2_goals')(team2_goals_branch)
    
    # Team 2 behinds prediction
    team2_behinds_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_norm])
    team2_behinds_branch = tf.keras.layers.Dense(64, activation='relu')(team2_behinds_branch)
    team2_behinds_branch = tf.keras.layers.Dense(32, activation='relu')(team2_behinds_branch)
    team2_behinds = BoundedOutputLayer(MIN_BEHINDS, MAX_BEHINDS, name='team2_behinds')(team2_behinds_branch)
    
    # Calculate team scores using custom layer
    team1_score = ScoreCalculationLayer(name='team1_score')([team1_goals, team1_behinds])
    team2_score = ScoreCalculationLayer(name='team2_score')([team2_goals, team2_behinds])
    
    # Calculate margin using custom layer
    margin = MarginCalculationLayer(name='calculated_margin')([team1_score, team2_score])
    
    # Team 1 win probability
    team1_win_prob = WinProbabilityLayer(name='team1_win_prob')(margin)
    
    # Team 2 win probability (1 - team1_win_prob)
    team2_win_prob = tf.keras.layers.Lambda(
        lambda x: 1 - x,
        name='team2_win_prob'
    )(team1_win_prob)
    
    # Draw probability
    draw_prob = DrawProbabilityLayer(name='draw_prob')(margin)
    
    # Reshape probabilities for concatenation
    team1_win_reshaped = tf.keras.layers.Reshape((1,))(team1_win_prob)
    draw_prob_reshaped = tf.keras.layers.Reshape((1,))(draw_prob)
    team2_win_reshaped = tf.keras.layers.Reshape((1,))(team2_win_prob)
    
    # Concatenate probabilities
    match_winner = tf.keras.layers.Concatenate(name='match_winner')([
        team1_win_reshaped,
        draw_prob_reshaped,
        team2_win_reshaped
    ])
    
    # Create model
    model = tf.keras.models.Model(
        inputs=[match_input, team1_players_input, team2_players_input, team1_enhanced_input, team2_enhanced_input],
        outputs=[match_winner, team1_goals, team1_behinds, team2_goals, team2_behinds, margin]
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss={
            'match_winner': 'categorical_crossentropy',
            'team1_goals': mse,
            'team1_behinds': mse,
            'team2_goals': mse,
            'team2_behinds': mse,
            'calculated_margin': mse
        },
        metrics={
            'match_winner': ['accuracy', tf.keras.metrics.Precision(name='precision'), 
                             tf.keras.metrics.Recall(name='recall')],
            'team1_goals': mae,
            'team1_behinds': mae,
            'team2_goals': mae,
            'team2_behinds': mae,
            'calculated_margin': mae
        }
    )
    
    return model

def main():
    """Main function to rebuild and save the model"""
    print("Rebuilding model with custom layers")
    
    # Load resources
    scaler, player_index, feature_size, num_players, weights_path = load_model_resources()
    
    # Build model
    model = build_custom_model(feature_size, num_players)
    
    # Try to load weights
    if weights_path and os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("The model will use randomly initialized weights")
    
    # Model summary
    model.summary()
    
    # Save the rebuilt model
    output_dir = os.path.join(PROJECT_ROOT, 'model/output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to model directory using Keras format
    model_path = os.path.join(output_dir, 'custom_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path} in Keras format")
    
    # Save weights separately
    weights_path = os.path.join(output_dir, 'custom_model.weights.h5')
    model.save_weights(weights_path)
    print(f"Model weights saved to {weights_path}")
    
    # Save to GUI resources directory for the SimplePredictor
    gui_resources_dir = os.path.join(PROJECT_ROOT, 'gui/resources')
    os.makedirs(gui_resources_dir, exist_ok=True)
    
    gui_model_path = os.path.join(gui_resources_dir, 'gui_model.keras')
    model.save(gui_model_path)
    print(f"GUI model saved to {gui_model_path} in Keras format")
    
    # Also save in h5 format for compatibility
    h5_model_path = os.path.join(gui_resources_dir, 'gui_model.h5')
    model.save(h5_model_path)
    print(f"GUI model also saved in H5 format to {h5_model_path}")
    
    print("Model rebuilding complete!")

if __name__ == "__main__":
    main() 