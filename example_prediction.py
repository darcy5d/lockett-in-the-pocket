#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from model.load_model import load_afl_model, prepare_match_inputs
from gui.simple_predictor import SimplePredictor

def main():
    """Example script showing how to use both the neural network model and SimplePredictor"""
    print("AFL Prediction Example")
    print("======================")

    # Method 1: Use the utility function to load the model
    print("\nMethod 1: Using the load_afl_model utility")
    print("-------------------------------------------")
    model, scaler, player_index, feature_cols = load_afl_model(
        model_path="model/output_compatible/model.h5",
        weights_path="model/output_compatible/model.weights.h5",
        scaler_path="model/output_compatible/scaler.joblib",
        player_index_path="model/output_compatible/player_index.json",
        feature_cols_path="model/output_compatible/feature_cols.json"
    )
    
    if model is None:
        print("Failed to load model directly. Try Method 2 with SimplePredictor.")
    else:
        print("Model loaded successfully!")
        
        # Example match data
        # For a real prediction, you would need real feature values and player indices
        match_features = np.zeros(len(feature_cols) if feature_cols else 119)
        team1_player_indices = [0] * 22  # Replace with real player indices
        team2_player_indices = [0] * 22  # Replace with real player indices
        
        # No enhanced features for this example (the model will use zeros)
        
        # Prepare inputs
        inputs = prepare_match_inputs(
            match_features=match_features,
            team1_players=team1_player_indices,
            team2_players=team2_player_indices,
            scaler=scaler
        )
        
        # Make prediction
        predictions = model.predict(inputs)
        
        # Process predictions
        match_winner_probs = predictions[0][0]  # [team1_win, draw, team2_win] probabilities
        team1_goals = predictions[1][0][0]      # team1 goals
        team1_behinds = predictions[2][0][0]    # team1 behinds
        team2_goals = predictions[3][0][0]      # team2 goals
        team2_behinds = predictions[4][0][0]    # team2 behinds
        margin = predictions[5][0][0]           # calculated margin
        
        # Display results
        print("\nPrediction Results:")
        print(f"Team 1 Goals: {team1_goals:.1f}")
        print(f"Team 1 Behinds: {team1_behinds:.1f}")
        print(f"Team 2 Goals: {team2_goals:.1f}")
        print(f"Team 2 Behinds: {team2_behinds:.1f}")
        print(f"Margin (Team 1 - Team 2): {margin:.1f}")
        
        winner_idx = np.argmax(match_winner_probs)
        winner = "Team 1" if winner_idx == 0 else ("Draw" if winner_idx == 1 else "Team 2")
        print(f"Predicted Winner: {winner}")
        print(f"Win Probabilities: Team 1: {match_winner_probs[0]:.2f}, Draw: {match_winner_probs[1]:.2f}, Team 2: {match_winner_probs[2]:.2f}")
    
    # Method 2: Use SimplePredictor with custom paths
    print("\nMethod 2: Using SimplePredictor")
    print("-------------------------------")
    # Create a new SimplePredictor
    predictor = SimplePredictor()
    
    # Override model directory with a Path object
    predictor.model_dir = Path(os.path.abspath("model/output_compatible"))
    
    # Reset all paths based on the new model_dir
    predictor.model_path = predictor.model_dir / "model.h5"
    predictor.model_weights_path = predictor.model_dir / "model.weights.h5"
    predictor.scaler_path = predictor.model_dir / "scaler.joblib"
    predictor.player_index_path = predictor.model_dir / "player_index.json"
    predictor.feature_cols_path = predictor.model_dir / "feature_cols.json"
    
    print(f"Looking for model at: {predictor.model_path}")
    print(f"Looking for weights at: {predictor.model_weights_path}")
    print(f"Looking for scaler at: {predictor.scaler_path}")
    print(f"Looking for player index at: {predictor.player_index_path}")
    
    # Re-load the preprocessing objects with the new paths
    predictor.load_preprocessing()
    
    # Try to load or build the model
    predictor.load_model()
    if not predictor.model:
        predictor.build_model()
    
    if predictor.model_loaded:
        print("SimplePredictor loaded successfully!")
        
        # Example match data
        team1_name = "Geelong"
        team2_name = "Richmond"
        venue = "M.C.G."
        
        # Example player lists - in reality, you would use real player names/IDs
        home_players = ["Player1", "Player2", "Player3"] + ["Player" + str(i) for i in range(4, 23)]
        away_players = ["Player24", "Player25", "Player26"] + ["Player" + str(i) for i in range(27, 46)]
        
        # Example player_names_dict - maps player names to IDs
        # In reality, you would use your actual player data
        player_names_dict = {f"Player{i}": str(i) for i in range(1, 100)}
        
        # Prepare input for prediction
        match_inputs = predictor.prepare_input(
            home_team=team1_name,
            away_team=team2_name,
            venue=venue,
            home_players=home_players,
            away_players=away_players,
            player_names_dict=player_names_dict
        )
        
        # Get prediction directly
        prediction_result = predictor.predict_match(
            team1_name=team1_name,
            team2_name=team2_name,
            venue=venue,
            team1_players=home_players,
            team2_players=away_players
        )
        
        # Display results
        print("\nSimplePredictor Results:")
        if "error" in prediction_result:
            print(f"Error: {prediction_result['error']}")
        else:
            print(f"Home Team ({team1_name}): {prediction_result['home_team_goals']}G {prediction_result['home_team_behinds']}B ({prediction_result['home_team_score']})")
            print(f"Away Team ({team2_name}): {prediction_result['away_team_goals']}G {prediction_result['away_team_behinds']}B ({prediction_result['away_team_score']})")
            print(f"Margin: {prediction_result['margin']:.1f}")
            print(f"Predicted Winner: {prediction_result['winner']}")
            print(f"Win Probabilities: {team1_name}: {prediction_result['home_team_win_probability']:.2f}, " + 
                  f"Draw: {prediction_result['draw_probability']:.2f}, {team2_name}: {prediction_result['away_team_win_probability']:.2f}")

    print("\nComparison:")
    print("Both approaches provide similar functionality but cater to different use cases:")
    print("1. Using load_afl_model directly is better for batch predictions and integration into other systems.")
    print("2. SimplePredictor provides a higher-level API with richer features for GUI applications.")
    print("Both use the same underlying model architecture with bounded outputs for realistic predictions.")

if __name__ == "__main__":
    main() 