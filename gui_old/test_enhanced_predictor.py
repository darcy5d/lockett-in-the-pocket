#!/usr/bin/env python3
import os
import sys
import json
import traceback
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import our enhanced predictor
try:
    from gui.enhanced_predictor import EnhancedPredictor
except ImportError as e:
    print(f"Error importing EnhancedPredictor: {e}")
    sys.exit(1)

def test_predictor():
    """Test the enhanced predictor with sample input data."""
    print("Creating EnhancedPredictor instance...")
    predictor = EnhancedPredictor()
    
    # Try to load preprocessing objects
    print("Loading preprocessing objects...")
    if not predictor.load_preprocessing():
        print("Warning: Failed to load preprocessing objects. Using defaults.")
    
    # Try to load the model first
    print("Attempting to load existing model...")
    if not predictor.load_model():
        print("Could not load existing model. Building new model...")
        if not predictor.build_model():
            print("Error: Failed to build model. Exiting.")
            return
    
    # Test with sample data for a match between two teams
    print("\n===== Making a sample prediction =====")
    
    # Sample teams
    team1_name = "Richmond"
    team2_name = "Collingwood"
    venue = "M.C.G."
    
    # Sample player IDs (just using sequential numbers for testing)
    team1_players = list(range(1, 23))  # 22 players
    team2_players = list(range(24, 46))  # 22 players
    
    print(f"Predicting match: {team1_name} vs {team2_name} at {venue}")
    print(f"Team 1 players: {len(team1_players)} players")
    print(f"Team 2 players: {len(team2_players)} players")
    
    # Make prediction
    try:
        match_features = predictor.prepare_match_features(team1_name, team2_name, venue)
        margin, match_winner_probs, team1_score, team2_score, debug_info = predictor.predict(
            match_features=match_features,
            team1_players=team1_players,
            team2_players=team2_players
        )
        
        if margin is None:
            print("Error: Prediction failed")
            return
            
        print("\n===== Prediction Results =====")
        # Extract prediction data
        team1_goals = debug_info.get('team1_goals', 0)
        team1_behinds = debug_info.get('team1_behinds', 0)
        team2_goals = debug_info.get('team2_goals', 0)
        team2_behinds = debug_info.get('team2_behinds', 0)
        
        # Determine winner based on margin
        if margin > 0:
            winner = team1_name
        elif margin < 0:
            winner = team2_name
            margin = abs(margin)  # Make margin positive for display
        else:
            winner = "Draw"
            
        # Print full prediction details
        print(f"Predicted Winner: {winner}")
        print(f"Predicted Score:")
        print(f"  {team1_name}: {team1_goals:.1f}.{team1_behinds:.1f} ({team1_score:.1f})")
        print(f"  {team2_name}: {team2_goals:.1f}.{team2_behinds:.1f} ({team2_score:.1f})")
        print(f"Predicted Margin: {abs(margin):.1f} points")
        
        # Print win probabilities
        if match_winner_probs is not None and len(match_winner_probs) == 3:
            print(f"Win Probability:")
            print(f"  {team1_name}: {match_winner_probs[0]*100:.1f}%")
            print(f"  Draw: {match_winner_probs[1]*100:.1f}%")
            print(f"  {team2_name}: {match_winner_probs[2]*100:.1f}%")
        
        # Print debug info
        print("\nDebug Information:")
        for key, value in debug_info.items():
            if isinstance(value, (float, int)):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, np.ndarray) and value.size <= 3:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value)}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_predictor() 