#!/usr/bin/env python3
import os
import sys
import json
import time
import tensorflow as tf
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing model functions
try:
    from model.neural_network_with_embeddings import load_and_prepare_data
    print("Successfully imported existing neural network model functions")
except ImportError:
    print("Warning: Could not import neural_network_with_embeddings, rebuilding from scratch")

# Import our enhanced predictor
from gui.enhanced_predictor import EnhancedPredictor, MAX_PLAYERS_PER_TEAM, EMBEDDING_DIM

def train_new_model():
    """Train a new model with bounded outputs using our EnhancedPredictor class."""
    print("\n===== Training New Enhanced Model with Bounded Outputs =====")
    
    # Create the predictor
    predictor = EnhancedPredictor()
    
    # Load preprocessing objects
    if not predictor.load_preprocessing():
        print("Warning: Failed to load preprocessing objects. Using defaults.")
    
    # Build the model architecture
    if not predictor.build_model():
        print("Error: Failed to build model. Exiting.")
        return False
    
    # Output paths
    output_dir = Path("model/output_enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "model.h5"
    
    # Try loading existing training data
    try:
        from model.neural_network_with_embeddings import load_and_prepare_data
        print("Loading existing training data...")
        
        # Assuming the load_and_prepare_data function returns the necessary data
        (X_match, team1_player_indices, team2_player_indices, team1_enhanced, 
         team2_enhanced, y_winner, y_margin, y_team1_goals, y_team1_behinds, 
         y_team2_goals, y_team2_behinds, scaler) = load_and_prepare_data()
        
        # Normalize the data
        X_match_normalized = scaler.transform(X_match)
        
        # Normalize enhanced player features
        from sklearn.preprocessing import StandardScaler
        team1_scaler = StandardScaler()
        team1_enhanced_normalized = team1_scaler.fit_transform(team1_enhanced)
        team2_scaler = StandardScaler()
        team2_enhanced_normalized = team2_scaler.fit_transform(team2_enhanced)
        
        # Split data into training and test sets
        from sklearn.model_selection import train_test_split
        (X_train_match, X_test_match, X_train_team1_players, X_test_team1_players,
         X_train_team2_players, X_test_team2_players, X_train_team1_enhanced, X_test_team1_enhanced,
         X_train_team2_enhanced, X_test_team2_enhanced, y_train_winner, y_test_winner,
         y_train_margin, y_test_margin, y_train_team1_goals, y_test_team1_goals,
         y_train_team1_behinds, y_test_team1_behinds, y_train_team2_goals, y_test_team2_goals,
         y_train_team2_behinds, y_test_team2_behinds) = train_test_split(
            X_match_normalized, team1_player_indices, team2_player_indices, 
            team1_enhanced_normalized, team2_enhanced_normalized, y_winner, y_margin, 
            y_team1_goals, y_team1_behinds, y_team2_goals, y_team2_behinds, 
            test_size=0.2, random_state=42
        )
        
        # One-hot encode winner categories for three classes
        # Convert 0, 0.5, 1 to proper indices for to_categorical
        from tensorflow.keras.utils import to_categorical
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
        
        print(f"Training data loaded with {len(X_train_match)} training samples and {len(X_test_match)} test samples")
        
        # Create training targets dictionary
        targets_dict = {
            'match_winner': y_train_winner, 
            'team1_goals': y_train_team1_goals, 
            'team1_behinds': y_train_team1_behinds,
            'team2_goals': y_train_team2_goals, 
            'team2_behinds': y_train_team2_behinds,
            'calculated_margin': y_train_margin  # Use the true margin for the calculated margin
        }

        # Test dictionary with targets
        test_targets_dict = {
            'match_winner': y_test_winner, 
            'team1_goals': y_test_team1_goals, 
            'team1_behinds': y_test_team1_behinds,
            'team2_goals': y_test_team2_goals, 
            'team2_behinds': y_test_team2_behinds,
            'calculated_margin': y_test_margin  # Use the true margin for the calculated margin
        }
        
        # Setup callbacks for training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, 
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor='val_loss', save_best_only=True, 
                save_weights_only=False, verbose=1
            )
        ]
        
        # Train the model
        print("\nTraining the model...")
        history = predictor.model.fit(
            [X_train_match, X_train_team1_players, X_train_team2_players, 
             X_train_team1_enhanced, X_train_team2_enhanced], 
            targets_dict,
            validation_data=(
                [X_test_match, X_test_team1_players, X_test_team2_players, 
                 X_test_team1_enhanced, X_test_team2_enhanced], 
                test_targets_dict
            ),
            epochs=20,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        print("\nEvaluating the model...")
        eval_result = predictor.model.evaluate(
            [X_test_match, X_test_team1_players, X_test_team2_players, 
             X_test_team1_enhanced, X_test_team2_enhanced], 
            test_targets_dict,
            verbose=1
        )
        
        # Save the model
        print(f"\nSaving model to {model_path}...")
        predictor.model.save(model_path)
        
        print("\nTraining complete!")
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_new_model() 