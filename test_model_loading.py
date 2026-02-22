#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import json
import joblib
from pathlib import Path
import traceback
import sys

# Add the model directory to the path so we can import the BoundedOutputLayer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))
try:
    from model.bounded_output_layer import BoundedOutputLayer
except ImportError:
    print("Failed to import BoundedOutputLayer from model directory")
    try:
        from bounded_output_layer import BoundedOutputLayer
        print("Imported BoundedOutputLayer directly")
    except ImportError:
        print("Failed to import BoundedOutputLayer. Creating a fallback implementation.")
        # Define a fallback BoundedOutputLayer
        class BoundedOutputLayer(tf.keras.layers.Layer):
            def __init__(self, max_value, min_value, **kwargs):
                super(BoundedOutputLayer, self).__init__(**kwargs)
                self.max_value = max_value
                self.min_value = min_value
                self.range = max_value - min_value
                
            def build(self, input_shape):
                self.kernel = self.add_weight(
                    name='kernel',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
                    trainable=True
                )
                self.bias = self.add_weight(
                    name='bias',
                    shape=(1,),
                    initializer='zeros',
                    trainable=True
                )
                super(BoundedOutputLayer, self).build(input_shape)
            
            def call(self, inputs):
                raw_output = tf.matmul(inputs, self.kernel) + self.bias
                sigmoid_output = tf.sigmoid(raw_output)
                scaled_output = sigmoid_output * self.range + self.min_value
                return scaled_output
            
            def compute_output_shape(self, input_shape):
                return (input_shape[0], 1)
                
            def get_config(self):
                config = super(BoundedOutputLayer, self).get_config()
                config.update({
                    'max_value': self.max_value,
                    'min_value': self.min_value
                })
                return config

# Constants
EMBEDDING_DIM = 32
MAX_PLAYERS_PER_TEAM = 22
MAX_GOALS = 25.0
MIN_GOALS = 5.0
MAX_BEHINDS = 15.0
MIN_BEHINDS = 3.0

# Define pooling layer
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

def test_model_loading():
    print("\n===== TESTING MODEL LOADING =====")
    # Define paths
    model_dir = Path('model/output')
    model_path = model_dir / 'model.h5'
    model_weights_path = model_dir / 'model.weights.h5'
    best_model_path = model_dir / 'best_model.h5'
    scaler_path = model_dir / 'scaler.joblib'
    player_index_path = model_dir / 'player_index.json'
    feature_cols_path = model_dir / 'feature_cols.json'
    
    # Check which model files exist
    print(f"Model exists: {model_path.exists()}")
    print(f"Weights exist: {model_weights_path.exists()}")
    print(f"Best model exists: {best_model_path.exists()}")
    print(f"Scaler exists: {scaler_path.exists()}")
    print(f"Player index exists: {player_index_path.exists()}")
    print(f"Feature cols exists: {feature_cols_path.exists()}")
    
    # Load preprocessing objects
    try:
        print("\nLoading preprocessing objects:")
        
        # Load scaler
        if scaler_path.exists():
            print("Loading scaler...")
            scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully")
        else:
            print("Scaler not found")
            scaler = None
        
        # Load player index
        if player_index_path.exists():
            print("Loading player index...")
            with open(player_index_path, 'r') as f:
                player_index = json.load(f)
            print(f"Player index loaded successfully with {len(player_index)} entries")
        else:
            print("Player index not found")
            player_index = {}
        
        # Load feature columns
        if feature_cols_path.exists():
            print("Loading feature columns...")
            with open(feature_cols_path, 'r') as f:
                feature_cols = json.load(f)
            print(f"Feature columns loaded successfully: {feature_cols}")
        else:
            print("Feature columns not found")
            feature_cols = []
    except Exception as e:
        print(f"Error loading preprocessing objects: {e}")
        traceback.print_exc()
    
    # Register the custom layers
    print("\nRegistering custom layers...")
    tf.keras.utils.get_custom_objects()['MeanPoolingLayer'] = MeanPoolingLayer
    tf.keras.utils.get_custom_objects()['BoundedOutputLayer'] = BoundedOutputLayer
    
    # Define MSE and MAE functions 
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
        
    def mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Define custom objects
    custom_objects = {
        'MeanPoolingLayer': MeanPoolingLayer,
        'BoundedOutputLayer': BoundedOutputLayer,
        'mse': mse,
        'mae': mae,
        'precision': tf.keras.metrics.Precision(),
        'recall': tf.keras.metrics.Recall(),
        'mean_absolute_error': mae,
        'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError()
    }
    
    # Try to load the model
    model = None
    try:
        print("\nTrying to load model.h5...")
        model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        
        # Try best_model.h5 instead
        try:
            print("\nTrying to load best_model.h5 instead...")
            model = tf.keras.models.load_model(str(best_model_path), custom_objects=custom_objects)
            print("Best model loaded successfully!")
        except Exception as e2:
            print(f"Error loading best model: {e2}")
            traceback.print_exc()
    
    # If model loaded, show structure
    if model:
        print("\nModel structure:")
        model.summary()
        
        print("\nModel inputs:")
        for i, inp in enumerate(model.inputs):
            print(f"  Input {i}: {inp.name} - {inp.shape}")
        
        print("\nModel outputs:")
        for i, out in enumerate(model.outputs):
            print(f"  Output {i}: {out.name} - {out.shape}")
        
        # Try making a prediction
        print("\nTrying to make a prediction...")
        try:
            # Create dummy input data
            feature_size = len(feature_cols) if feature_cols else model.inputs[0].shape[1]
            match_features = np.zeros((1, feature_size))
            team1_players = np.zeros((1, MAX_PLAYERS_PER_TEAM), dtype=np.int32)
            team2_players = np.zeros((1, MAX_PLAYERS_PER_TEAM), dtype=np.int32)
            
            # Create enhanced features
            metrics_size = 21
            team1_enhanced = np.zeros((1, metrics_size))
            team2_enhanced = np.zeros((1, metrics_size))
            
            # Make prediction
            predictions = model.predict([
                match_features,
                team1_players,
                team2_players,
                team1_enhanced,
                team2_enhanced
            ], verbose=1)
            
            print("\nPrediction output shapes:")
            for i, pred in enumerate(predictions):
                print(f"  Output {i}: {pred.shape}")
            
            # Extract prediction values
            print("\nPrediction values:")
            match_winner_probs = predictions[0][0]
            team1_goals = predictions[1][0][0]
            team1_behinds = predictions[2][0][0]
            team2_goals = predictions[3][0][0]
            team2_behinds = predictions[4][0][0]
            margin = predictions[5][0][0]
            
            print(f"  Team 1 Goals: {team1_goals:.1f}")
            print(f"  Team 1 Behinds: {team1_behinds:.1f}")
            print(f"  Team 2 Goals: {team2_goals:.1f}")
            print(f"  Team 2 Behinds: {team2_behinds:.1f}")
            print(f"  Margin: {margin:.1f}")
            print(f"  Win probabilities: Team 1: {match_winner_probs[0]:.2f}, Draw: {match_winner_probs[1]:.2f}, Team 2: {match_winner_probs[2]:.2f}")
            
            print("\nPrediction successful!")
        except Exception as e:
            print(f"Error making prediction: {e}")
            traceback.print_exc()
    
    print("\n===== MODEL LOADING TEST COMPLETE =====")

if __name__ == "__main__":
    test_model_loading() 