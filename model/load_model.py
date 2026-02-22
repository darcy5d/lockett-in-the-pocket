import os
import tensorflow as tf
import numpy as np
import json
import joblib
from pathlib import Path
from model.bounded_output_layer import BoundedOutputLayer
from tensorflow.keras.models import load_model

# Constants matching those in neural_network_with_embeddings.py
EMBEDDING_DIM = 32
MAX_PLAYERS_PER_TEAM = 22

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

def get_project_root():
    """Get the project root directory"""
    # Try different approaches to find the project root
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # This file is in model/, so the parent is the project root
    project_root = current_dir.parent
    
    # Verify by checking if certain expected directories exist
    if (project_root / 'model').exists() and (project_root / 'gui').exists():
        return project_root
    
    # If verification fails, return current directory as a fallback
    return current_dir

def load_afl_model(model_path=None, weights_path=None, scaler_path=None, player_index_path=None, feature_cols_path=None):
    """
    Load the AFL prediction model and its preprocessing objects
    
    Args:
        model_path (str, optional): Path to the model file (.h5)
        weights_path (str, optional): Path to the model weights file (.h5)
        scaler_path (str, optional): Path to the scaler file (.joblib)
        player_index_path (str, optional): Path to the player index file (.json)
        feature_cols_path (str, optional): Path to the feature columns file (.json)
        
    Returns:
        tuple: (model, scaler, player_index, feature_cols)
    """
    # Get project root
    project_root = get_project_root()
    
    # Define default paths if not provided
    output_dir = project_root / 'model/output'
    model_path = model_path or output_dir / 'model.h5'
    weights_path = weights_path or output_dir / 'model.weights.h5'
    scaler_path = scaler_path or output_dir / 'scaler.joblib'
    player_index_path = player_index_path or output_dir / 'player_index.json'
    feature_cols_path = feature_cols_path or output_dir / 'feature_cols.json'
    
    # Convert Path objects to strings if needed
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(weights_path, Path):
        weights_path = str(weights_path)
    if isinstance(scaler_path, Path):
        scaler_path = str(scaler_path)
    if isinstance(player_index_path, Path):
        player_index_path = str(player_index_path)
    if isinstance(feature_cols_path, Path):
        feature_cols_path = str(feature_cols_path)
    
    # Register custom objects
    custom_objects = {
        'MeanPoolingLayer': MeanPoolingLayer,
        'BoundedOutputLayer': BoundedOutputLayer,
    }
    
    # Load model
    model = None
    try:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model = load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully!")
        elif os.path.exists(weights_path):
            # If model.h5 doesn't exist but weights do, try to build model and load weights
            # This would require rebuilding the model architecture (not implemented here)
            print(f"Model file not found, but weights found at {weights_path}")
            print("Loading weights requires model architecture to be defined first")
            print("Please use SimplePredictor class for loading with weights only")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Load scaler
    scaler = None
    try:
        if os.path.exists(scaler_path):
            print(f"Loading scaler from {scaler_path}...")
            scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading scaler: {e}")
    
    # Load player index
    player_index = None
    try:
        if os.path.exists(player_index_path):
            print(f"Loading player index from {player_index_path}...")
            with open(player_index_path, 'r') as f:
                player_index = json.load(f)
            print(f"Player index loaded successfully with {len(player_index)} players!")
    except Exception as e:
        print(f"Error loading player index: {e}")
    
    # Load feature columns
    feature_cols = None
    try:
        if os.path.exists(feature_cols_path):
            print(f"Loading feature columns from {feature_cols_path}...")
            with open(feature_cols_path, 'r') as f:
                feature_cols = json.load(f)
            print(f"Feature columns loaded successfully with {len(feature_cols)} features!")
    except Exception as e:
        print(f"Error loading feature columns: {e}")
    
    return model, scaler, player_index, feature_cols

def prepare_match_inputs(match_features, team1_players, team2_players, team1_enhanced=None, team2_enhanced=None, scaler=None):
    """
    Prepare inputs for the model in the correct format
    
    Args:
        match_features (ndarray): Match features
        team1_players (list): List of player IDs for team 1
        team2_players (list): List of player IDs for team 2
        team1_enhanced (ndarray, optional): Enhanced features for team 1
        team2_enhanced (ndarray, optional): Enhanced features for team 2
        scaler (object, optional): StandardScaler object for normalizing match features
    
    Returns:
        tuple: Inputs ready for model prediction
    """
    # Ensure match features are 2D
    match_features = np.array(match_features).reshape(1, -1)
    
    # Apply scaler if provided
    if scaler is not None:
        match_features = scaler.transform(match_features)
    
    # Prepare player indices
    team1_players = np.array(team1_players).reshape(1, -1)
    team2_players = np.array(team2_players).reshape(1, -1)
    
    # Ensure proper shape and size
    if team1_players.shape[1] < MAX_PLAYERS_PER_TEAM:
        pad_width = ((0, 0), (0, MAX_PLAYERS_PER_TEAM - team1_players.shape[1]))
        team1_players = np.pad(team1_players, pad_width, 'constant')
    elif team1_players.shape[1] > MAX_PLAYERS_PER_TEAM:
        team1_players = team1_players[:, :MAX_PLAYERS_PER_TEAM]
    
    if team2_players.shape[1] < MAX_PLAYERS_PER_TEAM:
        pad_width = ((0, 0), (0, MAX_PLAYERS_PER_TEAM - team2_players.shape[1]))
        team2_players = np.pad(team2_players, pad_width, 'constant')
    elif team2_players.shape[1] > MAX_PLAYERS_PER_TEAM:
        team2_players = team2_players[:, :MAX_PLAYERS_PER_TEAM]
    
    # Handle enhanced features
    metrics_size = 21  # Default number of metrics
    
    if team1_enhanced is None:
        team1_enhanced = np.zeros((1, metrics_size))
    else:
        team1_enhanced = np.array(team1_enhanced).reshape(1, -1)
    
    if team2_enhanced is None:
        team2_enhanced = np.zeros((1, metrics_size))
    else:
        team2_enhanced = np.array(team2_enhanced).reshape(1, -1)
    
    return [match_features, team1_players, team2_players, team1_enhanced, team2_enhanced]

if __name__ == "__main__":
    # Example usage
    model, scaler, player_index, feature_cols = load_afl_model()
    
    if model is not None:
        print("\nModel successfully loaded and ready for predictions!")
        print(f"Model inputs: {[inp.name for inp in model.inputs]}")
        print(f"Model outputs: {[out.name for out in model.outputs]}")
    else:
        print("\nFailed to load model.") 