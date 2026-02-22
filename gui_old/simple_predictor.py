#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
import json
import joblib
from pathlib import Path
import random
import traceback
import pandas as pd
import datetime
import re
import keras
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_predictor')

# Add the model directory to the path so we can import the BoundedOutputLayer
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))
try:
    from bounded_output_layer import BoundedOutputLayer
except ImportError:
    # Define a fallback BoundedOutputLayer in case the import fails
    class BoundedOutputLayer(tf.keras.layers.Layer):
        def __init__(self, min_value, max_value, **kwargs):
            super(BoundedOutputLayer, self).__init__(**kwargs)
            self.min_value = min_value
            self.max_value = max_value
            
        def build(self, input_shape):
            super(BoundedOutputLayer, self).build(input_shape)
            
        def call(self, inputs):
            scaled_sigmoid = tf.sigmoid(inputs)
            return self.min_value + (self.max_value - self.min_value) * scaled_sigmoid
            
        def get_config(self):
            config = super(BoundedOutputLayer, self).get_config()
            config.update({
                'min_value': self.min_value,
                'max_value': self.max_value
            })
            return config

# Constants
EMBEDDING_DIM = 32
MAX_PLAYERS_PER_TEAM = 22
CURRENT_YEAR = 2023  # Current year for predictions

# Define these constants to match the main model
MAX_GOALS = 25.0  # Maximum realistic goals 
MAX_BEHINDS = 15.0  # Maximum realistic behinds
MIN_GOALS = 5.0    # Minimum realistic goals
MIN_BEHINDS = 3.0  # Minimum realistic behinds

class MeanPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, **kwargs):  # Use 32 directly to match exactly with training code
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

# Add import for FallbackPredictor
from fallback_predictor import FallbackPredictor

class SimplePredictor:
    def __init__(self):
        """Initialize the SimplePredictor"""
        # Standard setup
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.gui_resources_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
        self.model_dir = os.path.join(self.project_root, 'model/output')
        os.makedirs(self.gui_resources_dir, exist_ok=True)
        
        # Initialize attributes
        self.model = None
        self.scaler = None
        self.player_index = None
        self.feature_cols = None
        
        # Look for GUI model
        self.gui_model_path = os.path.join(self.gui_resources_dir, 'gui_model.h5')
        logger.info(f"Looking for GUI model at: {self.gui_model_path}")
        
        # Look for model in model directory
        self.model_path = os.path.join(self.model_dir, 'model.h5')
        self.weights_path = os.path.join(self.model_dir, 'model.weights.h5')
        logger.info(f"Looking for model at: {self.model_path}")
        logger.info(f"Looking for weights at: {self.weights_path}")
        
        # Initialize the fallback predictor
        self.fallback_predictor = FallbackPredictor()
        self.using_fallback = False
        
        # Initialize preprocessing
        self.initialize_team_venue_indices()
        
        # Try to load preprocessing data
        if self.load_preprocessing():
            # Try to load the model
            if self.load_model():
                logger.info("SimplePredictor initialized successfully with neural network model")
            else:
                logger.warning("Failed to initialize SimplePredictor with the real model. Using fallback prediction logic.")
                self.using_fallback = True
        else:
            logger.warning("Failed to load preprocessing data. Using fallback prediction logic.")
            self.using_fallback = True

    def initialize_team_venue_indices(self):
        """Initialize team and venue indices from available data."""
        # Initialize empty indices
        self.team_index = {}
        self.venue_index = {}
        
        # Standard AFL team names 
        team_names = [
            "Adelaide", "Brisbane Lions", "Carlton", "Collingwood", "Essendon", 
            "Fremantle", "Geelong", "Gold Coast", "Greater Western Sydney", 
            "Hawthorn", "Melbourne", "North Melbourne", "Port Adelaide", 
            "Richmond", "St Kilda", "Sydney", "West Coast", "Western Bulldogs"
        ]
        
        # Standard AFL venues
        venues = [
            "Adelaide Oval", "Bellerive Oval", "Carrara", "Docklands", 
            "Eureka Stadium", "Gabba", "Kardinia Park", "M.C.G.", 
            "Manuka Oval", "Marrara Oval", "Norwood Oval", "Perth Stadium", 
            "S.C.G.", "Summit Sports Park", "Sydney Showground", 
            "Traeger Park", "York Park"
        ]
        
        # Create index for teams (starting at index 0)
        for i, team in enumerate(team_names):
            self.team_index[team] = i
            
        # Create index for venues (starting at index after teams)
        offset = len(team_names)
        for i, venue in enumerate(venues):
            self.venue_index[venue] = i + offset

    def load_preprocessing(self):
        """
        Load preprocessing objects (scaler, player index, feature columns).
        
        Returns:
            bool: True if preprocessing objects loaded successfully, False otherwise
        """
        try:
            # Define paths to preprocessing objects
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            player_index_path = os.path.join(self.model_dir, 'player_index.json')
            feature_cols_path = os.path.join(self.model_dir, 'feature_cols.json')
            
            logger.info(f"Loading scaler from {scaler_path}...")
            
            # Load scaler
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning(f"Scaler not found at {scaler_path}")
                return False
            
            # Load player index
            logger.info(f"Loading player index from {player_index_path}...")
            if os.path.exists(player_index_path):
                with open(player_index_path, 'r') as f:
                    self.player_index = json.load(f)
                logger.info(f"Player index loaded successfully with {len(self.player_index)} entries")
            else:
                logger.warning(f"Player index not found at {player_index_path}")
                return False
            
            # Load feature columns
            logger.info(f"Loading feature columns from {feature_cols_path}...")
            if os.path.exists(feature_cols_path):
                with open(feature_cols_path, 'r') as f:
                    self.feature_cols = json.load(f)
                logger.info(f"Feature columns loaded successfully: {self.feature_cols}")
            else:
                logger.warning(f"Feature columns not found at {feature_cols_path}")
                return False
            
            logger.info("Preprocessing objects loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error loading preprocessing objects: {str(e)}")
            return False
    
    def load_model(self):
        """
        Load the trained model from file.
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Register only the custom layers actually used
            tf.keras.utils.get_custom_objects().update({
                'MeanPoolingLayer': MeanPoolingLayer,
                'BoundedOutputLayer': BoundedOutputLayer
            })
            custom_objects = {
                'MeanPoolingLayer': MeanPoolingLayer,
                'BoundedOutputLayer': BoundedOutputLayer
            }
            # Search for model in various locations and formats
            model_paths = [
                os.path.join(self.gui_resources_dir, 'gui_model'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model/output/custom_model'),
                os.path.join(self.gui_resources_dir, 'gui_model.keras'),
                os.path.join(self.gui_resources_dir, 'gui_model.h5'),
                os.path.join(self.gui_resources_dir, 'simple_predictor_model.h5'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model/output/custom_model.h5')
            ]
            for model_path in model_paths:
                if os.path.exists(model_path):
                    logger.info(f"Loading model from {model_path}")
                    try:
                        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                        logger.info("Model loaded successfully")
                        logger.info(f"Model input shape: {[inp.shape for inp in self.model.inputs]}")
                        logger.info(f"Model output shape: {[out.shape for out in self.model.outputs]}")
                        return True
                    except Exception as load_error:
                        logger.warning(f"Failed to load model from {model_path}: {str(load_error)}")
            logger.error("No model found in any of the expected locations")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}\n{traceback.format_exc()}")
            return False

    def build_model(self):
        """Build the model with the same architecture as neural_network_with_embeddings.py."""
        try:
            # Register the custom layers
            tf.keras.utils.get_custom_objects()['MeanPoolingLayer'] = MeanPoolingLayer
            tf.keras.utils.get_custom_objects()['BoundedOutputLayer'] = BoundedOutputLayer
            
            # Input layers with proper shapes
            match_features_size = len(self.feature_cols) if self.feature_cols else 119  # Updated default size if not available
            match_input = tf.keras.layers.Input(shape=(match_features_size,), name="match_input")
            team1_players_input = tf.keras.layers.Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team1_players_input", dtype=tf.int32)
            team2_players_input = tf.keras.layers.Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team2_players_input", dtype=tf.int32)
            
            # Get player stats metrics
            metrics = [
                'kicks', 'marks', 'handballs', 'goals', 'behinds', 
                'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances',
                'clangers', 'free_kicks_for', 'free_kicks_against', 
                'contested_possessions', 'uncontested_possessions', 'contested_marks',
                'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist',
                'percentage_of_game_played'
            ]
            
            # Add inputs for enhanced player features
            team1_enhanced_input = tf.keras.layers.Input(shape=(len(metrics),), name="team1_enhanced_input")
            team2_enhanced_input = tf.keras.layers.Input(shape=(len(metrics),), name="team2_enhanced_input")
            
            # Get the maximum player index for embedding dimension
            max_player_index = max(self.player_index.values()) if self.player_index else 10000
            
            # Player embedding layer
            player_embedding = tf.keras.layers.Embedding(
                input_dim=max_player_index + 1,  # +1 for unknown players (0 index)
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
            
            # Concatenate match features with team embeddings and enhanced features
            combined = tf.keras.layers.Concatenate()([
                match_input, 
                team1_aggregated, team2_aggregated,
                team1_enhanced_norm, team2_enhanced_norm
            ])
            
            # Shared layers
            x = tf.keras.layers.Dense(128, activation='relu')(combined)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            
            # 1. PRIMARY BRANCHES: Team goals and behinds prediction with bounded outputs
            # Branch for Team 1 goals prediction
            team1_goals_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_norm])
            team1_goals_branch = tf.keras.layers.Dense(64, activation='relu')(team1_goals_branch)
            team1_goals_branch = tf.keras.layers.Dense(32, activation='relu')(team1_goals_branch)
            # Use BoundedOutputLayer to ensure realistic range
            team1_goals = BoundedOutputLayer(MAX_GOALS, MIN_GOALS, name='team1_goals')(team1_goals_branch)
            
            # Branch for Team 1 behinds prediction
            team1_behinds_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_norm])
            team1_behinds_branch = tf.keras.layers.Dense(64, activation='relu')(team1_behinds_branch)
            team1_behinds_branch = tf.keras.layers.Dense(32, activation='relu')(team1_behinds_branch)
            # Use BoundedOutputLayer to ensure realistic range
            team1_behinds = BoundedOutputLayer(MAX_BEHINDS, MIN_BEHINDS, name='team1_behinds')(team1_behinds_branch)
            
            # Branch for Team 2 goals prediction
            team2_goals_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_norm])
            team2_goals_branch = tf.keras.layers.Dense(64, activation='relu')(team2_goals_branch)
            team2_goals_branch = tf.keras.layers.Dense(32, activation='relu')(team2_goals_branch)
            # Use BoundedOutputLayer to ensure realistic range
            team2_goals = BoundedOutputLayer(MAX_GOALS, MIN_GOALS, name='team2_goals')(team2_goals_branch)
            
            # Branch for Team 2 behinds prediction
            team2_behinds_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_norm])
            team2_behinds_branch = tf.keras.layers.Dense(64, activation='relu')(team2_behinds_branch)
            team2_behinds_branch = tf.keras.layers.Dense(32, activation='relu')(team2_behinds_branch)
            # Use BoundedOutputLayer to ensure realistic range
            team2_behinds = BoundedOutputLayer(MAX_BEHINDS, MIN_BEHINDS, name='team2_behinds')(team2_behinds_branch)
            
            # 2. DERIVED CALCULATIONS: Calculate scores and margin from goals and behinds
            # Calculate team scores from goals and behinds
            team1_score = tf.keras.layers.Lambda(lambda x: x[0]*6 + x[1], name='team1_score')([team1_goals, team1_behinds])
            team2_score = tf.keras.layers.Lambda(lambda x: x[0]*6 + x[1], name='team2_score')([team2_goals, team2_behinds])
            
            # Calculate derived margin from scores (team1_score - team2_score)
            calculated_margin = tf.keras.layers.Lambda(lambda x: x[0] - x[1], name='calculated_margin')([team1_score, team2_score])
            
            # 3. WINNER PREDICTION BRANCH: Directly derive winner from calculated margin
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
            
            # Concatenate to form the match winner output [team1_win, draw, team2_win]
            match_winner = tf.keras.layers.Concatenate(name='match_winner')([
                tf.reshape(team1_win_prob_norm, [-1, 1]),
                tf.reshape(draw_prob_norm, [-1, 1]),
                tf.reshape(team2_win_prob_norm, [-1, 1])
            ])
            
            # Create model with our bounded prediction outputs
            self.model = tf.keras.models.Model(
                inputs=[match_input, team1_players_input, team2_players_input, team1_enhanced_input, team2_enhanced_input], 
                outputs=[match_winner, team1_goals, team1_behinds, team2_goals, team2_behinds, calculated_margin]
            )
            
            # Compile the model
            self.model.compile(
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
                    'match_winner': ['accuracy', tf.keras.metrics.Precision(name='precision'), 
                                     tf.keras.metrics.Recall(name='recall')], 
                    'team1_goals': tf.keras.metrics.MeanAbsoluteError(name='mae'), 
                    'team1_behinds': tf.keras.metrics.MeanAbsoluteError(name='mae'), 
                    'team2_goals': tf.keras.metrics.MeanAbsoluteError(name='mae'), 
                    'team2_behinds': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                    'calculated_margin': tf.keras.metrics.MeanAbsoluteError(name='mae')
                }
            )
                
            return True
            
        except Exception as e:
            print(f"Error building model: {e}")
            self.model = None
            return False
    
    def prepare_input(self, home_team, away_team, venue, round_num=1, season=2023):
        """
        Prepare input data for the model.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            venue (str): Name of the venue
            round_num (int): Round number of the match
            season (int): Season year of the match
        
        Returns:
            numpy.ndarray or dict: Formatted input data ready for model prediction
        """
        try:
            # Get team and venue indices
            home_index = self.team_index.get(home_team, -1)
            away_index = self.team_index.get(away_team, -1)
            venue_index = self.venue_index.get(venue, -1)
            
            if home_index == -1 or away_index == -1 or venue_index == -1:
                unknown_entities = []
                if home_index == -1:
                    unknown_entities.append(f"home team '{home_team}'")
                if away_index == -1:
                    unknown_entities.append(f"away team '{away_team}'")
                if venue_index == -1:
                    unknown_entities.append(f"venue '{venue}'")
                
                error_msg = f"Unknown entities: {', '.join(unknown_entities)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Determine input format based on model inputs
            if hasattr(self.model, 'input_names'):
                input_names = self.model.input_names
                logger.debug(f"Model input names: {input_names}")
                
                # Check if the model expects a dictionary-based input
                if len(input_names) > 1:
                    # Structure for neural_network_with_embeddings.py model
                    if 'team1_input' in input_names and 'team2_input' in input_names:
                        # One-hot encode teams (18 teams in AFL)
                        team1_input = np.zeros((1, 18))
                        team2_input = np.zeros((1, 18))
                        
                        if 0 <= home_index < 18:
                            team1_input[0, home_index] = 1
                        
                        if 0 <= away_index < 18:
                            team2_input[0, away_index] = 1
                        
                        # Create venue input (1-hot encoded)
                        venue_input = np.zeros((1, len(self.venue_index)))
                        if 0 <= venue_index < len(self.venue_index):
                            venue_input[0, venue_index] = 1
                        
                        # Create match features
                        match_features = np.array([[
                            round_num,  # Round number
                            season      # Season year
                        ]])
                        
                        # Create input dictionary
                        model_input = {
                            'team1_input': team1_input,
                            'team2_input': team2_input,
                            'venue_input': venue_input,
                            'match_features': match_features
                        }
                        
                        # Check if model expects player inputs
                        player_inputs = [name for name in input_names if 'player' in name.lower()]
                        if player_inputs:
                            logger.warning("Model expects player inputs which are not provided. Using zeros.")
                            for input_name in player_inputs:
                                # Determine shape from model input
                                input_shape = self.model.get_input_shape_at(0)[input_names.index(input_name)]
                                model_input[input_name] = np.zeros((1,) + input_shape[1:])
                        
                        return model_input
                    
                    # Handle other multi-input models
                    logger.warning(f"Unknown multi-input model format. Creating default input structure.")
                    model_input = {}
                    for input_name in input_names:
                        # Determine input shape
                        input_layer = self.model.get_layer(input_name)
                        input_shape = input_layer.input_shape
                        if input_shape is None:
                            # Fallback to getting shape from model config
                            for layer in self.model.layers:
                                if layer.name == input_name:
                                    input_shape = layer.get_config().get('batch_input_shape')
                                    break
                        
                        # Create zeros with appropriate shape
                        if input_shape:
                            model_input[input_name] = np.zeros((1,) + input_shape[1:])
                        else:
                            logger.error(f"Could not determine shape for input {input_name}")
                            model_input[input_name] = np.zeros((1, 10))  # Default fallback
                    
                    return model_input
                
                # Single input model
                else:
                    # Get the expected input shape
                    input_shape = self.model.layers[0].input_shape
                    if isinstance(input_shape, list):
                        input_shape = input_shape[0]
                    
                    if input_shape is None:
                        # Fallback to a standard flat feature vector
                        logger.warning("Could not determine input shape. Using default feature vector.")
                        input_dim = 20  # Default dimension
                    else:
                        input_dim = input_shape[-1]  # Last dimension is feature dimension
                    
                    # Create flat feature vector
                    X = np.zeros((1, input_dim))
                    
                    # Fill known features
                    feature_idx = 0
                    
                    # Team indices (one-hot encoded)
                    if feature_idx + len(self.team_index) * 2 <= input_dim:
                        # Home team one-hot encoding
                        if 0 <= home_index < len(self.team_index):
                            X[0, feature_idx + home_index] = 1
                        
                        feature_idx += len(self.team_index)
                        
                        # Away team one-hot encoding
                        if 0 <= away_index < len(self.team_index):
                            X[0, feature_idx + away_index] = 1
                        
                        feature_idx += len(self.team_index)
                    else:
                        # Not enough space for one-hot encoding, use indices directly
                        if feature_idx + 2 <= input_dim:
                            X[0, feature_idx] = home_index
                            X[0, feature_idx + 1] = away_index
                            feature_idx += 2
                    
                    # Venue (one-hot encoded)
                    if feature_idx + len(self.venue_index) <= input_dim:
                        if 0 <= venue_index < len(self.venue_index):
                            X[0, feature_idx + venue_index] = 1
                        feature_idx += len(self.venue_index)
                    else:
                        # Use index directly
                        if feature_idx + 1 <= input_dim:
                            X[0, feature_idx] = venue_index
                            feature_idx += 1
                    
                    # Add round and season if there's space
                    if feature_idx + 2 <= input_dim:
                        X[0, feature_idx] = round_num
                        X[0, feature_idx + 1] = season
                    
                    return X
            
            # Fallback to legacy format if model input names not available
            else:
                logger.warning("Model input names not available. Using default feature format.")
                # Create a standard input vector with team & venue one-hot encoding
                # Total: 18 teams + 18 teams + venues + 2 numeric = 38 + num_venues
                num_teams = len(self.team_index)
                num_venues = len(self.venue_index)
                
                X = np.zeros((1, num_teams * 2 + num_venues + 2))
                
                # Home team (one-hot)
                if 0 <= home_index < num_teams:
                    X[0, home_index] = 1
                
                # Away team (one-hot)
                if 0 <= away_index < num_teams:
                    X[0, num_teams + away_index] = 1
                
                # Venue (one-hot)
                if 0 <= venue_index < num_venues:
                    X[0, num_teams * 2 + venue_index] = 1
                
                # Add round and season
                X[0, -2] = round_num
                X[0, -1] = season
                
                return X
        
        except Exception as e:
            logger.error(f"Error preparing input: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _generate_enhanced_features(self, players, team_name):
        """
        Generate enhanced features for a team based on player statistics.
        
        Args:
            players (list): List of player names
            team_name (str): Team name for logging
            
        Returns:
            ndarray: Enhanced features for the team with comprehensive stats
        """
        # All metrics we care about - matching what's in neural_network_with_embeddings.py
        metrics = [
            'kicks', 'marks', 'handballs', 'goals', 'behinds', 
            'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances',
            'clangers', 'free_kicks_for', 'free_kicks_against', 
            'contested_possessions', 'uncontested_possessions', 'contested_marks',
            'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist',
            'percentage_of_game_played'
        ]
        
        # Initialize features with the same number of metrics
        enhanced = np.zeros((1, len(metrics)))
        
        try:
            # Create basic team profile based on team name
            # These are estimates for AFL teams 
            if team_name in ["Richmond", "Geelong", "Brisbane Lions", "West Coast"]:
                # Traditionally strong scoring teams
                team_profile = {
                    'kicks': 220, 'marks': 95, 'handballs': 180, 'goals': 13.5, 'behinds': 10.8, 
                    'hit_outs': 40, 'tackles': 65, 'rebound_50s': 12, 'inside_50s': 55, 'clearances': 38,
                    'clangers': 45, 'free_kicks_for': 15, 'free_kicks_against': 15, 
                    'contested_possessions': 145, 'uncontested_possessions': 240, 'contested_marks': 12,
                    'marks_inside_50': 14, 'one_percenters': 45, 'bounces': 8, 'goal_assist': 11,
                    'percentage_of_game_played': 80
                }
            elif team_name in ["Collingwood", "Hawthorn", "Sydney", "Port Adelaide"]:
                # Balanced teams
                team_profile = {
                    'kicks': 210, 'marks': 90, 'handballs': 190, 'goals': 12.2, 'behinds': 11.5, 
                    'hit_outs': 38, 'tackles': 70, 'rebound_50s': 14, 'inside_50s': 52, 'clearances': 35,
                    'clangers': 40, 'free_kicks_for': 18, 'free_kicks_against': 18, 
                    'contested_possessions': 140, 'uncontested_possessions': 230, 'contested_marks': 11,
                    'marks_inside_50': 12, 'one_percenters': 48, 'bounces': 7, 'goal_assist': 10,
                    'percentage_of_game_played': 82
                }
            elif team_name in ["Western Bulldogs", "Melbourne", "Essendon", "Fremantle"]:
                # Midfield-focused teams
                team_profile = {
                    'kicks': 200, 'marks': 85, 'handballs': 200, 'goals': 11.8, 'behinds': 12.2, 
                    'hit_outs': 35, 'tackles': 75, 'rebound_50s': 15, 'inside_50s': 50, 'clearances': 40,
                    'clangers': 38, 'free_kicks_for': 16, 'free_kicks_against': 16, 
                    'contested_possessions': 150, 'uncontested_possessions': 220, 'contested_marks': 10,
                    'marks_inside_50': 11, 'one_percenters': 50, 'bounces': 9, 'goal_assist': 12,
                    'percentage_of_game_played': 81
                }
            else:
                # Default profile for other teams
                team_profile = {
                    'kicks': 195, 'marks': 80, 'handballs': 185, 'goals': 11.0, 'behinds': 11.0, 
                    'hit_outs': 33, 'tackles': 68, 'rebound_50s': 13, 'inside_50s': 48, 'clearances': 36,
                    'clangers': 42, 'free_kicks_for': 17, 'free_kicks_against': 17, 
                    'contested_possessions': 135, 'uncontested_possessions': 210, 'contested_marks': 9,
                    'marks_inside_50': 10, 'one_percenters': 46, 'bounces': 6, 'goal_assist': 9,
                    'percentage_of_game_played': 79
                }
            
            # Adjust based on number of players
            player_count = len(players)
            adjustment_factor = player_count / 22.0  # Standard AFL team size
            
            # Apply adjustments for each metric
            for i, metric in enumerate(metrics):
                if metric in team_profile:
                    enhanced[0, i] = team_profile[metric] * adjustment_factor
                else:
                    # Fallback if metric not in profile
                    enhanced[0, i] = 0
            
            # Normalize to expected ranges for key stats
            enhanced[0, metrics.index('goals')] = max(min(enhanced[0, metrics.index('goals')], 20), 5)  # Goals: 5-20
            enhanced[0, metrics.index('behinds')] = max(min(enhanced[0, metrics.index('behinds')], 15), 5)  # Behinds: 5-15
            
            print(f"Generated enhanced features for {team_name} with {len(metrics)} metrics")
            
        except Exception as e:
            print(f"Error generating enhanced features: {e}")
        
        return enhanced
    
    def get_team_stats(self, player_ids):
        """
        Generate team statistics based on the player IDs.
        This function leverages existing feature generation capability but works with player IDs.
        
        Args:
            player_ids (list): List of player IDs
            
        Returns:
            ndarray: Enhanced features for the team with comprehensive stats
        """
        if not player_ids:
            # If no players provided, return zeros
            metrics = [
                'kicks', 'marks', 'handballs', 'goals', 'behinds', 
                'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances',
                'clangers', 'free_kicks_for', 'free_kicks_against', 
                'contested_possessions', 'uncontested_possessions', 'contested_marks',
                'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist',
                'percentage_of_game_played'
            ]
            return np.zeros((1, len(metrics)))
        
        # Determine team name to use in feature generation (if possible)
        team_name = "Unknown"
        
        # Use the feature generation logic with player IDs instead of names
        # This reuses the same team-based statistical model
        return self._generate_enhanced_features(player_ids, team_name)
    
    def predict(self, match_features=None, team1_players=None, team2_players=None):
        """Make a prediction using the model. Return detailed prediction info."""
        if not self.model:
            print("No model loaded. Cannot make prediction.")
            return None, None, None, None, None
            
        try:
            # Prepare enhanced features from player stats if players provided
            print("Generating team enhanced features...")
            team1_enhanced = self.get_team_stats(team1_players) if team1_players else np.zeros((1, 21))
            team2_enhanced = self.get_team_stats(team2_players) if team2_players else np.zeros((1, 21))
            print(f"Team 1 enhanced shape: {team1_enhanced.shape}")
            print(f"Team 2 enhanced shape: {team2_enhanced.shape}")
            
            # Prepare match inputs
            print("Preparing match input...")
            if match_features is not None:
                print(f"Using provided match features with shape: {np.array(match_features).shape}")
                match_input_np = np.array([match_features]) if len(np.array(match_features).shape) == 1 else np.array(match_features)
            else:
                feature_size = len(self.feature_cols) if self.feature_cols else 119
                print(f"Creating empty match features with size {feature_size}")
                match_input_np = np.zeros((1, feature_size))
            
            # Get player indices for both teams, with zeros (unknown player index) for missing players
            print("Preparing player indices...")
            team1_players_indices = np.zeros((1, MAX_PLAYERS_PER_TEAM), dtype=int)
            team2_players_indices = np.zeros((1, MAX_PLAYERS_PER_TEAM), dtype=int)
            
            if team1_players:
                print(f"Processing {len(team1_players)} team 1 players...")
                for i, player_id in enumerate(team1_players):
                    if i < MAX_PLAYERS_PER_TEAM:
                        if str(player_id) in self.player_index:
                            team1_players_indices[0, i] = self.player_index[str(player_id)]
                        else:
                            print(f"Player {player_id} not found in player index - using 0")
            
            if team2_players:
                print(f"Processing {len(team2_players)} team 2 players...")
                for i, player_id in enumerate(team2_players):
                    if i < MAX_PLAYERS_PER_TEAM:
                        if str(player_id) in self.player_index:
                            team2_players_indices[0, i] = self.player_index[str(player_id)]
                        else:
                            print(f"Player {player_id} not found in player index - using 0")
            
            # Reshape enhanced features for model
            team1_enhanced_np = np.array(team1_enhanced).reshape(1, -1)
            team2_enhanced_np = np.array(team2_enhanced).reshape(1, -1)
            
            # Print model input shapes
            print("Model input shapes:")
            print(f"match_input: {match_input_np.shape}")
            print(f"team1_players: {team1_players_indices.shape}")
            print(f"team2_players: {team2_players_indices.shape}")
            print(f"team1_enhanced: {team1_enhanced_np.shape}")
            print(f"team2_enhanced: {team2_enhanced_np.shape}")
            
            # Check if model expects these input shapes
            if hasattr(self.model, 'inputs') and len(self.model.inputs) >= 5:
                print("Expected input shapes from model:")
                for i, inp in enumerate(self.model.inputs):
                    print(f"Input {i}: {inp.name} - {inp.shape}")
            
            # Make prediction with our new model
            print("Making model prediction...")
            predictions = self.model.predict([
                match_input_np, 
                team1_players_indices, 
                team2_players_indices,
                team1_enhanced_np,
                team2_enhanced_np
            ], verbose=0)
            
            # Print prediction shape information
            print("Prediction results shapes:")
            for i, p in enumerate(predictions):
                print(f"Prediction {i} shape: {p.shape}")
            
            # Unpack predictions based on model architecture
            print("Unpacking predictions...")
            match_winner_probs = predictions[0][0]  # [team1_win, draw, team2_win] probabilities
            team1_goals = predictions[1][0][0]  # team1 goals
            team1_behinds = predictions[2][0][0]  # team1 behinds
            team2_goals = predictions[3][0][0]  # team2 goals
            team2_behinds = predictions[4][0][0]  # team2 behinds
            margin = predictions[5][0][0]  # calculated margin
            
            # Calculate scores from goals and behinds
            team1_score = team1_goals * 6 + team1_behinds
            team2_score = team2_goals * 6 + team2_behinds
            
            # Debug information
            debug_info = {
                'calculated_margin': margin,
                'team1_goals': team1_goals,
                'team1_behinds': team1_behinds,
                'team2_goals': team2_goals,
                'team2_behinds': team2_behinds,
                'team1_score': team1_score,
                'team2_score': team2_score,
                'match_winner_probs': match_winner_probs
            }
            
            print("Prediction completed successfully")
            return margin, match_winner_probs, team1_score, team2_score, debug_info
            
        except Exception as e:
            print(f"ERROR in predict method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print("Exception details:")
            traceback.print_exc()
            return None, None, None, None, None

    def predict_match(self, match_data):
        """
        Predict the outcome of a match.
        
        Args:
            match_data (dict): Dictionary containing match information:
                - home_team: Name of the home team
                - away_team: Name of the away team
                - venue: Name of the venue
                - round_num: Round number (optional, defaults to 1)
                - season: Season year (optional, defaults to current year)
        
        Returns:
            dict: Prediction results containing:
                - home_goals: Predicted number of goals for home team
                - home_behinds: Predicted number of behinds for home team
                - away_goals: Predicted number of goals for away team
                - away_behinds: Predicted number of behinds for away team
                - home_score: Calculated total score for home team
                - away_score: Calculated total score for away team
                - margin: Predicted margin (home_score - away_score)
                - winner: Predicted winner ('home' or 'away')
                - win_probability: Probability of the predicted winner
                - error: Error message if prediction failed
        """
        # If using fallback predictor, delegate to it
        if self.using_fallback or not self.model:
            logger.info("Using fallback predictor for match prediction")
            return self.fallback_predictor.predict_match(match_data)
        
        # Otherwise, use the neural network model
        try:
            if not self.model:
                logger.error("Model not loaded. Call load_model() first.")
                return {"error": "Model not loaded"}
            
            # Extract match information
            home_team = match_data.get('home_team', '')
            away_team = match_data.get('away_team', '')
            venue = match_data.get('venue', '')
            round_num = match_data.get('round_num', 1)
            season = match_data.get('season', 2023)
            
            # Validate input
            if not home_team or not away_team or not venue:
                return {"error": "Missing team or venue information"}
            
            logger.info(f"Predicting match: {home_team} vs {away_team} at {venue}")
            
            # Prepare model input
            try:
                model_input = self.prepare_input(home_team, away_team, venue, round_num, season)
            except Exception as input_error:
                logger.error(f"Error preparing model input: {str(input_error)}\n{traceback.format_exc()}")
                return {"error": f"Error preparing input: {str(input_error)}"}
            
            # Log input shape for debugging
            if isinstance(model_input, dict):
                logger.debug(f"Model input shapes: {[(k, v.shape) for k, v in model_input.items()]}")
            else:
                logger.debug(f"Model input shape: {model_input.shape}")
            
            # Make prediction with robust error handling
            try:
                # Start with verbose=0 but can be changed for debugging
                predictions = self.model.predict(model_input, verbose=0)
                logger.debug(f"Raw prediction type: {type(predictions)}")
                
                # Log prediction outputs for debugging
                if isinstance(predictions, list):
                    logger.debug(f"Raw predictions: {len(predictions)} outputs")
                    for i, p in enumerate(predictions):
                        logger.debug(f"  Output {i} shape: {p.shape}, range: [{np.min(p)}, {np.max(p)}]")
                else:
                    logger.debug(f"Raw prediction shape: {predictions.shape}, range: [{np.min(predictions)}, {np.max(predictions)}]")
            except Exception as pred_error:
                logger.error(f"Error during model prediction: {str(pred_error)}\n{traceback.format_exc()}")
                logger.info("Falling back to statistical predictor")
                return self.fallback_predictor.predict_match(match_data)
            
            # Define reasonable ranges for AFL scores
            MAX_REALISTIC_GOALS = 25
            MAX_REALISTIC_BEHINDS = 20
            
            # Extract and process predictions based on custom model output format
            try:
                # For our new custom model, outputs are:
                # predictions[0] = match_winner [team1_win, draw, team2_win] probabilities
                # predictions[1] = team1_goals
                # predictions[2] = team1_behinds
                # predictions[3] = team2_goals
                # predictions[4] = team2_behinds
                # predictions[5] = calculated_margin
                
                # Extract values from outputs (our BoundedOutputLayer ensures reasonable values)
                home_goals = float(predictions[1][0][0])
                home_behinds = float(predictions[2][0][0])
                away_goals = float(predictions[3][0][0])
                away_behinds = float(predictions[4][0][0])
                
                # Get the margin
                margin = float(predictions[5][0][0])
                
                # Get win probabilities
                win_probs = predictions[0][0]
                home_win_prob = float(win_probs[0])
                draw_prob = float(win_probs[1]) if len(win_probs) > 1 else 0.0
                away_win_prob = float(win_probs[2]) if len(win_probs) > 2 else (1.0 - home_win_prob)
                
            except Exception as extract_error:
                logger.error(f"Error extracting prediction values: {str(extract_error)}\n{traceback.format_exc()}")
                return {"error": f"Error processing predictions: {str(extract_error)}"}
            
            # Calculate scores
            home_score = home_goals * 6 + home_behinds
            away_score = away_goals * 6 + away_behinds
            
            # Determine winner based on margin
            if margin > 0:
                winner = 'home'
                win_probability = home_win_prob
            elif margin < 0:
                winner = 'away'
                win_probability = away_win_prob
            else:
                # Draw - choose the team with higher probability
                if home_win_prob >= away_win_prob:
                    winner = 'home'
                    win_probability = home_win_prob
                else:
                    winner = 'away'
                    win_probability = away_win_prob
            
            # Format results with clean rounding
            result = {
                'home_goals': round(home_goals, 1),
                'home_behinds': round(home_behinds, 1),
                'away_goals': round(away_goals, 1),
                'away_behinds': round(away_behinds, 1),
                'home_score': round(home_score, 1),
                'away_score': round(away_score, 1),
                'margin': round(margin, 1),
                'winner': winner,
                'win_probability': round(win_probability, 3)
            }
            
            logger.info(f"Prediction result: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Unexpected error in predict_match: {str(e)}\n{traceback.format_exc()}")
            logger.info("Falling back to statistical predictor due to error")
            return self.fallback_predictor.predict_match(match_data)

# For testing
if __name__ == "__main__":
    predictor = SimplePredictor()
    print("Predictor test complete!") 