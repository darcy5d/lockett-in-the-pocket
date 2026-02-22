#!/usr/bin/env python3
import os
import sys
import json
import traceback
import tensorflow as tf
import numpy as np
from gui.simple_predictor import SimplePredictor, MeanPoolingLayer

# Import constants from simple_predictor
from gui.simple_predictor import MAX_PLAYERS_PER_TEAM, EMBEDDING_DIM

class BoundedOutputLambda(tf.keras.layers.Layer):
    """Custom Lambda layer that specifies output shape for bounded predictions."""
    def __init__(self, function, max_value, min_offset, **kwargs):
        super(BoundedOutputLambda, self).__init__(**kwargs)
        self.function = function
        self.max_value = max_value
        self.min_offset = min_offset
        
    def call(self, inputs):
        # Apply sigmoid and scale to appropriate range
        return self.function(inputs) * self.max_value + self.min_offset
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super(BoundedOutputLambda, self).get_config()
        config.update({
            "max_value": self.max_value,
            "min_offset": self.min_offset,
            "function": self.function.__name__ if hasattr(self.function, "__name__") else str(self.function)
        })
        return config

class ScoreCalculationLambda(tf.keras.layers.Layer):
    """Custom Lambda layer for calculating scores from goals and behinds."""
    def __init__(self, **kwargs):
        super(ScoreCalculationLambda, self).__init__(**kwargs)
        
    def call(self, inputs):
        # Calculate score as goals*6 + behinds
        goals, behinds = inputs
        return goals * 6 + behinds
        
    def compute_output_shape(self, input_shape):
        # Output shape is same as either input
        return input_shape[0]
        
    def get_config(self):
        config = super(ScoreCalculationLambda, self).get_config()
        return config

class MarginCalculationLambda(tf.keras.layers.Layer):
    """Custom Lambda layer for calculating margin between team scores."""
    def __init__(self, **kwargs):
        super(MarginCalculationLambda, self).__init__(**kwargs)
        
    def call(self, inputs):
        # Calculate margin as team1_score - team2_score
        team1_score, team2_score = inputs
        return team1_score - team2_score
        
    def compute_output_shape(self, input_shape):
        # Output shape is same as either input
        return input_shape[0]
        
    def get_config(self):
        config = super(MarginCalculationLambda, self).get_config()
        return config

class EnhancedPredictor(SimplePredictor):
    """Enhanced version of SimplePredictor with proper Lambda layer handling."""
    
    def __init__(self):
        """Initialize with additional attributes."""
        super().__init__()
        # Initialize team and venue indices if not set by parent
        if not hasattr(self, 'team_index'):
            self.team_index = {}
            
        if not hasattr(self, 'venue_index'):
            self.venue_index = {}
    
    def load_model(self):
        """Override the load_model method to use our custom loss functions."""
        try:
            if not self.model_path.exists():
                print(f"Model file not found at {self.model_path}")
                return False
                
            print(f"Attempting to load model from {self.model_path}...")
            
            # Define custom MSE and MAE functions
            def mse(y_true, y_pred):
                return tf.reduce_mean(tf.square(y_true - y_pred))
                
            def mae(y_true, y_pred):
                return tf.reduce_mean(tf.abs(y_true - y_pred))
            
            # Register custom layers first
            tf.keras.utils.get_custom_objects()['MeanPoolingLayer'] = MeanPoolingLayer
            tf.keras.utils.get_custom_objects()['BoundedOutputLambda'] = BoundedOutputLambda
            tf.keras.utils.get_custom_objects()['ScoreCalculationLambda'] = ScoreCalculationLambda
            tf.keras.utils.get_custom_objects()['MarginCalculationLambda'] = MarginCalculationLambda
            
            # Define custom objects dictionary
            custom_objects = {
                'MeanPoolingLayer': MeanPoolingLayer,
                'BoundedOutputLambda': BoundedOutputLambda,
                'ScoreCalculationLambda': ScoreCalculationLambda,
                'MarginCalculationLambda': MarginCalculationLambda,
                'mean_lambda': lambda x: tf.reduce_mean(x, axis=1),
                'mse': mse,
                'mae': mae,
                'precision': tf.keras.metrics.Precision(),
                'recall': tf.keras.metrics.Recall(),
                'mean_squared_error': mse,
                'mean_absolute_error': mae,
                'MSE': mse,
                'MAE': mae,
                'MeanSquaredError': tf.keras.metrics.MeanSquaredError(),
                'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError()
            }
            
            # Try to load with custom objects
            try:
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=custom_objects
                )
                print("Model loaded successfully!")
                return True
            except Exception as inner_e:
                print(f"First loading attempt failed: {inner_e}")
                # Continue to alternative approach
            
            # Try with compile=False
            try:
                print("Trying alternative loading approach with compile=False...")
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
                
                # Manually compile the model
                self.model.compile(
                    optimizer='adam',
                    loss={
                        'match_winner': 'categorical_crossentropy', 
                        'margin': mse,
                        'team1_goals': mse, 
                        'team1_behinds': mse, 
                        'team2_goals': mse, 
                        'team2_behinds': mse
                    },
                    metrics={
                        'match_winner': ['accuracy', tf.keras.metrics.Precision(name='precision'), 
                                         tf.keras.metrics.Recall(name='recall')], 
                        'margin': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                        'team1_goals': tf.keras.metrics.MeanAbsoluteError(name='mae'), 
                        'team1_behinds': tf.keras.metrics.MeanAbsoluteError(name='mae'), 
                        'team2_goals': tf.keras.metrics.MeanAbsoluteError(name='mae'), 
                        'team2_behinds': tf.keras.metrics.MeanAbsoluteError(name='mae')
                    }
                )
                
                print("Model loaded with alternative approach and recompiled!")
                return True
            except Exception as e2:
                print(f"Alternative loading approach also failed: {e2}")
            
            # If loading fails, try building a new model
            print("Model loading failed. Will need to build a new model.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def build_model(self):
        """Build the model with explicit output shapes for Lambda layers."""
        # Register the custom layers so they can be loaded from saved models
        tf.keras.utils.get_custom_objects()['MeanPoolingLayer'] = MeanPoolingLayer
        tf.keras.utils.get_custom_objects()['BoundedOutputLambda'] = BoundedOutputLambda
        tf.keras.utils.get_custom_objects()['ScoreCalculationLambda'] = ScoreCalculationLambda
        tf.keras.utils.get_custom_objects()['MarginCalculationLambda'] = MarginCalculationLambda
        
        # Input layers with proper shapes
        match_features_size = len(self.feature_cols) if self.feature_cols else 119
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
        
        # Define reasonable bounds for AFL match stats
        MAX_GOALS = 25.0  # Maximum realistic goals
        MAX_BEHINDS = 15.0  # Maximum realistic behinds
        
        # 1. Team 1 goals prediction with bounded outputs
        team1_goals_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_norm])
        team1_goals_branch = tf.keras.layers.Dense(64, activation='relu')(team1_goals_branch)
        team1_goals_branch = tf.keras.layers.Dense(32, activation='relu')(team1_goals_branch)
        team1_goals_raw = tf.keras.layers.Dense(1, activation='sigmoid', name='team1_goals_raw')(team1_goals_branch)
        # Use custom BoundedOutputLambda instead of regular Lambda
        team1_goals = BoundedOutputLambda(
            function=lambda x: x, 
            max_value=MAX_GOALS, 
            min_offset=5, 
            name='team1_goals'
        )(team1_goals_raw)
        
        # 2. Team 1 behinds prediction
        team1_behinds_branch = tf.keras.layers.Concatenate()([x, team1_enhanced_norm])
        team1_behinds_branch = tf.keras.layers.Dense(64, activation='relu')(team1_behinds_branch)
        team1_behinds_branch = tf.keras.layers.Dense(32, activation='relu')(team1_behinds_branch)
        team1_behinds_raw = tf.keras.layers.Dense(1, activation='sigmoid', name='team1_behinds_raw')(team1_behinds_branch)
        team1_behinds = BoundedOutputLambda(
            function=lambda x: x, 
            max_value=MAX_BEHINDS, 
            min_offset=3, 
            name='team1_behinds'
        )(team1_behinds_raw)
        
        # 3. Team 2 goals prediction
        team2_goals_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_norm])
        team2_goals_branch = tf.keras.layers.Dense(64, activation='relu')(team2_goals_branch)
        team2_goals_branch = tf.keras.layers.Dense(32, activation='relu')(team2_goals_branch)
        team2_goals_raw = tf.keras.layers.Dense(1, activation='sigmoid', name='team2_goals_raw')(team2_goals_branch)
        team2_goals = BoundedOutputLambda(
            function=lambda x: x, 
            max_value=MAX_GOALS, 
            min_offset=5, 
            name='team2_goals'
        )(team2_goals_raw)
        
        # 4. Team 2 behinds prediction
        team2_behinds_branch = tf.keras.layers.Concatenate()([x, team2_enhanced_norm])
        team2_behinds_branch = tf.keras.layers.Dense(64, activation='relu')(team2_behinds_branch)
        team2_behinds_branch = tf.keras.layers.Dense(32, activation='relu')(team2_behinds_branch)
        team2_behinds_raw = tf.keras.layers.Dense(1, activation='sigmoid', name='team2_behinds_raw')(team2_behinds_branch)
        team2_behinds = BoundedOutputLambda(
            function=lambda x: x, 
            max_value=MAX_BEHINDS, 
            min_offset=3, 
            name='team2_behinds'
        )(team2_behinds_raw)
        
        # 5. Calculate team scores from goals and behinds using custom Lambda
        team1_score = ScoreCalculationLambda(name='team1_score')([team1_goals, team1_behinds])
        team2_score = ScoreCalculationLambda(name='team2_score')([team2_goals, team2_behinds])
        
        # 6. Calculate margin from scores using custom Lambda
        calculated_margin = MarginCalculationLambda(name='calculated_margin')([team1_score, team2_score])
        
        # 7. Winner prediction from margin
        winner_logit = tf.keras.layers.Lambda(lambda x: x * 0.1, name='margin_scaled')(calculated_margin)
        team1_win_prob = tf.keras.layers.Lambda(lambda x: tf.sigmoid(x), name='team1_win_prob')(winner_logit)
        team2_win_prob = tf.keras.layers.Lambda(lambda x: 1 - tf.sigmoid(x), name='team2_win_prob')(winner_logit)
        draw_prob = tf.keras.layers.Lambda(lambda x: tf.exp(-tf.square(x) * 5.0), name='draw_prob')(winner_logit)
        
        # Normalize probabilities
        total_prob = tf.keras.layers.Lambda(lambda x: x[0] + x[1] + x[2], name='total_prob')([team1_win_prob, draw_prob, team2_win_prob])
        team1_win_prob_norm = tf.keras.layers.Lambda(lambda x: x[0] / x[1], name='team1_win_prob_norm')([team1_win_prob, total_prob])
        draw_prob_norm = tf.keras.layers.Lambda(lambda x: x[0] / x[1], name='draw_prob_norm')([draw_prob, total_prob])
        team2_win_prob_norm = tf.keras.layers.Lambda(lambda x: x[0] / x[1], name='team2_win_prob_norm')([team2_win_prob, total_prob])
        
        # Concatenate for final output
        match_winner = tf.keras.layers.Concatenate(name='match_winner')([
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 1]))(team1_win_prob_norm),
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 1]))(draw_prob_norm),
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 1]))(team2_win_prob_norm)
        ])
        
        # Create and compile model
        self.model = tf.keras.models.Model(
            inputs=[match_input, team1_players_input, team2_players_input, team1_enhanced_input, team2_enhanced_input], 
            outputs=[match_winner, team1_goals, team1_behinds, team2_goals, team2_behinds, calculated_margin]
        )
        
        # Compile with appropriate metrics
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
        
        print("Model built successfully with custom Lambda layers!")
        return True

    def prepare_match_features(self, team1_name, team2_name, venue):
        """Override to provide a simplified version of prepare_match_features."""
        # Create a vector of zeros with the right size
        feature_size = len(self.feature_cols) if self.feature_cols else 119
        match_features = np.zeros(feature_size)
        
        # If we have team and venue indices, use them
        if hasattr(self, 'team_index') and self.team_index:
            if team1_name in self.team_index:
                team1_idx = self.team_index[team1_name]
                match_features[team1_idx] = 1
                
            if team2_name in self.team_index:
                team2_idx = self.team_index[team2_name]
                match_features[team2_idx] = 1
                
            if hasattr(self, 'venue_index') and self.venue_index and venue in self.venue_index:
                venue_idx = self.venue_index[venue]
                match_features[venue_idx] = 1
        else:
            # Without indices, we just return the zeros vector
            print("Warning: No team/venue indices available. Using zeros vector for match features.")
            
        return match_features

# For testing
if __name__ == "__main__":
    predictor = EnhancedPredictor()
    predictor.build_model()
    print(predictor.model.summary())
    print("EnhancedPredictor test complete!") 