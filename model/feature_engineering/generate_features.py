#!/usr/bin/env python3
"""
Feature Engineering for AFL Match Prediction

This module generates features from the AFL match and player data
for use in predictive models. It builds on the utility functions
from the utils.data_processing module.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import project modules
from datafetch.load_afl_data import AFLDataLoader
from model.utils.data_processing import (
    clean_team_names,
    standardize_dates,
    calculate_match_result,
    process_player_data,
    calculate_team_form,
    create_match_features
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature_engineering')

# Define constants
FEATURE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

class AFLFeatureGenerator:
    """Class for generating features from AFL data for predictive modeling."""
    
    def __init__(self, data_cutoff_year: Optional[int] = None):
        """
        Initialize the feature generator.
        
        Args:
            data_cutoff_year: Optional year to use as cutoff for training data
        """
        self.loader = AFLDataLoader()
        self.matches_data = None
        self.player_data = {}
        self.odds_data = None
        self.data_cutoff_year = data_cutoff_year
        self.feature_sets = {}
    
    def load_data(self):
        """Load all required data."""
        logger.info("Loading match data...")
        self.matches_data = self.loader.load_matches()
        
        # Filter by cutoff year if specified
        if self.data_cutoff_year:
            self.matches_data = self.matches_data[self.matches_data['year'] <= self.data_cutoff_year]
        
        # Load a sample of player data for key players
        logger.info("Loading player data for key players...")
        
        # This would be a more sophisticated process in reality
        # For now, we'll just load a few random players
        player_list = self.loader.load_player_data()
        sample_size = min(50, len(player_list))
        sample_players = np.random.choice(player_list, sample_size)
        
        for player_name in sample_players:
            try:
                perf_data = self.loader.load_player_data(player_name, data_type='performance')
                personal_data = self.loader.load_player_data(player_name, data_type='personal')
                if perf_data is not None and personal_data is not None:
                    self.player_data[player_name] = process_player_data(perf_data, personal_data)
            except Exception as e:
                logger.warning(f"Error loading player data for {player_name}: {e}")
        
        logger.info("Loading odds data...")
        self.odds_data = self.loader.load_odds_data()
        
        logger.info("Data loading complete.")
    
    def generate_team_performance_features(self) -> pd.DataFrame:
        """
        Generate team performance features from match data.
        
        Returns:
            DataFrame with team performance features for each match
        """
        logger.info("Generating team performance features...")
        
        if self.matches_data is None:
            logger.error("Match data not loaded. Call load_data() first.")
            return pd.DataFrame()
        
        # Clean and preprocess match data
        match_features = create_match_features(self.matches_data)
        
        # Calculate team-level metrics for each match
        teams = []
        if 'team_1_team_name' in match_features.columns:
            teams.extend(match_features['team_1_team_name'].unique())
        if 'team_2_team_name' in match_features.columns:
            teams.extend(match_features['team_2_team_name'].unique())
        teams = list(set(teams))
        
        # Sort matches by date
        if 'date' in match_features.columns:
            match_features = match_features.sort_values('date')
        
        # For each match, calculate team form features
        recent_form_features = []
        
        for idx, match in match_features.iterrows():
            if 'date' not in match or pd.isna(match['date']):
                continue
                
            match_date = match['date']
            
            # Team 1 features
            if 'team_1_team_name' in match and not pd.isna(match['team_1_team_name']):
                team1_name = match['team_1_team_name']
                team1_form = calculate_team_form(match_features, team1_name, match_date)
                
                # Create row with match ID and team1 features
                form_row = {
                    'match_idx': idx,
                    'team_position': 1,
                    'team_name': team1_name,
                    'date': match_date
                }
                form_row.update({f'team1_{k}': v for k, v in team1_form.items()})
                recent_form_features.append(form_row)
            
            # Team 2 features
            if 'team_2_team_name' in match and not pd.isna(match['team_2_team_name']):
                team2_name = match['team_2_team_name']
                team2_form = calculate_team_form(match_features, team2_name, match_date)
                
                # Create row with match ID and team2 features
                form_row = {
                    'match_idx': idx,
                    'team_position': 2,
                    'team_name': team2_name,
                    'date': match_date
                }
                form_row.update({f'team2_{k}': v for k, v in team2_form.items()})
                recent_form_features.append(form_row)
        
        # Convert to DataFrame
        form_features_df = pd.DataFrame(recent_form_features)
        
        # Pivot the data to get features for both teams in a single row
        if not form_features_df.empty:
            form_features_df = form_features_df.pivot(index='match_idx', columns='team_position')
            form_features_df.columns = [f"{col[0]}_{col[1]}" for col in form_features_df.columns]
            form_features_df = form_features_df.reset_index()
            
            # Merge back with original match features
            match_features = match_features.reset_index(drop=True)
            match_features['match_idx'] = match_features.index
            match_features = pd.merge(match_features, form_features_df, on='match_idx', how='left')
        
        # Calculate head-to-head history
        # This would be more complex in reality
        
        self.feature_sets['team_performance'] = match_features
        logger.info("Team performance features generation complete.")
        
        return match_features
    
    def generate_player_features(self) -> pd.DataFrame:
        """
        Generate player-based features for matches.
        
        Returns:
            DataFrame with player-based features for each match
        """
        logger.info("Generating player-based features...")
        
        if self.matches_data is None or not self.player_data:
            logger.error("Match or player data not loaded. Call load_data() first.")
            return pd.DataFrame()
        
        # In reality, this would involve:
        # 1. Using lineup data to determine which players played in each match
        # 2. Calculating player performance metrics for key positions
        # 3. Aggregating to team level
        
        # For this simplified version, we'll create a placeholder DataFrame
        player_features = pd.DataFrame({
            'match_idx': range(len(self.matches_data)),
            'player_feature_placeholder': 0.0  # Placeholder
        })
        
        self.feature_sets['player_based'] = player_features
        logger.info("Player-based features generation complete.")
        
        return player_features
    
    def generate_venue_features(self) -> pd.DataFrame:
        """
        Generate venue-specific features.
        
        Returns:
            DataFrame with venue features for each match
        """
        logger.info("Generating venue features...")
        
        if self.matches_data is None:
            logger.error("Match data not loaded. Call load_data() first.")
            return pd.DataFrame()
        
        venue_features = self.matches_data[['venue']].copy() if 'venue' in self.matches_data.columns else pd.DataFrame()
        
        if not venue_features.empty:
            # Add match index
            venue_features['match_idx'] = range(len(venue_features))
            
            # One-hot encode venues
            venue_dummies = pd.get_dummies(venue_features['venue'], prefix='venue')
            venue_features = pd.concat([venue_features, venue_dummies], axis=1)
            
            # Calculate venue win rates for each team
            # This would be more complex in reality
        
        self.feature_sets['venue'] = venue_features
        logger.info("Venue features generation complete.")
        
        return venue_features
    
    def generate_match_context_features(self) -> pd.DataFrame:
        """
        Generate contextual features for matches (round, season, etc.)
        
        Returns:
            DataFrame with context features for each match
        """
        logger.info("Generating match context features...")
        
        if self.matches_data is None:
            logger.error("Match data not loaded. Call load_data() first.")
            return pd.DataFrame()
        
        # Extract relevant columns
        context_cols = ['year', 'round_num', 'date']
        context_cols = [col for col in context_cols if col in self.matches_data.columns]
        
        if not context_cols:
            logger.warning("No context columns found in match data.")
            return pd.DataFrame()
        
        context_features = self.matches_data[context_cols].copy()
        context_features['match_idx'] = range(len(context_features))
        
        # Convert date to datetime if it isn't already
        if 'date' in context_features.columns:
            context_features = standardize_dates(context_features, 'date')
        
        # Extract season progress (what percentage of the season has been played)
        if 'year' in context_features.columns and 'round_num' in context_features.columns:
            # Calculate max round number for each year
            max_rounds = context_features.groupby('year')['round_num'].max()
            
            # Merge back to get max round for each match's year
            context_features = context_features.merge(
                max_rounds.reset_index().rename(columns={'round_num': 'max_round'}),
                on='year', how='left'
            )
            
            # Calculate season progress
            context_features['season_progress'] = context_features['round_num'] / context_features['max_round']
        
        # Add days since last match for each team
        # This would be more complex in reality
        
        self.feature_sets['match_context'] = context_features
        logger.info("Match context features generation complete.")
        
        return context_features
    
    def generate_odds_features(self) -> pd.DataFrame:
        """
        Generate features from odds data.
        
        Returns:
            DataFrame with odds-based features for each match
        """
        logger.info("Generating odds features...")
        
        if self.matches_data is None or self.odds_data is None:
            logger.warning("Match or odds data not loaded. Odds features will be empty.")
            return pd.DataFrame()
        
        # For simplicity, we'll just create a placeholder
        # In reality, this would involve:
        # 1. Matching odds data to matches by date and teams
        # 2. Extracting relevant odds (pre-match, line, etc.)
        # 3. Calculating derived features (implied probability, etc.)
        
        odds_features = pd.DataFrame({
            'match_idx': range(len(self.matches_data)),
            'odds_feature_placeholder': 0.0  # Placeholder
        })
        
        self.feature_sets['odds'] = odds_features
        logger.info("Odds features generation complete.")
        
        return odds_features
    
    def combine_feature_sets(self) -> pd.DataFrame:
        """
        Combine all feature sets into a single DataFrame.
        
        Returns:
            DataFrame with all features combined
        """
        logger.info("Combining feature sets...")
        
        if not self.feature_sets:
            logger.error("No feature sets generated. Generate features first.")
            return pd.DataFrame()
        
        # Start with team performance features as the base
        if 'team_performance' in self.feature_sets:
            combined_features = self.feature_sets['team_performance'].copy()
        else:
            # Fall back to a DataFrame with match indices
            combined_features = pd.DataFrame({'match_idx': range(len(self.matches_data))})
        
        # Merge with other feature sets
        for feature_set_name, feature_df in self.feature_sets.items():
            if feature_set_name != 'team_performance' and not feature_df.empty:
                combined_features = pd.merge(
                    combined_features,
                    feature_df,
                    on='match_idx',
                    how='left'
                )
        
        # Add target variable if not already present
        if 'team_1_result' not in combined_features.columns and 'team_1_final_score' in combined_features.columns:
            combined_features = calculate_match_result(combined_features)
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        
        return combined_features
    
    def generate_all_features(self) -> pd.DataFrame:
        """
        Generate all feature sets and combine them.
        
        Returns:
            DataFrame with all features
        """
        logger.info("Generating all features...")
        
        # Load data if not already loaded
        if self.matches_data is None:
            self.load_data()
        
        # Generate individual feature sets
        self.generate_team_performance_features()
        self.generate_player_features()
        self.generate_venue_features()
        self.generate_match_context_features()
        self.generate_odds_features()
        
        # Combine feature sets
        combined_features = self.combine_feature_sets()
        
        # Save features to disk
        output_path = os.path.join(FEATURE_OUTPUT_DIR, 'match_features.csv')
        combined_features.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
        
        return combined_features

def main():
    """Main function to generate features."""
    feature_generator = AFLFeatureGenerator()
    features = feature_generator.generate_all_features()
    logger.info(f"Generated {features.shape[1]} features for {features.shape[0]} matches.")

if __name__ == "__main__":
    main() 