#!/usr/bin/env python3
"""
Utility functions for processing AFL data.

This module provides reusable functions for data cleaning, transformation, 
and feature engineering that can be used across different components of the project.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import List, Dict, Union, Optional, Tuple

def clean_team_names(df: pd.DataFrame, team_columns: List[str]) -> pd.DataFrame:
    """
    Standardize team names across datasets.
    
    Some historical teams have changed names over time, this function maps all variations
    to standard modern names.
    
    Args:
        df: DataFrame containing team names
        team_columns: List of column names containing team names
        
    Returns:
        DataFrame with standardized team names
    """
    # Team name mapping dictionary (historical names -> current names)
    team_mapping = {
        'Footscray': 'Western Bulldogs',
        'South Melbourne': 'Sydney',
        'Brisbane Bears': 'Brisbane Lions',
        'Kangaroos': 'North Melbourne',
        'University': 'University',  # Defunct team, keep as is
        'Fitzroy': 'Brisbane Lions',  # Merged with Brisbane Bears
    }
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Clean each team column
    for col in team_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].replace(team_mapping)
    
    return df_clean

def standardize_dates(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Convert date strings to standardized datetime objects.
    
    Args:
        df: DataFrame containing date strings
        date_column: Name of the column containing dates
        
    Returns:
        DataFrame with standardized dates
    """
    df_clean = df.copy()
    
    if date_column in df_clean.columns:
        # Try different date formats
        date_formats = ['%Y-%m-%d %H:%M', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']
        
        for date_format in date_formats:
            try:
                df_clean[date_column] = pd.to_datetime(df_clean[date_column], format=date_format, errors='coerce')
                break
            except:
                continue
        
        # Add derived date features
        if pd.api.types.is_datetime64_dtype(df_clean[date_column]):
            df_clean[f'{date_column}_year'] = df_clean[date_column].dt.year
            df_clean[f'{date_column}_month'] = df_clean[date_column].dt.month
            df_clean[f'{date_column}_day'] = df_clean[date_column].dt.day
            df_clean[f'{date_column}_dayofweek'] = df_clean[date_column].dt.dayofweek
    
    return df_clean

def calculate_match_result(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate match results (win/loss/draw) for both teams.
    
    Args:
        df: DataFrame containing match data with team scores
        
    Returns:
        DataFrame with added result columns
    """
    df_results = df.copy()
    
    # Check if required columns exist
    score_columns = ['team_1_final_score', 'team_2_final_score']
    if not all(col in df_results.columns for col in score_columns):
        # Try to calculate scores from goals and behinds
        if all(col in df_results.columns for col in ['team_1_final_goals', 'team_1_final_behinds', 
                                                   'team_2_final_goals', 'team_2_final_behinds']):
            df_results['team_1_final_score'] = df_results['team_1_final_goals'] * 6 + df_results['team_1_final_behinds']
            df_results['team_2_final_score'] = df_results['team_2_final_goals'] * 6 + df_results['team_2_final_behinds']
        else:
            raise ValueError("Cannot calculate match results: required score columns not found")
    
    # Calculate match result
    df_results['team_1_result'] = np.where(
        df_results['team_1_final_score'] > df_results['team_2_final_score'], 'win',
        np.where(df_results['team_1_final_score'] < df_results['team_2_final_score'], 'loss', 'draw')
    )
    
    df_results['team_2_result'] = np.where(
        df_results['team_2_final_score'] > df_results['team_1_final_score'], 'win',
        np.where(df_results['team_2_final_score'] < df_results['team_1_final_score'], 'loss', 'draw')
    )
    
    # Calculate margin
    df_results['margin'] = abs(df_results['team_1_final_score'] - df_results['team_2_final_score'])
    
    return df_results

def extract_round_number(round_str: str) -> int:
    """
    Extract numeric round number from round string (handles finals, etc.)
    
    Args:
        round_str: String representation of round (e.g., "Round 1", "Semi Final")
        
    Returns:
        Integer round number (finals are converted to rounds after regular season)
    """
    if pd.isna(round_str):
        return np.nan
    
    # Convert to string if not already
    round_str = str(round_str)
    
    # Regular season round
    match = re.search(r'(\d+)', round_str)
    if match:
        return int(match.group(1))
    
    # Finals mapping
    finals_mapping = {
        'qualifying final': 24,
        'elimination final': 24,
        'semi final': 25,
        'preliminary final': 26,
        'grand final': 27
    }
    
    for final_type, round_num in finals_mapping.items():
        if final_type in round_str.lower():
            return round_num
    
    return np.nan

def process_player_data(player_perf: pd.DataFrame, player_personal: pd.DataFrame) -> pd.DataFrame:
    """
    Process player data by combining performance and personal details.
    
    Args:
        player_perf: Player performance DataFrame
        player_personal: Player personal details DataFrame
        
    Returns:
        Processed player DataFrame
    """
    if player_perf.empty or player_personal.empty:
        return pd.DataFrame()
    
    # Combine the data
    player_data = player_perf.copy()
    
    # Add personal details as columns
    for col in player_personal.columns:
        if col not in player_data.columns and len(player_personal[col]) > 0:
            player_data[f'personal_{col}'] = player_personal[col].iloc[0]
    
    # Convert data types
    numeric_cols = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 
                    'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances',
                    'clangers', 'free_kicks_for', 'free_kicks_against', 'brownlow_votes']
    
    for col in numeric_cols:
        if col in player_data.columns:
            player_data[col] = pd.to_numeric(player_data[col], errors='coerce')
    
    return player_data

def calculate_team_form(match_data: pd.DataFrame, team_name: str, 
                        date: pd.Timestamp, n_matches: int = 5) -> Dict[str, float]:
    """
    Calculate team form metrics based on their recent matches.
    
    Args:
        match_data: DataFrame with all match data
        team_name: Name of the team to calculate form for
        date: Date to calculate form up to (exclude matches after this date)
        n_matches: Number of previous matches to consider
        
    Returns:
        Dictionary of form metrics
    """
    # Filter matches up to the given date
    prior_matches = match_data[match_data['date'] < date].copy()
    
    # Find matches where the team played
    team1_matches = prior_matches[prior_matches['team_1_team_name'] == team_name].copy()
    team2_matches = prior_matches[prior_matches['team_2_team_name'] == team_name].copy()
    
    # Combine and sort by date (most recent first)
    team1_matches['team_position'] = 1
    team2_matches['team_position'] = 2
    team_matches = pd.concat([team1_matches, team2_matches])
    team_matches = team_matches.sort_values('date', ascending=False).head(n_matches)
    
    if team_matches.empty:
        return {
            'recent_win_pct': np.nan,
            'recent_avg_score': np.nan,
            'recent_avg_score_against': np.nan,
            'recent_avg_margin': np.nan
        }
    
    # Calculate form metrics
    results = []
    scores = []
    scores_against = []
    margins = []
    
    for _, match in team_matches.iterrows():
        if match['team_position'] == 1:
            # Team played as team 1
            result = match['team_1_result'] if 'team_1_result' in match else np.nan
            score = match['team_1_final_score'] if 'team_1_final_score' in match else np.nan
            score_against = match['team_2_final_score'] if 'team_2_final_score' in match else np.nan
            margin = score - score_against if not np.isnan(score) and not np.isnan(score_against) else np.nan
        else:
            # Team played as team 2
            result = match['team_2_result'] if 'team_2_result' in match else np.nan
            score = match['team_2_final_score'] if 'team_2_final_score' in match else np.nan
            score_against = match['team_1_final_score'] if 'team_1_final_score' in match else np.nan
            margin = score - score_against if not np.isnan(score) and not np.isnan(score_against) else np.nan
        
        results.append(result)
        scores.append(score)
        scores_against.append(score_against)
        margins.append(margin)
    
    # Calculate averages
    win_pct = results.count('win') / len(results) if results else np.nan
    avg_score = np.nanmean(scores) if scores else np.nan
    avg_score_against = np.nanmean(scores_against) if scores_against else np.nan
    avg_margin = np.nanmean(margins) if margins else np.nan
    
    return {
        'recent_win_pct': win_pct,
        'recent_avg_score': avg_score,
        'recent_avg_score_against': avg_score_against,
        'recent_avg_margin': avg_margin
    }

def create_match_features(match_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for match prediction from raw match data.
    
    Args:
        match_data: DataFrame containing match data
        
    Returns:
        DataFrame with engineered features for match prediction
    """
    features = match_data.copy()
    
    # Ensure we have date as datetime
    if 'date' in features.columns:
        features = standardize_dates(features, 'date')
    
    # Calculate match results if not already present
    if 'team_1_result' not in features.columns:
        features = calculate_match_result(features)
    
    # Create round number feature if needed
    if 'round_num' in features.columns and not pd.api.types.is_numeric_dtype(features['round_num']):
        features['round_number'] = features['round_num'].apply(extract_round_number)
    
    # Standardize team names
    team_columns = [col for col in features.columns if 'team' in col and 'name' in col]
    features = clean_team_names(features, team_columns)
    
    # Create home/away indicator
    # In AFL, team_1 is traditionally the home team, but we should verify this in the data
    if 'venue' in features.columns:
        features['is_team1_home'] = True  # Simplified assumption for now
    
    # Calculate team form for each match (previous n matches)
    # This would be more complex in reality - needs careful handling of time-based features
    
    return features 