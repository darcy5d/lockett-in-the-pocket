import os
import glob
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Any, Optional

# Set the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Player statistics cache
_player_stats_cache = None
_player_index_cache = None

def load_player_index() -> Dict[str, int]:
    """
    Load the player index from the saved model files.
    
    Returns:
        Dict[str, int]: Dictionary mapping player IDs to indices
    """
    global _player_index_cache
    
    # Return cached version if available
    if _player_index_cache is not None:
        return _player_index_cache
    
    player_index_path = os.path.join(PROJECT_ROOT, 'model/output/player_index.json')
    
    try:
        with open(player_index_path, 'r') as f:
            player_index = json.load(f)
        
        # Convert string keys back to original format if needed
        _player_index_cache = player_index
        return player_index
    except Exception as e:
        print(f"Error loading player index: {e}")
        # Create a new player index if loading fails
        return create_player_index()

def create_player_index() -> Dict[str, int]:
    """
    Create player index lookup for embeddings.
    
    Returns:
        Dict[str, int]: Dictionary mapping player IDs to indices
    """
    global _player_index_cache
    
    print("\nCreating player index for embeddings...")
    
    # Get all player files
    player_performance_files = glob.glob(os.path.join(PROJECT_ROOT, 'afl_data/data/players/*_performance_details.csv'))
    
    # Create a set of all unique player IDs
    player_ids = set()
    for file in player_performance_files:
        # Extract player ID from filename 
        player_id = os.path.basename(file).split('_performance')[0]
        player_ids.add(player_id)
    
    # Create player index lookup
    player_index = {player_id: i+1 for i, player_id in enumerate(sorted(player_ids))}
    
    # Add 0 for unknown/missing players
    player_index['unknown'] = 0
    
    print(f"Created player index with {len(player_index)} players")
    _player_index_cache = player_index
    return player_index

def extract_player_stats() -> Dict[str, Dict[str, float]]:
    """
    Extract player performance statistics from player detail files.
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping player IDs to their performance stats
    """
    global _player_stats_cache
    
    # Return cached version if available
    if _player_stats_cache is not None:
        return _player_stats_cache
        
    print("\nExtracting player performance statistics...")
    
    # Get all player performance files
    player_performance_files = glob.glob(os.path.join(PROJECT_ROOT, 'afl_data/data/players/*_performance_details.csv'))
    
    # Dictionary to store player stats
    player_stats = {}
    
    # All metrics we care about
    metrics = [
        'kicks', 'marks', 'handballs', 'goals', 'behinds', 
        'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances',
        'clangers', 'free_kicks_for', 'free_kicks_against', 
        'contested_possessions', 'uncontested_possessions', 'contested_marks',
        'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist',
        'percentage_of_game_played'
    ]
    
    for file in player_performance_files:
        try:
            # Extract player ID from filename 
            player_id = os.path.basename(file).split('_performance')[0]
            
            # Read the performance data
            df = pd.read_csv(file)
            
            # Only process if it has enough required columns
            # We need at least the basic columns to be useful
            essential_columns = ['goals', 'behinds', 'kicks', 'marks', 'handballs']
            if not all(col in df.columns for col in essential_columns):
                continue
                
            # Get the recent games (last 5 years)
            recent_years = [2020, 2021, 2022, 2023, 2024, 2025]
            recent_games = df[df['year'].isin(recent_years)]
            
            # If no recent games, use all available data
            if len(recent_games) == 0:
                recent_games = df
            
            # Calculate averages if there's data
            if len(recent_games) > 0:
                # Create dictionary to store stats
                player_stat_dict = {'games_played': len(recent_games)}
                
                # Add player name if available
                if 'player_name' in df.columns:
                    player_stat_dict['name'] = df['player_name'].iloc[0]
                elif 'player_first_name' in df.columns and 'player_surname' in df.columns:
                    player_stat_dict['name'] = f"{df['player_first_name'].iloc[0]} {df['player_surname'].iloc[0]}"
                else:
                    player_stat_dict['name'] = f"Player {player_id}"
                
                # Process each metric
                for metric in metrics:
                    if metric in recent_games.columns:
                        # Calculate average, handling NaN values
                        avg_value = recent_games[metric].dropna().mean() 
                        player_stat_dict[f'avg_{metric}'] = avg_value if not np.isnan(avg_value) else 0
                    else:
                        # Set to 0 if column doesn't exist
                        player_stat_dict[f'avg_{metric}'] = 0
                
                # Store the statistics
                player_stats[player_id] = player_stat_dict
                
        except Exception as e:
            # Just skip any problem files
            print(f"Error processing {file}: {e}")
            continue
    
    print(f"Extracted performance stats for {len(player_stats)} players")
    _player_stats_cache = player_stats
    return player_stats

def get_player_stats_by_id(player_id: str) -> Dict[str, Any]:
    """
    Get performance statistics for a specific player.
    
    Args:
        player_id (str): The player ID to look up
        
    Returns:
        Dict[str, Any]: Dictionary of player statistics or empty dict if not found
    """
    player_stats = extract_player_stats()
    return player_stats.get(player_id, {})

def get_player_stats_for_team(player_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get performance statistics for all players in a team.
    
    Args:
        player_ids (List[str]): List of player IDs in the team
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing player statistics
    """
    player_stats = extract_player_stats()
    return [player_stats.get(player_id, {}) for player_id in player_ids]

def get_team_player_indices(player_ids: List[str], max_players: int = 22) -> np.ndarray:
    """
    Convert a list of player IDs to their embedding indices with padding.
    
    Args:
        player_ids (List[str]): List of player IDs
        max_players (int): Maximum number of players to include
        
    Returns:
        np.ndarray: Array of player indices for model input
    """
    player_index = load_player_index()
    
    # Convert player IDs to indices
    indices = [player_index.get(str(player_id), 0) for player_id in player_ids]
    
    # Pad or truncate to max_players
    if len(indices) < max_players:
        indices.extend([0] * (max_players - len(indices)))
    elif len(indices) > max_players:
        indices = indices[:max_players]
        
    return np.array(indices).reshape(1, -1)  # Shape for model input: (1, max_players)

def get_team_enhanced_features(player_ids: List[str]) -> np.ndarray:
    """
    Convert a list of player IDs to their enhanced performance features.
    
    Args:
        player_ids (List[str]): List of player IDs
        
    Returns:
        np.ndarray: Array of enhanced features for model input
    """
    player_stats = extract_player_stats()
    
    # Metrics to include
    metrics = [
        'kicks', 'marks', 'handballs', 'goals', 'behinds', 
        'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances',
        'clangers', 'free_kicks_for', 'free_kicks_against', 
        'contested_possessions', 'uncontested_possessions', 'contested_marks',
        'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist',
        'percentage_of_game_played'
    ]
    
    # Number of features is the number of metrics
    num_features = len(metrics)
    
    # Get stats for each player
    team_stats = [player_stats.get(player_id, {}) for player_id in player_ids]
    
    # Initialize the enhanced features
    enhanced_features = np.zeros((1, num_features))  # Shape for model input: (1, num_features)
    
    if team_stats:
        # Get games played for weighting (minimum 1 game to avoid division by zero)
        team_games = np.array([s.get('games_played', 1) for s in team_stats])
        team_games = np.where(team_games > 0, team_games, 1)
        team_weights = team_games / np.sum(team_games) if np.sum(team_games) > 0 else np.ones_like(team_games) / len(team_games)
        
        # Calculate weighted averages for each feature
        for j, field in enumerate(metrics):
            avg_field = f'avg_{field}'
            enhanced_features[0, j] = np.sum([s.get(avg_field, 0) * w for s, w in zip(team_stats, team_weights) if s])
    
    return enhanced_features

def get_player_names_by_ids(player_ids: List[str]) -> Dict[str, str]:
    """
    Get names of players by their IDs.
    
    Args:
        player_ids (List[str]): List of player IDs
        
    Returns:
        Dict[str, str]: Dictionary mapping player IDs to their names
    """
    player_stats = extract_player_stats()
    return {
        player_id: player_stats.get(player_id, {}).get('name', f"Unknown player {player_id}")
        for player_id in player_ids
    }

def search_players_by_name(name_query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for players by name.
    
    Args:
        name_query (str): Name to search for
        limit (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of matching player info dictionaries
    """
    player_stats = extract_player_stats()
    
    # Search for matches
    matches = []
    name_query = name_query.lower()
    
    for player_id, stats in player_stats.items():
        player_name = stats.get('name', '').lower()
        if name_query in player_name:
            matches.append({
                'id': player_id,
                'name': stats.get('name', f"Player {player_id}"),
                'games_played': stats.get('games_played', 0),
                'avg_goals': stats.get('avg_goals', 0),
                'avg_behinds': stats.get('avg_behinds', 0)
            })
    
    # Sort by games played (more games = more reliable stats)
    matches.sort(key=lambda x: x['games_played'], reverse=True)
    
    return matches[:limit]

def get_top_players_by_stat(stat_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get top players by a specific statistic.
    
    Args:
        stat_name (str): The statistic to sort by (e.g., 'avg_goals')
        limit (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of top player info dictionaries
    """
    player_stats = extract_player_stats()
    
    # Get players with this stat
    players_with_stat = []
    for player_id, stats in player_stats.items():
        if stat_name in stats and stats.get('games_played', 0) >= 10:  # Min 10 games for reliability
            players_with_stat.append({
                'id': player_id,
                'name': stats.get('name', f"Player {player_id}"),
                'games_played': stats.get('games_played', 0),
                stat_name: stats.get(stat_name, 0)
            })
    
    # Sort by the stat (higher = better)
    players_with_stat.sort(key=lambda x: x[stat_name], reverse=True)
    
    return players_with_stat[:limit] 