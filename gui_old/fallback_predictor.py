import os
import numpy as np
import json
import logging
import random
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fallback_predictor')

class FallbackPredictor:
    """
    A simple fallback predictor that uses basic statistics to predict AFL match outcomes
    when the neural network model cannot be loaded.
    """
    
    def __init__(self):
        """Initialize the fallback predictor"""
        self.resources_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
        self.team_stats = {}
        self.venues = {}
        self.load_team_stats()
        logger.info("Fallback predictor initialized")
    
    def load_team_stats(self):
        """Load team statistics from file or create default stats if file doesn't exist"""
        stats_path = os.path.join(self.resources_dir, 'team_stats.json')
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    self.team_stats = json.load(f)
                logger.info(f"Loaded team stats for {len(self.team_stats)} teams")
            except Exception as e:
                logger.error(f"Error loading team stats: {e}")
                self.create_default_stats()
        else:
            logger.warning("Team stats file not found, creating default stats")
            self.create_default_stats()
    
    def create_default_stats(self):
        """Create default team statistics based on historical AFL data"""
        # Define default team stats based on average AFL match scores
        # Format: {team_name: {'avg_goals': float, 'avg_behinds': float, 'home_advantage': float}}
        self.team_stats = {
            'Adelaide Crows': {'avg_goals': 12.5, 'avg_behinds': 11.8, 'home_advantage': 1.2},
            'Brisbane Lions': {'avg_goals': 14.2, 'avg_behinds': 12.5, 'home_advantage': 1.3},
            'Carlton': {'avg_goals': 11.8, 'avg_behinds': 10.5, 'home_advantage': 1.1},
            'Collingwood': {'avg_goals': 12.7, 'avg_behinds': 11.2, 'home_advantage': 1.15},
            'Essendon': {'avg_goals': 12.0, 'avg_behinds': 11.0, 'home_advantage': 1.1},
            'Fremantle': {'avg_goals': 11.5, 'avg_behinds': 10.8, 'home_advantage': 1.25},
            'Geelong Cats': {'avg_goals': 13.8, 'avg_behinds': 11.5, 'home_advantage': 1.2},
            'Gold Coast Suns': {'avg_goals': 10.5, 'avg_behinds': 10.2, 'home_advantage': 1.1},
            'Greater Western Sydney': {'avg_goals': 12.2, 'avg_behinds': 11.0, 'home_advantage': 1.15},
            'Hawthorn': {'avg_goals': 12.5, 'avg_behinds': 11.3, 'home_advantage': 1.1},
            'Melbourne': {'avg_goals': 13.0, 'avg_behinds': 11.8, 'home_advantage': 1.15},
            'North Melbourne': {'avg_goals': 10.2, 'avg_behinds': 9.8, 'home_advantage': 1.05},
            'Port Adelaide': {'avg_goals': 13.0, 'avg_behinds': 11.5, 'home_advantage': 1.2},
            'Richmond': {'avg_goals': 12.8, 'avg_behinds': 11.0, 'home_advantage': 1.15},
            'St Kilda': {'avg_goals': 11.5, 'avg_behinds': 10.8, 'home_advantage': 1.1},
            'Sydney Swans': {'avg_goals': 13.2, 'avg_behinds': 11.5, 'home_advantage': 1.2},
            'West Coast Eagles': {'avg_goals': 12.0, 'avg_behinds': 11.0, 'home_advantage': 1.3},
            'Western Bulldogs': {'avg_goals': 12.5, 'avg_behinds': 11.2, 'home_advantage': 1.1}
        }
        
        # Define default venue advantage
        self.venues = {
            'MCG': 1.05,
            'SCG': 1.1,
            'Adelaide Oval': 1.15,
            'Optus Stadium': 1.2,
            'Gabba': 1.15,
            'Marvel Stadium': 1.05,
            'GMHBA Stadium': 1.25,
            'GIANTS Stadium': 1.1,
            'Blundstone Arena': 1.1,
            'TIO Stadium': 1.05,
            'University of Tasmania Stadium': 1.15,
            'Metricon Stadium': 1.1
        }
        
        # Save the default stats
        os.makedirs(self.resources_dir, exist_ok=True)
        stats_path = os.path.join(self.resources_dir, 'team_stats.json')
        try:
            with open(stats_path, 'w') as f:
                json.dump(self.team_stats, f, indent=2)
            logger.info("Saved default team stats")
        except Exception as e:
            logger.error(f"Error saving default team stats: {e}")
    
    def predict_match(self, match_data):
        """
        Predict the outcome of a match using simple statistics.
        
        Args:
            match_data (dict): Dictionary containing match information:
                - home_team: Name of the home team
                - away_team: Name of the away team
                - venue: Name of the venue
                - round_num: Round number (optional)
                - season: Season year (optional)
        
        Returns:
            dict: Prediction results
        """
        # Extract match information
        home_team = match_data.get('home_team', '')
        away_team = match_data.get('away_team', '')
        venue = match_data.get('venue', '')
        
        # Validate input
        if not home_team or not away_team:
            return {"error": "Missing team information"}
        
        # Get team stats (use default values if team not found)
        default_stats = {'avg_goals': 12.0, 'avg_behinds': 11.0, 'home_advantage': 1.1}
        home_stats = self.team_stats.get(home_team, default_stats)
        away_stats = self.team_stats.get(away_team, default_stats)
        
        # Get venue advantage
        venue_advantage = self.venues.get(venue, 1.0)
        
        # Calculate home advantage
        home_advantage = home_stats['home_advantage'] * venue_advantage
        
        # Add some randomness to make predictions more realistic
        random_factor_home = random.uniform(0.85, 1.15)
        random_factor_away = random.uniform(0.85, 1.15)
        
        # Calculate predicted goals and behinds with home advantage and randomness
        home_goals = home_stats['avg_goals'] * home_advantage * random_factor_home
        home_behinds = home_stats['avg_behinds'] * home_advantage * random_factor_home
        
        away_goals = away_stats['avg_goals'] * random_factor_away
        away_behinds = away_stats['avg_behinds'] * random_factor_away
        
        # Calculate scores
        home_score = home_goals * 6 + home_behinds
        away_score = away_goals * 6 + away_behinds
        
        # Calculate margin
        margin = home_score - away_score
        
        # Determine winner
        if margin > 0:
            winner = 'home'
            # Calculate win probability (higher for larger margins)
            win_probability = 0.5 + min(0.45, abs(margin) / 100)
        elif margin < 0:
            winner = 'away'
            win_probability = 0.5 + min(0.45, abs(margin) / 100)
        else:
            # In case of a draw, randomly pick a winner with 50% probability
            if random.random() > 0.5:
                winner = 'home'
            else:
                winner = 'away'
            win_probability = 0.5
        
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
            'win_probability': round(win_probability, 3),
            'note': 'Prediction made using statistical fallback model'
        }
        
        logger.info(f"Fallback prediction for {home_team} vs {away_team}: {winner} wins by {abs(round(margin, 1))}")
        return result 