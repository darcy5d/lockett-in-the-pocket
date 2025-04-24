#!/usr/bin/env python3
"""
AFL Data Exploration Script

This script loads data from the AFL dataset and performs exploratory data analysis.
The goal is to understand the key characteristics of the data before model development.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import the data loader
from datafetch.load_afl_data import AFLDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('afl_data_exploration')

# Create output directory for plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

class AFLDataExplorer:
    """Class for exploring and analyzing AFL data."""
    
    def __init__(self):
        """Initialize the data explorer with the data loader."""
        self.loader = AFLDataLoader()
        self.matches_data = None
        self.player_data_sample = None
        self.odds_data = None
        
    def load_data(self):
        """Load the required datasets."""
        logger.info("Loading match data...")
        self.matches_data = self.loader.load_matches()
        
        logger.info("Loading a sample of player data...")
        # Get a list of available players
        player_list = self.loader.load_player_data()
        # Sample 10 players for exploration
        sample_players = np.random.choice(player_list, min(10, len(player_list)))
        self.player_data_sample = {}
        for player_name in sample_players:
            try:
                perf_data = self.loader.load_player_data(player_name, data_type='performance')
                personal_data = self.loader.load_player_data(player_name, data_type='personal')
                if perf_data is not None and personal_data is not None:
                    self.player_data_sample[player_name] = {
                        'performance': perf_data,
                        'personal': personal_data
                    }
            except Exception as e:
                logger.warning(f"Error loading player data for {player_name}: {e}")
        
        logger.info("Loading odds data...")
        self.odds_data = self.loader.load_odds_data()
    
    def explore_matches_data(self):
        """Explore and visualize the match data."""
        if self.matches_data is None:
            logger.error("Match data not loaded. Call load_data() first.")
            return
        
        logger.info("Exploring match data...")
        
        # Basic information
        logger.info(f"Match data shape: {self.matches_data.shape}")
        logger.info(f"Time period covered: {self.matches_data['year'].min()} to {self.matches_data['year'].max()}")
        
        # Missing values
        missing_values = self.matches_data.isnull().sum()
        missing_percent = (missing_values / len(self.matches_data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percent
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
        if not missing_df.empty:
            logger.info("Columns with missing values:")
            logger.info(f"\n{missing_df}")
        
        # Team statistics
        if 'team_1_team_name' in self.matches_data.columns:
            team1_counts = self.matches_data['team_1_team_name'].value_counts()
            team2_counts = self.matches_data['team_2_team_name'].value_counts()
            team_counts = pd.concat([team1_counts, team2_counts]).groupby(level=0).sum().sort_values(ascending=False)
            logger.info("Number of matches per team:")
            logger.info(f"\n{team_counts.head(10)}")
        
        # Score distribution
        if 'team_1_final_goals' in self.matches_data.columns and 'team_1_final_behinds' in self.matches_data.columns:
            # Calculate total scores
            self.matches_data['team_1_final_score'] = self.matches_data['team_1_final_goals'] * 6 + self.matches_data['team_1_final_behinds']
            self.matches_data['team_2_final_score'] = self.matches_data['team_2_final_goals'] * 6 + self.matches_data['team_2_final_behinds']
            
            # Score statistics
            team1_scores = self.matches_data['team_1_final_score']
            team2_scores = self.matches_data['team_2_final_score']
            all_scores = pd.concat([team1_scores, team2_scores])
            
            logger.info("Score statistics:")
            logger.info(f"Mean score: {all_scores.mean():.2f}")
            logger.info(f"Median score: {all_scores.median():.2f}")
            logger.info(f"Min score: {all_scores.min():.2f}")
            logger.info(f"Max score: {all_scores.max():.2f}")
            
            # Plot score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(all_scores, kde=True)
            plt.title('Distribution of Team Scores')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(PLOTS_DIR, 'score_distribution.png'))
            plt.close()
            
            # Plot scores over time
            if 'year' in self.matches_data.columns:
                scores_by_year = self.matches_data.groupby('year').agg({
                    'team_1_final_score': 'mean',
                    'team_2_final_score': 'mean'
                })
                
                all_scores_by_year = pd.DataFrame({
                    'year': scores_by_year.index,
                    'avg_score': (scores_by_year['team_1_final_score'] + scores_by_year['team_2_final_score']) / 2
                })
                
                plt.figure(figsize=(12, 6))
                sns.lineplot(x='year', y='avg_score', data=all_scores_by_year)
                plt.title('Average Score by Year')
                plt.xlabel('Year')
                plt.ylabel('Average Score')
                plt.savefig(os.path.join(PLOTS_DIR, 'scores_by_year.png'))
                plt.close()
        
        # Venue analysis
        if 'venue' in self.matches_data.columns:
            venue_counts = self.matches_data['venue'].value_counts()
            logger.info("Top 10 most common venues:")
            logger.info(f"\n{venue_counts.head(10)}")
            
            # Plot venue distribution
            plt.figure(figsize=(12, 8))
            venue_counts.head(15).plot(kind='bar')
            plt.title('Top 15 Most Common Venues')
            plt.xlabel('Venue')
            plt.ylabel('Number of Matches')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'venue_distribution.png'))
            plt.close()
    
    def explore_player_data(self):
        """Explore and visualize the player data."""
        if not self.player_data_sample:
            logger.error("Player data not loaded. Call load_data() first.")
            return
        
        logger.info("Exploring player data...")
        
        # Aggregate performance data
        all_performance = pd.concat([data['performance'] for data in self.player_data_sample.values()])
        
        logger.info(f"Sample player performance data shape: {all_performance.shape}")
        
        # Missing values
        missing_values = all_performance.isnull().sum()
        missing_percent = (missing_values / len(all_performance)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percent
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
        if not missing_df.empty:
            logger.info("Columns with missing values in player performance data:")
            logger.info(f"\n{missing_df}")
        
        # Key statistics
        numeric_cols = all_performance.select_dtypes(include=[np.number]).columns
        stats_summary = all_performance[numeric_cols].describe().T
        logger.info("Summary statistics for player performance metrics:")
        logger.info(f"\n{stats_summary[['mean', 'std', 'min', 'max']]}")
        
        # Plot distributions of key metrics
        key_metrics = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 'tackles']
        key_metrics = [col for col in key_metrics if col in all_performance.columns]
        
        if key_metrics:
            plt.figure(figsize=(15, 10))
            for i, metric in enumerate(key_metrics[:6], 1):  # Limit to 6 metrics
                plt.subplot(2, 3, i)
                sns.histplot(all_performance[metric].dropna(), kde=True)
                plt.title(f'Distribution of {metric.capitalize()}')
                plt.xlabel(metric.capitalize())
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'player_metrics_distribution.png'))
            plt.close()
            
        # Player career progression (for a single player)
        if self.player_data_sample:
            sample_player = list(self.player_data_sample.keys())[0]
            player_data = self.player_data_sample[sample_player]['performance']
            
            if 'disposals' in player_data.columns and 'year' in player_data.columns:
                plt.figure(figsize=(12, 6))
                sns.lineplot(x='year', y='disposals', data=player_data)
                plt.title(f'Career Progression - Disposals for {sample_player}')
                plt.xlabel('Year')
                plt.ylabel('Disposals')
                plt.savefig(os.path.join(PLOTS_DIR, 'player_career_progression.png'))
                plt.close()
    
    def explore_odds_data(self):
        """Explore and visualize the odds data."""
        if self.odds_data is None:
            logger.warning("Odds data not loaded or not available.")
            return
        
        logger.info("Exploring odds data...")
        
        # Basic information
        logger.info(f"Odds data shape: {self.odds_data.shape}")
        
        # Check for column names
        logger.info(f"Odds data columns: {self.odds_data.columns.tolist()}")
        
        # Missing values
        missing_values = self.odds_data.isnull().sum()
        missing_percent = (missing_values / len(self.odds_data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percent
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
        if not missing_df.empty:
            logger.info("Columns with missing values in odds data:")
            logger.info(f"\n{missing_df.head(10)}")  # Show only top 10
            
        # If the odds data has pricing columns, analyze them
        price_columns = [col for col in self.odds_data.columns if 'price' in col.lower()]
        if price_columns:
            for col in price_columns[:5]:  # Limit to 5 price columns
                if self.odds_data[col].dtype in [np.float64, np.int64]:
                    logger.info(f"Statistics for {col}:")
                    logger.info(f"Mean: {self.odds_data[col].mean():.2f}")
                    logger.info(f"Median: {self.odds_data[col].median():.2f}")
                    logger.info(f"Min: {self.odds_data[col].min():.2f}")
                    logger.info(f"Max: {self.odds_data[col].max():.2f}")
                    
                    # Plot price distribution
                    plt.figure(figsize=(10, 6))
                    sns.histplot(self.odds_data[col].dropna(), kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.savefig(os.path.join(PLOTS_DIR, f'odds_{col}_distribution.png'))
                    plt.close()

    def run_exploration(self):
        """Run all exploration steps."""
        self.load_data()
        self.explore_matches_data()
        self.explore_player_data()
        self.explore_odds_data()
        logger.info(f"Exploration complete. Plots saved to {PLOTS_DIR}")

def main():
    """Main function to run the exploration."""
    explorer = AFLDataExplorer()
    explorer.run_exploration()

if __name__ == "__main__":
    main() 