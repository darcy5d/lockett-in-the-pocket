#!/usr/bin/env python3
"""
AFL Betting Model - Main Script

This script runs the complete pipeline from data loading,
feature engineering, model training, to betting strategy evaluation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import project modules
from model.data_exploration.explore_data import AFLDataExplorer
from model.feature_engineering.generate_features import AFLFeatureGenerator
from model.models.baseline_model import AFLBaselineModel
from model.utils.betting_evaluation import (
    simulate_betting_strategies,
    plot_strategy_comparison
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'afl_model.log'))
    ]
)
logger = logging.getLogger('afl_model_main')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run AFL betting model pipeline')
    
    parser.add_argument('--explore', action='store_true', help='Run data exploration')
    parser.add_argument('--generate-features', action='store_true', help='Generate features')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate betting strategies')
    parser.add_argument('--cutoff-year', type=int, help='Year to use as cutoff for training data')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    return parser.parse_args()

def run_exploration():
    """Run data exploration."""
    logger.info("Starting data exploration...")
    explorer = AFLDataExplorer()
    explorer.run_exploration()
    logger.info("Data exploration complete.")

def run_feature_generation(cutoff_year=None):
    """Generate features for model training."""
    logger.info("Starting feature generation...")
    generator = AFLFeatureGenerator(data_cutoff_year=cutoff_year)
    features = generator.generate_all_features()
    logger.info(f"Feature generation complete. Generated {features.shape[1]} features for {features.shape[0]} matches.")
    return features

def run_model_training():
    """Train the baseline model."""
    logger.info("Starting model training...")
    model = AFLBaselineModel()
    
    # Load features
    features = model.load_features()
    if features.empty:
        logger.error("Failed to load features. Exiting.")
        return None
    
    # Prepare data
    X_train, X_val, y_train, y_val = model.prepare_data(test_size=0.2, time_based_split=True)
    if X_train is None:
        logger.error("Failed to prepare data. Exiting.")
        return None
    
    # Train model
    model.train_model()
    
    # Evaluate model
    metrics = model.evaluate_model()
    
    # Save model
    model.save_model()
    
    logger.info("Model training complete.")
    return model

def run_betting_evaluation(model=None):
    """Evaluate betting strategies using model predictions."""
    logger.info("Starting betting strategy evaluation...")
    
    # Load model if not provided
    if model is None:
        model = AFLBaselineModel()
        model.load_model()
    
    # Load features
    features = model.load_features()
    if features.empty:
        logger.error("Failed to load features. Exiting.")
        return
    
    # Prepare data
    _, X_val, _, y_val = model.prepare_data(test_size=0.2, time_based_split=True)
    if X_val is None:
        logger.error("Failed to prepare data. Exiting.")
        return
    
    # Generate predictions
    pred_probs = model.predict(X_val)
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'match_idx': X_val.index,
        'pred_probability': pred_probs
    })
    
    # In a real scenario, we would load actual odds data here
    # For this example, we'll create synthetic odds data
    # This is just for demonstration - in reality, you would use real odds data
    odds_data = pd.DataFrame({
        'match_idx': X_val.index,
        'odds': 1.0 / pred_probs + 0.1,  # Synthetic odds with bookmaker margin
        'actual_result': ['win' if result else 'loss' for result in y_val]
    })
    
    # Simulate different betting strategies
    strategy_results = simulate_betting_strategies(
        predictions=predictions,
        odds_data=odds_data,
        initial_bankroll=1000.0
    )
    
    # Plot strategy comparison
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'output',
        'strategy_comparison.png'
    )
    plot_strategy_comparison(strategy_results, output_path)
    
    logger.info("Betting strategy evaluation complete.")
    return strategy_results

def main():
    """Main function to run the entire pipeline."""
    args = parse_args()
    
    # Determine which steps to run
    run_all = args.all
    run_exp = args.explore or run_all
    run_feat = args.generate_features or run_all
    run_train = args.train or run_all
    run_eval = args.evaluate or run_all
    cutoff_year = args.cutoff_year
    
    # Create output directories
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'), exist_ok=True)
    
    # Run selected steps
    if run_exp:
        run_exploration()
    
    if run_feat:
        run_feature_generation(cutoff_year)
    
    model = None
    if run_train:
        model = run_model_training()
    
    if run_eval:
        run_betting_evaluation(model)
    
    logger.info("AFL betting model pipeline complete.")

if __name__ == "__main__":
    main() 