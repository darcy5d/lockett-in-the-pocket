#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_simple_predictor')

# Add the project root to the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Import the SimplePredictor
from gui.simple_predictor import SimplePredictor

def test_simple_predictor():
    """
    Test the SimplePredictor with a sample match prediction
    """
    logger.info("Testing SimplePredictor...")
    
    # Initialize the predictor
    predictor = SimplePredictor()
    
    # Define a test match
    test_match = {
        'home_team': 'Geelong Cats',
        'away_team': 'Sydney Swans',
        'venue': 'MCG',
        'round_num': 1,
        'season': 2023
    }
    
    # Make a prediction
    logger.info(f"Predicting match: {test_match}")
    result = predictor.predict_match(test_match)
    
    # Print the result
    logger.info("Prediction result:")
    for key, value in result.items():
        logger.info(f"  {key}: {value}")
    
    # Test another match
    test_match2 = {
        'home_team': 'Richmond',
        'away_team': 'Carlton',
        'venue': 'MCG',
        'round_num': 1,
        'season': 2023
    }
    
    logger.info(f"Predicting match: {test_match2}")
    result2 = predictor.predict_match(test_match2)
    
    # Print the result
    logger.info("Prediction result:")
    for key, value in result2.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    test_simple_predictor() 