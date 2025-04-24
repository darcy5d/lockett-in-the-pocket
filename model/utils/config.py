#!/usr/bin/env python3
"""
Configuration Manager

This module handles loading and accessing configuration settings for the model.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('config')

class ConfigManager:
    """
    Class to manage configuration settings for the model.
    
    This class loads configuration from a JSON file and provides
    methods to access and validate configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        if config_path is None:
            # Default config path
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config',
                'model_config.json'
            )
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary containing configuration settings
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file {self.config_path} not found. Using default settings.")
            return self._default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Dictionary containing default configuration settings
        """
        return {
            "model": {
                "algorithm": "xgboost",
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "random_state": 42
                }
            },
            "feature_engineering": {
                "team_form_matches": 5,
                "include_player_features": False,
                "include_odds_features": True,
                "standardize_features": True
            },
            "training": {
                "test_size": 0.2,
                "use_time_split": True,
                "cutoff_year": None
            },
            "betting": {
                "strategies": [
                    {
                        "name": "Flat (1%)",
                        "stake_method": "flat",
                        "flat_stake_percent": 1.0,
                        "threshold": 0.0
                    },
                    {
                        "name": "Kelly (50%)",
                        "stake_method": "kelly",
                        "kelly_fraction": 0.5,
                        "threshold": 0.0
                    }
                ],
                "initial_bankroll": 1000.0,
                "minimum_edge": 0.02
            },
            "paths": {
                "match_data": "afl_data/data/matches",
                "player_data": "afl_data/data/players",
                "lineups_data": "afl_data/data/lineups",
                "odds_data": "afl_data/odds_data/odds_data_2009_to_present.xlsx",
                "feature_output": "model/data/match_features.csv",
                "model_output": "model/output/baseline_model.pkl"
            }
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters from configuration.
        
        Returns:
            Dictionary of model parameters
        """
        return self.config.get('model', {}).get('parameters', {})
    
    def get_feature_engineering_params(self) -> Dict[str, Any]:
        """
        Get feature engineering parameters from configuration.
        
        Returns:
            Dictionary of feature engineering parameters
        """
        return self.config.get('feature_engineering', {})
    
    def get_training_params(self) -> Dict[str, Any]:
        """
        Get training parameters from configuration.
        
        Returns:
            Dictionary of training parameters
        """
        return self.config.get('training', {})
    
    def get_betting_params(self) -> Dict[str, Any]:
        """
        Get betting parameters from configuration.
        
        Returns:
            Dictionary of betting parameters
        """
        return self.config.get('betting', {})
    
    def get_paths(self) -> Dict[str, str]:
        """
        Get file path settings from configuration.
        
        Returns:
            Dictionary of file paths
        """
        return self.config.get('paths', {})
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save the current configuration to file.
        
        Args:
            config_path: Path to save the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        if config_path is None:
            config_path = self.config_path
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def update_config(self, new_config: Dict[str, Any], save: bool = True) -> bool:
        """
        Update configuration with new settings.
        
        Args:
            new_config: Dictionary containing new configuration settings
            save: Whether to save the updated configuration to file
            
        Returns:
            True if successful, False otherwise
        """
        # Update configuration
        self._recursive_update(self.config, new_config)
        
        # Save configuration if requested
        if save:
            return self.save_config()
        
        return True
    
    def _recursive_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary with another dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._recursive_update(d[k], v)
            else:
                d[k] = v

def get_config() -> ConfigManager:
    """
    Get a ConfigManager instance.
    
    Returns:
        ConfigManager instance
    """
    return ConfigManager()

def main():
    """Example usage of the ConfigManager."""
    config = get_config()
    print("Model parameters:")
    print(json.dumps(config.get_model_params(), indent=2))
    
    print("\nFeature engineering parameters:")
    print(json.dumps(config.get_feature_engineering_params(), indent=2))
    
    print("\nTraining parameters:")
    print(json.dumps(config.get_training_params(), indent=2))
    
    print("\nBetting parameters:")
    print(json.dumps(config.get_betting_params(), indent=2))
    
    print("\nFile paths:")
    print(json.dumps(config.get_paths(), indent=2))

if __name__ == "__main__":
    main() 