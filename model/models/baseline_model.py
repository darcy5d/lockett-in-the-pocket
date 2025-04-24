#!/usr/bin/env python3
"""
Baseline Model for AFL Match Prediction

This module provides a baseline model for predicting AFL match outcomes
using XGBoost. It includes model training, evaluation, and prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('baseline_model')

# Define constants
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
FEATURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

class AFLBaselineModel:
    """
    Baseline model for AFL match prediction.
    
    This class implements a binary classification model to predict
    whether team_1 will win a match.
    """
    
    def __init__(self, model_params: Optional[Dict] = None):
        """
        Initialize the baseline model.
        
        Args:
            model_params: Optional parameters for the XGBoost model
        """
        self.features_df = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.model = None
        self.feature_importance = None
        
        # Default model parameters
        self.model_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        # Update with user-provided parameters if any
        if model_params:
            self.model_params.update(model_params)
    
    def load_features(self, feature_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the feature data.
        
        Args:
            feature_path: Path to the features CSV file
            
        Returns:
            DataFrame with features
        """
        if feature_path is None:
            feature_path = os.path.join(FEATURE_DIR, 'match_features.csv')
        
        if not os.path.exists(feature_path):
            logger.error(f"Feature file {feature_path} does not exist. Generate features first.")
            return pd.DataFrame()
        
        logger.info(f"Loading features from {feature_path}")
        self.features_df = pd.read_csv(feature_path)
        return self.features_df
    
    def prepare_data(self, test_size: float = 0.2, time_based_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            test_size: Proportion of data to use for testing
            time_based_split: Whether to split data based on time (use recent matches for testing)
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        if self.features_df is None:
            logger.error("Features not loaded. Call load_features() first.")
            return None, None, None, None
        
        # Create target variable
        if 'team_1_result' in self.features_df.columns:
            self.features_df['target'] = (self.features_df['team_1_result'] == 'win').astype(int)
        else:
            logger.error("Could not find team_1_result column to create target variable.")
            return None, None, None, None
        
        # Select features and target
        feature_cols = self._select_features()
        if not feature_cols:
            logger.error("No valid features found.")
            return None, None, None, None
        
        X = self.features_df[feature_cols]
        y = self.features_df['target']
        
        # Split into training and validation sets
        if time_based_split and 'date' in self.features_df.columns:
            # Sort by date
            sorted_indices = self.features_df['date'].sort_values().index
            split_idx = int(len(sorted_indices) * (1 - test_size))
            train_indices = sorted_indices[:split_idx]
            val_indices = sorted_indices[split_idx:]
            
            self.X_train = X.loc[train_indices]
            self.y_train = y.loc[train_indices]
            self.X_val = X.loc[val_indices]
            self.y_val = y.loc[val_indices]
        else:
            # Random split
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        logger.info(f"Data prepared: {self.X_train.shape[0]} training samples, {self.X_val.shape[0]} validation samples")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def _select_features(self) -> List[str]:
        """
        Select features for model training, excluding target and ID columns.
        
        Returns:
            List of feature column names
        """
        if self.features_df is None:
            return []
        
        # Columns to exclude
        exclude_patterns = [
            'team_1_result', 'team_2_result',  # Target variables
            'match_idx', 'index',  # ID columns
            'date', 'team_name',  # Date and team name columns
            'target'  # Target variable
        ]
        
        # Select numeric columns excluding the patterns above
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not any(excl in col for excl in exclude_patterns)]
        
        logger.info(f"Selected {len(feature_cols)} features for modeling")
        return feature_cols
    
    def train_model(self) -> xgb.XGBClassifier:
        """
        Train the XGBoost model.
        
        Returns:
            Trained XGBoost classifier
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared. Call prepare_data() first.")
            return None
        
        logger.info("Training XGBoost model...")
        
        # Initialize and train the model
        self.model = xgb.XGBClassifier(**self.model_params)
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
            verbose=False
        )
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Model training complete.")
        return self.model
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            return {}
        
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_val, y_pred)
        auc = roc_auc_score(self.y_val, y_pred_proba)
        loss = log_loss(self.y_val, y_pred_proba)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_val, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'log_loss': loss
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'confusion_matrix.png'))
        plt.close()
        
        # Plot feature importance
        self._plot_feature_importance()
        
        return metrics
    
    def _plot_feature_importance(self, top_n: int = 20):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not available.")
            return
        
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'feature_importance.png'))
        plt.close()
    
    def save_model(self, model_path: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            return
        
        if model_path is None:
            model_path = os.path.join(MODEL_OUTPUT_DIR, 'baseline_model.pkl')
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(self.model, model_path)
        
        # Save feature importance
        if self.feature_importance is not None:
            self.feature_importance.to_csv(os.path.join(MODEL_OUTPUT_DIR, 'feature_importance.csv'), index=False)
    
    def load_model(self, model_path: Optional[str] = None) -> xgb.XGBClassifier:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded XGBoost classifier
        """
        if model_path is None:
            model_path = os.path.join(MODEL_OUTPUT_DIR, 'baseline_model.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} does not exist.")
            return None
        
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            logger.error("Model not trained or loaded. Call train_model() or load_model() first.")
            return np.array([])
        
        return self.model.predict_proba(X)[:, 1]

def main():
    """Main function to train and evaluate the baseline model."""
    # Initialize model
    model = AFLBaselineModel()
    
    # Load features
    features = model.load_features()
    if features.empty:
        logger.error("Failed to load features. Exiting.")
        return
    
    # Prepare data
    X_train, X_val, y_train, y_val = model.prepare_data(test_size=0.2, time_based_split=True)
    if X_train is None:
        logger.error("Failed to prepare data. Exiting.")
        return
    
    # Train model
    model.train_model()
    
    # Evaluate model
    metrics = model.evaluate_model()
    
    # Save model
    model.save_model()
    
    logger.info("Baseline model training and evaluation complete.")

if __name__ == "__main__":
    main() 