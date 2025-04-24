#!/usr/bin/env python3
"""
Random Forest Model for AFL Match Prediction

This module provides a Random Forest model for predicting AFL match outcomes.
It includes model training, evaluation, and prediction.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('random_forest_model')

# Define constants
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
FEATURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

class AFLRandomForestModel:
    """
    Random Forest model for AFL match prediction.
    
    This class implements a binary classification model to predict
    whether team_1 will win a match.
    """
    
    def __init__(self, model_params: Optional[Dict] = None):
        """
        Initialize the Random Forest model.
        
        Args:
            model_params: Optional parameters for the Random Forest model
        """
        self.features_df = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.model = None
        
        # Default model parameters
        self.model_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
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
    
    def prepare_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            test_size: Proportion of data to use for testing
            
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
    
    def train_model(self) -> RandomForestClassifier:
        """
        Train the Random Forest model.
        
        Returns:
            Trained Random Forest classifier
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not prepared. Call prepare_data() first.")
            return None
        
        logger.info("Training Random Forest model...")
        
        # Initialize and train the model
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(self.X_train, self.y_train)
        
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
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'log_loss': loss
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
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
            model_path = os.path.join(MODEL_OUTPUT_DIR, 'random_forest_model.pkl')
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(self.model, model_path)
    
    def load_model(self, model_path: Optional[str] = None) -> RandomForestClassifier:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded Random Forest classifier
        """
        if model_path is None:
            model_path = os.path.join(MODEL_OUTPUT_DIR, 'random_forest_model.pkl')
        
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
    """Main function to train and evaluate the Random Forest model."""
    # Initialize model
    model = AFLRandomForestModel()
    
    # Load features
    features = model.load_features()
    if features.empty:
        logger.error("Failed to load features. Exiting.")
        return
    
    # Prepare data
    X_train, X_val, y_train, y_val = model.prepare_data(test_size=0.2)
    if X_train is None:
        logger.error("Failed to prepare data. Exiting.")
        return
    
    # Train model
    model.train_model()
    
    # Evaluate model
    metrics = model.evaluate_model()
    
    # Save model
    model.save_model()
    
    logger.info("Random Forest model training and evaluation complete.")

if __name__ == "__main__":
    main() 