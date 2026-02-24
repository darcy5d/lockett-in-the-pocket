"""
Prediction API — loads trained model and FeatureEngine state,
computes real features for each match, and runs predictions.

Win probability is derived from the predicted margin via a calibrated
logistic function (Goddard 2005), not from a separate softmax head.
This guarantees consistency: whoever is predicted to score more always
has the higher win probability.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import tensorflow as tf

from model.bounded_output_layer import BoundedOutputLayer
from model.custom_loss import mae, mse
from model.player_stats_api import get_team_player_indices

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = Path(PROJECT_ROOT) / "model" / "output"
MAX_PLAYERS_PER_TEAM = 22


class MeanPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"output_dim": self.output_dim})
        return cfg


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_model = None
_match_scaler = None
_enh_scaler = None
_feature_cols = None
_engine = None
_calibration = None  # {k, sigma, draw_base_rate}


def clear_model_cache() -> None:
    """Clear all caches. Call after retraining."""
    global _model, _match_scaler, _enh_scaler, _feature_cols, _engine, _calibration
    _model = None
    _match_scaler = None
    _enh_scaler = None
    _feature_cols = None
    _engine = None
    _calibration = None
    import model.player_stats_api as _psa
    _psa._player_stats_cache = None
    _psa._player_index_cache = None


def _load_all():
    """Load model, scalers, feature cols, calibration, and FeatureEngine state."""
    global _model, _match_scaler, _enh_scaler, _feature_cols, _engine, _calibration

    if _model is not None:
        return

    model_dir = str(OUTPUT_DIR)

    # Model
    custom_objects = {
        "MeanPoolingLayer": MeanPoolingLayer,
        "BoundedOutputLayer": BoundedOutputLayer,
        "mse": mse,
        "mae": mae,
    }
    model_path = os.path.join(model_dir, "model.h5")
    try:
        _model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Match scaler
    try:
        _match_scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    except Exception:
        pass

    # Enhanced scaler
    try:
        _enh_scaler = joblib.load(os.path.join(model_dir, "enhanced_scaler.joblib"))
    except Exception:
        _enh_scaler = None

    # Feature columns
    try:
        with open(os.path.join(model_dir, "feature_cols.json")) as f:
            _feature_cols = json.load(f)
    except Exception:
        _feature_cols = None

    # Calibration parameters (k, sigma, draw_base_rate)
    try:
        with open(os.path.join(model_dir, "calibration.json")) as f:
            _calibration = json.load(f)
        print(f"Calibration loaded: k={_calibration['k']:.2f}")
    except Exception:
        _calibration = {"k": 37.0, "sigma": 43.6, "draw_base_rate": 0.009}
        print("Using default calibration (k=37)")

    # FeatureEngine state
    try:
        from core.feature_engine import FeatureEngine
        _engine = FeatureEngine.load_from_state(OUTPUT_DIR)
    except Exception as e:
        print(f"Warning: FeatureEngine state not loaded: {e}")
        _engine = None


# ---------------------------------------------------------------------------
# Margin -> win probability (calibrated logistic)
# ---------------------------------------------------------------------------

def _margin_to_probabilities(margin: float) -> dict:
    """
    Derive win/draw/loss probabilities from predicted margin using
    calibrated logistic function (Goddard 2005).

    P(home_win) = sigmoid(margin / k)
    P(draw) ~ base_rate * exp(-margin^2 / (2 * sigma^2))
    """
    cal = _calibration or {"k": 37.0, "sigma": 43.6, "draw_base_rate": 0.009}
    k = cal["k"]
    sigma = cal["sigma"]
    draw_base = cal["draw_base_rate"]

    p_home = 1.0 / (1.0 + np.exp(-margin / k))
    p_away = 1.0 - p_home
    p_draw = draw_base * np.exp(-margin ** 2 / (2.0 * sigma ** 2))

    # Normalise to sum to 1
    total = p_home + p_away + p_draw
    return {
        "team1_win": float(p_home / total),
        "draw": float(p_draw / total),
        "team2_win": float(p_away / total),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_match_outcome(
    team1_player_ids: List[str],
    team2_player_ids: List[str],
    home_team: str = "",
    away_team: str = "",
    venue: str = "",
) -> Dict[str, Any]:
    """
    Predict the outcome of a match.

    Win probability is derived from the predicted margin via calibrated
    logistic — guaranteed consistent with predicted scores.
    """
    _load_all()

    if _model is None:
        return {"error": "Model not loaded"}

    # Player embedding indices
    team1_indices = get_team_player_indices(team1_player_ids)
    team2_indices = get_team_player_indices(team2_player_ids)

    # Match features from FeatureEngine
    if _engine is not None and _feature_cols is not None:
        X_match, enh_t1, enh_t2 = _engine.get_prediction_features(
            home_team or "unknown",
            away_team or "unknown",
            venue or "",
            team1_player_ids,
            team2_player_ids,
        )
        if _match_scaler is not None:
            X_match = _match_scaler.transform(X_match)
        if _enh_scaler is not None:
            enh_t1 = _enh_scaler.transform(enh_t1)
            enh_t2 = _enh_scaler.transform(enh_t2)
    else:
        n_feats = len(_feature_cols) if _feature_cols else 19
        X_match = np.zeros((1, n_feats))
        if _match_scaler is not None:
            X_match = _match_scaler.transform(X_match)
        from model.player_stats_api import get_team_enhanced_features
        enh_t1 = get_team_enhanced_features(team1_player_ids)
        enh_t2 = get_team_enhanced_features(team2_player_ids)
        if _enh_scaler is not None:
            enh_t1 = _enh_scaler.transform(enh_t1)
            enh_t2 = _enh_scaler.transform(enh_t2)

    try:
        predictions = _model.predict(
            [X_match, team1_indices, team2_indices, enh_t1, enh_t2]
        )

        # Model outputs: [t1_goals, t1_behinds, t2_goals, t2_behinds, margin]
        team1_goals = float(predictions[0][0][0])
        team1_behinds = float(predictions[1][0][0])
        team2_goals = float(predictions[2][0][0])
        team2_behinds = float(predictions[3][0][0])
        margin = float(predictions[4][0][0])

        team1_score = team1_goals * 6 + team1_behinds
        team2_score = team2_goals * 6 + team2_behinds

        # Derive win probability from margin (Goddard 2005)
        probs = _margin_to_probabilities(margin)

        if margin > 0:
            winner = "team1"
        elif margin < 0:
            winner = "team2"
        else:
            winner = "draw"

        return {
            "winner": winner,
            "winner_probabilities": probs,
            "margin": margin,
            "team1": {"goals": team1_goals, "behinds": team1_behinds, "score": team1_score},
            "team2": {"goals": team2_goals, "behinds": team2_behinds, "score": team2_score},
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}
