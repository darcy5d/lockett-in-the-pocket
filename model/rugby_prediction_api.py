"""
Rugby League Prediction API — per-competition lazy-loaded models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from core.competition_config import get_competition
from core.rugby_feature_engine import RugbyFeatureEngine

_models: dict[str, tf.keras.Model] = {}
_scalers: dict[str, Any] = {}
_feature_cols: dict[str, list] = {}
_engines: dict[str, RugbyFeatureEngine] = {}


def clear_rugby_model_cache(competition_id: str | None = None) -> None:
    global _models, _scalers, _feature_cols, _engines
    if competition_id:
        _models.pop(competition_id, None)
        _scalers.pop(competition_id, None)
        _feature_cols.pop(competition_id, None)
        _engines.pop(competition_id, None)
    else:
        _models.clear()
        _scalers.clear()
        _feature_cols.clear()
        _engines.clear()


def _load_rugby_model(competition_id: str) -> bool:
    global _models, _scalers, _feature_cols, _engines
    if competition_id in _models:
        return True
    cfg = get_competition(competition_id)
    if not cfg:
        return False
    output_dir = Path(cfg.get("output_dir", str(PROJECT_ROOT / "model" / "output" / competition_id)))
    model_path = output_dir / "model.h5"
    if not model_path.exists():
        return False
    try:
        _models[competition_id] = tf.keras.models.load_model(model_path)
        _scalers[competition_id] = joblib.load(output_dir / "scaler.joblib")
        with open(output_dir / "feature_cols.json") as f:
            _feature_cols[competition_id] = json.load(f)
        _engines[competition_id] = RugbyFeatureEngine(competition_id)
        _engines[competition_id].load_matches(2020, 2025)
        _engines[competition_id].compute_training_features()
        return True
    except Exception as e:
        print(f"Rugby model load error ({competition_id}): {e}")
        return False


def predict_rugby_match(
    competition_id: str,
    home_player_ids: list[str],
    away_player_ids: list[str],
    home_team: str = "",
    away_team: str = "",
    venue: str = "",
) -> dict[str, Any]:
    """
    Predict match outcome for a competition.

    Returns dict with team1_score, team2_score, margin, p_home_win, p_draw, p_away_win.
    """
    if not _load_rugby_model(competition_id):
        return {"error": f"Model not trained for {competition_id}. Run model/rugby_train.py --competition {competition_id} first."}
    engine = _engines[competition_id]
    scaler = _scalers[competition_id]
    model = _models[competition_id]
    feats = engine.get_prediction_features(
        home_team or "Home", away_team or "Away", venue or "",
        home_player_ids, away_player_ids,
    )
    X = scaler.transform(feats)
    preds = model.predict(X, verbose=0)
    # Outputs: [t1_tries, t2_tries, t1_kick, t2_kick, t1_score, t2_score, margin]
    s1 = float(preds[4].flatten()[0])
    s2 = float(preds[5].flatten()[0])
    margin = float(preds[6].flatten()[0])
    p_home = 1.0 / (1.0 + np.exp(-margin / 10.0))
    return {
        "team1_score": round(s1, 1),
        "team2_score": round(s2, 1),
        "margin": round(margin, 1),
        "p_home_win": round(p_home, 3),
        "p_draw": 0.0,
        "p_away_win": round(1 - p_home, 3),
    }
