"""
NRL Prediction API — delegates to rugby_prediction_api with competition=nrl.
"""

from __future__ import annotations

from typing import Any

from model.rugby_prediction_api import clear_rugby_model_cache, predict_rugby_match


def clear_nrl_model_cache() -> None:
    clear_rugby_model_cache("nrl")


def predict_nrl_match(
    home_player_ids: list[str],
    away_player_ids: list[str],
    home_team: str = "",
    away_team: str = "",
    venue: str = "",
) -> dict[str, Any]:
    """Predict NRL match outcome."""
    return predict_rugby_match(
        competition_id="nrl",
        home_player_ids=home_player_ids,
        away_player_ids=away_player_ids,
        home_team=home_team,
        away_team=away_team,
        venue=venue,
    )
