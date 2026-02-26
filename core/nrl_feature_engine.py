"""
NRL Feature Engine — delegates to RugbyFeatureEngine('nrl').
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.rugby_feature_engine import RUGBY_FEATURE_COLS, RugbyFeatureEngine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
NRL_DATA = _PROJECT_ROOT / "nrl_data" / "data"
NRL_OUTPUT = _PROJECT_ROOT / "model" / "output" / "nrl"

# Re-export for backward compatibility
NRL_FEATURE_COLS = RUGBY_FEATURE_COLS


class NRLFeatureEngine(RugbyFeatureEngine):
    """NRL feature engine — delegates to RugbyFeatureEngine('nrl')."""

    def __init__(
        self,
        match_dir: Optional[Path] = None,
        lineup_dir: Optional[Path] = None,
        presence_path: Optional[Path] = None,
    ):
        super().__init__(
            competition_id="nrl",
            match_dir=match_dir or (NRL_DATA / "matches"),
            lineup_dir=lineup_dir or (NRL_DATA / "lineups"),
            presence_path=presence_path or (
                (NRL_DATA / "player_presence_elo_nrl.json")
                if (NRL_DATA / "player_presence_elo_nrl.json").exists()
                else (NRL_DATA / "player_presence_elo.json")
            ),
        )
