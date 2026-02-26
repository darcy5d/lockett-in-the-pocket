"""
NRL Data Service — thin wrapper around RugbyDataService("nrl").

Backward compatibility for existing NRL predictor code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.rugby_data_service import RugbyDataService


class NRLDataService(RugbyDataService):
    """NRL data service — delegates to RugbyDataService('nrl')."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        player_index_path: Optional[Path] = None,
    ):
        super().__init__(
            competition_id="nrl",
            data_dir=data_dir,
            player_index_path=player_index_path,
        )
