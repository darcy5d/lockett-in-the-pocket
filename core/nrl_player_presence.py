"""
NRL Player Presence — delegates to rugby_player_presence.compute_presence('nrl').
"""

from __future__ import annotations

from pathlib import Path

from core.rugby_player_presence import compute_presence as _compute_presence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
NRL_DATA = _PROJECT_ROOT / "nrl_data" / "data"


def compute_presence(
    match_dir: Path | None = None,
    lineup_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, float]:
    """Compute NRL player presence ELO. Delegates to rugby_player_presence."""
    # rugby_player_presence uses player_presence_elo_nrl.json by default
    return _compute_presence(
        competition_id="nrl",
        match_dir=match_dir or (NRL_DATA / "matches"),
        lineup_dir=lineup_dir or (NRL_DATA / "lineups"),
        output_path=output_path or (NRL_DATA / "player_presence_elo_nrl.json"),
    )
