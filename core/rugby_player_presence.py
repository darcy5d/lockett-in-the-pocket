"""
Rugby League Player Presence — per-competition ELO from outcomes.

Learns player presence ELO: when this player is in the 17,
how does the team perform vs expected (from team ELO)?

Output: player_presence_elo_{competition_id}.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "nrl_data" / "data"
PRESENCE_K = 16
TEAM_ELO_K = 32
ELO_START = 1500

from core.competition_config import get_competition_slugs, slug_matches_file, slug_matches_lineup_file


def _expected_margin(elo1: float, elo2: float) -> float:
    diff = elo1 - elo2
    return 20 * (diff / 400)


def _expected_score(elo1: float, elo2: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400))


def _update_elo(elo: float, actual: float, expected: float, k: float) -> float:
    return elo + k * (actual - expected)


def _fingerprint(files: list[Path]) -> list[list]:
    """Return JSON-serializable fingerprint: [[path_str, mtime], ...]."""
    return [[str(f), f.stat().st_mtime] for f in sorted(files, key=lambda p: str(p))]


def compute_presence(
    competition_id: str,
    match_dir: Path | None = None,
    lineup_dir: Path | None = None,
    output_path: Path | None = None,
    force_recompute: bool = False,
) -> dict[str, float]:
    """
    Compute player presence ELO for a competition.

    Filters matches/lineups by competition slugs.
    Returns player_id -> presence ELO.
    Uses fingerprint-based cache: if input files unchanged, loads from disk.
    Set force_recompute=True to bypass cache (e.g. after fetching new data).
    """
    match_dir = match_dir or (DATA_DIR / "matches")
    lineup_dir = lineup_dir or (DATA_DIR / "lineups")
    output_path = output_path or (DATA_DIR / f"player_presence_elo_{competition_id}.json")
    meta_path = output_path.parent / f"player_presence_meta_{competition_id}.json"

    slugs = get_competition_slugs(competition_id)
    match_files = [
        f for f in sorted(match_dir.glob("matches_*.csv"))
        if "2026" not in f.name and slug_matches_file(f.name, slugs)
    ]
    lineup_files = [
        f for f in sorted(lineup_dir.glob("lineup_details_*.csv"))
        if slug_matches_lineup_file(f.name, slugs)
    ]
    if not match_files or not lineup_files:
        return {}

    current_fingerprint = _fingerprint(match_files + lineup_files)

    if not force_recompute and output_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("fingerprint") == current_fingerprint:
                with open(output_path) as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    matches = pd.concat([pd.read_csv(f) for f in match_files], ignore_index=True)
    lineups = pd.concat([pd.read_csv(f) for f in lineup_files], ignore_index=True)

    matches = matches.sort_values(["year", "round_num", "match_id"])
    team_elos: dict[str, float] = defaultdict(lambda: ELO_START)
    player_elos: dict[str, float] = defaultdict(lambda: ELO_START)
    player_games: dict[str, int] = defaultdict(int)

    for _, row in matches.iterrows():
        t1 = str(row.get("team_1_team_name", ""))
        t2 = str(row.get("team_2_team_name", ""))
        s1 = int(row.get("team_1_score", 0) or 0)
        s2 = int(row.get("team_2_score", 0) or 0)
        mid = str(row.get("match_id", ""))
        if not t1 or not t2 or not mid:
            continue

        margin = s1 - s2
        win1 = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        e1 = team_elos[t1]
        e2 = team_elos[t2]
        exp_win1 = _expected_score(e1, e2)
        exp_margin = _expected_margin(e1, e2)

        team_elos[t1] = _update_elo(e1, win1, exp_win1, TEAM_ELO_K)
        team_elos[t2] = _update_elo(e2, 1 - win1, 1 - exp_win1, TEAM_ELO_K)

        match_lineups = lineups[lineups["match_id"] == mid]
        t1_players = match_lineups[match_lineups["team"] == t1]["player_id"].tolist()
        t2_players = match_lineups[match_lineups["team"] == t2]["player_id"].tolist()

        t1_residual = margin - exp_margin
        t2_residual = -t1_residual

        for pid in t1_players:
            pid = str(pid)
            player_elos[pid] = _update_elo(player_elos[pid], t1_residual, 0, PRESENCE_K)
            player_games[pid] += 1
        for pid in t2_players:
            pid = str(pid)
            player_elos[pid] = _update_elo(player_elos[pid], t2_residual, 0, PRESENCE_K)
            player_games[pid] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dict(player_elos), f, indent=2)
    with open(meta_path, "w") as f:
        json.dump(
            {"fingerprint": current_fingerprint, "competition_id": competition_id},
            f,
            indent=2,
        )
    return dict(player_elos)
