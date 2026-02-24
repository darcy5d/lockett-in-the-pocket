#!/usr/bin/env python3
"""
Compare two data directories (default: afl_data/data vs afl_data_old/data).

Reports:
  - Match counts per year for each source + overlap
  - Lineup row counts + overlap
  - Player stats: file counts, column parity, value-level comparison for overlapping games
  - Overall coverage percentages

Usage:
  python scripts/compare_data_sources.py [--year-from 2020] [--year-to 2025]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = _PROJECT_ROOT / "afl_data" / "data"
AFLTABLES_DIR = _PROJECT_ROOT / "afl_data_old" / "data"

MATCH_KEY = ["year", "round_num", "team_1_team_name", "team_2_team_name"]
LINEUP_KEY = ["year", "round_num", "team_name"]
PLAYER_STAT_KEY = ["team", "year", "round"]
PLAYER_STAT_NUMERIC = [
    "kicks", "marks", "handballs", "disposals", "goals", "behinds",
    "hit_outs", "tackles", "rebound_50s", "inside_50s", "clearances",
    "clangers", "free_kicks_for", "free_kicks_against", "brownlow_votes",
    "contested_possessions", "uncontested_possessions", "contested_marks",
    "marks_inside_50", "one_percenters", "bounces", "goal_assist",
    "percentage_of_game_played",
]


def _load_matches(data_dir: Path, year_from: int, year_to: int) -> pd.DataFrame | None:
    match_dir = data_dir / "matches"
    if not match_dir.exists():
        return None
    dfs = []
    for y in range(year_from, year_to + 1):
        p = match_dir / f"matches_{y}.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                dfs.append(df)
            except Exception:
                pass
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    for c in ["team_1_final_goals", "team_2_final_goals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["team_1_final_goals", "team_2_final_goals"], how="all")
    df["round_num"] = df["round_num"].astype(str)
    if "year" not in df.columns and "year_from_file" in df.columns:
        df["year"] = df["year_from_file"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    return df


def _load_lineups(data_dir: Path) -> pd.DataFrame | None:
    lineup_dir = data_dir / "lineups"
    if not lineup_dir.exists():
        return None
    files = sorted(lineup_dir.glob("team_lineups_*.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["round_num"] = df["round_num"].astype(str)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    return df


def _load_player_index(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all *_performance_details.csv into {player_key: DataFrame}."""
    player_dir = data_dir / "players"
    if not player_dir.exists():
        return {}
    files = sorted(glob.glob(str(player_dir / "*_performance_details.csv")))
    result = {}
    for f in files:
        key = Path(f).stem.replace("_performance_details", "")
        try:
            df = pd.read_csv(f, dtype=str)
            result[key] = df
        except Exception:
            pass
    return result


def _match_keys(df: pd.DataFrame) -> set[tuple]:
    if df is None or not all(c in df.columns for c in MATCH_KEY):
        return set()
    return set(
        tuple(row[c] for c in MATCH_KEY)
        for _, row in df[MATCH_KEY].drop_duplicates().iterrows()
    )


def _lineup_keys(df: pd.DataFrame) -> set[tuple]:
    if df is None or not all(c in df.columns for c in LINEUP_KEY):
        return set()
    return set(
        tuple(row[c] for c in LINEUP_KEY)
        for _, row in df[LINEUP_KEY].drop_duplicates().iterrows()
    )


def _normalise_player_key(raw_key: str) -> str:
    """
    Normalise player filename key for matching across naming schemes.
    Old: 'heeney_isaac_05051996' (lastname_firstname_DOB)
    New: 'isaac_heeney' (firstname_lastname from URL)
    Normalised: 'heeney_isaac' (lastname_firstname, no DOB)
    """
    parts = raw_key.split("_")
    # Remove DOB suffix (8-digit numeric part)
    clean = [p for p in parts if not (len(p) == 8 and p.isdigit())]
    if len(clean) >= 2:
        # Try to detect if it's firstname_lastname (new) or lastname_firstname (old)
        # Heuristic: old format has DOB stripped, so it was already lastname_firstname
        # New format is firstname_lastname — reverse it
        # We can't know for sure, so normalise to sorted order for matching
        return "_".join(sorted(clean))
    return "_".join(clean)


def _compare_player_stats(current_idx: dict, at_idx: dict) -> None:
    """Compare player stats between current and AFL Tables sources."""
    current_keys = set(current_idx.keys())
    at_keys = set(at_idx.keys())

    # Direct filename match
    direct_common = current_keys & at_keys
    print("Player stats comparison:")
    print(f"  Current players (files): {len(current_keys)}")
    print(f"  AFL Tables players (files): {len(at_keys)}")
    print(f"  Direct filename match: {len(direct_common)}")

    # Normalised matching (handle different naming conventions)
    cur_norm: dict[str, list[str]] = {}
    for k in current_keys:
        nk = _normalise_player_key(k)
        cur_norm.setdefault(nk, []).append(k)
    at_norm: dict[str, list[str]] = {}
    for k in at_keys:
        nk = _normalise_player_key(k)
        at_norm.setdefault(nk, []).append(k)
    norm_common = set(cur_norm.keys()) & set(at_norm.keys())
    print(f"  Name-matched (normalised): {len(norm_common)}")

    current_total_rows = sum(len(df) for df in current_idx.values())
    at_total_rows = sum(len(df) for df in at_idx.values())
    print(f"  Current total game rows: {current_total_rows}")
    print(f"  AFL Tables total game rows: {at_total_rows}")

    if not norm_common:
        print("  No overlapping players to compare values.")
        return

    # Column parity check
    cur_sample = next(iter(current_idx.values()))
    at_sample = next(iter(at_idx.values()))
    cur_cols = set(cur_sample.columns)
    at_cols = set(at_sample.columns)
    missing_in_at = cur_cols - at_cols
    extra_in_at = at_cols - cur_cols
    if missing_in_at:
        print(f"  Columns in current but missing in AFL Tables: {sorted(missing_in_at)}")
    if extra_in_at:
        print(f"  Columns in AFL Tables but not in current: {sorted(extra_in_at)}")

    # Value comparison for name-matched players
    match_count = 0
    diff_count = 0
    total_compared = 0
    stat_diffs: dict[str, int] = {s: 0 for s in PLAYER_STAT_NUMERIC}
    checked = 0
    for nk in sorted(norm_common):
        if checked >= 200:
            break
        cur_key = cur_norm[nk][0]
        at_key = at_norm[nk][0]
        df_c = current_idx[cur_key].copy()
        df_a = at_idx[at_key].copy()
        for c in PLAYER_STAT_KEY:
            if c in df_c.columns:
                df_c[c] = df_c[c].astype(str)
            if c in df_a.columns:
                df_a[c] = df_a[c].astype(str)
        key_cols = [c for c in PLAYER_STAT_KEY if c in df_c.columns and c in df_a.columns]
        if not key_cols:
            continue
        merged = df_c.merge(df_a, on=key_cols, suffixes=("_cur", "_at"), how="inner")
        match_count += len(merged)
        for stat in PLAYER_STAT_NUMERIC:
            c_cur = f"{stat}_cur"
            c_at = f"{stat}_at"
            if c_cur in merged.columns and c_at in merged.columns:
                v_cur = pd.to_numeric(merged[c_cur], errors="coerce").fillna(0)
                v_at = pd.to_numeric(merged[c_at], errors="coerce").fillna(0)
                n_diff = int((v_cur != v_at).sum())
                stat_diffs[stat] += n_diff
                diff_count += n_diff
                total_compared += len(merged)
        checked += 1

    print(f"\n  Value comparison ({checked} name-matched players sampled):")
    print(f"    Game rows matched (by team+year+round): {match_count}")
    if total_compared > 0:
        pct_match = 100.0 * (1 - diff_count / total_compared)
        print(f"    Total stat comparisons: {total_compared}")
        print(f"    Differences: {diff_count} ({pct_match:.1f}% match)")
        top_diffs = sorted(stat_diffs.items(), key=lambda x: -x[1])[:5]
        if any(d > 0 for _, d in top_diffs):
            print(f"    Top differing columns: {[(k, v) for k, v in top_diffs if v > 0]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare afl_data vs afl_data_afltables")
    ap.add_argument("--year-from", type=int, default=2020)
    ap.add_argument("--year-to", type=int, default=2025)
    args = ap.parse_args()

    print("Data directories:")
    print(f"  Current (akareen): {CURRENT_DIR}")
    print(f"  AFL Tables:        {AFLTABLES_DIR}")
    print()

    # Matches
    df_current = _load_matches(CURRENT_DIR, args.year_from, args.year_to)
    df_at = _load_matches(AFLTABLES_DIR, args.year_from, args.year_to)

    if df_current is not None:
        print("Current matches:")
        print(f"  Total: {len(df_current)}")
        if "year" in df_current.columns:
            print(df_current.groupby("year").size().to_string())
    else:
        print("Current: no match data found.")
    print()

    if df_at is not None:
        print("AFL Tables matches:")
        print(f"  Total: {len(df_at)}")
        if "year" in df_at.columns:
            print(df_at.groupby("year").size().to_string())
    else:
        print("AFL Tables: no match data found.")
    print()

    match_overlap_pct = 0.0
    if df_current is not None and df_at is not None:
        k_current = _match_keys(df_current)
        k_at = _match_keys(df_at)
        only_current = k_current - k_at
        only_at = k_at - k_current
        common = k_current & k_at
        if k_current:
            match_overlap_pct = 100.0 * len(common) / len(k_current)
        print("Match overlap (key = year, round_num, team_1, team_2):")
        print(f"  In both:     {len(common)}")
        print(f"  Only current: {len(only_current)}")
        print(f"  Only AFL Tables: {len(only_at)}")
        print(f"  Coverage: {match_overlap_pct:.1f}% of current matches found in AFL Tables")
        if only_current and len(only_current) <= 20:
            print("  Only in current:", sorted(only_current)[:20])
        elif only_current:
            print("  Only in current (first 10):", sorted(only_current)[:10])
        if only_at and len(only_at) <= 20:
            print("  Only in AFL Tables:", sorted(only_at)[:20])
        elif only_at:
            print("  Only in AFL Tables (first 10):", sorted(only_at)[:10])
    print()

    # Lineups
    lineup_current = _load_lineups(CURRENT_DIR)
    lineup_at = _load_lineups(AFLTABLES_DIR)

    lineup_overlap_pct = 0.0
    if lineup_current is not None:
        print("Current lineups:")
        print(f"  Rows: {len(lineup_current)}")
    else:
        print("Current: no lineup data.")
    if lineup_at is not None:
        print("AFL Tables lineups:")
        print(f"  Rows: {len(lineup_at)}")
    else:
        print("AFL Tables: no lineup data.")
    print()

    if lineup_current is not None and lineup_at is not None:
        lk_current = _lineup_keys(lineup_current)
        lk_at = _lineup_keys(lineup_at)
        only_l_current = lk_current - lk_at
        only_l_at = lk_at - lk_current
        common_l = lk_current & lk_at
        if lk_current:
            lineup_overlap_pct = 100.0 * len(common_l) / len(lk_current)
        print("Lineup overlap (key = year, round_num, team_name):")
        print(f"  In both:     {len(common_l)}")
        print(f"  Only current: {len(only_l_current)}")
        print(f"  Only AFL Tables: {len(only_l_at)}")
        print(f"  Coverage: {lineup_overlap_pct:.1f}% of current lineups found in AFL Tables")
    print()

    # Player stats
    print("Loading player stats (may take a moment)...")
    current_players = _load_player_index(CURRENT_DIR)
    at_players = _load_player_index(AFLTABLES_DIR)
    _compare_player_stats(current_players, at_players)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Match coverage:  {match_overlap_pct:.1f}%")
    print(f"  Lineup coverage: {lineup_overlap_pct:.1f}%")
    print(f"  Player files:    current={len(current_players)}, AFL Tables={len(at_players)}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
