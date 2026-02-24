#!/usr/bin/env python3
"""
Populate afl.db (SQLite) from CSV data.

Source can be afl_data_afltables/data (AFL Tables scraped) or afl_data/data (current).
Matches and lineups are inserted; optionally player_games from afl_data/data/players.

Usage:
  python datafetch/populate_sqlite_from_csv.py [--source afl_data_afltables] [--output afl_data/data/afl.db] [--players]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys_path = _PROJECT_ROOT
if str(_PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_PROJECT_ROOT))

from core.afl_data_store import init_sqlite_schema, STAT_FIELDS

MATCH_COLS = [
    "year", "round_num", "team_1_team_name", "team_2_team_name", "date", "venue", "attendance",
    "team_1_q1_goals", "team_1_q1_behinds", "team_1_q2_goals", "team_1_q2_behinds",
    "team_1_q3_goals", "team_1_q3_behinds", "team_1_final_goals", "team_1_final_behinds",
    "team_2_q1_goals", "team_2_q1_behinds", "team_2_q2_goals", "team_2_q2_behinds",
    "team_2_q3_goals", "team_2_q3_behinds", "team_2_final_goals", "team_2_final_behinds",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="afl_data", help="Directory name under project root (e.g. afl_data)")
    ap.add_argument("--output", default=None, help="Output afl.db path (default: <source>/data/afl.db)")
    ap.add_argument("--players", action="store_true", help="Also populate player_games from afl_data/data/players (only if source has players)")
    args = ap.parse_args()

    source_dir = _PROJECT_ROOT / args.source / "data"
    if not source_dir.exists():
        raise SystemExit(f"Source not found: {source_dir}")

    out_path = Path(args.output) if args.output else source_dir / "afl.db"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    init_sqlite_schema(out_path)
    conn = __import__("sqlite3").connect(out_path)

    try:
        # Matches
        match_dir = source_dir / "matches"
        if match_dir.exists():
            conn.execute("DELETE FROM matches")
            for p in sorted(match_dir.glob("matches_*.csv")):
                df = pd.read_csv(p)
                df["round_num"] = df["round_num"].astype(str)
                for c in MATCH_COLS:
                    if c not in df.columns:
                        df[c] = None
                df = df[[c for c in MATCH_COLS if c in df.columns]]
                df.to_sql("matches", conn, if_exists="append", index=False)
            print(f"  Inserted matches from {match_dir}")
        else:
            print("  No matches dir, skipping.")

        # Lineups
        lineup_dir = source_dir / "lineups"
        if lineup_dir.exists():
            conn.execute("DELETE FROM lineups")
            files = sorted(lineup_dir.glob("team_lineups_*.csv"))
            for f in files:
                df = pd.read_csv(f)
                df["round_num"] = df["round_num"].astype(str)
                for c in ["year", "date", "round_num", "team_name", "players"]:
                    if c not in df.columns:
                        df[c] = ""
                df[["year", "date", "round_num", "team_name", "players"]].to_sql(
                    "lineups", conn, if_exists="append", index=False
                )
            print(f"  Inserted lineups from {lineup_dir} ({len(files)} files)")
        else:
            print("  No lineups dir, skipping.")

        # Player games (from source_dir/players)
        if args.players:
            player_dir = source_dir / "players"
            if player_dir.exists():
                conn.execute("DELETE FROM player_games")
                files = glob.glob(str(player_dir / "*_performance_details.csv"))
                count = 0
                for f in files:
                    pid = Path(f).stem.replace("_performance_details", "")
                    try:
                        df = pd.read_csv(f)
                        df["player_id"] = pid
                        if "round" in df.columns and "round_num" not in df.columns:
                            df["round_num"] = df["round"].astype(str)
                        cols = ["player_id", "year", "round_num", "team", "opponent", "games_played", "result", "jersey_num"]
                        for s in STAT_FIELDS:
                            if s in df.columns:
                                cols.append(s)
                        df = df[[c for c in cols if c in df.columns]]
                        df.to_sql("player_games", conn, if_exists="append", index=False)
                        count += 1
                    except Exception:
                        pass
                print(f"  Inserted player_games from {player_dir} ({count} players)")
            else:
                print("  No players dir, skipping.")
        conn.commit()
    finally:
        conn.close()

    print(f"Done. DB: {out_path}")


if __name__ == "__main__":
    main()
