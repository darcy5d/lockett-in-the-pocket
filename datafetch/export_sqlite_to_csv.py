#!/usr/bin/env python3
"""
Export afl.db (SQLite) to CSV files under a target directory.

Useful for backup or for tools that expect the CSV layout.

Usage:
  python datafetch/export_sqlite_to_csv.py [--db afl_data/data/afl.db] [--out afl_data_export/data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = _PROJECT_ROOT / "afl_data" / "data" / "afl.db"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SQLite AFL DB to CSV files")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="Path to afl.db")
    ap.add_argument("--out", default=None, help="Output directory (default: <db_dir>/csv_export)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    out_dir = Path(args.out) if args.out else db_path.parent / "csv_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    matches_dir = out_dir / "matches"
    lineups_dir = out_dir / "lineups"
    players_dir = out_dir / "players"
    matches_dir.mkdir(parents=True, exist_ok=True)
    lineups_dir.mkdir(parents=True, exist_ok=True)
    players_dir.mkdir(parents=True, exist_ok=True)

    conn = __import__("sqlite3").connect(db_path)

    try:
        # Matches by year
        df = pd.read_sql_query("SELECT * FROM matches ORDER BY year, date", conn)
        if not df.empty:
            for year, group in df.groupby("year"):
                path = matches_dir / f"matches_{int(year)}.csv"
                group.to_csv(path, index=False)
                print(f"  Wrote {path} ({len(group)} rows)")
        else:
            print("  No matches in DB.")

        # Lineups
        df = pd.read_sql_query("SELECT year, date, round_num, team_name, players FROM lineups ORDER BY year, date", conn)
        if not df.empty:
            df["_key"] = df["team_name"].astype(str).str.replace(" ", "_").str.lower()
            for team_key, group in df.groupby("_key"):
                path = lineups_dir / f"team_lineups_{team_key}.csv"
                group[["year", "date", "round_num", "team_name", "players"]].to_csv(path, index=False)
                print(f"  Wrote {path} ({len(group)} rows)")
        else:
            print("  No lineups in DB.")

        # Player games → one file per player (performance_details style)
        df = pd.read_sql_query("SELECT * FROM player_games ORDER BY player_id, year, games_played", conn)
        if not df.empty:
            for pid, group in df.groupby("player_id"):
                path = players_dir / f"{pid}_performance_details.csv"
                group.drop(columns=["player_id"], errors="ignore").to_csv(path, index=False)
                print(f"  Wrote {path} ({len(group)} rows)")
        else:
            print("  No player_games in DB.")
    finally:
        conn.close()

    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
