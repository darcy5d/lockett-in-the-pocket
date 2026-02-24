#!/usr/bin/env python3
"""
Gap update: scrape only new rounds/seasons from AFL Tables and append to
afl_data_afltables/data/ and optionally to afl_data/data/afl.db.

Determines the latest (year, round) in the existing data, then scrapes from
the next round onward (or next season if current season is complete).

Usage:
  python datafetch/update_from_afl_tables.py [--source-dir afl_data_afltables] [--db afl_data/data/afl.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from datafetch.afl_tables_scraper import scrape_season, write_matches_csv, write_lineups_csv, write_player_stats_csv
from core.afl_data_store import AFLDataStore, DB_FILENAME


def _latest_in_csv(data_dir: Path) -> tuple[int, str]:
    """Return (year, round_num) of the latest match in CSV match files."""
    match_dir = data_dir / "matches"
    if not match_dir.exists():
        return (0, "")
    max_year = 0
    max_round = ""
    for p in match_dir.glob("matches_*.csv"):
        try:
            y = int(p.stem.replace("matches_", ""))
            if y > max_year:
                max_year = y
        except ValueError:
            continue
    if max_year == 0:
        return (0, "")
    df = pd.read_csv(match_dir / f"matches_{max_year}.csv")
    if df.empty or "round_num" not in df.columns:
        return (max_year, "")
    df["round_num"] = df["round_num"].astype(str)
    # Prefer numeric round for comparison
    numeric = pd.to_numeric(df["round_num"], errors="coerce")
    if numeric.notna().any():
        max_round_val = numeric.max()
        max_round = str(int(max_round_val))
    else:
        max_round = df["round_num"].iloc[-1] if len(df) else ""
    return (max_year, max_round)


def _latest_in_db(db_path: Path) -> tuple[int, str]:
    """Return (year, round_num) of the latest match in SQLite."""
    if not db_path.exists():
        return (0, "")
    store = AFLDataStore.from_sqlite(db_path)
    df = store.load_matches(1897, 2030)
    if df.empty or "year" not in df.columns or "round_num" not in df.columns:
        return (0, "")
    max_year = int(df["year"].max())
    df_y = df[df["year"] == max_year]
    numeric = pd.to_numeric(df_y["round_num"], errors="coerce")
    if numeric.notna().any():
        max_round = str(int(numeric.max()))
    else:
        max_round = df_y["round_num"].iloc[-1] if len(df_y) else ""
    return (max_year, max_round)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gap update from AFL Tables")
    ap.add_argument("--source-dir", default="afl_data", help="CSV data dir name under project root")
    ap.add_argument("--db", default=None, help="SQLite path (default: afl_data/data/afl.db)")
    ap.add_argument("--lineups", action="store_true")
    ap.add_argument("--player-stats", action="store_true")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = _PROJECT_ROOT / args.source_dir / "data"
    db_path = Path(args.db) if args.db else _PROJECT_ROOT / "afl_data" / "data" / DB_FILENAME

    # Latest in CSV
    csv_year, csv_round = _latest_in_csv(data_dir)
    # Latest in DB if present
    db_year, db_round = _latest_in_db(db_path) if db_path.exists() else (0, "")

    # Use the later of the two
    if db_year > csv_year or (db_year == csv_year and (db_round and (not csv_round or (str(csv_round).isdigit() and str(db_round).isdigit() and int(db_round) >= int(csv_round))))):
        start_year, start_round = db_year, db_round
    else:
        start_year, start_round = csv_year, csv_round

    current_year = pd.Timestamp.now().year
    # If we have data up to current year, scrape from start_year (might need new rounds)
    year_from = start_year
    year_to = current_year

    print(f"Latest data: year={start_year} round={start_round}")
    print(f"Scraping {year_from}–{year_to} (new rounds/seasons only would require scraper support; scraping full years for now).")

    matches_by_year = {}
    all_lineup_rows = []
    all_player_stat_rows = []
    for year in range(year_from, year_to + 1):
        print(f"Scraping {year}...")
        matches, lineup_rows, player_stat_rows = scrape_season(
            year, fetch_lineups=args.lineups, fetch_player_stats=args.player_stats, delay=args.delay,
        )
        for m in matches:
            m.pop("match_stats_url", None)
        matches_by_year[year] = matches
        all_lineup_rows.extend(lineup_rows)
        all_player_stat_rows.extend(player_stat_rows)

    if not matches_by_year:
        print("No new data.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    matches_dir = data_dir / "matches"
    lineups_dir = data_dir / "lineups"
    players_dir = data_dir / "players"
    write_matches_csv(matches_by_year, matches_dir)
    if all_lineup_rows:
        write_lineups_csv(all_lineup_rows, lineups_dir)
    if all_player_stat_rows:
        write_player_stats_csv(all_player_stat_rows, players_dir)

    if db_path.exists():
        print("Re-populating SQLite from updated CSVs...")
        import subprocess
        subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "datafetch" / "populate_sqlite_from_csv.py"),
                "--source", args.source_dir,
                "--output", str(db_path),
            ],
            cwd=str(_PROJECT_ROOT),
            check=True,
        )

    print("Done.")


if __name__ == "__main__":
    main()
