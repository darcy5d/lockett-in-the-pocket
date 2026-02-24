#!/usr/bin/env python3
"""
Full rebuild: re-scrape AFL Tables for all seasons (or a year range) and write to
afl_data/data/ and optionally populate afl_data/data/afl.db.

Runs round-by-round to keep requests small and resumable. Supports --resume-from-year
and --resume-from-round to continue after a failure.

Usage:
  python datafetch/rebuild_from_afl_tables.py [--year-from 1990] [--year-to 2025] [--lineups] [--player-stats] [--populate-db]
  python datafetch/rebuild_from_afl_tables.py --resume-from-year 2005 --resume-from-round 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from datafetch.afl_tables_scraper import (
    scrape_season,
    write_matches_csv,
    write_lineups_csv,
    write_player_stats_csv,
)
OUTPUT_BASE = _PROJECT_ROOT / "afl_data" / "data"
DEFAULT_DB = _PROJECT_ROOT / "afl_data" / "data" / "afl.db"

ALL_ROUNDS = (
    ["Opening Round"]
    + [str(r) for r in range(1, 28)]
    + ["Qualifying Final", "Elimination Final", "Semi Final", "Preliminary Final", "Grand Final"]
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Full rebuild from AFL Tables (round-by-round)")
    ap.add_argument("--year-from", type=int, default=1990)
    ap.add_argument("--year-to", type=int, default=2025)
    ap.add_argument("--lineups", action="store_true", help="Fetch lineups")
    ap.add_argument("--player-stats", action="store_true", help="Fetch per-player stats")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--resume-from-year", type=int, default=None, help="Skip years before this")
    ap.add_argument("--resume-from-round", type=str, default=None, help="Skip rounds before this (within resume year)")
    ap.add_argument("--populate-db", action="store_true", help="After scrape, populate afl_data/data/afl.db")
    args = ap.parse_args()

    matches_dir = OUTPUT_BASE / "matches"
    lineups_dir = OUTPUT_BASE / "lineups"
    players_dir = OUTPUT_BASE / "players"

    resume_year = args.resume_from_year
    resume_round = args.resume_from_round
    skipping = resume_year is not None

    for year in range(args.year_from, args.year_to + 1):
        if skipping and year < resume_year:
            continue

        for rnd in ALL_ROUNDS:
            if skipping and year == resume_year and resume_round is not None:
                if rnd != resume_round:
                    continue
                else:
                    skipping = False

            label = f"{year} round {rnd}"
            print(f"Scraping {label}...")
            try:
                matches, lineup_rows, player_stat_rows = scrape_season(
                    year,
                    fetch_lineups=args.lineups,
                    fetch_player_stats=args.player_stats,
                    delay=args.delay,
                    round_filter=rnd,
                )
            except Exception as e:
                print(f"  ERROR scraping {label}: {e}")
                continue

            for m in matches:
                m.pop("match_stats_url", None)

            if matches:
                write_matches_csv({year: matches}, matches_dir)
            if lineup_rows:
                write_lineups_csv(lineup_rows, lineups_dir)
            if player_stat_rows:
                write_player_stats_csv(player_stat_rows, players_dir)

        if skipping and year >= resume_year:
            skipping = False

    if args.populate_db:
        print("Populating SQLite...")
        import subprocess
        subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "datafetch" / "populate_sqlite_from_csv.py"),
                "--source", "afl_data",
                "--output", str(DEFAULT_DB),
            ],
            cwd=str(_PROJECT_ROOT),
            check=True,
        )

    print("Done.")


if __name__ == "__main__":
    main()
