#!/usr/bin/env python3
"""
Full Rugby League historical data pipeline — scrape Tier 1, Tier 2, and UK from RLP.

Tier 1: NSWRFL, NSWRL, ARL, Super League AU 1997, NRL
Tier 2: NSW Cup, QLD Cup, reserve grade, BRL
UK: Super League UK, Championship

Uses nrl_competition_history.md lineage mapping.
Storage slugs: super-league-au, super-league-uk, championship-uk for disambiguation.

Usage:
  python datafetch/rebuild_nrl_from_rlp.py                    # Tier 1 + 2
  python datafetch/rebuild_nrl_from_rlp.py --tiers 1,2,uk    # Include UK
  python datafetch/rebuild_nrl_from_rlp.py --competition nrl # NRL lineage only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

DATA_DIR = _PROJECT_ROOT / "nrl_data" / "data"
MATCH_DIR = DATA_DIR / "matches"

from datafetch.rlp_scraper import run_scrape


def _migrate_super_league_au() -> None:
    """Rename matches_super-league_1997_1997.csv -> matches_super-league-au_1997_1997.csv if needed."""
    old_path = MATCH_DIR / "matches_super-league_1997_1997.csv"
    new_path = MATCH_DIR / "matches_super-league-au_1997_1997.csv"
    if old_path.exists() and not new_path.exists():
        old_path.rename(new_path)
        print(f"Migrated {old_path.name} -> {new_path.name}")

# Lineage: dict with slug, y1, y2, optional output_slug, year_format
LINEAGE_TIER1 = [
    {"slug": "nswrfl", "y1": 1908, "y2": 1983},
    {"slug": "nswrl", "y1": 1984, "y2": 1994},
    {"slug": "arl", "y1": 1995, "y2": 1997},
    {"slug": "super-league", "y1": 1997, "y2": 1997, "output_slug": "super-league-au"},
    {"slug": "nrl", "y1": 1998, "y2": 2025},
]

LINEAGE_TIER2 = [
    {"slug": "nswrfl-reserve-grade", "y1": 1908, "y2": 1983},
    {"slug": "nswrl-reserve-grade", "y1": 1984, "y2": 1997},
    {"slug": "nswrl-first-division", "y1": 1998, "y2": 2007},
    {"slug": "nsw-cup", "y1": 2008, "y2": 2025},
    {"slug": "qrl", "y1": 1909, "y2": 1930},
    {"slug": "brl", "y1": 1930, "y2": 1997},
    {"slug": "qld-cup", "y1": 1996, "y2": 2025},
]

LINEAGE_UK = [
    {"slug": "super-league", "y1": 1996, "y2": 2025, "output_slug": "super-league-uk", "year_format": "split"},
    {"slug": "championship", "y1": 2009, "y2": 2025, "output_slug": "championship-uk", "year_format": "split"},
]

# Competition -> lineage entries
COMPETITION_LINEAGE = {
    "nrl": LINEAGE_TIER1,
    "nsw-cup": [e for e in LINEAGE_TIER2 if e["slug"] in ("nsw-cup", "nswrl-first-division", "nswrl-reserve-grade", "nswrfl-reserve-grade")],
    "qld-cup": [e for e in LINEAGE_TIER2 if e["slug"] in ("qld-cup", "brl", "qrl")],
    "uk-super-league": LINEAGE_UK[:1],
    "uk-championship": LINEAGE_UK[1:],
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scrape Rugby League Tier 1, Tier 2, and UK from RLP."
    )
    ap.add_argument(
        "--tiers",
        type=str,
        default="1,2",
        help="Comma-separated: 1, 2, uk (default: 1,2)",
    )
    ap.add_argument(
        "--competition",
        type=str,
        default=None,
        help="Override: only scrape this competition (nrl, nsw-cup, qld-cup, uk-super-league, uk-championship)",
    )
    ap.add_argument(
        "--year-from",
        type=int,
        default=None,
        help="Override: only scrape from this year (filters lineage)",
    )
    ap.add_argument(
        "--year-to",
        type=int,
        default=None,
        help="Override: only scrape to this year (filters lineage)",
    )
    ap.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Override: only scrape this slug (e.g. nrl)",
    )
    args = ap.parse_args()

    _migrate_super_league_au()

    if args.competition:
        lineage = COMPETITION_LINEAGE.get(args.competition, [])
        if not lineage:
            print(f"Unknown competition: {args.competition}")
            return
    else:
        tiers_wanted = set(t.strip() for t in args.tiers.split(",") if t.strip())
        lineage = []
        if "1" in tiers_wanted:
            lineage.extend(LINEAGE_TIER1)
        if "2" in tiers_wanted:
            lineage.extend(LINEAGE_TIER2)
        if "uk" in tiers_wanted:
            lineage.extend(LINEAGE_UK)
        if not lineage:
            print("No tiers selected. Use --tiers 1,2,uk or --competition nrl")
            return

    total_matches = 0
    total_lineups = 0

    for entry in lineage:
        slug = entry["slug"]
        y1, y2 = entry["y1"], entry["y2"]
        if args.slug and slug != args.slug:
            continue
        if args.year_from is not None and y2 < args.year_from:
            continue
        if args.year_to is not None and y1 > args.year_to:
            continue

        y_from = max(y1, args.year_from or y1)
        y_to = min(y2, args.year_to or y2)
        if y_from > y_to:
            continue

        out_slug = entry.get("output_slug")
        year_fmt = entry.get("year_format", "single")
        print(f"\n--- {out_slug or slug} ---")
        run_scrape(
            slug=slug,
            year_from=y_from,
            year_to=y_to,
            output_slug=out_slug,
            year_format=year_fmt,
        )

    # Summary
    match_dir = _PROJECT_ROOT / "nrl_data" / "data" / "matches"
    lineup_dir = _PROJECT_ROOT / "nrl_data" / "data" / "lineups"
    for p in match_dir.glob("matches_*.csv"):
        if "2026" in p.name:
            continue
        try:
            total_matches += len(pd.read_csv(p))
        except Exception:
            pass
    for p in lineup_dir.glob("lineup_details_*.csv"):
        try:
            total_lineups += len(pd.read_csv(p))
        except Exception:
            pass

    print(f"\n--- Done ---")
    print(f"Total matches: {total_matches}")
    print(f"Total lineup entries: {total_lineups}")


if __name__ == "__main__":
    main()
