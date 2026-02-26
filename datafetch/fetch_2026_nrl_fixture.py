#!/usr/bin/env python3
"""
Fetch the 2026 NRL fixture and write matches_2026.csv.

Primary source: League Unlimited (leagueunlimited.com).
Fallback: RLP (rugbyleagueproject.org) if League Unlimited structure unavailable.

Output schema matches historical NRL matches:
    round_num, venue, date, year, attendance,
    team_1_team_name, team_1_score, team_1_tries, team_1_goals, team_1_fg,
    team_2_team_name, team_2_score, team_2_tries, team_2_goals, team_2_fg

For future rounds the score columns are left empty.

Usage:
    python datafetch/fetch_2026_nrl_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.nrl_mappings import NRLTeamMapper, NRLVenueMapper

OUTPUT_PATH = _PROJECT_ROOT / "nrl_data" / "data" / "matches" / "matches_2026.csv"
LEAGUE_UNLIMITED_BASE = "https://leagueunlimited.com/competition/show/national-rugby-league/2026/draw"
RLP_BASE = "https://www.rugbyleagueproject.org/seasons/nrl-2026"
YEAR = 2026

_team_mapper = NRLTeamMapper()
_venue_mapper = NRLVenueMapper()

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

NRL_SCORE_COLS = [
    "team_1_score", "team_1_tries", "team_1_goals", "team_1_fg",
    "team_2_score", "team_2_tries", "team_2_goals", "team_2_fg",
]


def _fetch_from_rlp() -> list[dict]:
    """Fetch 2026 fixture from RLP round summaries."""
    from datafetch.rlp_scraper import scrape_round, discover_rounds

    matches = []
    rounds = discover_rounds("nrl", YEAR)
    if not rounds:
        return []

    for rn in rounds:
        m, _ = scrape_round("nrl", YEAR, rn)
        for row in m:
            matches.append({
                "round_num": str(rn),
                "venue": row.get("venue", ""),
                "date": row.get("date", ""),
                "year": YEAR,
                "attendance": row.get("attendance", ""),
                "team_1_team_name": _team_mapper.to_internal(row.get("team_1_team_name", "")),
                "team_2_team_name": _team_mapper.to_internal(row.get("team_2_team_name", "")),
                "team_1_score": row.get("team_1_score"),
                "team_1_tries": row.get("team_1_tries"),
                "team_1_goals": row.get("team_1_goals"),
                "team_1_fg": row.get("team_1_fg"),
                "team_2_score": row.get("team_2_score"),
                "team_2_tries": row.get("team_2_tries"),
                "team_2_goals": row.get("team_2_goals"),
                "team_2_fg": row.get("team_2_fg"),
            })
    return matches


def _fetch_from_league_unlimited() -> list[dict]:
    """Fetch 2026 fixture from League Unlimited. Structure may vary."""
    matches = []
    for rn in range(1, 28):
        url = f"{LEAGUE_UNLIMITED_BASE}/{rn}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if "/matches/" in href:
                    text = a.get_text(strip=True)
                    if " v " in text or " vs " in text.lower():
                        parts = text.replace(" vs ", " v ").split(" v ", 1)
                        if len(parts) == 2:
                            t1 = _team_mapper.to_internal(parts[0].strip())
                            t2 = _team_mapper.to_internal(parts[1].strip())
                            matches.append({
                                "round_num": str(rn),
                                "venue": "",
                                "date": "",
                                "year": YEAR,
                                "attendance": "",
                                "team_1_team_name": t1,
                                "team_2_team_name": t2,
                                "team_1_score": None,
                                "team_1_tries": None,
                                "team_1_goals": None,
                                "team_1_fg": None,
                                "team_2_score": None,
                                "team_2_tries": None,
                                "team_2_goals": None,
                                "team_2_fg": None,
                            })
        except Exception:
            pass
    return matches


def fetch_and_save(output_path: Path = OUTPUT_PATH) -> Path:
    """Fetch fixture, transform, and save CSV."""
    matches = _fetch_from_league_unlimited()
    if not matches:
        matches = _fetch_from_rlp()
    if not matches:
        matches = []

    df = pd.DataFrame(matches)
    if not df.empty:
        cols = ["round_num", "venue", "date", "year", "attendance",
                "team_1_team_name", "team_1_score", "team_1_tries", "team_1_goals", "team_1_fg",
                "team_2_team_name", "team_2_score", "team_2_tries", "team_2_goals", "team_2_fg"]
        if "competition" in df.columns:
            df = df.drop(columns=["competition", "match_id"], errors="ignore")
        df = df[[c for c in cols if c in df.columns]]
    else:
        df = pd.DataFrame(columns=[
            "round_num", "venue", "date", "year", "attendance",
            "team_1_team_name", "team_1_score", "team_1_tries", "team_1_goals", "team_1_fg",
            "team_2_team_name", "team_2_score", "team_2_tries", "team_2_goals", "team_2_fg",
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} matches to {output_path}")
    return output_path


if __name__ == "__main__":
    fetch_and_save()
