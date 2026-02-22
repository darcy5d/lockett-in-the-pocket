#!/usr/bin/env python3
"""
Fetch the 2026 AFL fixture and write matches_2026.csv in akareen-compatible format.

Primary source: fixturedownload.com (free CSV, no auth).
Output schema matches existing matches_YYYY.csv:
    round_num, venue, date, year, attendance,
    team_1_team_name, team_1_q1_goals, team_1_q1_behinds, ...
    team_1_final_goals, team_1_final_behinds,
    team_2_team_name, team_2_q1_goals, ...
    team_2_final_goals, team_2_final_behinds

For future rounds the score columns are left empty (NaN).

Usage:
    python datafetch/fetch_2026_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.mappings import TeamNameMapper, VenueMapper

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FIXTURE_URL = "https://fixturedownload.com/results/afl-2026"
OUTPUT_PATH = _PROJECT_ROOT / "afl_data" / "data" / "matches" / "matches_2026.csv"
YEAR = 2026

# fixturedownload column names
FD_ROUND = "Round Number"
FD_DATE = "Date"
FD_LOCATION = "Location"
FD_HOME = "Home Team"
FD_AWAY = "Away Team"
FD_RESULT = "Result"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_team_mapper = TeamNameMapper()
_venue_mapper = VenueMapper()

SCORE_COLS = [
    "team_1_q1_goals", "team_1_q1_behinds",
    "team_1_q2_goals", "team_1_q2_behinds",
    "team_1_q3_goals", "team_1_q3_behinds",
    "team_1_final_goals", "team_1_final_behinds",
    "team_2_q1_goals", "team_2_q1_behinds",
    "team_2_q2_goals", "team_2_q2_behinds",
    "team_2_q3_goals", "team_2_q3_behinds",
    "team_2_final_goals", "team_2_final_behinds",
]


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _scrape_fixture_table(url: str, timeout: int = 30) -> pd.DataFrame:
    """
    Scrape the fixture table from fixturedownload.com HTML page.
    Returns a DataFrame with columns matching the HTML table headers.
    """
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError("No table found on page — site structure may have changed.")
    headers_el = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:  # skip header row
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append(cells)
    return pd.DataFrame(rows, columns=headers_el)


def _parse_round(round_str: str) -> str:
    """Map round string from fixturedownload to akareen format."""
    r = str(round_str).strip()
    mapping = {
        "OR": "Opening Round",
        "QF1": "Qualifying Final",
        "EF1": "Elimination Final",
        "SF1": "Semi Final",
        "SF2": "Semi Final",
        "PF1": "Preliminary Final",
        "PF2": "Preliminary Final",
        "GF": "Grand Final",
        "WC": "Wildcard Finals",
    }
    if r in mapping:
        return mapping[r]
    # Numeric rounds
    try:
        return str(int(r))
    except ValueError:
        return r


def transform_fixture(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise scraped fixturedownload DataFrame into akareen schema."""
    for col in [FD_ROUND, FD_DATE, FD_LOCATION, FD_HOME, FD_AWAY]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found. "
                             f"Got: {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        round_num = _parse_round(row[FD_ROUND])
        date = str(row[FD_DATE]).strip()
        location_raw = str(row[FD_LOCATION]).strip()
        venue = _venue_mapper.to_internal(location_raw)
        home_raw = str(row[FD_HOME]).strip()
        away_raw = str(row[FD_AWAY]).strip()
        team1 = _team_mapper.to_internal(home_raw)
        team2 = _team_mapper.to_internal(away_raw)

        record: dict = {
            "round_num": round_num,
            "venue": venue,
            "date": date,
            "year": YEAR,
            "attendance": None,
            "team_1_team_name": team1,
            "team_2_team_name": team2,
        }
        for col in SCORE_COLS:
            record[col] = None

        rows.append(record)

    result = pd.DataFrame(rows)
    cols = [
        "round_num", "venue", "date", "year", "attendance",
        "team_1_team_name",
        "team_1_q1_goals", "team_1_q1_behinds",
        "team_1_q2_goals", "team_1_q2_behinds",
        "team_1_q3_goals", "team_1_q3_behinds",
        "team_1_final_goals", "team_1_final_behinds",
        "team_2_team_name",
        "team_2_q1_goals", "team_2_q1_behinds",
        "team_2_q2_goals", "team_2_q2_behinds",
        "team_2_q3_goals", "team_2_q3_behinds",
        "team_2_final_goals", "team_2_final_behinds",
    ]
    return result[cols]


def fetch_and_save(output_path: Path = OUTPUT_PATH) -> Path:
    """Download fixture, transform, and save CSV. Returns output path."""
    print(f"Scraping 2026 fixture from {FIXTURE_URL}…")
    try:
        raw_df = _scrape_fixture_table(FIXTURE_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch fixture: {e}") from e

    print(f"Found {len(raw_df)} matches. Transforming…")
    df = transform_fixture(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} matches to {output_path}")
    return output_path


def validate_output(path: Path) -> None:
    """Sanity-check the generated CSV."""
    df = pd.read_csv(path)
    print(f"\nValidation: {len(df)} rows, {df['round_num'].nunique()} rounds")
    print("Teams found:")
    all_teams = set(df["team_1_team_name"]) | set(df["team_2_team_name"])
    for t in sorted(all_teams):
        print(f"  {t}")
    print("Venues found:")
    for v in sorted(df["venue"].dropna().unique()):
        print(f"  {v}")


if __name__ == "__main__":
    path = fetch_and_save()
    validate_output(path)
