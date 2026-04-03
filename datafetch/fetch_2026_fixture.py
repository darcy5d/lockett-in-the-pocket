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
import time
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


def _scrape_fixture_table(url: str, timeout: int = 60, retries: int = 3) -> pd.DataFrame:
    """
    Scrape the fixture table from fixturedownload.com HTML page.
    Returns a DataFrame with columns matching the HTML table headers.
    Uses longer timeout and retries to handle slow/unreliable responses.
    """
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))  # 2s, 4s backoff
            else:
                raise
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


def _parse_fixturedownload_result(result_str: str) -> tuple[int | None, int | None]:
    """
    Parse fixturedownload Result column (e.g., "119 - 65") into home/away scores.
    
    Args:
        result_str: Result string from fixturedownload (e.g., "119 - 65", "-", or empty)
    
    Returns:
        tuple: (home_score, away_score) or (None, None) if no valid result
    """
    if not result_str or pd.isna(result_str):
        return None, None
        
    result_str = str(result_str).strip()
    
    # Check for valid result format: "119 - 65"
    if " - " in result_str:
        try:
            parts = result_str.split(" - ")
            if len(parts) == 2:
                home_score = int(parts[0].strip())
                away_score = int(parts[1].strip())
                # Basic validation - AFL scores are typically 0-200
                if 0 <= home_score <= 300 and 0 <= away_score <= 300:
                    return home_score, away_score
        except (ValueError, IndexError):
            pass
    
    return None, None


def _convert_total_score_to_goals_behinds(total_score: int) -> tuple[int, int]:
    """
    Convert total AFL score back to approximate goals.behinds format.
    
    This is an approximation since we don't have the actual breakdown,
    but it provides reasonable estimates for display purposes.
    
    Args:
        total_score: Total AFL score (goals * 6 + behinds)
        
    Returns:
        tuple: (goals, behinds) approximation
    """
    if total_score < 0:
        return 0, 0
        
    # Simple approximation: assume roughly 1 behind per 2 goals on average
    # Goals account for roughly 80-85% of total score in typical matches
    estimated_goals = round(total_score / 7)  # Rough estimate
    estimated_behinds = total_score - (estimated_goals * 6)
    
    # Ensure behinds are non-negative
    if estimated_behinds < 0:
        estimated_goals = total_score // 6
        estimated_behinds = total_score % 6
        
    return estimated_goals, estimated_behinds


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

        # Parse result if available (e.g., "119 - 65" for completed matches)
        home_score, away_score = _parse_fixturedownload_result(row.get(FD_RESULT, ""))
        
        record: dict = {
            "round_num": round_num,
            "venue": venue,
            "date": date,
            "year": YEAR,
            "attendance": None,
            "team_1_team_name": team1,
            "team_2_team_name": team2,
        }
        
        # Initialize all score columns to None
        for col in SCORE_COLS:
            record[col] = None
            
        # If we have valid results from fixturedownload, use them
        if home_score is not None and away_score is not None:
            # Convert total scores to goals.behinds approximations
            home_goals, home_behinds = _convert_total_score_to_goals_behinds(home_score)
            away_goals, away_behinds = _convert_total_score_to_goals_behinds(away_score)
            
            # Set final scores (most important for display)
            record["team_1_final_goals"] = home_goals
            record["team_1_final_behinds"] = home_behinds
            record["team_2_final_goals"] = away_goals
            record["team_2_final_behinds"] = away_behinds
            
            print(f"FIXTUREDOWNLOAD RESULT: {team1} {home_goals}.{home_behinds} ({home_score}) vs {team2} {away_goals}.{away_behinds} ({away_score})")
        else:
            print(f"FIXTUREDOWNLOAD: No result for {team1} vs {team2} (Result: '{row.get(FD_RESULT, '')}')")

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
