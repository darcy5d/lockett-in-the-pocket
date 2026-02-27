#!/usr/bin/env python3
"""
Fetch the 2026 NRL fixture and write matches_2026.csv.

Primary source: League Unlimited (leagueunlimited.com).
Fallback: RLP (rugbyleagueproject.org) if League Unlimited structure unavailable.

Output schema matches historical NRL matches:
    round_num, venue, date, year, attendance,
    team_1_team_name, team_1_score, team_1_tries, team_1_goals, team_1_fg,
    team_2_team_name, team_2_score, team_2_tries, team_2_goals, team_2_fg,
    live_update_id (for lineup scraper)

For future rounds the score columns are left empty.

Usage:
    python datafetch/fetch_2026_nrl_fixture.py
"""

from __future__ import annotations

import re
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

# Day abbreviations for parsing team2 from "Team2 Sun 1 Mar 1:15PM"
DAY_ABBREVS = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
MONTH_MAP = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

# Short names for main draw page format: "Pre Match Knights Cowboys Venue"
LU_SHORT_TO_INTERNAL = {
    "Knights": "Newcastle", "Cowboys": "North Qld", "Bulldogs": "Canterbury",
    "Dragons": "St Geo Illa", "Storm": "Melbourne", "Eels": "Parramatta",
    "Warriors": "Warriors", "Roosters": "Sydney", "Broncos": "Brisbane",
    "Panthers": "Penrith", "Sharks": "Cronulla", "Titans": "Gold Coast",
    "Sea Eagles": "Manly", "Raiders": "Canberra", "Dolphins": "Dolphins",
    "Rabbitohs": "South Sydney", "Tigers": "Wests Tigers",
}


def _parse_date(date_str: str) -> str:
    """Parse 'Sun 1 Mar 1:15PM' -> '2026-03-01 13:15'."""
    if not date_str or not date_str.strip():
        return ""
    s = date_str.strip()
    # Match: Day D MMM H:MMAM/PM or H:MM AM/PM
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{1,2}):(\d{2})\s*([AP]M)", s, re.I)
    if not m:
        return ""
    day, mon_str, hour, minute, ampm = m.groups()
    month = MONTH_MAP.get(mon_str.capitalize(), 1)
    hour = int(hour)
    minute = int(minute)
    if ampm.upper() == "PM" and hour < 12:
        hour += 12
    elif ampm.upper() == "AM" and hour == 12:
        hour = 0
    return f"{YEAR}-{month:02d}-{int(day):02d} {hour:02d}:{minute:02d}"


def _extract_team2_and_datetime(rest: str) -> tuple[str, str]:
    """From 'Team2 Sun 1 Mar 1:15PM' or 'Team2Sun 1 Mar 1:15PM' extract team2 and datetime."""
    # League Unlimited sometimes omits space before day (e.g. "CowboysSun 1 Mar")
    m = re.search(r"(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s*\d{1,2}\s+[A-Za-z]{3}\s*\d{1,2}:\d{2}\s*[AP]M", rest, re.I)
    if m:
        idx = m.start()
        team2 = rest[:idx].strip()
        dt = rest[idx:].strip()
        return team2, dt
    return rest.strip(), ""


def _parse_short_format(text: str, href: str) -> dict | None:
    """
    Parse main draw page format:
    'Sun 1 Mar1:15PMPre Match Knights Cowboys Allegiant Stadium, Paradise / Las Vegas NV'
    """
    if " have a bye" in text or " BYE" in text.upper():
        return None
    if "Pre Match" not in text and "Pre-Match" not in text:
        return None
    # Extract venue: last part before # or end (venue has comma and location)
    venue_part = ""
    if "#" in text:
        text = text.split("#")[0].strip()
    # Match: Day D MonTimePre Match Team1 Team2 Venue
    m = re.search(
        r"(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+(\d{1,2})\s+([A-Za-z]{3})\s*(\d{1,2}):(\d{2})\s*([AP]M)\s*(?:Pre[- ]?Match)?\s*(.+)",
        text, re.I
    )
    if not m:
        return None
    day_abbrev, day_num, mon_str, hour, minute, ampm = m.groups()[:6]
    rest = m.group(7).strip()
    # rest = "Knights Cowboys Allegiant Stadium, Paradise / Las Vegas NV"
    # Find venue: starts with a known venue or has comma (location suffix)
    venue_raw = ""
    teams_part = rest
    for v in ("Allegiant Stadium", "AAMI Park", "Go Media Stadium", "Suncorp Stadium",
              "Ocean Protect Stadium", "4 Pines Park", "CommBank Stadium",
              "McDonald Jones Stadium", "Queensland Country Bank Stadium",
              "Allianz Stadium", "GIO Stadium Canberra", "Polytec Stadium",
              "Cbus Super Stadium", "WIN Stadium", "Carrington Park", "TIO Stadium",
              "Sky Stadium", "Kayo Stadium", "Accor Stadium", "Optus Stadium",
              "Leichhardt Oval", "Jubilee Stadium", "Netstrata Jubilee Stadium",
              "Campbelltown Sports Stadium"):
        if v in rest:
            idx = rest.find(v)
            teams_part = rest[:idx].strip()
            venue_raw = rest[idx:].split(",")[0].strip()
            break
    if not venue_raw:
        parts = rest.rsplit(",", 1)
        if len(parts) == 2:
            teams_part = parts[0].strip()
            venue_raw = parts[1].strip().split("/")[0].strip()
        else:
            return None
    # teams_part = "Knights Cowboys" - split into two teams
    words = teams_part.split()
    t1 = t2 = ""
    for i in range(len(words), 0, -1):
        first = " ".join(words[:i])
        second = " ".join(words[i:])
        if first in LU_SHORT_TO_INTERNAL and second in LU_SHORT_TO_INTERNAL:
            t1 = LU_SHORT_TO_INTERNAL[first]
            t2 = LU_SHORT_TO_INTERNAL[second]
            break
    if not t1 or not t2:
        return None
    month = MONTH_MAP.get(mon_str.capitalize(), 1)
    h, mi = int(hour), int(minute)
    if ampm.upper() == "PM" and h < 12:
        h += 12
    elif ampm.upper() == "AM" and h == 12:
        h = 0
    date = f"{YEAR}-{month:02d}-{int(day_num):02d} {h:02d}:{mi:02d}"
    venue = _venue_mapper.to_internal(venue_raw) if venue_raw else venue_raw
    live_id = ""
    if m := re.search(r"/live-update/show/(\d+)/(\d+)", href):
        live_id = f"{m.group(1)}/{m.group(2)}"
    return {
        "round_num": "",
        "venue": venue,
        "date": date,
        "year": YEAR,
        "attendance": "",
        "team_1_team_name": t1,
        "team_2_team_name": t2,
        "team_1_score": None, "team_1_tries": None, "team_1_goals": None, "team_1_fg": None,
        "team_2_score": None, "team_2_tries": None, "team_2_goals": None, "team_2_fg": None,
        "live_update_id": live_id,
    }


def _parse_match_link(text: str, href: str) -> dict | None:
    """
    Parse link text like:
    'Newcastle Knights v North Queensland Cowboys Sun 1 Mar 1:15PM at Allegiant Stadium, Paradise / Las Vegas NV'
    Returns match dict or None for byes.
    """
    if " have a bye" in text:
        return None
    if " v " not in text and " vs " not in text.lower():
        return None
    text = text.replace(" vs ", " v ")
    parts = text.split(" v ", 1)
    if len(parts) != 2:
        return None
    team1_raw = parts[0].strip()
    rest = parts[1].strip()
    if " at " not in rest:
        return None
    team2_datetime, venue_part = rest.split(" at ", 1)
    team2_raw, date_str = _extract_team2_and_datetime(team2_datetime)
    venue_raw = venue_part.split(",")[0].strip() if venue_part else ""
    t1 = _team_mapper.to_internal(team1_raw)
    t2 = _team_mapper.to_internal(team2_raw)
    venue = _venue_mapper.to_internal(venue_raw) if venue_raw else venue_raw
    date = _parse_date(date_str) if date_str else ""
    # Extract live_update_id from href: /live-update/show/4586/28350
    live_id = ""
    m = re.search(r"/live-update/show/(\d+)/(\d+)", href)
    if m:
        live_id = f"{m.group(1)}/{m.group(2)}"
    return {
        "round_num": "",  # Filled by caller
        "venue": venue,
        "date": date,
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
        "live_update_id": live_id,
    }


def _extract_round_from_header(text: str) -> str | None:
    """Parse 'Round 1' or 'Round 3 Multicultural Round' -> '1' or '3'."""
    m = re.search(r"Round\s+(\d+)", text, re.I)
    return m.group(1) if m else None


def _fetch_from_league_unlimited() -> list[dict]:
    """Fetch 2026 fixture from League Unlimited. Uses /live-update/show/ links."""
    matches: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    # Fetch pages that cover all rounds (draw/2 returns multiple rounds)
    urls = [
        f"{LEAGUE_UNLIMITED_BASE}",
        f"{LEAGUE_UNLIMITED_BASE}/2",
        f"{LEAGUE_UNLIMITED_BASE}/14",
        f"{LEAGUE_UNLIMITED_BASE}/27",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            current_round = "1"
            for elem in soup.find_all(["h4", "a"]):
                if elem.name == "h4":
                    rn = _extract_round_from_header(elem.get_text(strip=True))
                    if rn:
                        current_round = rn
                    continue
                if elem.name != "a":
                    continue
                href = elem.get("href", "")
                if "/live-update/show/" not in href:
                    continue
                text = elem.get_text(strip=True)
                match = _parse_match_link(text, href)
                if match is None:
                    match = _parse_short_format(text, href)
                if match is None:
                    continue
                match["round_num"] = current_round
                key = (current_round, match["team_1_team_name"], match["team_2_team_name"])
                if key not in seen:
                    seen.add(key)
                    matches.append(match)
        except Exception:
            pass
    return matches


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
                "live_update_id": "",
            })
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
                "team_2_team_name", "team_2_score", "team_2_tries", "team_2_goals", "team_2_fg",
                "live_update_id"]
        if "competition" in df.columns:
            df = df.drop(columns=["competition", "match_id"], errors="ignore")
        df = df[[c for c in cols if c in df.columns]]
    else:
        df = pd.DataFrame(columns=[
            "round_num", "venue", "date", "year", "attendance",
            "team_1_team_name", "team_1_score", "team_1_tries", "team_1_goals", "team_1_fg",
            "team_2_team_name", "team_2_score", "team_2_tries", "team_2_goals", "team_2_fg",
            "live_update_id",
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} matches to {output_path}")
    return output_path


if __name__ == "__main__":
    fetch_and_save()
