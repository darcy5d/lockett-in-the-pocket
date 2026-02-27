#!/usr/bin/env python3
"""
Fetch the 2026 UK Super League fixture from League Unlimited.

Output: matches_super-league-uk_2026.csv (per competition_config)

Usage:
    python datafetch/fetch_2026_super_league_fixture.py
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

from core.competition_config import get_competition
from core.uk_mappings import TeamMapper, VenueMapper

OUTPUT_PATH = _PROJECT_ROOT / "nrl_data" / "data" / "matches" / "matches_super-league-uk_2026.csv"
LEAGUE_UNLIMITED_BASE = "https://leagueunlimited.com/competition/show/super-league/2026/draw"
YEAR = 2026

_team_mapper = TeamMapper()
_venue_mapper = VenueMapper()

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
MONTH_MAP = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

# Super League short names from "Pre Match Castleford Huddersfield Venue"
LU_SHORT_TO_INTERNAL = {
    "Castleford": "Castleford", "Huddersfield": "Huddersfield", "Hull": "Hull",
    "Hull FC": "Hull", "York": "York", "Warrington": "Warrington",
    "Wakefield": "Wakefield", "Catalans": "Catalans", "St Helens": "St Helens",
    "Hull KR": "Hull KR", "Leeds": "Leeds", "Bradford": "Bradford",
    "Toulouse": "Toulouse", "Wigan": "Wigan", "Leigh": "Leigh",
}


def _parse_date(date_str: str) -> str:
    if not date_str or not date_str.strip():
        return ""
    s = date_str.strip()
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{1,2}):(\d{2})\s*([AP]M)", s, re.I)
    if not m:
        return ""
    day, mon_str, hour, minute, ampm = m.groups()
    month = MONTH_MAP.get(mon_str.capitalize(), 1)
    h, mi = int(hour), int(minute)
    if ampm.upper() == "PM" and h < 12:
        h += 12
    elif ampm.upper() == "AM" and h == 12:
        h = 0
    return f"{YEAR}-{month:02d}-{int(day):02d} {h:02d}:{mi:02d}"


def _extract_team2_and_datetime(rest: str) -> tuple[str, str]:
    m = re.search(r"(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s*\d{1,2}\s+[A-Za-z]{3}\s*\d{1,2}:\d{2}\s*[AP]M", rest, re.I)
    if m:
        return rest[: m.start()].strip(), rest[m.start() :].strip()
    return rest.strip(), ""


def _parse_short_format(text: str, href: str) -> dict | None:
    if " have a bye" in text or " BYE" in text.upper():
        return None
    if "Pre Match" not in text and "Pre-Match" not in text and "Full Time" not in text:
        return None
    if "#" in text:
        text = text.split("#")[0].strip()
    m = re.search(
        r"(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+(\d{1,2})\s+([A-Za-z]{3})\s*(\d{1,2}):(\d{2})\s*([AP]M)\s*(?:Pre[- ]?Match|Full Time)?\s*(.+)",
        text, re.I
    )
    if not m:
        return None
    day_abbrev, day_num, mon_str, hour, minute, ampm = m.groups()[:6]
    rest = m.group(7).strip()
    venue_raw = ""
    teams_part = rest
    for v in ("The Brick Community Stadium", "OneBore Stadium", "MKM Stadium",
              "Halliwell Jones Stadium", "Stade Gilbert Brutus", "Allegiant Stadium",
              "Odsal Stadium", "Wheldon Road", "John Smith's Stadium"):
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
    if live_m := re.search(r"/live-update/show/(\d+)/(\d+)", href):
        live_id = f"{live_m.group(1)}/{live_m.group(2)}"
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
    if " have a bye" in text:
        return None
    if " v " not in text and " vs " not in text.lower():
        return None
    text = text.replace(" vs ", " v ")
    parts = text.split(" v ", 1)
    if len(parts) != 2 or " at " not in parts[1]:
        return None
    team1_raw = parts[0].strip()
    rest = parts[1].strip()
    team2_datetime, venue_part = rest.split(" at ", 1)
    team2_raw, date_str = _extract_team2_and_datetime(team2_datetime)
    venue_raw = venue_part.split(",")[0].strip() if venue_part else ""
    t1 = _team_mapper.to_internal(team1_raw)
    t2 = _team_mapper.to_internal(team2_raw)
    venue = _venue_mapper.to_internal(venue_raw) if venue_raw else venue_raw
    date = _parse_date(date_str) if date_str else ""
    live_id = ""
    if m := re.search(r"/live-update/show/(\d+)/(\d+)", href):
        live_id = f"{m.group(1)}/{m.group(2)}"
    return {
        "round_num": "", "venue": venue, "date": date, "year": YEAR, "attendance": "",
        "team_1_team_name": t1, "team_2_team_name": t2,
        "team_1_score": None, "team_1_tries": None, "team_1_goals": None, "team_1_fg": None,
        "team_2_score": None, "team_2_tries": None, "team_2_goals": None, "team_2_fg": None,
        "live_update_id": live_id,
    }


def _extract_round_from_header(text: str) -> str | None:
    m = re.search(r"Round\s+(\d+)", text, re.I)
    return m.group(1) if m else None


def _fetch_from_league_unlimited() -> list[dict]:
    matches: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    urls = [LEAGUE_UNLIMITED_BASE, f"{LEAGUE_UNLIMITED_BASE}/2", f"{LEAGUE_UNLIMITED_BASE}/14"]
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


def fetch_and_save(output_path: Path = OUTPUT_PATH) -> Path:
    matches = _fetch_from_league_unlimited()
    df = pd.DataFrame(matches) if matches else pd.DataFrame()
    if not df.empty:
        cols = ["round_num", "venue", "date", "year", "attendance",
                "team_1_team_name", "team_1_score", "team_1_tries", "team_1_goals", "team_1_fg",
                "team_2_team_name", "team_2_score", "team_2_tries", "team_2_goals", "team_2_fg",
                "live_update_id"]
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
