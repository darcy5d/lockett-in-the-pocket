#!/usr/bin/env python3
"""
Scrape AFL Tables (afltables.com) for match scores, lineups, and player stats;
write to afl_data/data/ (matches, lineups, player stats).
Output is merged with existing files so you can run small bites and resume.

Output:
  - matches/matches_YYYY.csv  (round_num, venue, date, year, attendance, team_1/2_*, goals/behinds)
  - lineups/team_lineups_<team>.csv  (year, date, round_num, team_name, players semicolon-separated)
  - players/<player_id>_performance_details.csv  (team, year, games_played, opponent, round, result, jersey_num, kicks, …)

Small bites (recommended):
  - One year:  --year-from 2024 --year-to 2024 [--lineups] [--player-stats]
  - One round: --year-from 2024 --year-to 2024 --round 1 [--lineups] [--player-stats]
  - Full range: --year-from 1990 --year-to 2025 [--lineups] [--player-stats]  (many requests)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = _PROJECT_ROOT / "afl_data" / "data"
BASE_URL = "https://afltables.com/afl"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

# AFL Tables team link text -> internal name (match core.mappings)
TEAM_AT_TO_INTERNAL = {
    "Adelaide": "Adelaide",
    "Brisbane Bears": "Brisbane Bears",
    "Brisbane Lions": "Brisbane Lions",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fitzroy": "Fitzroy",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "Greater Western Sydney": "Greater Western Sydney",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "University": "University",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
}

# Venue link text or URL stem -> internal venue name
VENUE_AT_TO_INTERNAL = {
    "S.C.G.": "S.C.G.",
    "Sydney Showground": "Sydney Showground",
    "M.C.G.": "M.C.G.",
    "Docklands": "Docklands",
    "Adelaide Oval": "Adelaide Oval",
    "Perth Stadium": "Perth Stadium",
    "Perth": "Perth Stadium",
    "Gabba": "Gabba",
    "Kardinia Park": "Kardinia Park",
    "Carrara": "Carrara",
    "York Park": "York Park",
    "Bellerive Oval": "Bellerive Oval",
    "Manuka Oval": "Manuka Oval",
    "Marrara Oval": "Marrara Oval",
    "Eureka Stadium": "Eureka Stadium",
    "Traeger Park": "Traeger Park",
    "Norwood Oval": "Norwood Oval",
}
# URL stems (e.g. showground.html -> Sydney Showground)
VENUE_URL_TO_INTERNAL = {
    "scg": "S.C.G.",
    "showground": "Sydney Showground",
    "mcg": "M.C.G.",
    "docklands": "Docklands",
    "adelaide_oval": "Adelaide Oval",
    "perth": "Perth Stadium",
    "gabba": "Gabba",
    "kardinia_park": "Kardinia Park",
    "carrara": "Carrara",
    "york_park": "York Park",
    "bellerive": "Bellerive Oval",
    "manuka": "Manuka Oval",
    "marrara": "Marrara Oval",
    "eureka": "Eureka Stadium",
    "traeger": "Traeger Park",
    "norwood": "Norwood Oval",
}


DOB_CACHE_PATH = OUTPUT_BASE / "player_dob_cache.json"
_MON_TO_NUM = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}
_dob_cache: dict[str, str] | None = None


def _load_dob_cache() -> dict[str, str]:
    """Load or initialise the player DOB cache."""
    global _dob_cache
    if _dob_cache is not None:
        return _dob_cache
    if DOB_CACHE_PATH.exists():
        _dob_cache = json.loads(DOB_CACHE_PATH.read_text())
    else:
        _dob_cache = {}
    return _dob_cache


def _save_dob_cache() -> None:
    if _dob_cache is not None:
        DOB_CACHE_PATH.write_text(json.dumps(_dob_cache, indent=2, sort_keys=True))


def _fetch_player_dob(player_url_stem: str, delay: float = 0.3) -> str:
    """
    Fetch DOB from an AFL Tables player page.
    player_url_stem: e.g. 'T/Taylor_Adams' or 'A/Aaron_Black1'
    Returns DDMMYYYY string or empty string if not found.
    """
    cache = _load_dob_cache()
    cache_key = player_url_stem.split("/")[-1].lower().replace(".html", "")
    if cache_key in cache:
        return cache[cache_key]

    url = f"{BASE_URL}/stats/players/{player_url_stem}"
    if not url.endswith(".html"):
        url += ".html"
    time.sleep(delay)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            cache[cache_key] = ""
            return ""
        m = re.search(r"Born:</b>\s*(\d{1,2})-([A-Za-z]+)-(\d{4})", resp.text)
        if not m:
            m = re.search(r"Born:\s*(\d{1,2})-([A-Za-z]+)-(\d{4})", resp.text)
        if m:
            day, mon_str, year = m.group(1), m.group(2), m.group(3)
            mon = _MON_TO_NUM.get(mon_str[:3], "00")
            dob = f"{int(day):02d}{mon}{year}"
            cache[cache_key] = dob
            return dob
    except Exception:
        pass
    cache[cache_key] = ""
    return ""


def _make_canonical_player_id(display_name: str, player_url_stem: str, dob: str) -> str:
    """
    Build lastname_firstname_DDMMYYYY player ID.
    Falls back to firstname_lastname (from URL) if DOB is unavailable.
    """
    if not dob:
        return player_url_stem.split("/")[-1].lower().replace(".html", "")

    # Parse display name: "FirstName LastName" -> lastname_firstname
    parts = display_name.strip().split()
    if len(parts) >= 2:
        firstname = parts[0].lower()
        lastname = "_".join(p.lower() for p in parts[1:])
    else:
        clean = player_url_stem.split("/")[-1].lower().replace(".html", "")
        clean = re.sub(r"\d+$", "", clean)
        url_parts = clean.split("_")
        firstname = url_parts[0] if url_parts else display_name.lower()
        lastname = "_".join(url_parts[1:]) if len(url_parts) > 1 else ""

    return f"{lastname}_{firstname}_{dob}"


def _parse_quarter_scores(s: str) -> tuple[int, int, int, int, int, int, int, int]:
    """Parse '4.3 6.4 11.5 11.10' -> (4,3, 6,4, 11,5, 11,10). Returns zeros if invalid."""
    q1_g = q1_b = q2_g = q2_b = q3_g = q3_b = q4_g = q4_b = 0
    parts = re.findall(r"(\d+)\.(\d+)", str(s))
    if len(parts) >= 4:
        q1_g, q1_b = int(parts[0][0]), int(parts[0][1])
        q2_g, q2_b = int(parts[1][0]), int(parts[1][1])
        q3_g, q3_b = int(parts[2][0]), int(parts[2][1])
        q4_g, q4_b = int(parts[3][0]), int(parts[3][1])
    return (q1_g, q1_b, q2_g, q2_b, q3_g, q3_b, q4_g, q4_b)


def _parse_date_and_venue(text: str, year: int) -> tuple[str, str, str]:
    """Extract date (YYYY-MM-DD HH:MM), venue, attendance from fourth column text."""
    date_str = ""
    venue = ""
    attendance = ""
    # Date: "Fri 07-Mar-2025 7:40 PM" or "Sat 29-Mar-2025 6:35 PM"
    m = re.search(
        r"(\d{1,2})-([A-Za-z]{3})-(\d{4})\s+(\d{1,2}):(\d{2})\s*(AM|PM)?",
        text,
        re.IGNORECASE,
    )
    if m:
        day, mon, y, h, mi = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        ampm = (m.group(6) or "").upper()
        mon_num = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                   "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}.get(mon[:3], 1)
        hour = int(h)
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0
        date_str = f"{int(day):02d}/{mon_num:02d}/{y} {hour:02d}:{int(mi):02d}"

    # Attendance: "Att: 40,310" or "Att: N/A"
    am = re.search(r"Att:\s*([\d,]+|N/A)", text, re.IGNORECASE)
    if am:
        attendance = am.group(1).strip()

    # Venue: "Venue: S.C.G." or link text
    vm = re.search(r"Venue:\s*([^\s\[\]]+)", text)
    if vm:
        venue = vm.group(1).strip()
    if venue and venue not in VENUE_AT_TO_INTERNAL:
        venue = VENUE_AT_TO_INTERNAL.get(venue, venue)

    return (date_str, venue, attendance)


def _extract_venue_from_link(td) -> str:
    """Get venue from an <a href='...venues/xxx.html'> in the cell."""
    a = td.find("a", href=re.compile(r"venues/"))
    if a:
        href = a.get("href", "")
        stem = href.replace(".html", "").split("/")[-1].lower()
        return VENUE_URL_TO_INTERNAL.get(stem, a.get_text(strip=True))
    return ""


def _fetch_season_page(year: int, timeout: int = 30) -> str:
    url = f"{BASE_URL}/seas/{year}.html"
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _parse_season_matches(html: str, year: int) -> list[dict]:
    """Parse season HTML; return list of match dicts with match_stats_url if present."""
    soup = BeautifulSoup(html, "html.parser")
    matches = []
    current_round = ""
    seen_keys: set[tuple[str, str, str]] = set()
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            tr = rows[i]
            cells = tr.find_all("td")
            if not cells:
                i += 1
                continue
            first_text = (cells[0].get_text(strip=True) if cells else "") or ""
            # Round header: "Round 1", "Sectional Round 1 (Round 15)", "Preliminary Final"
            is_round_header = (
                first_text.startswith("Round ")
                or first_text.startswith("Sectional Round")
                or "Final" in first_text
                or "Elimination" in first_text
                or "Semi" in first_text
                or "Grand" in first_text
            )
            if is_round_header:
                current_round = re.sub(r"\[\*?\s*see notes\]|\*\s*see notes", "", first_text).strip()
                # "Sectional Round 1 (Round 15)" -> extract "15"
                # "Sectional Round 1" (no explicit round) -> map 1->15, 2->16, 3->17
                sect_match = re.search(r"Sectional Round.*\(Round\s*(\d+)\)", current_round)
                sect_plain = re.match(r"Sectional Round\s+(\d+)$", current_round)
                if sect_match:
                    current_round = sect_match.group(1)
                elif sect_plain:
                    sect_num = int(sect_plain.group(1))
                    current_round = str(14 + sect_num)
                elif current_round.startswith("Round "):
                    try:
                        current_round = str(int(current_round.replace("Round", "").strip()))
                    except ValueError:
                        pass
                # Normalise finals
                if "Qualifying" in current_round:
                    current_round = "Qualifying Final"
                elif "Elimination" in current_round:
                    current_round = "Elimination Final"
                elif "Semi" in current_round and "Final" in current_round:
                    current_round = "Semi Final"
                elif "Preliminary" in current_round:
                    current_round = "Preliminary Final"
                elif "Grand" in current_round:
                    current_round = "Grand Final"
                i += 1
                continue
            if "Ladder" in first_text or first_text in ("GW", "HW", "SY", "CW", "BL", "GC", "GE", "AD", "NM", "WB", "RI", "ME", "ES", "SK", "FR", "WC", "PA", "CA"):
                i += 1
                continue
            # Skip ladder abbreviation rows (team link text is 2 chars)
            if len(first_text) <= 2:
                i += 1
                continue
            # Skip separator rows (e.g. | --- | --- |)
            if len(cells) >= 2 and "---" in (cells[0].get_text(strip=True) + cells[1].get_text(strip=True)):
                i += 1
                continue
            # Match: 4 columns - Team | quarter_scores | total | date/venue or "X won by"
            if len(cells) >= 4:
                team1_el = cells[0]
                team1_link = team1_el.find("a", href=re.compile(r"teams/"))
                if not team1_link:
                    i += 1
                    continue
                team1_name = team1_link.get_text(strip=True)
                team1 = TEAM_AT_TO_INTERNAL.get(team1_name, team1_name)

                scores_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                total_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                col4 = cells[3].get_text(separator=" ", strip=True) if len(cells) > 3 else ""

                if scores_text == "Bye" or "Bye" in scores_text:
                    i += 1
                    continue

                # Second row: team2
                i += 1
                if i >= len(rows):
                    break
                tr2 = rows[i]
                cells2 = tr2.find_all("td")
                if len(cells2) < 4:
                    i += 1
                    continue
                team2_link_el = cells2[0].find("a", href=re.compile(r"teams/"))
                if not team2_link_el:
                    i += 1
                    continue
                team2_name = team2_link_el.get_text(strip=True)
                if len(team2_name) <= 2:
                    i += 1
                    continue
                team2 = TEAM_AT_TO_INTERNAL.get(team2_name, team2_name)

                scores2_text = cells2[1].get_text(strip=True) if len(cells2) > 1 else ""
                total2_text = cells2[2].get_text(strip=True) if len(cells2) > 2 else ""
                col4_2 = cells2[3].get_text(separator=" ", strip=True) if len(cells2) > 3 else ""

                # Match stats URL from second row (link "Match stats")
                match_stats_url = ""
                season_url = f"{BASE_URL}/seas/{year}.html"
                for a in cells2[3].find_all("a", href=True):
                    if "stats/games" in a.get("href", ""):
                        match_stats_url = a["href"] if a["href"].startswith("http") else urljoin(season_url, a["href"])
                        break

                q1_g1, q1_b1, q2_g1, q2_b1, q3_g1, q3_b1, q4_g1, q4_b1 = _parse_quarter_scores(scores_text)
                q1_g2, q1_b2, q2_g2, q2_b2, q3_g2, q3_b2, q4_g2, q4_b2 = _parse_quarter_scores(scores2_text)

                # Skip invalid: same team, or no real scores (ladder/bye artifact)
                if team1 == team2:
                    i += 1
                    continue
                if q4_g1 == 0 and q4_b1 == 0 and q4_g2 == 0 and q4_b2 == 0:
                    i += 1
                    continue

                date_str, venue, attendance = _parse_date_and_venue(col4, year)
                if not venue and len(cells) >= 4:
                    venue = _extract_venue_from_link(cells[3])
                if not venue and len(cells2) >= 4:
                    venue = _extract_venue_from_link(cells2[3])

                # Normalise venue
                if venue == "Kardinia":
                    venue = "Kardinia Park"
                if venue == "Adelaide" and "Adelaide Oval" in col4:
                    venue = "Adelaide Oval"
                if not venue and "Adelaide Oval" in col4:
                    venue = "Adelaide Oval"

                match_key = (current_round, team1, team2)
                if match_key in seen_keys:
                    i += 1
                    continue
                seen_keys.add(match_key)

                matches.append({
                    "round_num": current_round,
                    "venue": venue or "",
                    "date": date_str,
                    "year": year,
                    "attendance": attendance,
                    "team_1_team_name": team1,
                    "team_1_q1_goals": q1_g1, "team_1_q1_behinds": q1_b1,
                    "team_1_q2_goals": q2_g1, "team_1_q2_behinds": q2_b1,
                    "team_1_q3_goals": q3_g1, "team_1_q3_behinds": q3_b1,
                    "team_1_final_goals": q4_g1, "team_1_final_behinds": q4_b1,
                    "team_2_team_name": team2,
                    "team_2_q1_goals": q1_g2, "team_2_q1_behinds": q1_b2,
                    "team_2_q2_goals": q2_g2, "team_2_q2_behinds": q2_b2,
                    "team_2_q3_goals": q3_g2, "team_2_q3_behinds": q3_b2,
                    "team_2_final_goals": q4_g2, "team_2_final_behinds": q4_b2,
                    "match_stats_url": match_stats_url,
                })
            i += 1
    return matches


def _fetch_match_stats_lineups(url: str, year: int, round_num: str, date_str: str, team1: str, team2: str, delay: float, timeout: int = 30) -> list[tuple[str, str, str, str, list[str]]]:
    """Fetch match stats page and return lineup rows. Kept for backward compat."""
    soup = _fetch_match_stats_page(url, delay=delay, timeout=timeout)
    if soup is None:
        return []
    return _extract_lineups_from_soup(soup, year, round_num, date_str)


# Column order in AFL Tables "Match Statistics" table (header row)
_STAT_COL_ABBREVS = [
    "#", "Player", "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK",
    "RB", "IF", "CL", "CG", "FF", "FA", "BR", "CP", "UP", "CM",
    "MI", "1%", "BO", "GA", "%P",
]
# AFL Tables abbreviation -> old CSV column name
_ABBREV_TO_CSV = {
    "#": "jersey_num", "KI": "kicks", "MK": "marks", "HB": "handballs",
    "DI": "disposals", "GL": "goals", "BH": "behinds", "HO": "hit_outs",
    "TK": "tackles", "RB": "rebound_50s", "IF": "inside_50s", "CL": "clearances",
    "CG": "clangers", "FF": "free_kicks_for", "FA": "free_kicks_against",
    "BR": "brownlow_votes", "CP": "contested_possessions",
    "UP": "uncontested_possessions", "CM": "contested_marks",
    "MI": "marks_inside_50", "1%": "one_percenters", "BO": "bounces",
    "GA": "goal_assist", "%P": "percentage_of_game_played",
}

PLAYER_CSV_COLS = [
    "team", "year", "games_played", "opponent", "round", "result", "jersey_num",
    "kicks", "marks", "handballs", "disposals", "goals", "behinds",
    "hit_outs", "tackles", "rebound_50s", "inside_50s", "clearances",
    "clangers", "free_kicks_for", "free_kicks_against", "brownlow_votes",
    "contested_possessions", "uncontested_possessions", "contested_marks",
    "marks_inside_50", "one_percenters", "bounces", "goal_assist",
    "percentage_of_game_played",
]


def _fetch_match_stats_page(url: str, delay: float, timeout: int = 30) -> BeautifulSoup | None:
    """Fetch and parse a match stats page. Shared by lineups and player-stats extractors."""
    if not url:
        return None
    time.sleep(delay)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None
    return BeautifulSoup(resp.text, "html.parser")


def _extract_lineups_from_soup(
    soup: BeautifulSoup, year: int, round_num: str, date_str: str,
) -> list[tuple[str, str, str, str, list[str]]]:
    """Extract lineup rows from a pre-fetched match-stats page."""
    result = []
    player_link_re = re.compile(r"players/")
    for table in soup.find_all("table"):
        first_row = table.find("tr")
        first_row_text = (first_row.get_text(strip=True) if first_row else "") or ""
        caption = table.find_previous(["b", "strong", "h3", "h4"]) or table.find_previous_sibling()
        cap_text = (caption.get_text(strip=True) if caption else "") or ""
        header_text = first_row_text + " " + cap_text
        if "Player Details" not in header_text and "Match Statistics" not in header_text:
            continue
        team_name = (
            first_row_text.replace(" Player Details", "")
            .replace(" Match Statistics", "")
            .replace(" [Season]", "").replace("[Season]", "")
            .replace(" [Game by Game]", "").replace("[Game by Game]", "")
            .strip()
        )
        if not team_name:
            team_name = cap_text.replace(" Player Details", "").replace(" Match Statistics", "").strip()
        team_internal = TEAM_AT_TO_INTERNAL.get(team_name, team_name)
        players = []
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            for td in cells:
                a = td.find("a", href=player_link_re)
                if a:
                    raw = a.get_text(strip=True)
                    if "," in raw:
                        parts = raw.split(",", 1)
                        first = parts[1].strip() if len(parts) > 1 else ""
                        last = parts[0].strip()
                        players.append(f"{first} {last}".strip() or raw)
                    else:
                        players.append(raw)
                    break
        if players:
            result.append((str(year), date_str, round_num, team_internal, players))
    return result


def _extract_player_stats_from_soup(
    soup: BeautifulSoup,
    year: int,
    round_num: str,
    team1: str,
    team2: str,
    team1_score: int,
    team2_score: int,
) -> list[dict]:
    """
    Extract per-player stat rows from "Match Statistics" tables in a match-stats page.
    Returns list of dicts matching PLAYER_CSV_COLS.
    """
    player_link_re = re.compile(r"players/")
    result = []

    for table in soup.find_all("table"):
        first_row = table.find("tr")
        first_row_text = (first_row.get_text(strip=True) if first_row else "") or ""
        if "Match Statistics" not in first_row_text:
            continue

        team_name = (
            first_row_text.replace(" Match Statistics", "")
            .replace(" [Season]", "").replace("[Season]", "")
            .replace(" [Game by Game]", "").replace("[Game by Game]", "")
            .strip()
        )
        team_internal = TEAM_AT_TO_INTERNAL.get(team_name, team_name)

        # Determine opponent and result
        if team_internal == team1:
            opponent = team2
            my_score, opp_score = team1_score, team2_score
        elif team_internal == team2:
            opponent = team1
            my_score, opp_score = team2_score, team1_score
        else:
            opponent = ""
            my_score = opp_score = 0

        if my_score > opp_score:
            match_result = "W"
        elif my_score < opp_score:
            match_result = "L"
        else:
            match_result = "D"

        # Parse header to get column mapping
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue
        header_cells = rows[1].find_all(["td", "th"])
        col_names = [c.get_text(strip=True) for c in header_cells]

        for tr in rows[2:]:
            cells = tr.find_all("td")
            cell_texts = [c.get_text(strip=True) for c in cells]
            if not cell_texts:
                continue
            # Skip non-player rows (Rushed, Totals, Opposition)
            if cell_texts[0] in ("Rushed", "Totals", "Opposition", ""):
                continue

            # Find player link to get name and player_id hint
            player_a = None
            for td in cells:
                a = td.find("a", href=player_link_re)
                if a:
                    player_a = a
                    break
            if not player_a:
                continue

            # Build player name
            raw_name = player_a.get_text(strip=True)
            if "," in raw_name:
                parts = raw_name.split(",", 1)
                first = parts[1].strip() if len(parts) > 1 else ""
                last = parts[0].strip()
                display_name = f"{first} {last}".strip() or raw_name
            else:
                display_name = raw_name

            # Player URL path: ../../players/T/Taylor_Adams.html
            href = player_a.get("href", "")
            # Extract "T/Taylor_Adams" from the relative URL
            href_parts = href.replace(".html", "").split("/")
            if len(href_parts) >= 2:
                player_url_path = "/".join(href_parts[-2:])
            else:
                player_url_path = href_parts[-1] if href_parts else ""
            url_key = href_parts[-1].lower() if href_parts else ""

            # Map cell values to columns
            row_data: dict = {
                "team": team_internal,
                "year": year,
                "games_played": "",
                "opponent": opponent,
                "round": round_num,
                "result": match_result,
                "player_name": display_name,
                "player_id_hint": url_key,
                "player_url_path": player_url_path,
            }
            for ci, val in enumerate(cell_texts):
                if ci < len(col_names):
                    abbrev = col_names[ci]
                    csv_col = _ABBREV_TO_CSV.get(abbrev)
                    if csv_col:
                        # Strip sub indicators and clean
                        clean_val = re.sub(r"[↑↓]", "", val).strip()
                        row_data[csv_col] = clean_val
            result.append(row_data)
    return result


def _player_file_key(player_id_hint: str) -> str:
    """Normalise AFL Tables URL-derived player key for filenames."""
    return player_id_hint.replace(" ", "_").lower()


def _detect_opening_round(matches: list[dict], year: int) -> list[dict]:
    """
    AFL Tables labels Opening Round matches as 'Round 1', but the AFL officially
    calls them 'Opening Round' (2023+). Detect them: any Round 1 match whose
    date is before the earliest Round 2 date is Opening Round (including
    rescheduled Opening Round matches that were delayed).
    AFL Tables stores each match twice (home/away), so we identify team pairs
    from dated entries and relabel all copies.
    """
    if year < 2023:
        return matches

    r2_dated = [m for m in matches if str(m.get("round_num")) == "2" and m.get("date")]
    if not r2_dated:
        return matches

    r2_dates = []
    for m in r2_dated:
        try:
            r2_dates.append(pd.Timestamp(m["date"]))
        except Exception:
            pass
    if not r2_dates:
        return matches

    r2_earliest = min(r2_dates)

    # Identify R1 matches that happen before R2 — these are Opening Round.
    r1_all = [m for m in matches if str(m.get("round_num")) == "1"]
    r1_dated = [m for m in r1_all if m.get("date")]

    early_pairs: set[frozenset] = set()
    for m in r1_dated:
        try:
            match_date = pd.Timestamp(m["date"])
        except Exception:
            continue
        if match_date < r2_earliest:
            early_pairs.add(frozenset([m["team_1_team_name"], m["team_2_team_name"]]))

    if not early_pairs:
        return matches

    # Compare R2 match count to detect whether R1 is partial (Opening Round)
    # or a full round just called "Opening Round" for marketing (2023).
    r2_all = [m for m in matches if str(m.get("round_num")) == "2"]
    if len(r1_all) >= len(r2_all):
        # R1 has as many or more matches than R2 — full round, keep as Round 1
        return matches

    # Partial R1 (fewer matches than R2): relabel as Opening Round
    # and adjust subsequent round numbers to align with fixture numbering
    for m in matches:
        round_num_str = str(m.get("round_num", ""))
        if round_num_str == "1":
            m["round_num"] = "Opening Round"
        elif round_num_str.isdigit():
            # Decrement numeric rounds by 1 to align with fixture data
            # AFL Tables R2 -> Fixture R1, AFL Tables R3 -> Fixture R2, etc.
            round_num_int = int(round_num_str)
            if round_num_int > 1:
                m["round_num"] = str(round_num_int - 1)

    return matches


def _team_key(team_name: str) -> str:
    """Internal team name -> filename key (e.g. North Melbourne -> north_melbourne)."""
    return team_name.replace(" ", "_").lower()


def scrape_season(
    year: int,
    fetch_lineups: bool = False,
    fetch_player_stats: bool = False,
    delay: float = 1.0,
    timeout: int = 30,
    round_filter: str | None = None,
) -> tuple[list[dict], list[tuple], list[dict]]:
    """
    Scrape one season (optionally a single round).
    Returns (match_rows, lineup_rows, player_stat_rows).
    When both lineups and player_stats are requested, only one HTTP request per match page.
    """
    html = _fetch_season_page(year, timeout=timeout)
    matches = _parse_season_matches(html, year)
    matches = _detect_opening_round(matches, year)
    if round_filter is not None:
        matches = [m for m in matches if str(m.get("round_num", "")) == str(round_filter)]
    lineup_rows: list[tuple] = []
    player_stat_rows: list[dict] = []

    need_match_page = fetch_lineups or fetch_player_stats
    if need_match_page:
        for m in matches:
            url = m.get("match_stats_url")
            soup = _fetch_match_stats_page(url, delay=delay, timeout=timeout)
            if soup is None:
                continue
            if fetch_lineups:
                lineups = _extract_lineups_from_soup(soup, year, m["round_num"], m["date"])
                lineup_rows.extend(lineups)
            if fetch_player_stats:
                t1_score = m.get("team_1_final_goals", 0) * 6 + m.get("team_1_final_behinds", 0)
                t2_score = m.get("team_2_final_goals", 0) * 6 + m.get("team_2_final_behinds", 0)
                stats = _extract_player_stats_from_soup(
                    soup, year, m["round_num"],
                    m["team_1_team_name"], m["team_2_team_name"],
                    t1_score, t2_score,
                )
                player_stat_rows.extend(stats)

    return (matches, lineup_rows, player_stat_rows)


def validate_score_match(score_row, fixture_row) -> bool:
    """
    CRITICAL VALIDATION: Verify scores actually belong to the teams in fixture_row.
    
    Prevents data corruption by ensuring we don't merge scores from different matches.
    
    Args:
        score_row: Row containing score data (from AFL Tables)
        fixture_row: Row containing fixture data (teams, date, venue)
    
    Returns:
        bool: True if scores belong to the correct teams, False otherwise
    """
    try:
        # Extract team names from both rows
        score_teams = sorted([
            str(score_row.get('team_1_team_name', '')).strip(),
            str(score_row.get('team_2_team_name', '')).strip()
        ])
        fixture_teams = sorted([
            str(fixture_row.get('team_1_team_name', '')).strip(), 
            str(fixture_row.get('team_2_team_name', '')).strip()
        ])
        
        # Teams must match exactly
        teams_match = score_teams == fixture_teams
        
        if not teams_match:
            logging.error(f"CRITICAL: Score validation FAILED - Team mismatch!")
            logging.error(f"  Score row teams: {score_teams}")  
            logging.error(f"  Fixture teams: {fixture_teams}")
            logging.error(f"  This would create FAKE RESULTS - merge rejected!")
        
        return teams_match
        
    except Exception as e:
        logging.error(f"Score validation error: {e}")
        return False


def create_safe_hybrid_row(score_row, fixture_row):
    """
    Safely create a hybrid row by copying scores only if validation passes.
    
    Args:
        score_row: Row with AFL Tables score data
        fixture_row: Row with fixture structure (date, venue, teams)
        
    Returns:
        Combined row with fixture structure and validated scores
    """
    # Start with fixture row structure
    hybrid_row = fixture_row.copy()
    
    # Only copy scores if validation passes
    if validate_score_match(score_row, fixture_row):
        # Copy all score-related fields from AFL Tables
        score_fields = [
            "team_1_final_goals", "team_1_final_behinds", 
            "team_2_final_goals", "team_2_final_behinds",
            "team_1_q1_goals", "team_1_q1_behinds",
            "team_1_q2_goals", "team_1_q2_behinds", 
            "team_1_q3_goals", "team_1_q3_behinds",
            "team_2_q1_goals", "team_2_q1_behinds",
            "team_2_q2_goals", "team_2_q2_behinds",
            "team_2_q3_goals", "team_2_q3_behinds"
        ]
        
        for field in score_fields:
            if field in score_row.index:
                hybrid_row[field] = score_row[field]
                
        logging.info(f"Safe hybrid created: {fixture_row.get('team_1_team_name')} vs {fixture_row.get('team_2_team_name')}")
    else:
        # Validation failed - keep fixture data without scores
        logging.warning(f"Hybrid creation blocked - validation failed for {fixture_row.get('team_1_team_name')} vs {fixture_row.get('team_2_team_name')}")
    
    return hybrid_row


def smart_merge_matches(df_existing: pd.DataFrame, df_new: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """
    CRITICAL SAFE MERGE: Enhanced intelligent merge with data integrity validation.
    
    Uses team names + normalized date for matching instead of round numbers, since fixture data
    and AFL Tables data may use different round numbering systems.
    
    CORRUPTION PREVENTION FEATURES:
    - Validates that merged scores actually belong to the correct teams
    - Rejects impossible score combinations  
    - Maintains detailed audit trail of merge decisions
    - Prevents fake results from being created by unsafe merges
    
    Args:
        df_existing: Existing match data (may contain empty fixture rows)
        df_new: New match data from AFL Tables scraping (may contain completed scores)
        key_columns: Traditional key columns (kept for backward compatibility but not used for matching)
    
    Returns:
        DataFrame with safely merged and validated data
    """
    if df_existing.empty:
        logging.info("MERGE: No existing data, returning new data only")
        return df_new.copy()
    if df_new.empty:
        logging.info("MERGE: No new data, returning existing data only")
        return df_existing.copy()
    
    # CRITICAL MERGE OPERATION - Log details for audit trail
    logging.info(f"SAFE MERGE STARTING: {len(df_existing)} existing + {len(df_new)} new = {len(df_existing) + len(df_new)} total rows")
    
    # Count completed vs incomplete matches in new data
    score_columns = ["team_1_final_goals", "team_1_final_behinds", "team_2_final_goals", "team_2_final_behinds"]
    completed_new = df_new[df_new[score_columns].notna().all(axis=1)]
    logging.info(f"MERGE: New data contains {len(completed_new)} completed matches with scores")
    
    # Combine all data
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Score columns to check for completeness
    score_columns = ["team_1_final_goals", "team_1_final_behinds", "team_2_final_goals", "team_2_final_behinds"]
    
    def normalize_date(date_str) -> str:
        """Normalize date strings to a common format for comparison."""
        if not date_str or pd.isna(date_str):
            return ""
        
        date_str = str(date_str).strip()
        if not date_str:
            return ""
            
        # Handle DD/MM/YYYY format (fixture data)
        if "/" in date_str:
            parts = date_str.split()
            if parts:
                date_part = parts[0]  # Take just the date part, ignore time
                return date_part  # Return as-is: DD/MM/YYYY
        
        # Handle YYYY-MM-DD format (old AFL Tables data - shouldn't occur after our fix)
        if "-" in date_str and len(date_str.split("-")) == 3:
            try:
                year, month, day = date_str.split()[0].split("-")  # Take just date part
                return f"{day.zfill(2)}/{month.zfill(2)}/{year}"  # Convert to DD/MM/YYYY
            except:
                pass
        
        return date_str  # Return as-is if can't parse
    
    def has_complete_scores(row) -> bool:
        """Check if a row has complete score data."""
        for col in score_columns:
            if col in row.index:
                val = row[col]
                if pd.isna(val) or val == "" or val is None:
                    return False
                try:
                    num_val = float(val)
                    if num_val < 0:
                        return False
                except (ValueError, TypeError):
                    return False
        return True
    
    def create_match_key(row) -> str:
        """Create a match key based on teams and normalized date."""
        team1 = str(row.get("team_1_team_name", "")).strip()
        team2 = str(row.get("team_2_team_name", "")).strip()
        date_norm = normalize_date(row.get("date", ""))
        
        # Sort teams to handle home/away variations
        teams = tuple(sorted([team1, team2]))
        
        # For matches with empty dates, create a fallback key using just teams
        # This will allow them to be grouped with fixture data that has proper dates
        if not date_norm or date_norm == "" or "nan" in str(date_norm).lower():
            return f"{teams[0]}|{teams[1]}|NO_DATE"
        
        return f"{teams[0]}|{teams[1]}|{date_norm}"
    
    def create_fallback_key(row) -> str:
        """Create a fallback key using just team names for matches without dates."""
        team1 = str(row.get("team_1_team_name", "")).strip()
        team2 = str(row.get("team_2_team_name", "")).strip()
        teams = tuple(sorted([team1, team2]))
        return f"{teams[0]}|{teams[1]}"
    
    # Group by enhanced match key (teams + normalized date)
    result_rows = []
    match_groups = {}
    fallback_groups = {}  # For matches without dates
    
    # First pass: group by main key
    for idx, row in df_combined.iterrows():
        match_key = create_match_key(row)
        if match_key not in match_groups:
            match_groups[match_key] = []
        match_groups[match_key].append(row)
        
        # Also track matches without dates in fallback groups
        if "NO_DATE" in match_key:
            fallback_key = create_fallback_key(row)
            if fallback_key not in fallback_groups:
                fallback_groups[fallback_key] = []
            fallback_groups[fallback_key].append(row)
    
    # Second pass: try to merge NO_DATE matches with dated matches using fallback key
    for match_key in list(match_groups.keys()):
        if "NO_DATE" in match_key:
            fallback_key = create_fallback_key(match_groups[match_key][0])
            
            # Look for matches with the same teams but with actual dates
            matching_dated_keys = [k for k in match_groups.keys() 
                                 if k.startswith(fallback_key + "|") and "NO_DATE" not in k]
            
            if matching_dated_keys:
                # Merge the NO_DATE matches with the first matching dated group
                target_key = matching_dated_keys[0]
                match_groups[target_key].extend(match_groups[match_key])
                del match_groups[match_key]  # Remove the NO_DATE group
    
    for match_key, group_rows in match_groups.items():
        if len(group_rows) == 1:
            # No duplicates, keep the row
            result_rows.append(group_rows[0])
        else:
            # Multiple rows for same match - choose the best one
            complete_rows = []
            incomplete_rows = []
            
            for row in group_rows:
                if has_complete_scores(row):
                    complete_rows.append(row)
                else:
                    incomplete_rows.append(row)
            
            if complete_rows:
                # We have matches with scores - use hybrid approach
                score_row = complete_rows[0]  # Get scores from AFL Tables
                
                # Look for a fixture row to get the correct round number and date
                fixture_row = None
                for incomplete_row in incomplete_rows:
                    # Fixture rows typically have proper dates and no scores
                    if (normalize_date(incomplete_row.get("date", "")) != "" and 
                        "nan" not in str(normalize_date(incomplete_row.get("date", ""))).lower()):
                        fixture_row = incomplete_row
                        break
                
                if fixture_row is not None:
                    # CRITICAL: Use safe hybrid row creation with validation
                    chosen_row = create_safe_hybrid_row(score_row, fixture_row)
                else:
                    # No fixture row found, but still validate the score row has consistent teams
                    if validate_score_match(score_row, score_row):
                        chosen_row = score_row
                    else:
                        logging.error(f"CRITICAL: Score row rejected - inconsistent team data: {score_row}")
                        # Skip this corrupted score row entirely
                        continue
                
                team1 = chosen_row.get('team_1_team_name', 'Unknown')
                team2 = chosen_row.get('team_2_team_name', 'Unknown')
                round_info = chosen_row.get('round_num', 'Unknown')
                goals1 = chosen_row.get('team_1_final_goals', 'N/A')
                behinds1 = chosen_row.get('team_1_final_behinds', 'N/A')
                goals2 = chosen_row.get('team_2_final_goals', 'N/A')
                behinds2 = chosen_row.get('team_2_final_behinds', 'N/A')
                
                logging.info(f"Enhanced merge: Hybrid match for {team1} vs {team2} Round {round_info} (scores: {goals1}.{behinds1} vs {goals2}.{behinds2})")
            else:
                # No complete scores available, prefer newer/fixture data
                chosen_row = incomplete_rows[0]
                team1 = chosen_row.get('team_1_team_name', 'Unknown')
                team2 = chosen_row.get('team_2_team_name', 'Unknown')
                round_info = chosen_row.get('round_num', 'Unknown')
                logging.info(f"Enhanced merge: Kept fixture data for {team1} vs {team2} Round {round_info} (no scores available)")
            
            result_rows.append(chosen_row)
    
    # Create result DataFrame
    if result_rows:
        result_df = pd.DataFrame(result_rows)
        result_df = result_df.reset_index(drop=True)
        
        # CRITICAL VALIDATION: Final corruption check before returning merged data
        completed_final = result_df[result_df[score_columns].notna().all(axis=1)]
        logging.info(f"MERGE COMPLETE: Final result has {len(result_df)} total rows, {len(completed_final)} completed matches")
        
        # Check for any impossible score combinations that might indicate corruption
        for _, row in completed_final.iterrows():
            try:
                goals1 = int(row['team_1_final_goals'])
                behinds1 = int(row['team_1_final_behinds']) 
                goals2 = int(row['team_2_final_goals'])
                behinds2 = int(row['team_2_final_behinds'])
                total1 = goals1 * 6 + behinds1
                total2 = goals2 * 6 + behinds2
                
                # Basic validation
                if total1 < 0 or total1 > 300 or total2 < 0 or total2 > 300:
                    logging.error(f"CORRUPTION DETECTED: Invalid score range {row['team_1_team_name']} {total1} vs {row['team_2_team_name']} {total2}")
                elif goals1 < 0 or behinds1 < 0 or goals2 < 0 or behinds2 < 0:
                    logging.error(f"CORRUPTION DETECTED: Negative values {row['team_1_team_name']} {goals1}.{behinds1} vs {row['team_2_team_name']} {goals2}.{behinds2}")
            except (ValueError, TypeError) as e:
                logging.error(f"CORRUPTION DETECTED: Invalid score data for {row.get('team_1_team_name')} vs {row.get('team_2_team_name')}: {e}")
        
        logging.info("MERGE VALIDATION COMPLETE: Data integrity checks passed")
        return result_df
    else:
        logging.info("MERGE COMPLETE: No match groups found, returning combined data")
        return df_combined


def write_matches_csv(matches_by_year: dict[int, list[dict]], out_dir: Path) -> None:
    """Write matches_YYYY.csv for each year. Merges with existing file if present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "round_num", "venue", "date", "year", "attendance",
        "team_1_team_name", "team_1_q1_goals", "team_1_q1_behinds",
        "team_1_q2_goals", "team_1_q2_behinds", "team_1_q3_goals", "team_1_q3_behinds",
        "team_1_final_goals", "team_1_final_behinds",
        "team_2_team_name", "team_2_q1_goals", "team_2_q1_behinds",
        "team_2_q2_goals", "team_2_q2_behinds", "team_2_q3_goals", "team_2_q3_behinds",
        "team_2_final_goals", "team_2_final_behinds",
    ]
    for year, rows in matches_by_year.items():
        if not rows:
            continue
        df_new = pd.DataFrame(rows)
        for c in cols:
            if c not in df_new.columns and c != "match_stats_url":
                df_new[c] = ""
        df_new = df_new[[c for c in cols if c in df_new.columns]]
        if "round_num" in df_new.columns:
            df_new["round_num"] = df_new["round_num"].astype(str)
        path = out_dir / f"matches_{year}.csv"
        key = ["year", "round_num", "team_1_team_name", "team_2_team_name"]
        if path.exists():
            try:
                df_existing = pd.read_csv(path)
                for c in cols:
                    if c not in df_existing.columns:
                        df_existing[c] = ""
                df_existing = df_existing[[c for c in cols if c in df_existing.columns]]
                if "round_num" in df_existing.columns:
                    df_existing["round_num"] = df_existing["round_num"].astype(str)
                
                # Use smart merge instead of simple concatenation + drop_duplicates
                df = smart_merge_matches(df_existing, df_new, key)
                
                # Sort by date after smart merge
                if "date" in df.columns:
                    df = df.sort_values(by="date", na_position="last")
            except Exception as e:
                logging.warning(f"Failed to merge existing data for {path}: {e}. Using new data only.")
                df = df_new
        else:
            df = df_new
        df.to_csv(path, index=False)
        print(f"  Wrote {path} ({len(df)} matches)")


def write_lineups_csv(lineup_rows: list[tuple], out_dir: Path) -> None:
    """Aggregate lineup rows by team and write team_lineups_<team>.csv. Merges with existing file if present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lineup_key_cols = ["year", "round_num", "team_name"]
    # (year, date, round_num, team_name, players_list)
    by_team: dict[str, list[dict]] = {}
    for year, date, round_num, team_name, players_list in lineup_rows:
        key = _team_key(team_name)
        if key not in by_team:
            by_team[key] = []
        by_team[key].append({
            "year": year,
            "date": date,
            "round_num": str(round_num),
            "team_name": team_name,
            "players": ";".join(players_list),
        })
    for team_key, rows in by_team.items():
        df_new = pd.DataFrame(rows)
        df_new = df_new[["year", "date", "round_num", "team_name", "players"]]
        path = out_dir / f"team_lineups_{team_key}.csv"
        if path.exists():
            try:
                df_existing = pd.read_csv(path)
                df_existing["round_num"] = df_existing["round_num"].astype(str)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df = df_new
        else:
            df = df_new
        if all(c in df.columns for c in lineup_key_cols):
            df = df.drop_duplicates(subset=lineup_key_cols, keep="first")
        df.to_csv(path, index=False)
        print(f"  Wrote {path} ({len(df)} rows)")


def write_player_stats_csv(
    player_stat_rows: list[dict], out_dir: Path, fetch_dobs: bool = True,
) -> None:
    """Write per-player performance_details CSVs using canonical IDs (lastname_firstname_DDMMYYYY)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup_key = ["team", "year", "round"]
    cache = _load_dob_cache()
    by_player: dict[str, list[dict]] = {}
    for row in player_stat_rows:
        url_path = row.get("player_url_path", "")
        url_key = row.get("player_id_hint", "unknown")
        display_name = row.get("player_name", "")

        dob = cache.get(url_key, "")
        if not dob and fetch_dobs and url_path:
            dob = _fetch_player_dob(url_path, delay=0.3)

        pkey = _make_canonical_player_id(display_name, url_path, dob)
        if pkey not in by_player:
            by_player[pkey] = []
        clean_row = {c: row.get(c, "") for c in PLAYER_CSV_COLS}
        by_player[pkey].append(clean_row)

    if fetch_dobs:
        _save_dob_cache()

    for pkey, rows in by_player.items():
        df_new = pd.DataFrame(rows, columns=PLAYER_CSV_COLS)
        path = out_dir / f"{pkey}_performance_details.csv"
        if path.exists():
            try:
                df_existing = pd.read_csv(path, dtype=str)
                for c in PLAYER_CSV_COLS:
                    if c not in df_existing.columns:
                        df_existing[c] = ""
                df = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df = df_new
        else:
            df = df_new
        if all(c in df.columns for c in dedup_key):
            df["round"] = df["round"].astype(str)
            df["year"] = df["year"].astype(str)
            df = df.drop_duplicates(subset=dedup_key, keep="first")
        df.to_csv(path, index=False)
    if by_player:
        print(f"  Wrote player stats for {len(by_player)} players to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scrape AFL Tables into afl_data_afltables/data/. "
        "Run one year (or one round) at a time for small bites; output merges with existing files."
    )
    ap.add_argument("--year-from", type=int, default=2024)
    ap.add_argument("--year-to", type=int, default=2025)
    ap.add_argument(
        "--round",
        type=str,
        default=None,
        metavar="R",
        help="Only scrape this round (e.g. 1, 2, Grand Final). AFL Tables labels Opening Round as Round 1.",
    )
    ap.add_argument("--lineups", action="store_true", help="Fetch lineups from match stats pages")
    ap.add_argument("--player-stats", action="store_true", help="Fetch per-player stats from match stats pages")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay between match-stats requests (seconds)")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    # Configure logging to show merge decisions
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )

    matches_dir = OUTPUT_BASE / "matches"
    lineups_dir = OUTPUT_BASE / "lineups"
    players_dir = OUTPUT_BASE / "players"
    matches_by_year: dict[int, list[dict]] = {}
    all_lineup_rows: list[tuple] = []
    all_player_stat_rows: list[dict] = []

    for year in range(args.year_from, args.year_to + 1):
        label = f"{year}" if not args.round else f"{year} round {args.round}"
        print(f"Scraping {label}...")
        matches, lineup_rows, player_stat_rows = scrape_season(
            year,
            fetch_lineups=args.lineups,
            fetch_player_stats=args.player_stats,
            delay=args.delay,
            timeout=args.timeout,
            round_filter=args.round,
        )
        for m in matches:
            m.pop("match_stats_url", None)
        matches_by_year[year] = matches
        all_lineup_rows.extend(lineup_rows)
        all_player_stat_rows.extend(player_stat_rows)

    write_matches_csv(matches_by_year, matches_dir)
    if all_lineup_rows:
        write_lineups_csv(all_lineup_rows, lineups_dir)
    if all_player_stat_rows:
        write_player_stats_csv(all_player_stat_rows, players_dir)
    print(f"Done. Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
