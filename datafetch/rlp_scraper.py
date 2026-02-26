#!/usr/bin/env python3
"""
RLP Match + Lineup Scraper for NRL lineage competitions.

Scrapes Rugby League Project round summaries:
  /seasons/{slug}-{year}/round-{n}/summary.html

Output:
  - nrl_data/data/matches/matches_{slug}_{year}.csv (or unified with competition col)
  - nrl_data/data/lineups/lineup_details.csv (match_id, team, position, player_id, tries, goals, fg)

NRL lineage slugs: nswrfl, nswrl, arl, super-league (1997 Aus), nrl

Usage:
  python datafetch/rlp_scraper.py --slug nrl --year-from 2024 --year-to 2025
  python datafetch/rlp_scraper.py --slug nrl --year-from 2025 --year-to 2025 --round 1
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = _PROJECT_ROOT / "nrl_data" / "data"
BASE_URL = "https://www.rugbyleagueproject.org"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
DELAY = 0.5

NRL_LINEAGE_SLUGS = ["nswrfl", "nswrl", "arl", "super-league", "nrl"]


def fetch(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  Error {url}: {e}")
        return None


def _extract_player_id(href: str) -> str | None:
    m = re.search(r"/players/(\d+)", href)
    return m.group(1) if m else None


def _parse_scorers(text: str) -> tuple[int, int, int]:
    """Parse scorer text for tries, goals, fg. Returns (tries, goals, fg)."""
    tries = goals = fg = 0
    parts = re.split(r"\s*;\s*", text)
    try_part = parts[0] if parts else ""
    goal_part = parts[1] if len(parts) > 1 else ""
    tries = sum(int(m.group(1)) for m in re.finditer(r"(\d+)\s*,", try_part))
    if re.search(r"[,\s][A-Za-z.']+\s+tries?(?:\s|;|$)", try_part):
        tries += 1
    if tries == 0 and "tries" in try_part.lower():
        tries = try_part.count(",") + 1
    goals = sum(int(m.group(1)) for m in re.finditer(r"(\d+)\s*goals?", goal_part, re.I))
    if re.search(r"[,\s][A-Za-z.']+\s+goals?(?:\s|;|$)", goal_part, re.I):
        goals += 1
    if goals == 0 and "goal" in goal_part.lower():
        goals = goal_part.count(",") + 1
    if "field goal" in text.lower() or re.search(r"\bfg\b", text, re.I):
        fg = 1
    return (tries, goals, fg)


def _parse_match_result_line(soup_tag, year: int) -> dict | None:
    """
    Parse a result line like:
    [Canberra] 30 ([scorers]) defeated [Warriors] 8 ([scorers]) at [Venue]. Date: ...
    """
    text = soup_tag.get_text()
    if "defeated" not in text and "drew" not in text.lower():
        return None

    # Team names from links: /seasons/.../team/summary.html or ../team/summary.html (not round)
    teams = []
    for a in soup_tag.find_all("a", href=True):
        href = a.get("href", "")
        if "/round-" in href:
            continue
        if "/round-" in href:
            continue
        if re.search(r"(?:\.\./|/seasons/[^/]+/)([a-z0-9\-]+)/summary\.html", href):
            name = a.get_text(strip=True)
            if name and name not in ("Data", "Venues", "Rounds", "Results", "Referees", "Players", "Coaches", "Summary") and not name.startswith(">"):
                teams.append(name)
    if len(teams) < 2:
        return None

    team_1 = teams[0]
    team_2 = teams[1]

    # Scores: "Team1 30 (scorers) defeated Team2 8 (scorers)" or "Team1 (R) 30 ... defeated Team2 (R) 22 ..."
    m1 = re.search(r"(\d+)\s*\(", text)  # First score before (
    m2 = re.search(r"defeated\s+.*?(\d+)\s*\(", text, re.DOTALL)  # Second score (.*? skips team name incl. "(R)")
    if not m1 or not m2:
        return None
    score_1 = int(m1.group(1))
    score_2 = int(m2.group(1))

    # Scorers in parentheses (first two paren groups after scores)
    t1_scorers = t2_scorers = ""
    parens = re.findall(r"\(([^)]+)\)", text)
    if len(parens) >= 2:
        t1_scorers = parens[0]
        t2_scorers = parens[1]
    t1_tries, t1_goals, t1_fg = _parse_scorers(t1_scorers)
    t2_tries, t2_goals, t2_fg = _parse_scorers(t2_scorers)

    # Venue: link with /matches/ in href, skip ">" or referee links
    venue = ""
    for a in soup_tag.find_all("a", href=True):
        if "/matches/" in a.get("href", ""):
            v = a.get_text(strip=True)
            if v and v != ">" and len(v) > 2:
                venue = v
                break

    # Date, Halftime, Penalties, Referee, Crowd
    date_match = re.search(r"Date:\s*([^.]+?)(?:\.|Kickoff|$)", text)
    date_str = date_match.group(1).strip() if date_match else ""
    ht_match = re.search(r"Halftime:\s*([^.]+?)(?:\.|Penalties|$)", text)
    halftime = ht_match.group(1).strip() if ht_match else ""
    pen_match = re.search(r"Penalties:\s*([^.]+?)(?:\.|Referee|$)", text)
    penalties = pen_match.group(1).strip() if pen_match else ""
    ref_match = re.search(r"Referee:\s*\[([^\]]+)\]", text)
    referee = ref_match.group(1).strip() if ref_match else ""
    crowd_match = re.search(r"Crowd:\s*([\d,]+)", text)
    attendance = crowd_match.group(1).replace(",", "") if crowd_match else ""

    return {
        "team_1_team_name": team_1,
        "team_1_score": score_1,
        "team_1_tries": t1_tries,
        "team_1_goals": t1_goals,
        "team_1_fg": t1_fg,
        "team_2_team_name": team_2,
        "team_2_score": score_2,
        "team_2_tries": t2_tries,
        "team_2_goals": t2_goals,
        "team_2_fg": t2_fg,
        "venue": venue,
        "date": date_str,
        "halftime": halftime,
        "penalties": penalties,
        "referee": referee,
        "attendance": attendance,
    }


def _parse_lineup_line(soup_tag) -> tuple[str, list[dict]]:
    """
    Parse lineup: [Team]: [Player](id), [Player](id), ...
    Team link: /seasons/{slug}-{year}/{team}/summary.html
    Player links: /players/{id}
    Returns (team_name, [{player_id, player_name}])
    """
    team_name = ""
    players = []
    for a in soup_tag.find_all("a", href=True):
        href = a.get("href", "")
        pid = _extract_player_id(href)
        if pid:
            name = a.get_text(strip=True)
            if name:
                players.append({"player_id": pid, "player_name": name})
        elif re.search(r"(?:\.\./|/seasons/[^/]+/)([a-z0-9\-]+)/summary\.html", href) and "/round-" not in href:
            team_name = a.get_text(strip=True)
    return (team_name, players)


def _to_split_year(year: int) -> str:
    """Convert year to RLP split-year format (e.g. 1996 -> 1996-97)."""
    next_yy = (year % 100) + 1
    return f"{year}-{next_yy:02d}"


def scrape_round(
    slug: str,
    year: int,
    round_num: int | str,
    year_suffix: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Scrape one round summary. Returns (matches, lineup_entries).
    year_suffix: for UK split-year (e.g. "1996-97"). If None, use single year.
    """
    round_str = str(round_num).lower().replace(" ", "-")
    url_year = year_suffix if year_suffix else str(year)
    url = f"{BASE_URL}/seasons/{slug}-{url_year}/round-{round_str}/summary.html"
    html = fetch(url)
    if not html:
        return [], []

    soup = BeautifulSoup(html, "html.parser")
    matches = []
    lineup_entries = []

    # Collect match result divs (exclude huge page wrapper: text < 1200 chars)
    blocks = []
    for elem in soup.find_all("div"):
        text = elem.get_text()
        if ("defeated" in text or "drew" in text.lower()) and len(text) < 1200:
            m = _parse_match_result_line(elem, year)
            if m:
                blocks.append(("result", elem))
        else:
            team_name, players = _parse_lineup_line(elem)
            if team_name and len(players) >= 10 and len(text) < 800:
                blocks.append(("lineup", elem))

    match_idx = 0
    for btype, elem in blocks:
        if btype == "result":
            m = _parse_match_result_line(elem, year)
            if m:
                m["competition"] = slug
                m["year"] = year
                m["round_num"] = str(round_num)
                matches.append(m)
                match_idx = len(matches) - 1
        else:
            team_name, players = _parse_lineup_line(elem)
            if team_name and players and matches:
                for p in players:
                    lineup_entries.append({
                        "match_idx": match_idx,
                        "competition": slug,
                        "year": year,
                        "round_num": str(round_num),
                        "team": team_name,
                        "player_id": p["player_id"],
                        "player_name": p["player_name"],
                    })

    return matches, lineup_entries


def discover_rounds(slug: str, year: int, year_suffix: str | None = None) -> list[str | int]:
    """Fetch round list from season round-1 page (has links to all rounds)."""
    url_year = year_suffix if year_suffix else str(year)
    url = f"{BASE_URL}/seasons/{slug}-{url_year}/round-1/summary.html"
    html = fetch(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    rounds = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        m = re.search(r"round-([a-z0-9\-]+)/summary\.html", href)
        if m:
            rn = m.group(1)
            if rn.isdigit():
                rounds.append(int(rn))
            else:
                rounds.append(rn)
    return sorted(set(rounds), key=lambda x: (0 if isinstance(x, int) else 1, str(x)))


def run_scrape(
    slug: str,
    year_from: int,
    year_to: int,
    round_num: int | None = None,
    output_dir: Path | None = None,
    output_slug: str | None = None,
    year_format: str = "single",
) -> None:
    """
    output_slug: when set, write to matches_{output_slug}_*.csv (for super-league-au vs super-league-uk).
    year_format: "single" or "split". When split, use year_suffix for UK URLs (1996 -> 1996-97).
    """
    out_slug = output_slug or slug
    output_dir = output_dir or OUTPUT_BASE
    match_dir = output_dir / "matches"
    lineup_dir = output_dir / "lineups"
    match_dir.mkdir(parents=True, exist_ok=True)
    lineup_dir.mkdir(parents=True, exist_ok=True)

    all_matches = []
    all_lineups = []

    for year in range(year_from, year_to + 1):
        year_suffix = _to_split_year(year) if year_format == "split" else None
        rounds = [round_num] if round_num is not None else discover_rounds(slug, year, year_suffix)
        if not rounds:
            url_part = f"{slug}-{year_suffix or year}"
            print(f"  {url_part}: no rounds found")
            continue

        for rn in rounds:
            matches, lineups = scrape_round(slug, year, rn, year_suffix)
            base_idx = len(all_matches)
            for i, m in enumerate(matches):
                m["match_id"] = f"{out_slug}_{year}_r{rn}_m{i}"
                all_matches.append(m)
            for le in lineups:
                idx = base_idx + le["match_idx"]
                if idx < len(all_matches):
                    le["match_id"] = all_matches[idx]["match_id"]
                all_lineups.append(le)
            url_part = f"{slug}-{year_suffix or year}"
            print(f"  {url_part} round {rn}: {len(matches)} matches")
            time.sleep(DELAY)

    if all_matches:
        df = pd.DataFrame(all_matches)
        out_path = match_dir / f"matches_{out_slug}_{year_from}_{year_to}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} matches)")

    if all_lineups:
        df = pd.DataFrame(all_lineups)
        out_path = lineup_dir / f"lineup_details_{out_slug}_{year_from}_{year_to}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} lineup entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", type=str, default="nrl", help="Competition slug")
    ap.add_argument("--year-from", type=int, default=2024)
    ap.add_argument("--year-to", type=int, default=2025)
    ap.add_argument("--round", type=int, default=None, help="Single round to scrape")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    run_scrape(
        slug=args.slug,
        year_from=args.year_from,
        year_to=args.year_to,
        round_num=args.round,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
