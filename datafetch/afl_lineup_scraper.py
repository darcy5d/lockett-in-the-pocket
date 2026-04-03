#!/usr/bin/env python3
"""
Scrape AFL team lineups from afl.com.au/matches/team-lineups.

The page is fully JS-rendered (React), so we use Playwright (headless Chromium)
to load the page, expand lineups, and parse player/position/team data from the
DOM's aria labels (e.g. "Brodie Grundy. ruck. Sydney Swans.").

Output:
  - afl_data/data/lineups/team_lineups_<team>.csv  (appends 2026 row per team)
  - afl_data/data/lineups/lineup_positions_2026.csv (round, team, player, position)
  - afl_data/data/lineups/lineup_meta.json          (last_updated timestamp)

Usage:
  python datafetch/afl_lineup_scraper.py [--round OR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.mappings import TEAM_EXTERNAL_TO_INTERNAL

LINEUPS_URL = "https://www.afl.com.au/matches/team-lineups"
LINEUP_DIR = _PROJECT_ROOT / "afl_data" / "data" / "lineups"
MATCH_DIR = _PROJECT_ROOT / "afl_data" / "data" / "matches"
META_PATH = LINEUP_DIR / "lineup_meta.json"

TEAM_DISPLAY_TO_FILE_KEY: dict[str, str] = {
    "Adelaide Crows": "adelaide",
    "Brisbane Lions": "brisbane_lions",
    "Carlton": "carlton",
    "Collingwood": "collingwood",
    "Essendon": "essendon",
    "Fremantle": "fremantle",
    "Geelong Cats": "geelong",
    "Gold Coast SUNS": "gold_coast",
    "GWS GIANTS": "greater_western_sydney",
    "Hawthorn": "hawthorn",
    "Melbourne": "melbourne",
    "North Melbourne": "north_melbourne",
    "Port Adelaide": "port_adelaide",
    "Richmond": "richmond",
    "St Kilda": "st_kilda",
    "Sydney Swans": "sydney",
    "West Coast Eagles": "west_coast",
    "Western Bulldogs": "western_bulldogs",
}

EMERGENCY_POSITIONS = {"undefined"}

ROUND_LABEL_MAP = {
    "Opening Round": "Opening Round",
    "OR": "Opening Round",
}


def _team_display_to_internal(display: str) -> str:
    """Map AFL.com.au display name to internal match CSV name."""
    return TEAM_EXTERNAL_TO_INTERNAL.get(display.strip(), display.strip())


def _team_display_to_file_key(display: str) -> str:
    """Map AFL.com.au display name to lineup CSV filename key."""
    return TEAM_DISPLAY_TO_FILE_KEY.get(display.strip(), display.strip().lower().replace(" ", "_"))


def _parse_player_label(label: str) -> tuple[str, str, str] | None:
    """
    Parse DOM label like "Brodie Grundy. ruck. Sydney Swans."
    Returns (player_name, position, team_display) or None.
    """
    parts = label.split(".")
    if len(parts) < 3:
        return None
    player_name = parts[0].strip()
    position = parts[1].strip()
    team_display = parts[2].strip()
    if not player_name or not team_display:
        return None
    return (player_name, position, team_display)


def _detect_round(page) -> str | None:
    """Detect which round is currently displayed from the round selector buttons."""
    try:
        active = page.locator('[class*="active"][class*="round"], [aria-pressed="true"]').first
        if active.count():
            return active.inner_text().strip()
    except Exception:
        pass
    return None


def scrape_lineups(
    round_filter: str | None = None,
    year: int = 2026,
    timeout_ms: int = 30000,
) -> dict:
    """
    Scrape AFL.com.au team lineups using Playwright.

    Parameters
    ----------
    round_filter : str, optional
        Round to scrape (e.g. "OR", "1", "Opening Round"). If None, scrapes
        whichever round the page defaults to (typically the current/next round).
    year : int
        Season year (for CSV output).
    timeout_ms : int
        Page load timeout in milliseconds.

    Returns
    -------
    dict with keys: matches (list of dicts), round_num, teams_updated, error
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"error": "playwright not installed. Run: pip install playwright && playwright install chromium"}

    matches: dict[str, dict[str, list]] = {}
    round_num = round_filter or ""
    match_headers: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        print(f"Navigating to {LINEUPS_URL} ...")
        page.goto(LINEUPS_URL, wait_until="networkidle", timeout=timeout_ms)
        page.wait_for_timeout(2000)

        if round_filter:
            mapped = ROUND_LABEL_MAP.get(round_filter, round_filter)
            btn_text = "OR" if mapped == "Opening Round" else mapped
            try:
                round_btn = page.locator(f'button:text-is("{btn_text}")').first
                if round_btn.count():
                    round_btn.click()
                    page.wait_for_timeout(2000)
                    print(f"Selected round: {btn_text}")
            except Exception as e:
                print(f"Warning: could not select round '{btn_text}': {e}")

        expand_btn = page.locator('text="Expand All Lineups"').first
        if expand_btn.count():
            expand_btn.click()
            page.wait_for_timeout(3000)
            print("Clicked 'Expand All Lineups'")

        match_links = page.locator('a[href*="/matches/"]').all()
        for link in match_links:
            text = link.inner_text().strip()
            if " v " in text:
                parts = text.split(" v ", 1)
                if len(parts) == 2:
                    match_headers.append({
                        "home": parts[0].strip().split("\n")[0].strip(),
                        "away": parts[1].strip().split("\n")[0].strip(),
                    })

        player_elements = page.locator('[class*="lineup"] [aria-label], [data-testid*="lineup"] [aria-label]').all()
        if not player_elements:
            player_elements = page.locator('[aria-label]').all()

        parsed_count = 0
        for el in player_elements:
            try:
                label = el.get_attribute("aria-label") or ""
            except Exception:
                continue
            if not label:
                name_attr = ""
                try:
                    name_attr = el.get_attribute("name") or ""
                except Exception:
                    pass
                if not name_attr:
                    continue
                label = name_attr

            parsed = _parse_player_label(label)
            if not parsed:
                continue
            player_name, position, team_display = parsed

            if team_display not in matches:
                matches[team_display] = {"players": [], "emergencies": []}

            entry = {"name": player_name, "position": position}
            if position in EMERGENCY_POSITIONS:
                matches[team_display]["emergencies"].append(entry)
            else:
                matches[team_display]["players"].append(entry)
            parsed_count += 1

        if parsed_count == 0:
            all_generics = page.locator('[role="generic"]').all()
            for el in all_generics:
                try:
                    name_text = el.get_attribute("name") or el.inner_text().strip()
                except Exception:
                    continue
                if not name_text:
                    continue
                parsed = _parse_player_label(name_text)
                if parsed:
                    player_name, position, team_display = parsed
                    if team_display not in matches:
                        matches[team_display] = {"players": [], "emergencies": []}
                    entry = {"name": player_name, "position": position}
                    if position in EMERGENCY_POSITIONS:
                        matches[team_display]["emergencies"].append(entry)
                    else:
                        matches[team_display]["players"].append(entry)
                    parsed_count += 1

        round_el = page.locator('h2:has-text("Thursday"), h2:has-text("Friday"), h2:has-text("Saturday"), h2:has-text("Sunday"), h2:has-text("Monday")').first
        last_updated_el = page.locator('text=/Last updated/').first
        last_updated = ""
        if last_updated_el.count():
            try:
                last_updated = last_updated_el.inner_text().strip()
            except Exception:
                pass

        browser.close()

    if not matches:
        return {"error": "No lineup data found on page", "matches": [], "round_num": round_num, "teams_updated": []}

    if not round_num:
        if round_filter:
            round_num = ROUND_LABEL_MAP.get(round_filter, round_filter)
        else:
            round_num = "Opening Round"

    round_num = ROUND_LABEL_MAP.get(round_num, round_num)

    teams_updated = _write_lineup_csvs(matches, round_num, year)
    _write_positions_csv(matches, round_num, year)
    _write_meta(round_num, last_updated, len(matches))

    print(f"\nScraped {parsed_count} players across {len(matches)} teams for round '{round_num}'")
    for team, data in sorted(matches.items()):
        print(f"  {team}: {len(data['players'])} players + {len(data['emergencies'])} emergencies")

    return {
        "matches": [
            {
                "team": team,
                "team_internal": _team_display_to_internal(team),
                "players": len(data["players"]),
                "emergencies": len(data["emergencies"]),
            }
            for team, data in matches.items()
        ],
        "round_num": round_num,
        "teams_updated": teams_updated,
        "last_updated": last_updated,
        "error": None,
    }


def _write_lineup_csvs(
    matches: dict[str, dict[str, list]],
    round_num: str,
    year: int,
) -> list[str]:
    """Append/update lineup rows in per-team CSV files. Returns list of file keys updated."""
    LINEUP_DIR.mkdir(parents=True, exist_ok=True)
    updated = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    for team_display, data in matches.items():
        players = data["players"]
        if not players:
            continue

        file_key = _team_display_to_file_key(team_display)
        team_internal = _team_display_to_internal(team_display)
        csv_path = LINEUP_DIR / f"team_lineups_{file_key}.csv"

        player_names = [p["name"] for p in players]
        players_str = ";".join(player_names)

        new_row = pd.DataFrame([{
            "year": year,
            "date": now_str,
            "round_num": round_num,
            "team_name": team_internal,
            "players": players_str,
        }])

        if csv_path.exists():
            existing = pd.read_csv(csv_path, dtype=str)
            mask = (
                (existing["year"].astype(str) == str(year))
                & (existing["round_num"].astype(str).str.strip() == round_num)
            )
            existing = existing[~mask]
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row

        combined.to_csv(csv_path, index=False)
        updated.append(file_key)
        print(f"  Updated {csv_path.name}: {len(player_names)} players for {round_num}")

    return updated


def _write_positions_csv(
    matches: dict[str, dict[str, list]],
    round_num: str,
    year: int,
) -> None:
    """Write/update detailed positions CSV for future analysis."""
    pos_path = LINEUP_DIR / "lineup_positions_2026.csv"
    rows = []
    for team_display, data in matches.items():
        team_internal = _team_display_to_internal(team_display)
        for p in data["players"]:
            rows.append({
                "year": year,
                "round_num": round_num,
                "team": team_internal,
                "player_name": p["name"],
                "position": p["position"],
                "is_emergency": False,
            })
        for p in data["emergencies"]:
            rows.append({
                "year": year,
                "round_num": round_num,
                "team": team_internal,
                "player_name": p["name"],
                "position": "emergency",
                "is_emergency": True,
            })

    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if pos_path.exists():
        existing = pd.read_csv(pos_path, dtype=str)
        mask = (
            (existing["year"].astype(str) == str(year))
            & (existing["round_num"].astype(str).str.strip() == round_num)
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(pos_path, index=False)
    print(f"  Positions saved to {pos_path.name}: {len(rows)} entries")


def _write_meta(round_num: str, last_updated_text: str, team_count: int) -> None:
    """Write metadata JSON for freshness tracking."""
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "last_fetched": datetime.now(timezone.utc).isoformat(),
        "round_num": round_num,
        "teams_with_lineups": team_count,
        "afl_last_updated": last_updated_text,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def get_lineup_meta() -> dict | None:
    """Read lineup metadata (for server freshness display)."""
    if not META_PATH.exists():
        return None
    try:
        with open(META_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Scrape AFL team lineups from afl.com.au")
    ap.add_argument("--round", default=None, help="Round to scrape (e.g. OR, 1, 2). Default: current round.")
    ap.add_argument("--year", type=int, default=2026, help="Season year")
    ap.add_argument("--timeout", type=int, default=30000, help="Page load timeout (ms)")
    args = ap.parse_args()

    result = scrape_lineups(
        round_filter=args.round,
        year=args.year,
        timeout_ms=args.timeout,
    )

    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print(f"\nDone. Updated {len(result['teams_updated'])} team lineup files.")


if __name__ == "__main__":
    main()
