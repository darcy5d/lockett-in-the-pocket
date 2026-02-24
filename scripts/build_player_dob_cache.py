#!/usr/bin/env python3
"""
Scrape DOB for every player from AFL Tables player pages.
Writes afl_data/data/player_dob_cache.json mapping file stem -> DOB (DDMMYYYY).

Usage:
  python scripts/build_player_dob_cache.py [--delay 0.5]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYERS_DIR = _PROJECT_ROOT / "afl_data" / "data" / "players"
CACHE_PATH = _PROJECT_ROOT / "afl_data" / "data" / "player_dob_cache.json"
BASE_URL = "https://afltables.com/afl/stats/players"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

MON_TO_NUM = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def _afl_tables_titlecase(part: str) -> str:
    """Title-case a name part, handling Mc/Mac/O/D prefixes as AFL Tables does."""
    t = part.title()
    # McX -> McGrath, McCreery, etc.
    if t.startswith("Mc") and len(t) > 2:
        t = "Mc" + t[2].upper() + t[3:]
    # MacX -> MacGregor, etc. (but not Mackin, Mackay which stay as-is)
    # AFL Tables is inconsistent here; try both
    # OX -> O'Brien stored as OBrien on AFL Tables
    if t.startswith("O") and len(t) > 1 and t[1].islower():
        # Try OBrien format (no apostrophe, capital after O)
        pass  # Title() already does this: obrien -> Obrien, which is wrong
    if part.startswith("o") and len(part) > 1:
        t = "O" + part[1:].title()
        if len(t) > 1:
            t = "O" + t[1].upper() + t[2:]
    if part.startswith("d") and len(part) > 1 and part[1] != "e":
        t = "D" + part[1:].title()
        if len(t) > 1:
            t = "D" + t[1].upper() + t[2:]
    return t


def stem_to_url(stem: str) -> str:
    """Convert file stem (e.g. 'aaron_black1') to AFL Tables URL."""
    parts = stem.split("_")
    titled = [_afl_tables_titlecase(p) for p in parts]
    name = "_".join(titled)
    first_letter = titled[0][0]
    return f"{BASE_URL}/{first_letter}/{name}.html"


def fetch_dob(url: str, timeout: int = 15) -> str | None:
    """Fetch player page and extract DOB as DDMMYYYY string."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        m = re.search(r"Born:</b>\s*(\d{1,2})-([A-Za-z]+)-(\d{4})", resp.text)
        if not m:
            m = re.search(r"Born:\s*(\d{1,2})-([A-Za-z]+)-(\d{4})", resp.text)
        if not m:
            return None
        day, mon_str, year = m.group(1), m.group(2), m.group(3)
        mon = MON_TO_NUM.get(mon_str[:3], "00")
        return f"{int(day):02d}{mon}{year}"
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build player DOB cache from AFL Tables")
    ap.add_argument("--delay", type=float, default=0.3, help="Delay between requests (seconds)")
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    # Load existing cache
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
        print(f"Loaded existing cache: {len(cache)} entries")

    # Collect all player file stems
    files = sorted(glob.glob(str(PLAYERS_DIR / "*_performance_details.csv")))
    stems = [Path(f).stem.replace("_performance_details", "") for f in files]
    print(f"Player files: {len(stems)}")

    # Filter to those not already cached
    to_fetch = [s for s in stems if s not in cache]
    print(f"Need to fetch: {len(to_fetch)}")

    fetched = 0
    failed = 0
    for i, stem in enumerate(to_fetch):
        url = stem_to_url(stem)
        dob = fetch_dob(url, timeout=args.timeout)
        if dob:
            cache[stem] = dob
            fetched += 1
        else:
            cache[stem] = ""
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(to_fetch)}: fetched={fetched}, failed={failed}")
            CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))

        time.sleep(args.delay)

    # Final save
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
    print(f"\nDone. Cache: {len(cache)} entries ({fetched} new, {failed} failed)")
    print(f"Saved to {CACHE_PATH}")

    empty = sum(1 for v in cache.values() if not v)
    print(f"Players with DOB: {len(cache) - empty}, without: {empty}")


if __name__ == "__main__":
    main()
