#!/usr/bin/env python3
"""
Build RLP player DOB cache for mapping League Unlimited names to RLP player_ids.

For each unique player_id in lineup_details CSV files, fetches RLP player summary page,
extracts DOB ("Born Friday, 14th November, 1997" -> DDMMYYYY) and name.
Builds reverse index (normalised_name, dob) -> player_id for lineup scraper.

Usage:
  python scripts/build_rlp_player_dob_cache.py [--competition nrl] [--delay 0.5]
"""

from __future__ import annotations

import argparse
from typing import Callable
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = _PROJECT_ROOT / "nrl_data" / "data" / "rlp_player_dob_cache.json"
LINEUP_DIR = _PROJECT_ROOT / "nrl_data" / "data" / "lineups"
RLP_BASE = "https://www.rugbyleagueproject.org"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _fetch_player_dob(player_id: str, timeout: int = 15) -> tuple[str | None, str | None]:
    """Fetch RLP player page. Returns (name, dob) or (None, None)."""
    url = f"{RLP_BASE}/players/{player_id}/summary.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None, None
        text = r.text
        # Name from title: "Nathan Cleary - Playing Career - RLP"
        name_match = re.search(r"<title>([^\-]+)\s*\-", text)
        name = name_match.group(1).strip() if name_match else None
        # Fallback: og:title or h1
        if not name:
            og_match = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', text)
            if og_match:
                name = og_match.group(1).split(" - ")[0].strip()
        if not name:
            h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", text, re.I)
            if h1_match:
                name = h1_match.group(1).strip()
        # DOB: "Born Friday, 14th November, 1997"
        dob_match = re.search(
            r"Born\s+(?:[A-Za-z]+,\s*)?(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),\s*(\d{4})",
            text, re.I
        )
        if not dob_match:
            # Fallback: "14 November 1997"
            dob_match = re.search(r"(\d{1,2})[/\s]+([A-Za-z]+)[/\s]+(\d{4})", text, re.I)
        if not dob_match:
            # Fallback: ISO "1997-11-14"
            iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
            if iso_match:
                y, m, d = iso_match.groups()
                return name, f"{int(d):02d}{m}{y}"
        if dob_match:
            day, mon_str, year = dob_match.groups()[:3]
            mon = MONTH_MAP.get(mon_str.lower()[:3], "01")
            return name, f"{int(day):02d}{mon}{year}"
        return name, None
    except Exception:
        return None, None


def _normalise_name(name: str) -> list[str]:
    """Return variants for matching: full name, surname, First Last."""
    if not name or not name.strip():
        return []
    name = name.strip()
    parts = name.split()
    variants = [name]
    if len(parts) >= 2:
        variants.append(parts[-1])  # surname
        variants.append(f"{parts[-1]} {parts[0]}")  # Surname First
    return list(dict.fromkeys(variants))


def fix_empty_entries(
    cache_path: Path | None = None,
    delay: float = 0.5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Re-fetch cache entries with empty name. Returns count fixed."""
    cache_path = cache_path or CACHE_PATH
    if not cache_path.exists():
        return 0
    with open(cache_path) as fp:
        cache = json.load(fp)
    empty_ids = [pid for pid, d in cache.items() if not (d.get("name") or "").strip()]
    if not empty_ids:
        print("No empty entries to fix.")
        return 0
    print(f"Re-fetching {len(empty_ids)} entries with empty name...")
    fixed = 0
    for i, pid in enumerate(sorted(empty_ids)):
        name, dob = _fetch_player_dob(str(pid))
        if name or dob:
            cache[str(pid)] = {"name": name or "", "dob": dob or ""}
            fixed += 1
        if progress_callback:
            progress_callback(i + 1, len(empty_ids))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(empty_ids)}...")
        time.sleep(delay)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as fp:
        json.dump(cache, fp, indent=2, sort_keys=True)
    print(f"Fixed {fixed} entries.")
    return fixed


def build_cache(
    competition: str = "nrl",
    delay: float = 0.5,
    lineup_dir: Path | None = None,
    cache_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    fix_empty: bool = False,
    rebuild: bool = False,
) -> dict:
    lineup_dir = lineup_dir or LINEUP_DIR
    cache_path = cache_path or CACHE_PATH

    # Slug patterns for competition (from competition_config)
    slugs = {
        "nrl": ["nrl", "nswrfl", "nswrl", "arl"],
        "super-league-uk": ["super-league-uk"],
        "nsw-cup": ["nsw-cup"],
        "qld-cup": ["qld-cup"],
        "championship-uk": ["championship-uk"],
    }.get(competition, [competition] if competition else ["nrl"])
    slug_set = set(slugs)

    # Collect unique player_ids from lineup files
    player_ids: set[str] = set()
    for f in lineup_dir.glob("lineup_details_*.csv"):
        rest = f.stem.replace("lineup_details_", "")
        parts = rest.split("_")
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            slug = "_".join(parts[:-2])
            if slug not in slug_set:
                continue
        try:
            df = pd.read_csv(f, usecols=["player_id"])
            player_ids.update(df["player_id"].astype(str).dropna().unique())
        except Exception:
            pass

    player_ids.discard("")
    player_ids.discard("unknown")

    # Load existing cache
    cache: dict = {}
    if cache_path.exists():
        try:
            with open(cache_path) as fp:
                cache = json.load(fp)
        except Exception:
            pass

    # Rebuild: remove entries for players we're about to re-fetch (preserves other competitions)
    if rebuild:
        for pid in player_ids:
            cache.pop(str(pid), None)

    # Optionally fix entries with empty name
    if fix_empty:
        fix_empty_entries(cache_path=cache_path, delay=delay, progress_callback=progress_callback)
        with open(cache_path) as fp:
            cache = json.load(fp)

    # Fetch DOB for missing players
    missing = [pid for pid in sorted(player_ids) if str(pid) not in cache]
    total = len(missing)
    print(f"Fetching DOB for {total} players (of {len(player_ids)} total)...")
    for i, pid in enumerate(missing):
        name, dob = _fetch_player_dob(str(pid))
        cache[str(pid)] = {"name": name or "", "dob": dob or ""}
        current = i + 1
        if progress_callback:
            progress_callback(current, total)
        if current % 50 == 0:
            print(f"  {current}/{total}...")
        time.sleep(delay)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as fp:
        json.dump(cache, fp, indent=2, sort_keys=True)
    print(f"Saved {len(cache)} entries to {cache_path}")

    return cache


def build_reverse_index(cache: dict) -> dict:
    """Build (normalised_name, dob) -> player_id for lookup."""
    reverse: dict[str, str] = {}
    for pid, data in cache.items():
        name = data.get("name", "")
        dob = data.get("dob", "")
        if not name:
            continue
        for variant in _normalise_name(name):
            key = (variant.lower(), dob)
            if key not in reverse:
                reverse[f"{variant.lower()}:::{dob}"] = pid
            # Also without DOB for name-only match when unique
            key_only = f"{variant.lower()}:::"
            if key_only not in reverse:
                reverse[key_only] = pid  # Last one wins for name-only
    return reverse


def main():
    ap = argparse.ArgumentParser(description="Build RLP player DOB cache")
    ap.add_argument("--competition", default="nrl", help="nrl or super-league-uk")
    ap.add_argument("--delay", type=float, default=0.5, help="Delay between requests")
    ap.add_argument("--fix-empty", action="store_true", help="Re-fetch entries with empty name")
    ap.add_argument("--rebuild", action="store_true", help="Full rebuild: re-fetch all players from lineup files")
    args = ap.parse_args()
    build_cache(
        competition=args.competition,
        delay=args.delay,
        fix_empty=args.fix_empty,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
