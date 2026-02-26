#!/usr/bin/env python3
"""
RLP Calendar Discovery — build competition_slugs_by_year.json.

Iterates ANZ years (1908–2026, single year) and Europe years (1980-81–2025, split-year).
Fetches /calendar/{year}/comps.html, extracts competition links (/competitions/ID),
follows redirects to /seasons/{slug}-{year}/ to resolve slugs.

Output: competition_slugs_by_year.json — maps year (str) → list of slugs.

Usage:
    python scripts/rlp_calendar_discovery.py [--output PATH] [--year-from Y] [--year-to Y]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.rugbyleagueproject.org"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
DELAY = 0.6  # Be polite to RLP

# NRL lineage slugs (from nrl_competition_history) — for --nrl-only fast probe
NRL_LINEAGE_SLUGS = ["nswrfl", "nswrl", "arl", "super-league", "nrl"]


def fetch(url: str, follow_redirects: bool = False) -> requests.Response | None:
    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
            allow_redirects=follow_redirects,
        )
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"  Error {url}: {e}")
        return None


def get_anz_years(year_from: int, year_to: int) -> list[tuple[str, str]]:
    """(year_key, url_suffix) for ANZ single-year format."""
    out = []
    for y in range(year_from, year_to + 1):
        out.append((str(y), f"{y}"))
    return out


def get_europe_years(year_from: int, year_to: int) -> list[tuple[str, str]]:
    """(year_key, url_suffix) for Europe split-year format (e.g. 1980-81)."""
    out = []
    for y in range(year_from, year_to + 1):
        next_y = (y % 100) + 1
        suffix = f"{y}-{next_y:02d}"
        out.append((suffix, suffix))
    return out


def slug_from_competition_url(comp_url: str) -> str | None:
    """
    Follow competition URL redirect to /seasons/{slug}-{year}/ and extract slug.
    """
    full = comp_url if comp_url.startswith("http") else BASE + comp_url
    r = fetch(full, follow_redirects=True)
    if not r:
        return None
    # Final URL e.g. https://.../seasons/nrl-2026/summary.html
    m = re.search(r"/seasons/([a-z0-9\-]+)-\d", r.url)
    if m:
        return m.group(1)
    return None


def discover_slugs_for_year(url_suffix: str, year_key: str) -> set[str]:
    """
    Fetch calendar comps page and extract slugs from competition links.
    """
    url = f"{BASE}/calendar/{url_suffix}/comps.html"
    r = fetch(url)
    if not r:
        return set()

    soup = BeautifulSoup(r.text, "html.parser")
    comp_ids = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        m = re.match(r"/competitions/(\d+)(?:\?|$|/)", href)
        if m:
            comp_ids.add(href)

    slugs = set()
    for href in comp_ids:
        slug = slug_from_competition_url(href)
        if slug:
            slugs.add(slug)
        time.sleep(DELAY)

    return slugs


def probe_season_exists(slug: str, year: int) -> bool:
    """Check if /seasons/{slug}-{year}/round-1/summary.html exists."""
    url = f"{BASE}/seasons/{slug}-{year}/round-1/summary.html"
    r = fetch(url)
    if not r:
        return False
    if "404" in r.text or "not found" in r.text.lower():
        return False
    return True


def discover_nrl_lineage_only(year_from: int, year_to: int) -> dict[str, list[str]]:
    """Fast probe: only check NRL lineage slugs per year."""
    result: dict[str, list[str]] = {}
    for year in range(year_from, year_to + 1):
        slugs = []
        for slug in NRL_LINEAGE_SLUGS:
            if probe_season_exists(slug, year):
                slugs.append(slug)
            time.sleep(DELAY * 0.5)  # Shorter delay for probe
        if slugs:
            result[str(year)] = slugs
    return result


def main():
    ap = argparse.ArgumentParser(description="RLP calendar discovery")
    ap.add_argument("--output", "-o", type=Path, default=None, help="Output JSON path")
    ap.add_argument("--year-from", type=int, default=1908, help="ANZ start year")
    ap.add_argument("--year-to", type=int, default=2026, help="ANZ end year")
    ap.add_argument("--europe-from", type=int, default=1980, help="Europe start year")
    ap.add_argument("--europe-to", type=int, default=2025, help="Europe end year")
    ap.add_argument("--anz-only", action="store_true", help="Only ANZ, skip Europe")
    ap.add_argument("--nrl-only", action="store_true", help="Fast: only probe NRL lineage slugs")
    args = ap.parse_args()

    out_path = args.output or Path(__file__).resolve().parent.parent / "nrl_data" / "data" / "competition_slugs_by_year.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, list[str]] = {}

    if args.nrl_only:
        print("Probing NRL lineage slugs only...")
        result = discover_nrl_lineage_only(args.year_from, args.year_to)
    else:
        # ANZ single-year
        print("Discovering ANZ competitions...")
        for year_key, url_suffix in get_anz_years(args.year_from, args.year_to):
            slugs = discover_slugs_for_year(url_suffix, year_key)
            if slugs:
                result[year_key] = sorted(slugs)
                print(f"  {year_key}: {len(slugs)} slugs")
            time.sleep(DELAY)

        # Europe split-year
        if not args.anz_only:
            print("Discovering Europe competitions...")
            for year_key, url_suffix in get_europe_years(args.europe_from, args.europe_to):
                if year_key not in result:
                    result[year_key] = []
                slugs = discover_slugs_for_year(url_suffix, year_key)
                if slugs:
                    existing = set(result[year_key])
                    existing.update(slugs)
                    result[year_key] = sorted(existing)
                    print(f"  {year_key}: +{len(slugs)} slugs")
                time.sleep(DELAY)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
