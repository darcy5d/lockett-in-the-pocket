#!/usr/bin/env python3
"""
Probe Rugby League Project (RLP) to determine data bounds for each competition.

Target competitions: NRL, Super League, NSW Cup, QLD Cup, WSL (Women's Super League), RFL Championship

URL patterns observed:
- Calendar: https://www.rugbyleagueproject.org/calendar/{year}/comps.html
- Season round: https://www.rugbyleagueproject.org/seasons/{comp}-{year}/round-1/summary.html
- Season data: https://www.rugbyleagueproject.org/seasons/{comp}-{year}/data.html

Competition slugs (from 2026 calendar):
- nrl, super-league, nsw-cup, qld-cup
- women-s-super-league (WSL)
- championship (RFL Championship)

Historical names may differ: NSWRL (pre-NRL), BRL (pre-QLD Cup), etc.
"""

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.rugbyleagueproject.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def fetch(url: str, delay: float = 0.5) -> str | None:
    """Fetch URL, return HTML or None on failure."""
    time.sleep(delay)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  Error: {e}")
        return None


def comp_slug_from_calendar(year: int) -> dict[str, str]:
    """Check calendar page for competition links; extract slugs."""
    url = f"{BASE}/calendar/{year}/comps.html"
    html = fetch(url)
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href=True)
    slugs = {}
    for a in links:
        href = a.get("href", "")
        m = re.search(r"/seasons/([a-z0-9\-]+)-" + str(year) + r"/", href)
        if m:
            slug = m.group(1)
            if slug not in slugs:
                slugs[slug] = href
    return slugs


def season_exists(comp_slug: str, year: int) -> bool:
    """Check if /seasons/{comp}-{year}/ exists (e.g. round-1 or summary)."""
    # Try round-1 first
    url = f"{BASE}/seasons/{comp_slug}-{year}/round-1/summary.html"
    html = fetch(url)
    if not html:
        return False
    # Check for 404 or "not found" in content
    if "404" in html or "not found" in html.lower():
        return False
    # Also try data.html for data completeness
    return True


def find_first_year(comp_slug: str, start: int = 1895, end: int = 2030) -> int | None:
    """Binary-ish search for first year competition exists."""
    for y in range(start, end + 1):
        if season_exists(comp_slug, y):
            return y
    return None


def find_last_year(comp_slug: str, start: int = 2026) -> int | None:
    """Find latest year (scan backward from current)."""
    for y in range(start, 1894, -1):
        if season_exists(comp_slug, y):
            return y
    return None


# Competition slug mapping (RLP uses different names over time)
COMPETITIONS = {
    "NRL": ["nrl"],  # 1998+; pre-1998 was NSWRL
    "Super League": ["super-league"],
    "NSW Cup": ["nsw-cup", "nswrl-reserve-grade", "nswrl-first-division"],
    "QLD Cup": ["qld-cup", "brisbane-rugby-league", "brl"],
    "WSL": ["women-s-super-league", "wsl"],
    "RFL Championship": ["championship", "second-division", "league-1"],
}


def probe_competition(name: str, slugs: list[str]) -> tuple[int | None, int | None]:
    """Probe first and last year for a competition."""
    first = last = None
    for slug in slugs:
        # Try recent year first (2026, 2025, 2024...)
        for y in [2026, 2025, 2024, 2023, 2022]:
            if season_exists(slug, y):
                last = y
                break
        if last is None:
            continue
        # Find first year (scan from 1908 upward)
        for y in range(1908, last + 1):
            if season_exists(slug, y):
                first = y
                break
        if first is not None:
            break
    return first, last


def main():
    print("=" * 60)
    print("Rugby League Project - Data Bounds Probe")
    print("=" * 60)

    # 1. Check 2026 calendar for available comp slugs
    print("\n1. Competitions in 2026 calendar:")
    slugs_2026 = comp_slug_from_calendar(2026)
    for s in sorted(slugs_2026.keys()):
        print(f"   - {s}")

    # 2. Probe each target competition
    # NRL: nrl (1998+), nswrl (1908-1997), arl (1995-97), super-league (1997 SL)
    results = {}
    targets = [
        ("NRL (incl. NSWRL)", ["nrl", "nswrl"]),
        ("Super League", ["super-league"]),
        ("NSW Cup", ["nsw-cup", "nswrl-reserve-grade", "nswrl-first-division"]),
        ("QLD Cup", ["qld-cup", "brl"]),
        ("WSL (Women's Super League)", ["women-s-super-league"]),
        ("RFL Championship", ["championship", "second-division"]),
    ]

    print("\n2. Probing competition bounds (first year, latest year):")
    for name, slugs in targets:
        first, last = probe_competition(name, slugs)
        results[name] = (first, last)
        print(f"   {name}: first={first}, latest={last}")

    # 3. Sample a match page to verify data structure
    print("\n3. Sampling NRL 2025 Round 1 match structure:")
    url = f"{BASE}/seasons/nrl-2025/round-1/summary.html"
    html = fetch(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        # Count match result lines (team scores)
        text = soup.get_text()
        if "Canberra" in text and "30" in text and "Warriors" in text:
            print("   OK: Match data present (Canberra 30 v Warriors 8)")
        # Check for match detail link
        match_links = soup.find_all("a", href=re.compile(r"/matches/\d+"))
        if match_links:
            print(f"   Found {len(match_links)} match links")
        # Check for player links
        player_links = soup.find_all("a", href=re.compile(r"/players/\d+"))
        print(f"   Found {len(player_links)} player links")

    # 4. Try match detail page
    print("\n4. Sampling match detail page:")
    match_url = f"{BASE}/matches/103171"  # Canberra v Warriors R1 2025
    html = fetch(match_url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        if "Canberra" in text and "Warriors" in text:
            print("   OK: Match detail page loads")
        else:
            print("   Content sample:", text[:500] if text else "empty")

    print("\n" + "=" * 60)
    print("Summary - Data Bounds:")
    print("-" * 60)
    for name, (first, last) in results.items():
        if first and last:
            print(f"  {name}: {first} - {last}")
        else:
            print(f"  {name}: No data found (check slug)")
    print("=" * 60)


if __name__ == "__main__":
    main()
