#!/usr/bin/env python3
"""
Scrape NRL/rugby lineups from League Unlimited Teams pages.

Fetches /live-update/show/{round_id}/{match_id}/Teams for each fixture match,
parses jersey 1-22 and player names, maps to RLP player_ids via DOB cache.
Output: lineup_details_{slug}_{year}_{year}.csv

Usage:
  python datafetch/league_unlimited_lineup_scraper.py [--competition nrl] [--round 1]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.competition_config import get_competition, get_competition_slugs
from core.nrl_mappings import NRLTeamMapper

LU_BASE = "https://leagueunlimited.com"
RLP_BASE = "https://www.rugbyleagueproject.org"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
CACHE_PATH = _PROJECT_ROOT / "nrl_data" / "data" / "rlp_player_dob_cache.json"
DATA_DIR = _PROJECT_ROOT / "nrl_data" / "data"

_team_mapper = NRLTeamMapper()


def _load_dob_cache() -> dict:
    """Load {player_id: {name, dob}}."""
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH) as f:
        return json.load(f)


def _normalize_name_for_match(name: str) -> str:
    """Normalize name for matching: apostrophes, accents, lowercase."""
    if not name:
        return ""
    # Replace apostrophe variants (e.g. Mata'afa, O'Brien) with plain form
    s = re.sub(r"['\u2019\u2018]", "", (name or "").strip().lower())
    return s


def _build_name_to_player_id(cache: dict) -> dict:
    """Build (name_lower, dob) -> player_id and name_lower -> player_id for unique names."""
    by_name_dob: dict[tuple[str, str], list[str]] = {}
    by_name: dict[str, list[str]] = {}
    for pid, data in cache.items():
        name = (data.get("name") or "").strip()
        dob = (data.get("dob") or "").strip()
        if not name:
            continue
        # Variants: "Kalyn Ponga", "Ponga", "Kalyn Ponga" (normalized: apostrophes removed)
        parts = name.split()
        variants = [name]
        if len(parts) >= 2:
            variants.append(parts[-1])
            variants.append(f"{parts[-1]} {parts[0]}")
        for v in variants:
            vlo = v.lower().strip()
            if not vlo:
                continue
            key = (vlo, dob)
            if key not in by_name_dob:
                by_name_dob[key] = []
            by_name_dob[key].append(pid)
            if vlo not in by_name:
                by_name[vlo] = []
            by_name[vlo].append((pid, dob))
            # Add normalized variant (no apostrophes) for "Mata'afa" -> "mataafa"
            vlo_norm = _normalize_name_for_match(v)
            if vlo_norm and vlo_norm != vlo:
                key_norm = (vlo_norm, dob)
                if key_norm not in by_name_dob:
                    by_name_dob[key_norm] = []
                by_name_dob[key_norm].append(pid)
                if vlo_norm not in by_name:
                    by_name[vlo_norm] = []
                by_name[vlo_norm].append((pid, dob))
    return by_name_dob, by_name


def _fetch_rlp_by_slug(slug: str, timeout: int = 15) -> tuple[str | None, str | None, str | None]:
    """
    Fetch RLP player page by slug (e.g. toa-mataafa). Returns (player_id, name, dob) or (None, None, None).
    RLP may use /players/{slug}/summary.html or redirect to /players/{id}/summary.html.
    """
    if not slug or not slug.strip():
        return None, None, None
    slug = slug.strip().lower()
    url = f"{RLP_BASE}/players/{slug}/summary.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return None, None, None
        # Player ID from final URL: /players/12345/summary.html
        final_url = r.url
        id_match = re.search(r"/players/(\d+)/summary\.html", final_url)
        player_id = id_match.group(1) if id_match else None
        text = r.text
        # Name from title
        name_match = re.search(r"<title>([^\-]+)\s*\-", text)
        name = name_match.group(1).strip() if name_match else None
        # DOB
        dob_match = re.search(
            r"Born\s+(?:[A-Za-z]+,\s*)?(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),\s*(\d{4})",
            text, re.I
        )
        dob = None
        if dob_match:
            day, mon_str, year = dob_match.groups()
            mon = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
                   "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"}.get(mon_str.lower()[:3], "01")
            dob = f"{int(day):02d}{mon}{year}"
        # If URL had slug not numeric, try to get ID from page (e.g. canonical link)
        if not player_id and name:
            canon_match = re.search(r'<link[^>]+rel="canonical"[^>]+href="[^"]*?/players/(\d+)/', text)
            if canon_match:
                player_id = canon_match.group(1)
        return (player_id, name, dob) if player_id else (None, name, dob)
    except Exception:
        return None, None, None


def _resolve_player_id(
    name: str,
    slug: str | None,
    cache: dict,
    by_name_dob: dict,
    by_name: dict,
    rlp_delay: float = 0.3,
) -> str:
    """Map League Unlimited name/slug to RLP player_id. Tries slug->RLP first, then name->cache."""
    name = (name or "").strip()
    if not name and not slug:
        return "unknown"
    # 1. Try name -> cache first (fast, no network)
    nlo = name.lower() if name else ""
    nlo_norm = _normalize_name_for_match(name) if name else ""
    for key_lo in ([nlo, nlo_norm] if nlo_norm != nlo else [nlo]):
        if not key_lo:
            continue
        for (n, dob), pids in by_name_dob.items():
            if n == key_lo and len(pids) == 1:
                return pids[0]
        if key_lo in by_name:
            candidates = by_name[key_lo]
            if len(candidates) == 1:
                return candidates[0][0]
            return candidates[0][0]
    # 2. Try surname only
    if name:
        parts = name.split()
        if len(parts) >= 2:
            surname = parts[-1].lower()
            surname_norm = _normalize_name_for_match(parts[-1])
            for s in ([surname, surname_norm] if surname_norm != surname else [surname]):
                if s in by_name and len(by_name[s]) == 1:
                    return by_name[s][0][0]
    # 3. Try slug -> RLP lookup (discover new players, only when cache miss)
    if slug:
        pid, rlp_name, dob = _fetch_rlp_by_slug(slug)
        if pid:
            cache[pid] = {"name": rlp_name or name, "dob": dob or ""}
            _save_cache(cache)
            return pid
        time.sleep(rlp_delay)
    return "unknown"


def _save_cache(cache: dict) -> None:
    """Persist cache to disk."""
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _parse_teams_page(html: str, team_display_to_internal: dict[str, str]) -> list[tuple[str, str, int, str | None]]:
    """
    Parse Teams page. Returns [(team_internal, player_name, jersey, slug)] in order.
    Slug from href /people/show/toa-mataafa -> toa-mataafa.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str, int, str | None]] = []
    current_team: str | None = None
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            # Team header: | Knights | or | Cowboys |
            first_text = (cells[0].get_text(strip=True) if cells else "").strip()
            if first_text and first_text in team_display_to_internal:
                current_team = team_display_to_internal[first_text]
                continue
            # Player row: | 1 | Kalyn Ponga | (link href=/people/show/kalyn-ponga)
            if len(cells) >= 2 and current_team:
                try:
                    jersey = int(cells[0].get_text(strip=True))
                except ValueError:
                    continue
                name_cell = cells[1]
                link = name_cell.find("a", href=True)
                slug: str | None = None
                if link:
                    href = link.get("href", "")
                    # /people/show/toa-mataafa -> toa-mataafa
                    if "/people/show/" in href:
                        slug = href.split("/people/show/")[-1].strip("/").split("/")[0]
                    name = link.get_text(strip=True)
                else:
                    name = name_cell.get_text(strip=True)
                if name and 1 <= jersey <= 99:
                    results.append((current_team, name, jersey, slug or None))
    return results


def _get_team_display_to_internal(competition_id: str) -> dict[str, str]:
    """Map League Unlimited team display names (e.g. Knights, Cowboys, York) to internal."""
    cfg = get_competition(competition_id)
    mod_name = cfg.get("mapper_module", "core.nrl_mappings")
    mod = __import__(mod_name, fromlist=["TEAM_INTERNAL_TO_DISPLAY"])
    internal_to_display = getattr(mod, "TEAM_INTERNAL_TO_DISPLAY", {})
    result: dict[str, str] = {}
    for internal, display in internal_to_display.items():
        result[display] = internal
        result[internal] = internal  # League Unlimited often uses internal name (York, Castleford)
        parts = display.split()
        if len(parts) >= 2:
            short = parts[-1]
            if short not in result:
                result[short] = internal
            first = parts[0]
            if first not in result:
                result[first] = internal  # "York" from "York Knights", "Castleford" from "Castleford Tigers"
        elif display not in result:
            result[display] = internal
    return result


def scrape_lineups(
    competition_id: str = "nrl",
    year: int = 2026,
    round_filter: str | int | None = None,
    fixture_path: Path | None = None,
    output_path: Path | None = None,
    delay: float = 0.5,
) -> Path:
    cfg = get_competition(competition_id)
    if not cfg:
        raise ValueError(f"Unknown competition: {competition_id}")
    slug = "nrl" if competition_id == "nrl" else "super-league-uk"
    fixture_path = fixture_path or (DATA_DIR / "matches" / cfg.get("fixture_filename", "matches_2026.csv"))
    output_path = output_path or (DATA_DIR / "lineups" / f"lineup_details_{slug}_{year}_{year}.csv")

    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}. Run fixture fetch first.")

    df = pd.read_csv(fixture_path, dtype=str)
    if "live_update_id" not in df.columns:
        raise ValueError("Fixture missing live_update_id. Re-run fixture fetch.")

    if round_filter is not None:
        rn = str(round_filter).strip()
        df = df[df["round_num"].astype(str).str.strip() == rn]
    if df.empty:
        print("No matches to scrape.")
        return output_path

    cache = _load_dob_cache()
    by_name_dob, by_name = _build_name_to_player_id(cache)
    team_map = _get_team_display_to_internal(competition_id)

    all_rows: list[dict] = []
    match_idx_per_round: dict[str, int] = {}

    for _, row in df.iterrows():
        live_id = str(row.get("live_update_id", "")).strip()
        if not live_id or "/" not in live_id:
            continue
        round_num = str(row.get("round_num", "")).strip()
        t1 = str(row.get("team_1_team_name", "")).strip()
        t2 = str(row.get("team_2_team_name", "")).strip()
        round_key = round_num
        if round_key not in match_idx_per_round:
            match_idx_per_round[round_key] = 0
        m_idx = match_idx_per_round[round_key]
        match_id = f"{slug}_{year}_r{round_num}_m{m_idx}"
        match_idx_per_round[round_key] += 1

        url = f"{LU_BASE}/live-update/show/{live_id}/Teams"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"  Skip {live_id}: {e}")
            continue

        players = _parse_teams_page(r.text, team_map)
        for team_internal, player_name, jersey, player_slug in players:
            pid = _resolve_player_id(
                player_name, player_slug, cache, by_name_dob, by_name, rlp_delay=delay
            )
            all_rows.append({
                "match_idx": 0,
                "competition": slug,
                "year": year,
                "round_num": round_num,
                "team": team_internal,
                "player_id": pid,
                "player_name": player_name,
                "match_id": match_id,
            })
        time.sleep(delay)

    if not all_rows:
        print("No lineup data scraped.")
        return output_path

    out_df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Replace existing rows for scraped match_ids, then append new data
    if output_path.exists():
        existing = pd.read_csv(output_path)
        scraped_match_ids = set(out_df["match_id"].unique())
        existing = existing[~existing["match_id"].isin(scraped_match_ids)]
        combined = pd.concat([existing, out_df], ignore_index=True)
    else:
        combined = out_df
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(all_rows)} lineup entries to {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default="nrl")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--round", default=None, help="Only scrape this round")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()
    scrape_lineups(
        competition_id=args.competition,
        year=args.year,
        round_filter=args.round,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
