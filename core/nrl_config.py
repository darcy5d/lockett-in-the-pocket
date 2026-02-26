"""
NRL competition tier config — from nrl_competition_history.md.

Tier 1: National top tier (NRL predictor uses this only).
Tier 2: State second tier (NSW Cup, QLD Cup, reserve grade) — for future predictors.
"""

from __future__ import annotations

from typing import Optional

# Slug must match filename pattern: matches_{slug}_{y1}_{y2}.csv
# super-league-au = Australia 1997 (RLP super-league); super-league-uk = UK
TIER1_SLUGS = frozenset([
    "nswrfl", "nswrl", "arl", "super-league", "super-league-au", "nrl",
])

TIER2_SLUGS = frozenset([
    # NSW
    "nswrfl-reserve-grade", "nswrl-reserve-grade",
    "nswrl-first-division", "nsw-cup",
    # QLD
    "qrl", "brl", "qld-cup",
])


def slug_from_match_filename(name: str) -> Optional[str]:
    """Extract slug from matches_{slug}_{y1}_{y2}.csv"""
    if not name.startswith("matches_") or not name.endswith(".csv"):
        return None
    rest = name[8:-4]  # strip "matches_" and ".csv"
    parts = rest.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])  # slug may contain hyphens
    return None


def is_tier1_file(name: str) -> bool:
    slug = slug_from_match_filename(name)
    return slug in TIER1_SLUGS if slug else False


def is_tier1_lineup_file(name: str) -> bool:
    """Extract slug from lineup_details_{slug}_{y1}_{y2}.csv"""
    if not name.startswith("lineup_details_") or not name.endswith(".csv"):
        return False
    rest = name[15:-4]
    parts = rest.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        slug = "_".join(parts[:-2])
        return slug in TIER1_SLUGS
    return False
