"""
Competition registry for Rugby League predictors.

Each competition has its own slugs (storage filenames), fixture source, and model output.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPETITIONS: dict[str, dict] = {
    "nrl": {
        "display": "NRL",
        "slugs": ["nswrfl", "nswrl", "arl", "super-league-au", "nrl"],
        "year_format": "single",
        "fixture_source": "league_unlimited",
        "output_dir": str(_PROJECT_ROOT / "model" / "output" / "nrl"),
        "fixture_filename": "matches_2026.csv",
        "mapper_module": "core.nrl_mappings",
    },
    "nsw-cup": {
        "display": "NSW Cup",
        "slugs": [
            "nsw-cup",
            "nswrl-first-division",
            "nswrl-reserve-grade",
            "nswrfl-reserve-grade",
        ],
        "year_format": "single",
        "fixture_source": "league_unlimited",
        "output_dir": str(_PROJECT_ROOT / "model" / "output" / "nsw-cup"),
        "fixture_filename": "matches_nsw-cup_2026.csv",
        "mapper_module": "core.nsw_cup_mappings",
    },
    "qld-cup": {
        "display": "QLD Cup",
        "slugs": ["qld-cup", "brl", "qrl"],
        "year_format": "single",
        "fixture_source": "league_unlimited",
        "output_dir": str(_PROJECT_ROOT / "model" / "output" / "qld-cup"),
        "fixture_filename": "matches_qld-cup_2026.csv",
        "mapper_module": "core.qld_cup_mappings",
    },
    "au-state-cup": {
        "display": "AU State Cup (NSW+QLD)",
        "slugs": [
            "nsw-cup",
            "nswrl-first-division",
            "nswrl-reserve-grade",
            "nswrfl-reserve-grade",
            "qld-cup",
            "brl",
            "qrl",
        ],
        "year_format": "single",
        "fixture_source": "league_unlimited",
        "output_dir": str(_PROJECT_ROOT / "model" / "output" / "au-state-cup"),
        "fixture_filename": "matches_au-state-cup_2026.csv",
        "mapper_module": "core.au_state_cup_mappings",
    },
    "uk-super-league": {
        "display": "UK Super League",
        "slugs": ["super-league-uk"],
        "year_format": "split",
        "fixture_source": "rlp",
        "output_dir": str(_PROJECT_ROOT / "model" / "output" / "uk-super-league"),
        "fixture_filename": "matches_super-league-uk_2026.csv",
        "mapper_module": "core.uk_mappings",
    },
    "uk-championship": {
        "display": "UK Championship",
        "slugs": ["championship-uk", "second-division-uk"],
        "year_format": "split",
        "fixture_source": "league_unlimited",
        "output_dir": str(_PROJECT_ROOT / "model" / "output" / "uk-championship"),
        "fixture_filename": "matches_championship-uk_2026.csv",
        "mapper_module": "core.uk_mappings",
    },
}


def get_competition(comp_id: str) -> dict | None:
    return COMPETITIONS.get(comp_id)


def get_all_competition_ids() -> list[str]:
    return list(COMPETITIONS.keys())


def get_competition_slugs(comp_id: str) -> frozenset:
    cfg = COMPETITIONS.get(comp_id)
    if not cfg:
        return frozenset()
    return frozenset(cfg.get("slugs", []))


def slug_matches_file(filename: str, allowed_slugs: frozenset) -> bool:
    """Check if match filename's slug is in allowed_slugs."""
    if not filename.startswith("matches_") or not filename.endswith(".csv"):
        return False
    rest = filename[8:-4]
    parts = rest.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        slug = "_".join(parts[:-2])
        return slug in allowed_slugs
    return False


def slug_matches_lineup_file(filename: str, allowed_slugs: frozenset) -> bool:
    """Check if lineup filename's slug is in allowed_slugs."""
    if not filename.startswith("lineup_details_") or not filename.endswith(".csv"):
        return False
    rest = filename[15:-4]
    parts = rest.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        slug = "_".join(parts[:-2])
        return slug in allowed_slugs
    return False
