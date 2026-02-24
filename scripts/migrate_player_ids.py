#!/usr/bin/env python3
"""
Rename player performance files from firstname_lastname to lastname_firstname_DDMMYYYY
using the DOB cache built by build_player_dob_cache.py.

Usage:
  python scripts/migrate_player_ids.py [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYERS_DIR = _PROJECT_ROOT / "afl_data" / "data" / "players"
CACHE_PATH = _PROJECT_ROOT / "afl_data" / "data" / "player_dob_cache.json"


def stem_to_canonical_id(stem: str, dob: str) -> str | None:
    """
    Convert firstname_lastname stem + DOB -> lastname_firstname_DDMMYYYY.
    Handles compound names and numeric disambiguators.
    """
    if not dob:
        return None

    clean = re.sub(r"\d+$", "", stem)
    parts = clean.split("_")
    if len(parts) < 2:
        return None

    firstname = parts[0]
    lastname = "_".join(parts[1:])
    return f"{lastname}_{firstname}_{dob}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Rename player files to lastname_firstname_DDMMYYYY format")
    ap.add_argument("--dry-run", action="store_true", help="Print renames without executing")
    args = ap.parse_args()

    if not CACHE_PATH.exists():
        raise SystemExit(f"DOB cache not found: {CACHE_PATH}\nRun: python scripts/build_player_dob_cache.py")

    cache = json.loads(CACHE_PATH.read_text())
    print(f"DOB cache: {len(cache)} entries")

    files = sorted(glob.glob(str(PLAYERS_DIR / "*_performance_details.csv")))
    print(f"Player files: {len(files)}")

    renamed = 0
    skipped = 0
    already_correct = 0
    conflicts: dict[str, list[str]] = {}

    planned_renames: list[tuple[Path, Path]] = []

    for f in files:
        path = Path(f)
        stem = path.stem.replace("_performance_details", "")

        # Check if already in old format (has 8-digit DOB suffix)
        parts = stem.split("_")
        if len(parts) >= 3 and len(parts[-1]) == 8 and parts[-1].isdigit():
            already_correct += 1
            continue

        dob = cache.get(stem, "")
        if not dob:
            skipped += 1
            if not args.dry_run:
                print(f"  SKIP (no DOB): {stem}")
            continue

        new_id = stem_to_canonical_id(stem, dob)
        if not new_id:
            skipped += 1
            continue

        new_path = PLAYERS_DIR / f"{new_id}_performance_details.csv"

        if new_path == path:
            already_correct += 1
            continue

        # Track conflicts
        new_name = new_path.name
        conflicts.setdefault(new_name, []).append(stem)
        planned_renames.append((path, new_path))

    # Check for conflicts (multiple old files mapping to same new name)
    real_conflicts = {k: v for k, v in conflicts.items() if len(v) > 1}
    if real_conflicts:
        print(f"\nWARNING: {len(real_conflicts)} filename conflicts:")
        for target, sources in sorted(real_conflicts.items())[:10]:
            print(f"  {target} <- {sources}")

    print(f"\nSummary:")
    print(f"  Already correct: {already_correct}")
    print(f"  To rename: {len(planned_renames)}")
    print(f"  Skipped (no DOB): {skipped}")
    print(f"  Conflicts: {len(real_conflicts)}")

    if args.dry_run:
        print("\n[DRY RUN] Sample renames:")
        for old, new in planned_renames[:10]:
            print(f"  {old.name} -> {new.name}")
        return

    # Execute renames
    for old_path, new_path in planned_renames:
        if new_path.exists():
            # Merge: append rows from old to existing new (dedup by team+year+round)
            print(f"  MERGE: {old_path.name} -> {new_path.name}")
            with open(new_path, "a") as dst, open(old_path) as src:
                lines = src.readlines()
                if lines:
                    dst.writelines(lines[1:])  # skip header
            old_path.unlink()
        else:
            old_path.rename(new_path)
        renamed += 1

    print(f"\nRenamed: {renamed}")


if __name__ == "__main__":
    main()
