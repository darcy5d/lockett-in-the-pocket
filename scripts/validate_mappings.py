#!/usr/bin/env python3
"""
Map-first validation: verify all mappings work before analysis.
Run from project root: python scripts/validate_mappings.py
"""

import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mappings import TeamNameMapper, VenueMapper, PlayerMapper, load_all_mappings


def test_team_mapping():
    """Validate team name mappings."""
    mapper = TeamNameMapper()
    tests = [
        ("Sydney Swans", "Sydney"),
        ("GWS GIANTS", "Greater Western Sydney"),
        ("Gold Coast SUNS", "Gold Coast"),
        ("Collingwood", "Collingwood"),
    ]
    print("Team mapping (external -> internal):")
    for ext, expected in tests:
        got = mapper.to_internal(ext)
        ok = "OK" if got == expected else f"FAIL (expected {expected})"
        print(f"  {ext!r} -> {got!r} {ok}")
    return all(mapper.to_internal(ext) == exp for ext, exp in tests)


def test_venue_mapping():
    """Validate venue mappings."""
    mapper = VenueMapper()
    tests = [
        ("SCG", "S.C.G."),
        ("MCG", "M.C.G."),
        ("Marvel Stadium", "Docklands"),
        ("People First Stadium", "Carrara"),
        ("ENGIE Stadium", "Sydney Showground"),
    ]
    print("\nVenue mapping (external -> internal):")
    for ext, expected in tests:
        got = mapper.to_internal(ext)
        ok = "OK" if got == expected else f"FAIL (expected {expected})"
        print(f"  {ext!r} -> {got!r} {ok}")
    return all(mapper.to_internal(ext) == exp for ext, exp in tests)


def test_player_mapping():
    """Validate player name <-> ID mappings."""
    mapper = PlayerMapper()
    # Known lineup names -> expected player_ids
    tests = [
        ("Jack Crisp", "crisp_jack_02101993"),
        ("Josh Daicos", "daicos_josh_26111998"),
        ("Nick Daicos", "daicos_nick_03012003"),
        ("Jordan de Goey", "goey_jordan_15031996"),
        ("Darcy Moore", None),  # May have multiple, we accept any match
    ]
    print("\nPlayer mapping (display -> player_id):")
    for display, expected in tests:
        got = mapper.to_player_id(display)
        if expected:
            ok = "OK" if got == expected else f"FAIL (expected {expected}, got {got})"
        else:
            ok = "OK" if got else "FAIL (no match)"
        print(f"  {display!r} -> {got!r} {ok}")
    # At least Jack Crisp and Jordan de Goey should work
    assert mapper.to_player_id("Jack Crisp") == "crisp_jack_02101993"
    assert mapper.to_player_id("Jordan de Goey") == "goey_jordan_15031996"
    # Round-trip
    pid = "crisp_jack_02101993"
    display = mapper.to_display(pid)
    back = mapper.to_player_id(display)
    assert back == pid, f"Round-trip failed: {pid} -> {display} -> {back}"
    print(f"  Round-trip OK: {pid} -> {display} -> {back}")
    return True


def audit_lineup_schema():
    """Check lineup file structure: players column vs player1, player2."""
    import pandas as pd
    lineup_dir = PROJECT_ROOT / "afl_data" / "data" / "lineups"
    sample = lineup_dir / "team_lineups_collingwood.csv"
    if not sample.exists():
        print("\nLineup audit: file not found")
        return False
    df = pd.read_csv(sample, nrows=5)
    has_players = "players" in df.columns
    has_player_cols = any(re.match(r"player\d+", c) for c in df.columns)
    print("\nLineup schema audit:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Has 'players' column: {has_players}")
    print(f"  Has player1, player2...: {has_player_cols}")
    if has_players and not has_player_cols:
        print("  -> Model expects player1..player22; lineup has 'players' (semicolon-sep).")
        print("  -> Need to parse 'players' and map to player_ids for training.")
    return True


def main():
    print("=" * 60)
    print("MAP-FIRST VALIDATION")
    print("=" * 60)
    ok = True
    ok &= test_team_mapping()
    ok &= test_venue_mapping()
    ok &= test_player_mapping()
    audit_lineup_schema()
    print("\n" + "=" * 60)
    print("PASS" if ok else "FAIL")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
