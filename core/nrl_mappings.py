"""
NRL mappings — RLP team/venue names to display format.

RLP uses internal names (e.g. St Geo Illa, Sydney, North Qld).
Display names for UI (e.g. St George Illawarra, Sydney Roosters, North Queensland).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# TEAM MAPPING (RLP internal -> display)
# ---------------------------------------------------------------------------

TEAM_INTERNAL_TO_DISPLAY: dict[str, str] = {
    "Brisbane": "Brisbane Broncos",
    "Canberra": "Canberra Raiders",
    "Canterbury": "Canterbury-Bankstown Bulldogs",
    "Cronulla": "Cronulla-Sutherland Sharks",
    "Dolphins": "Dolphins",
    "Manly": "Manly-Warringah Sea Eagles",
    "Melbourne": "Melbourne Storm",
    "Newcastle": "Newcastle Knights",
    "North Qld": "North Queensland Cowboys",
    "Parramatta": "Parramatta Eels",
    "Penrith": "Penrith Panthers",
    "South Sydney": "South Sydney Rabbitohs",
    "St Geo Illa": "St George Illawarra Dragons",
    "Sydney": "Sydney Roosters",
    "Warriors": "New Zealand Warriors",
    "Wests Tigers": "Wests Tigers",
    # Historic / 1997
    "St George": "St George Dragons",
    "Illawarra": "Illawarra Steelers",
    "Balmain": "Balmain Tigers",
    "Western Suburbs": "Western Suburbs Magpies",
    "North Sydney": "North Sydney Bears",
    "Gold Coast": "Gold Coast Titans",
    "Hunter": "Hunter Mariners",
    "Perth": "Perth Reds",
    "Adelaide": "Adelaide Rams",
    "South Qld": "South Queensland Crushers",
}

TEAM_EXTERNAL_TO_INTERNAL: dict[str, str] = {v: k for k, v in TEAM_INTERNAL_TO_DISPLAY.items()}
for k in TEAM_INTERNAL_TO_DISPLAY:
    if k not in TEAM_EXTERNAL_TO_INTERNAL:
        TEAM_EXTERNAL_TO_INTERNAL[k] = k
# League Unlimited short form
TEAM_EXTERNAL_TO_INTERNAL["Warriors"] = "Warriors"

# ---------------------------------------------------------------------------
# VENUE MAPPING
# ---------------------------------------------------------------------------

VENUE_INTERNAL_TO_DISPLAY: dict[str, str] = {
    "Allegiant Stadium": "Allegiant Stadium",
    "Allianz Stadium": "Allianz Stadium",
    "AAMI Park": "AAMI Park",
    "Campbelltown Sports Stadium": "Campbelltown Sports Stadium",
    "CommBank Stadium": "CommBank Stadium",
    "Netstrata Jubilee Stadium": "Netstrata Jubilee Stadium",
    "4 Pines Park": "4 Pines Park",
    "Brookvale Oval": "Brookvale Oval",
    "Suncorp Stadium": "Suncorp Stadium",
    "McDonald Jones Stadium": "McDonald Jones Stadium",
    "Queensland Country Bank Stadium": "Queensland Country Bank Stadium",
    "Western Sydney Stadium": "CommBank Stadium",
    # League Unlimited 2026 venues
    "Go Media Stadium": "Go Media Stadium",
    "Ocean Protect Stadium": "Ocean Protect Stadium",
    "GIO Stadium Canberra": "GIO Stadium Canberra",
    "Polytec Stadium": "Polytec Stadium",
    "Cbus Super Stadium": "Cbus Super Stadium",
    "WIN Stadium": "WIN Stadium",
    "Carrington Park": "Carrington Park",
    "TIO Stadium": "TIO Stadium",
    "Sky Stadium": "Sky Stadium",
    "One New Zealand Stadium": "One New Zealand Stadium",
    "Kayo Stadium": "Kayo Stadium",
    "Accor Stadium": "Accor Stadium",
    "Optus Stadium": "Optus Stadium",
    "Leichhardt Oval": "Leichhardt Oval",
}

VENUE_EXTERNAL_TO_INTERNAL: dict[str, str] = {v: k for k, v in VENUE_INTERNAL_TO_DISPLAY.items()}
# League Unlimited uses "Jubilee Stadium" for Netstrata Jubilee Stadium
VENUE_EXTERNAL_TO_INTERNAL["Jubilee Stadium"] = "Netstrata Jubilee Stadium"

# ---------------------------------------------------------------------------
# 1997/1998 CLUB MAPPING (for ELO propagation)
# ---------------------------------------------------------------------------

# 1997 ARL/SL clubs -> 1998 NRL clubs
# Direct: same name
# Mergers: St George+Illawarra -> St Geo Illa, Balmain+Wests -> Wests Tigers, Norths+Manly -> Manly (Northern Eagles 1999)
# Folded: Hunter, Perth, Adelaide, South Qld, Gold Coast Chargers
# New: Melbourne

CLUB_1997_TO_1998: dict[str, str] = {
    "Brisbane": "Brisbane",
    "Canberra": "Canberra",
    "Canterbury": "Canterbury",
    "Cronulla": "Cronulla",
    "Manly": "Manly",
    "Newcastle": "Newcastle",
    "North Qld": "North Qld",
    "Parramatta": "Parramatta",
    "Penrith": "Penrith",
    "South Sydney": "South Sydney",
    "St George": "St Geo Illa",
    "Illawarra": "St Geo Illa",
    "Sydney": "Sydney",
    "Warriors": "Warriors",
    "Balmain": "Wests Tigers",
    "Western Suburbs": "Wests Tigers",
    "North Sydney": "Manly",
    "Hunter": "Newcastle",
    "Perth": "Brisbane",
    "Adelaide": "Melbourne",
    "South Qld": "Brisbane",
    "Gold Coast": "Melbourne",
}

# ---------------------------------------------------------------------------
# Mapper classes
# ---------------------------------------------------------------------------


class NRLTeamMapper:
    """Map NRL/RLP team names to display."""

    def to_internal(self, external: str) -> str:
        return TEAM_EXTERNAL_TO_INTERNAL.get(external.strip(), external.strip())

    def to_display(self, internal: str) -> str:
        return TEAM_INTERNAL_TO_DISPLAY.get(internal.strip(), internal.strip())

    def all_internal(self) -> list[str]:
        return list(TEAM_INTERNAL_TO_DISPLAY.keys())


class NRLVenueMapper:
    """Map NRL venue names."""

    def to_internal(self, external: str) -> str:
        return VENUE_EXTERNAL_TO_INTERNAL.get(external.strip(), external.strip())

    def to_display(self, internal: str) -> str:
        return VENUE_INTERNAL_TO_DISPLAY.get(internal.strip(), internal.strip())


# Aliases for competition-agnostic imports
TeamMapper = NRLTeamMapper
VenueMapper = NRLVenueMapper


class NRLPlayerMapper:
    """
    RLP uses numeric player IDs (/players/12345).
    Map player_id to display name from lineup data or player index.
    """

    def __init__(self, player_index_path: Optional[Path] = None):
        self._index_path = player_index_path or (
            _PROJECT_ROOT / "model" / "output" / "nrl" / "player_index.json"
        )
        self._id_to_display: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        import json
        with open(self._index_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            self._id_to_display = {k: v for k, v in data.items() if k != "unknown"}

    def to_display(self, player_id: str) -> str:
        return self._id_to_display.get(str(player_id), f"Player {player_id}")


PlayerMapper = NRLPlayerMapper
