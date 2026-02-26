"""
NSW Cup mappings — RLP team/venue names to display format.

NSW Cup teams include reserve grade (Canberra (R), Newtown (R)) and standalone clubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# NSW Cup / reserve grade team names (RLP internal -> display)
TEAM_INTERNAL_TO_DISPLAY: dict[str, str] = {
    # NRL-affiliated reserve grade
    "Canberra": "Canberra Raiders",
    "Canterbury": "Canterbury-Bankstown Bulldogs",
    "Cronulla": "Cronulla-Sutherland Sharks",
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
    "Brisbane": "Brisbane Broncos",
    "Dolphins": "Dolphins",
    "Gold Coast": "Gold Coast Titans",
    # Standalone NSW Cup clubs
    "Newtown": "Newtown Jets",
    "North Sydney": "North Sydney Bears",
    "Western Suburbs": "Western Suburbs Magpies",
    "Blacktown": "Blacktown Workers Sea Eagles",
    "Mount Pritchard": "Mount Pritchard Mounties",
    "Wentworthville": "Wentworthville Magpies",
}

TEAM_EXTERNAL_TO_INTERNAL = {v: k for k, v in TEAM_INTERNAL_TO_DISPLAY.items()}
for k in TEAM_INTERNAL_TO_DISPLAY:
    if k not in TEAM_EXTERNAL_TO_INTERNAL:
        TEAM_EXTERNAL_TO_INTERNAL[k] = k

VENUE_INTERNAL_TO_DISPLAY: dict[str, str] = {
    "Henson Park": "Henson Park",
    "Campbelltown Sports Stadium": "Campbelltown Sports Stadium",
    "Leichhardt Oval": "Leichhardt Oval",
    "CommBank Stadium": "CommBank Stadium",
    "4 Pines Park": "4 Pines Park",
    "Netstrata Jubilee Stadium": "Netstrata Jubilee Stadium",
    "McDonald Jones Stadium": "McDonald Jones Stadium",
    "Suncorp Stadium": "Suncorp Stadium",
    "AAMI Park": "AAMI Park",
}

VENUE_EXTERNAL_TO_INTERNAL = {v: k for k, v in VENUE_INTERNAL_TO_DISPLAY.items()}
for k in VENUE_INTERNAL_TO_DISPLAY:
    if k not in VENUE_EXTERNAL_TO_INTERNAL:
        VENUE_EXTERNAL_TO_INTERNAL[k] = k


class TeamMapper:
    def to_internal(self, external: str) -> str:
        return TEAM_EXTERNAL_TO_INTERNAL.get(external.strip(), external.strip())

    def to_display(self, internal: str) -> str:
        return TEAM_INTERNAL_TO_DISPLAY.get(internal.strip(), internal.strip())

    def all_internal(self) -> list[str]:
        return list(TEAM_INTERNAL_TO_DISPLAY.keys())


class VenueMapper:
    def to_internal(self, external: str) -> str:
        return VENUE_EXTERNAL_TO_INTERNAL.get(external.strip(), external.strip())

    def to_display(self, internal: str) -> str:
        return VENUE_INTERNAL_TO_DISPLAY.get(internal.strip(), internal.strip())

    def all_internal(self) -> list[str]:
        return list(VENUE_INTERNAL_TO_DISPLAY.keys())


class PlayerMapper:
    """RLP player_id -> display name. Uses competition-specific player_index."""

    def __init__(self, player_index_path: Optional[Path] = None):
        self._index_path = player_index_path or (
            _PROJECT_ROOT / "model" / "output" / "nsw-cup" / "player_index.json"
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
