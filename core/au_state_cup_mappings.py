"""
AU State Cup mappings — merged NSW Cup + QLD Cup.

Combines team and venue mappings from both competitions for the combined
au-state-cup model (NSW + QLD second-tier state leagues, same rules).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Merge NSW + QLD team names (RLP internal -> display)
# Overlapping keys (Brisbane, North Qld, Gold Coast) map to same display
_NSW_TEAMS = {
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
    "Newtown": "Newtown Jets",
    "North Sydney": "North Sydney Bears",
    "Western Suburbs": "Western Suburbs Magpies",
    "Blacktown": "Blacktown Workers Sea Eagles",
    "Mount Pritchard": "Mount Pritchard Mounties",
    "Wentworthville": "Wentworthville Magpies",
}
_QLD_TEAMS = {
    "Redcliffe": "Redcliffe Dolphins",
    "Wynnum Manly": "Wynnum Manly Seagulls",
    "Ipswich": "Ipswich Jets",
    "Townsville": "Townsville Blackhawks",
    "Mackay": "Mackay Cutters",
    "Central": "Central Queensland Capras",
    "Northern": "Northern Pride",
    "Souths Logan": "Souths Logan Magpies",
    "Burleigh": "Burleigh Bears",
    "Tweed Heads": "Tweed Seagulls",
    "Sunshine Coast": "Sunshine Coast Falcons",
    "Norths Devils": "Norths Devils",
    "Western Clydesdales": "Western Clydesdales",
    "Brisbane Tigers": "Brisbane Tigers",
    "PNG Hunters": "PNG Hunters",
}
TEAM_INTERNAL_TO_DISPLAY: dict[str, str] = {**_NSW_TEAMS, **_QLD_TEAMS}

TEAM_EXTERNAL_TO_INTERNAL = {v: k for k, v in TEAM_INTERNAL_TO_DISPLAY.items()}
for k in TEAM_INTERNAL_TO_DISPLAY:
    if k not in TEAM_EXTERNAL_TO_INTERNAL:
        TEAM_EXTERNAL_TO_INTERNAL[k] = k

# Merge NSW + QLD venues
_NSW_VENUES = {
    "Henson Park": "Henson Park",
    "Campbelltown Sports Stadium": "Campbelltown Sports Stadium",
    "Leichhardt Oval": "Leichhardt Oval",
    "CommBank Stadium": "CommBank Stadium",
    "4 Pines Park": "4 Pines Park",
    "Netstrata Jubilee Stadium": "Netstrata Jubilee Stadium",
    "McDonald Jones Stadium": "McDonald Jones Stadium",
    "Suncorp Stadium": "Suncorp Stadium",
    "AAMI Park": "AAMI Park",
    "Go Media Stadium": "Go Media Stadium",
    "St Marys Leagues Stadium": "St Marys Leagues Stadium",
    "Belmore Sports Ground": "Belmore Sports Ground",
    "Cessnock Sports Ground": "Cessnock Sports Ground",
    "Mt Smart": "Mt Smart",
    "Lidcombe Oval": "Lidcombe Oval",
    "WIN Stadium": "WIN Stadium",
    "Redfern Oval": "Redfern Oval",
    "GIO Stadium Canberra": "GIO Stadium Canberra",
    "Seabrook Reserve": "Seabrook Reserve",
    "Polytec Stadium": "Polytec Stadium",
    "North Sydney Oval": "North Sydney Oval",
    "Parker Street Oval": "Parker Street Oval",
    "Michael Cronin Oval": "Michael Cronin Oval",
    "Collegians Sports Complex": "Collegians Sports Complex",
    "Scully Park": "Scully Park",
    "James Hardie Centre of Excellence": "James Hardie Centre of Excellence",
    "Newcastle Centre of Excellence": "Newcastle Centre of Excellence",
}
_QLD_VENUES = {
    "Queensland Country Bank Stadium": "Queensland Country Bank Stadium",
    "Dolphin Stadium": "Dolphin Stadium",
    "BMD Kougari Oval": "BMD Kougari Oval",
    "Kayo Stadium": "Kayo Stadium",
    "Seagulls Sports Complex": "Seagulls Sports Complex",
    "Bishop Park": "Bishop Park",
    "Toowoomba Sports Ground": "Toowoomba Sports Ground",
    "Richardson Park": "Richardson Park",
    "Santos National Football Stadium": "Santos National Football Stadium",
    "Totally Workwear Stadium": "Totally Workwear Stadium",
    "Barlow Park": "Barlow Park",
    "Davies Park": "Davies Park",
    "UAA Park": "UAA Park",
    "Sunshine Coast Stadium": "Sunshine Coast Stadium",
    "Jack Manski Oval": "Jack Manski Oval",
    "BB Print Stadium": "BB Print Stadium",
    "Browne Park": "Browne Park",
}
VENUE_INTERNAL_TO_DISPLAY: dict[str, str] = {**_NSW_VENUES, **_QLD_VENUES}

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
    """Loads from both NSW and QLD player indices."""

    def __init__(self, player_index_path: Optional[Path] = None):
        self._id_to_display: dict[str, str] = {}
        if player_index_path and player_index_path.exists():
            self._load_path(player_index_path)
        else:
            for sub in ("nsw-cup", "qld-cup"):
                p = _PROJECT_ROOT / "model" / "output" / sub / "player_index.json"
                if p.exists():
                    self._load_path(p)

    def _load_path(self, path: Path) -> None:
        import json
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                if k != "unknown" and k not in self._id_to_display:
                    self._id_to_display[k] = v

    def to_display(self, player_id: str) -> str:
        return self._id_to_display.get(str(player_id), f"Player {player_id}")
