"""
UK Super League and Championship mappings — RLP team/venue names to display format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# UK Super League & Championship teams
TEAM_INTERNAL_TO_DISPLAY: dict[str, str] = {
    "St Helens": "St Helens",
    "Wigan": "Wigan Warriors",
    "Leeds": "Leeds Rhinos",
    "Warrington": "Warrington Wolves",
    "Hull": "Hull FC",
    "Hull KR": "Hull KR",
    "Catalans": "Catalans Dragons",
    "Castleford": "Castleford Tigers",
    "Huddersfield": "Huddersfield Giants",
    "Salford": "Salford Red Devils",
    "Wakefield": "Wakefield Trinity",
    "Leigh": "Leigh Leopards",
    "London": "London Broncos",
    "Bradford": "Bradford Bulls",
    "Featherstone": "Featherstone Rovers",
    "Toulouse": "Toulouse Olympique",
    "Sheffield": "Sheffield Eagles",
    "Widnes": "Widnes Vikings",
    "Halifax": "Halifax Panthers",
    "Batley": "Batley Bulldogs",
    "Dewsbury": "Dewsbury Rams",
    "York": "York Knights",
    "Newcastle": "Newcastle Thunder",
    "Whitehaven": "Whitehaven",
    "Barrow": "Barrow Raiders",
    # Championship
    "Oldham": "Oldham RLFC",
    "Midlands Hurricanes": "Midlands Hurricanes",
    "Workington Town": "Workington Town",
    "Doncaster": "Doncaster RLFC",
    "Hunslet": "Hunslet RLFC",
    "Keighley": "Keighley Cougars",
    "Goole Vikings": "Goole Vikings",
    "Swinton": "Swinton Lions",
    "Rochdale": "Rochdale Hornets",
    "North Wales Crusaders": "North Wales Crusaders",
}

TEAM_EXTERNAL_TO_INTERNAL = {v: k for k, v in TEAM_INTERNAL_TO_DISPLAY.items()}
for k in TEAM_INTERNAL_TO_DISPLAY:
    if k not in TEAM_EXTERNAL_TO_INTERNAL:
        TEAM_EXTERNAL_TO_INTERNAL[k] = k
# League Unlimited short forms (internal names used as display)
for internal in TEAM_INTERNAL_TO_DISPLAY:
    TEAM_EXTERNAL_TO_INTERNAL[internal] = internal
# League Unlimited Championship display names
TEAM_EXTERNAL_TO_INTERNAL["Salford RLFC"] = "Salford"
TEAM_EXTERNAL_TO_INTERNAL["Oldham RLFC"] = "Oldham"
TEAM_EXTERNAL_TO_INTERNAL["Doncaster RLFC"] = "Doncaster"
TEAM_EXTERNAL_TO_INTERNAL["Hunslet RLFC"] = "Hunslet"
TEAM_EXTERNAL_TO_INTERNAL["Keighley Cougars"] = "Keighley"
TEAM_EXTERNAL_TO_INTERNAL["Swinton Lions"] = "Swinton"
TEAM_EXTERNAL_TO_INTERNAL["Rochdale Hornets"] = "Rochdale"
TEAM_EXTERNAL_TO_INTERNAL["Whitehaven RLFC"] = "Whitehaven"
TEAM_EXTERNAL_TO_INTERNAL["Batley Bulldogs"] = "Batley"

VENUE_INTERNAL_TO_DISPLAY: dict[str, str] = {
    "Totally Wicked Stadium": "Totally Wicked Stadium",
    "DW Stadium": "DW Stadium",
    "Headingley": "Headingley Stadium",
    "Halliwell Jones Stadium": "Halliwell Jones Stadium",
    "MKM Stadium": "MKM Stadium",
    "Craven Park": "Craven Park (Hull KR)",
    "Stade Gilbert Brutus": "Stade Gilbert Brutus",
    "Wheldon Road": "Wheldon Road",
    "John Smith's Stadium": "John Smith's Stadium",
    "AJ Bell Stadium": "AJ Bell Stadium",
    "Belle Vue": "Belle Vue",
    "Leigh Sports Village": "Leigh Sports Village",
    "Plough Lane": "Plough Lane",
    # Championship
    "CorpAcq Stadium": "CorpAcq Stadium",
    "Avery Fields": "Avery Fields",
    "Northern Competitions Stadium": "Northern Competitions Stadium",
    "Eco-Power Stadium": "Eco-Power Stadium",
    "South Leeds Stadium": "South Leeds Stadium",
    "Cougar Park": "Cougar Park",
    "The Cherry Red Records Stadium": "The Cherry Red Records Stadium",
    "The Ortus REC": "The Ortus REC",
    "Morson Stadium": "Morson Stadium",
    "Crown Oil Arena": "Crown Oil Arena",
    "FLAIR Stadium": "FLAIR Stadium",
    "Steel City Stadium": "Steel City Stadium",
    "Fox's Biscuits Stadium": "Fox's Biscuits Stadium",
    "Crow Trees Ground": "Crow Trees Ground",
    "Derwent Park": "Derwent Park",
    "DCBL Stadium": "DCBL Stadium",
    "The Shay Stadium": "The Shay Stadium",
    "Victoria Pleasure Grounds": "Victoria Pleasure Grounds",
    "Boundary Park": "Boundary Park",
    "Bower Fold": "Bower Fold",
    "Stadiwm Eirias": "Stadiwm Eirias",
    "Kufflink Stadium": "Kufflink Stadium",
}

VENUE_EXTERNAL_TO_INTERNAL = {v: k for k, v in VENUE_INTERNAL_TO_DISPLAY.items()}
for k in VENUE_INTERNAL_TO_DISPLAY:
    if k not in VENUE_EXTERNAL_TO_INTERNAL:
        VENUE_EXTERNAL_TO_INTERNAL[k] = k
# League Unlimited 2026 venues
VENUE_EXTERNAL_TO_INTERNAL["The Brick Community Stadium"] = "DW Stadium"
VENUE_EXTERNAL_TO_INTERNAL["OneBore Stadium"] = "Wheldon Road"
VENUE_EXTERNAL_TO_INTERNAL["Odsal Stadium"] = "Odsal Stadium"
if "Odsal Stadium" not in VENUE_INTERNAL_TO_DISPLAY:
    VENUE_INTERNAL_TO_DISPLAY["Odsal Stadium"] = "Odsal Stadium"


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
    def __init__(self, player_index_path: Optional[Path] = None):
        self._index_path = player_index_path or (
            _PROJECT_ROOT / "model" / "output" / "uk-super-league" / "player_index.json"
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
