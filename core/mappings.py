"""
Map-first mapping layer for AFL Match Predictor.

All external data sources use different naming conventions. This module
provides canonical mappings so the rest of the pipeline uses consistent
internal identifiers. Map first, analyze next.

Mappings:
- Team names: fixturedownload, lineup files, match CSVs, display
- Venues: fixturedownload, match CSVs, display
- Players: lineup "FirstName LastName" <-> player_index "lastname_firstname_DDMMYYYY"
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

# Project root (parent of core/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# TEAM NAME MAPPING
# ---------------------------------------------------------------------------
# Internal names (match CSVs, lineup team_name): Sydney, Brisbane Lions, etc.
# External (fixturedownload): Sydney Swans, Brisbane Lions, GWS GIANTS, etc.

TEAM_INTERNAL_TO_DISPLAY: dict[str, str] = {
    "Adelaide": "Adelaide Crows",
    "Brisbane Lions": "Brisbane Lions",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong Cats",
    "Gold Coast": "Gold Coast SUNS",
    "Greater Western Sydney": "GWS GIANTS",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney Swans",
    "West Coast": "West Coast Eagles",
    "Western Bulldogs": "Western Bulldogs",
}

# Reverse: external/display -> internal (for fixture ingestion)
TEAM_EXTERNAL_TO_INTERNAL: dict[str, str] = {
    v: k for k, v in TEAM_INTERNAL_TO_DISPLAY.items()
}
# Add self-references for names that match
for k in TEAM_INTERNAL_TO_DISPLAY:
    if k not in TEAM_EXTERNAL_TO_INTERNAL:
        TEAM_EXTERNAL_TO_INTERNAL[k] = k


# ---------------------------------------------------------------------------
# VENUE MAPPING
# ---------------------------------------------------------------------------
# Internal (match CSV venue column): S.C.G., M.C.G., Docklands, etc.
# External (fixturedownload): SCG, MCG, Marvel Stadium, People First Stadium, ENGIE Stadium

VENUE_INTERNAL_TO_DISPLAY: dict[str, str] = {
    # Metro grounds
    "S.C.G.": "SCG",
    "M.C.G.": "MCG",
    "Docklands": "Marvel Stadium",
    "Carrara": "People First Stadium",
    "Sydney Showground": "ENGIE Stadium",
    "Gabba": "Gabba",
    "Adelaide Oval": "Adelaide Oval",
    "Perth Stadium": "Optus Stadium",
    "Kardinia Park": "GMHBA Stadium",
    # Regional / historic
    "Bellerive Oval": "Bellerive Oval",
    "Eureka Stadium": "Eureka Stadium",
    "Manuka Oval": "Manuka Oval",
    "Marrara Oval": "Marrara Oval",
    "Norwood Oval": "Norwood Oval",
    "Traeger Park": "Traeger Park",
    "York Park": "York Park",
    # 2026 new / renamed venues (internal name = display, no alias)
    "Barossa Park": "Barossa Park",
    "Ninja Stadium": "Ninja Stadium",
    "Hands Oval": "Hands Oval",
    "Corroboree Group Oval Manuka": "Corroboree Group Oval Manuka",
    "TIO Stadium": "TIO Stadium",
    "TIO Traeger Park": "TIO Traeger Park",
    "UTAS Stadium": "UTAS Stadium",
}

# External -> internal (for fixture ingestion)
VENUE_EXTERNAL_TO_INTERNAL: dict[str, str] = {
    "SCG": "S.C.G.",
    "MCG": "M.C.G.",
    "Marvel Stadium": "Docklands",
    "People First Stadium": "Carrara",
    "ENGIE Stadium": "Sydney Showground",
    "Gabba": "Gabba",
    "Adelaide Oval": "Adelaide Oval",
    "Optus Stadium": "Perth Stadium",
    "GMHBA Stadium": "Kardinia Park",
    "Bellerive Oval": "Bellerive Oval",
    "Manuka Oval": "Manuka Oval",
    "Norwood Oval": "Norwood Oval",
    # 2026 venues — these already come through correctly from fixturedownload
    "Barossa Park": "Barossa Park",
    "Ninja Stadium": "Ninja Stadium",
    "Hands Oval": "Hands Oval",
    "Corroboree Group Oval Manuka": "Corroboree Group Oval Manuka",
    "TIO Stadium": "TIO Stadium",
    "TIO Traeger Park": "TIO Traeger Park",
    "UTAS Stadium": "UTAS Stadium",
}
# Self-refs for any internal names not already mapped
for _k in VENUE_INTERNAL_TO_DISPLAY:
    if _k not in VENUE_EXTERNAL_TO_INTERNAL:
        VENUE_EXTERNAL_TO_INTERNAL[VENUE_INTERNAL_TO_DISPLAY.get(_k, _k)] = _k


# ---------------------------------------------------------------------------
# Mapper classes
# ---------------------------------------------------------------------------

class TeamNameMapper:
    """Map team names between internal, display, and external formats."""

    def to_internal(self, external: str) -> str:
        """Convert external/display name to internal (match CSV) format."""
        return TEAM_EXTERNAL_TO_INTERNAL.get(external.strip(), external.strip())

    def to_display(self, internal: str) -> str:
        """Convert internal name to display format."""
        return TEAM_INTERNAL_TO_DISPLAY.get(internal.strip(), internal.strip())

    def all_internal(self) -> list[str]:
        """Return all known internal team names."""
        return list(TEAM_INTERNAL_TO_DISPLAY.keys())


class VenueMapper:
    """Map venue names between internal (match CSV) and external formats."""

    def to_internal(self, external: str) -> str:
        """Convert external name to internal (match CSV) format."""
        return VENUE_EXTERNAL_TO_INTERNAL.get(external.strip(), external.strip())

    def to_display(self, internal: str) -> str:
        """Convert internal name to display format."""
        return VENUE_INTERNAL_TO_DISPLAY.get(internal.strip(), internal.strip())

    def all_internal(self) -> list[str]:
        """Return all known internal venue names."""
        return list(VENUE_INTERNAL_TO_DISPLAY.keys())


class PlayerMapper:
    """
    Map between lineup format ("FirstName LastName") and player_index format
    ("lastname_firstname_DDMMYYYY").

    Built from player_index.json and optionally player performance filenames.
    """

    def __init__(self, player_index_path: Optional[Path] = None):
        self._player_index_path = player_index_path or (
            _PROJECT_ROOT / "model" / "output" / "player_index.json"
        )
        self._id_to_display: dict[str, str] = {}
        self._display_to_ids: dict[str, list[str]] = {}
        self._load()

    def _parse_player_id(self, player_id: str) -> tuple[str, str, str]:
        """
        Parse player_id (lastname_firstname_DDMMYYYY) into (lastname, firstname, dob).
        Handles compound last names (e.g. zerk-thatcher_brandon_25081998).
        """
        parts = player_id.split("_")
        if len(parts) < 3:
            return ("", "", "")
        dob = parts[-1]
        if not (len(dob) == 8 and dob.isdigit()):
            return ("", "", "")
        firstname = parts[-2]
        lastname = "_".join(parts[:-2])
        return (lastname, firstname, dob)

    def _to_display_name(self, lastname: str, firstname: str) -> str:
        """Format as 'FirstName LastName' with proper capitalization."""
        def cap(s: str) -> str:
            return s.replace("_", "-").replace("-", " ").title().replace(" ", "-")
        return f"{cap(firstname)} {cap(lastname)}"

    def _load(self) -> None:
        """Build id<->display mappings from player_index.json."""
        if not self._player_index_path.exists():
            return
        with open(self._player_index_path, "r") as f:
            index = json.load(f)
        for player_id in index.keys():
            if player_id == "unknown":
                continue
            lastname, firstname, _ = self._parse_player_id(player_id)
            if not lastname or not firstname:
                continue
            display = self._to_display_name(lastname, firstname)
            self._id_to_display[player_id] = display
            key = display.lower().strip()
            if key not in self._display_to_ids:
                self._display_to_ids[key] = []
            self._display_to_ids[key].append(player_id)

        # For duplicates (e.g. Gary Ablett snr/jnr), prefer most recent (latest DOB)
        for key in self._display_to_ids:
            ids = self._display_to_ids[key]
            if len(ids) > 1:
                ids.sort(key=lambda x: x.split("_")[-1], reverse=True)

    def to_display(self, player_id: str) -> str:
        """Convert player_id to 'FirstName LastName' display format."""
        return self._id_to_display.get(player_id, player_id)

    def _normalize_display_for_lookup(self, name: str) -> str:
        """Normalize 'Jordan de Goey' -> 'Jordan Goey' (player_id uses primary surname)."""
        skip = {"de", "van", "von", "la", "le"}
        parts = name.lower().strip().split()
        filtered = [p for p in parts if p not in skip]
        return " ".join(filtered) if filtered else name.lower()

    def to_player_id(self, display_name: str) -> Optional[str]:
        """
        Convert 'FirstName LastName' to player_id.
        Returns first match; for duplicates, returns most recent (by DOB).
        Handles compound surnames: "Jordan de Goey" -> goey_jordan_*
        """
        key = display_name.strip().lower()
        ids = self._display_to_ids.get(key)
        if ids:
            return ids[0]
        # Compound surnames: "Jordan de Goey" -> "Jordan Goey" -> goey_jordan_*
        normalized = self._normalize_display_for_lookup(display_name)
        ids = self._display_to_ids.get(normalized)
        if ids:
            return ids[0]
        # Try "LastWord FirstWord" for reversed format
        parts = key.split()
        if len(parts) >= 2:
            alt_key = f"{parts[-1]} {parts[0]}"
            ids = self._display_to_ids.get(alt_key)
            if ids:
                return ids[0]
            # Fallback: exact match on any display name
            for pid, disp in self._id_to_display.items():
                if disp.lower() == key or disp.lower() == normalized:
                    return pid
        return None

    def to_player_ids(self, display_names: list[str]) -> list[str]:
        """Convert list of display names to player_ids. Unknown -> skip or use 'unknown'."""
        result = []
        for name in display_names:
            pid = self.to_player_id(name)
            result.append(pid if pid else "unknown")
        return result

    def all_player_ids(self) -> list[str]:
        """Return all known player_ids (excluding 'unknown')."""
        return [k for k in self._id_to_display.keys() if k != "unknown"]

    def search_by_name(self, query: str, limit: int = 20) -> list[tuple[str, str]]:
        """Search players by name; returns [(player_id, display_name), ...]."""
        q = query.lower().strip()
        matches = []
        for pid, disp in self._id_to_display.items():
            if q in disp.lower():
                matches.append((pid, disp))
        matches.sort(key=lambda x: x[1].lower())
        return matches[:limit]

    def lineup_string_to_player_ids(self, players_str: str) -> list[str]:
        """
        Parse lineup 'players' column (semicolon-separated 'FirstName LastName')
        into list of player_ids. Unknown names -> 'unknown'.
        """
        if not players_str or (isinstance(players_str, float) and str(players_str) == "nan"):
            return []
        names = [n.strip() for n in str(players_str).split(";") if n.strip()]
        return self.to_player_ids(names)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_all_mappings(
    player_index_path: Optional[Path] = None,
) -> tuple[TeamNameMapper, VenueMapper, PlayerMapper]:
    """Load all mappers. Map-first entry point."""
    teams = TeamNameMapper()
    venues = VenueMapper()
    players = PlayerMapper(player_index_path)
    return teams, venues, players
