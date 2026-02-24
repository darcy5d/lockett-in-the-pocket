"""
Shared data service — single source of truth for teams, grounds, lineups, players.

All I/O goes through here. Uses core.mappings for all name/ID conversions.
League-aware: accepts an optional data_dir for multi-league support later.
"""

from __future__ import annotations

import functools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from core.mappings import PlayerMapper, TeamNameMapper, VenueMapper

# Placeholder time used by fixturedownload.com for unconfirmed timeslots (UTC)
_PLACEHOLDER_TIME = "02:00"

# Default data root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "afl_data" / "data"


class DataService:
    """
    League-aware data service. Instantiate once per app; results are cached.

    Parameters
    ----------
    data_dir : Path, optional
        Root of the league's data directory (e.g. afl_data/data).
        Defaults to afl_data/data.
    player_index_path : Path, optional
        Path to player_index.json. Defaults to model/output/player_index.json.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        player_index_path: Optional[Path] = None,
    ):
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._lineup_dir = self._data_dir / "lineups"
        self._match_dir = self._data_dir / "matches"
        self._store = None
        if os.environ.get("USE_AFL_SQLITE") == "1" and (self._data_dir / "afl.db").exists():
            from core.afl_data_store import AFLDataStore
            self._store = AFLDataStore.from_path(self._data_dir)
        self._player_mapper = PlayerMapper(player_index_path)
        self._venue_mapper = VenueMapper()
        self._team_mapper = TeamNameMapper()
        self._fixture_2026_path = self._match_dir / "matches_2026.csv"
        self._fixture_meta_path = self._match_dir / "fixture_2026_meta.json"
        # Eager-load teams and grounds (cheap; used on every page load)
        self._teams: list[str] = self._load_teams()
        self._grounds: list[str] = self._load_grounds()
        # Lazy-load fixture — only when first requested
        self._fixture_df: Optional[pd.DataFrame] = None
        self._lineup_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------

    def _load_teams(self) -> list[str]:
        if self._store is not None:
            teams = self._store.get_teams_from_lineups()
            if teams:
                return teams
            # Fall back to CSV if DB has no lineups
        files = sorted(self._lineup_dir.glob("team_lineups_*.csv"))
        teams = []
        for f in files:
            raw = f.stem.replace("team_lineups_", "")
            teams.append(raw)
        return sorted(teams)

    def get_teams(self) -> list[str]:
        """Return sorted list of internal team keys (e.g. 'adelaide', 'brisbane_lions')."""
        return self._teams

    def get_team_display_names(self) -> list[dict]:
        """Return [{value, label}] for dropdowns, using title-cased keys as display."""
        return [
            {"value": t, "label": t.replace("_", " ").title()}
            for t in self._teams
        ]

    # ------------------------------------------------------------------
    # Grounds / Venues
    # ------------------------------------------------------------------

    def _load_grounds(self) -> list[str]:
        if self._store is not None:
            grounds = self._store.get_grounds_from_matches()
            if grounds:
                return grounds
            # Fall back to CSV if DB has no matches
        files = sorted(self._match_dir.glob("matches_*.csv"))
        venues: set[str] = set()
        for f in files:
            try:
                df = pd.read_csv(f, usecols=["venue"])
                venues.update(df["venue"].dropna().unique())
            except Exception:
                pass
        return sorted(venues)

    def get_grounds(self) -> list[str]:
        """Return sorted list of internal venue names (e.g. 'S.C.G.', 'M.C.G.')."""
        return self._grounds

    def get_grounds_display(self) -> list[dict]:
        """Return [{value, label}] with display-friendly venue names."""
        return [
            {
                "value": g,
                "label": self._venue_mapper.to_display(g),
            }
            for g in self._grounds
        ]

    # ------------------------------------------------------------------
    # Lineups
    # ------------------------------------------------------------------

    def _lineup_file(self, team_key: str) -> Path:
        return self._lineup_dir / f"team_lineups_{team_key}.csv"

    def _get_lineup_df_for_team(self, team_key: str) -> pd.DataFrame:
        """Lineup rows for this team (from store or CSV)."""
        if self._store is not None:
            if self._lineup_df is None:
                self._lineup_df = self._store.load_lineups()
            if self._lineup_df.empty:
                return pd.DataFrame()
            norm = self._lineup_df["team_name"].astype(str).str.replace(" ", "_").str.lower()
            return self._lineup_df[norm == team_key].copy()
        path = self._lineup_file(team_key)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    @functools.lru_cache(maxsize=64)
    def get_last_lineup(self, team_key: str) -> list[dict]:
        """
        Return the most recent known lineup for a team.

        Returns a list of {display_name, player_id} dicts.
        player_id may be 'unknown' if not in player_index.
        """
        df = self._get_lineup_df_for_team(team_key)
        if df.empty:
            return []
        if "year" not in df.columns:
            return []
        latest_year = df["year"].max()
        df_year = df[df["year"] == latest_year].copy()
        try:
            df_year["_round_num"] = pd.to_numeric(df_year["round_num"], errors="coerce")
            latest_round = df_year["_round_num"].max()
            row = df_year[df_year["_round_num"] == latest_round].iloc[-1]
        except Exception:
            row = df_year.iloc[-1]
        players_str = row.get("players", "")
        if pd.isna(players_str) or not players_str:
            return []
        names = [n.strip() for n in str(players_str).split(";") if n.strip()]
        return [
            {
                "display_name": name,
                "player_id": self._player_mapper.to_player_id(name) or "unknown",
            }
            for name in names
        ]

    @functools.lru_cache(maxsize=64)
    def get_season_players(self, team_key: str) -> list[dict]:
        """
        Return all unique players for this team in the current year.

        Returns a list of {display_name, player_id} dicts sorted by display name.
        """
        df = self._get_lineup_df_for_team(team_key)
        if df.empty:
            return []
        year = datetime.now().year
        df_year = df[df["year"] == year]
        seen: set[str] = set()
        players: list[dict] = []
        for _, row in df_year.iterrows():
            players_str = row.get("players", "")
            if pd.isna(players_str) or not players_str:
                continue
            for name in str(players_str).split(";"):
                name = name.strip()
                if name and name not in seen:
                    seen.add(name)
                    pid = self._player_mapper.to_player_id(name) or "unknown"
                    players.append({"display_name": name, "player_id": pid})
        return sorted(players, key=lambda p: p["display_name"].lower())

    # ------------------------------------------------------------------
    # 2026 Fixture
    # ------------------------------------------------------------------

    def _load_fixture(self) -> pd.DataFrame:
        """Load and cache the 2026 fixture CSV."""
        if self._fixture_df is None:
            if not self._fixture_2026_path.exists():
                self._fixture_df = pd.DataFrame()
            else:
                self._fixture_df = pd.read_csv(self._fixture_2026_path, dtype=str)
        return self._fixture_df

    def _invalidate_fixture_cache(self) -> None:
        """Force fixture to reload on next access (call after refresh)."""
        self._fixture_df = None

    def _round_sort_key(self, r: str) -> tuple[int, str]:
        """Sort key: Opening Round first, then numeric rounds."""
        if r == "Opening Round":
            return (0, "")
        try:
            return (1, str(int(r)).zfill(3))
        except ValueError:
            return (2, r)

    def _is_time_confirmed(self, date_str: str) -> bool:
        """Return False if the time portion is the fixturedownload placeholder (02:00)."""
        if not date_str or pd.isna(date_str):
            return False
        time_part = str(date_str).strip().split(" ")[-1] if " " in str(date_str) else ""
        return time_part != _PLACEHOLDER_TIME

    def get_fixture_last_updated(self) -> Optional[str]:
        """Return ISO timestamp string when fixture was last fetched, or None."""
        if not self._fixture_meta_path.exists():
            return None
        try:
            with open(self._fixture_meta_path) as f:
                meta = json.load(f)
            return meta.get("last_fetched")
        except Exception:
            return None

    def save_fixture_meta(self) -> None:
        """Write fixture metadata after a fresh fetch."""
        meta = {"last_fetched": datetime.now(timezone.utc).isoformat()}
        with open(self._fixture_meta_path, "w") as f:
            json.dump(meta, f)

    def get_fixture_rounds(self) -> list[str]:
        """
        Return sorted round labels from matches_2026.csv.
        Order: Opening Round first, then 1, 2, … 24.
        """
        df = self._load_fixture()
        if df.empty or "round_num" not in df.columns:
            return []
        rounds = df["round_num"].dropna().unique().tolist()
        return sorted(rounds, key=self._round_sort_key)

    def get_fixture_round(self, round_num: str) -> list[dict]:
        """
        Return all matches in the given round, structured for the UI.

        Each match dict:
            round_num, date, time_confirmed, venue, venue_display,
            home_team_key, home_team_display,
            away_team_key, away_team_display
        """
        df = self._load_fixture()
        if df.empty:
            return []
        mask = df["round_num"].str.strip() == round_num.strip()
        subset = df[mask].copy()
        matches = []
        for _, row in subset.iterrows():
            date_str = str(row.get("date", "")).strip()
            venue_internal = str(row.get("venue", "")).strip()
            home_display = str(row.get("team_1_team_name", "")).strip()
            away_display = str(row.get("team_2_team_name", "")).strip()

            # Convert internal team name → lineup file key (lowercase, underscored)
            home_key = home_display.lower().replace(" ", "_")
            away_key = away_display.lower().replace(" ", "_")

            matches.append({
                "round_num": round_num,
                "date": date_str,
                "time_confirmed": self._is_time_confirmed(date_str),
                "venue": venue_internal,
                "venue_display": self._venue_mapper.to_display(venue_internal),
                "home_team_key": home_key,
                "home_team_display": home_display,
                "away_team_key": away_key,
                "away_team_display": away_display,
            })
        return matches

    # ------------------------------------------------------------------
    # Player search
    # ------------------------------------------------------------------

    def search_players(self, query: str, limit: int = 30) -> list[dict]:
        """
        Search all known players by display name fragment.

        Returns [{player_id, display_name}] sorted alphabetically.
        """
        results = self._player_mapper.search_by_name(query, limit=limit)
        return [{"player_id": pid, "display_name": disp} for pid, disp in results]

    def player_id_to_display(self, player_id: str) -> str:
        return self._player_mapper.to_display(player_id)

    def display_to_player_id(self, display_name: str) -> Optional[str]:
        return self._player_mapper.to_player_id(display_name)

    def lineup_string_to_player_ids(self, players_str: str) -> list[str]:
        return self._player_mapper.lineup_string_to_player_ids(players_str)


# Module-level default instance (AFL, lazy-loaded)
_default: Optional[DataService] = None


def get_default_service() -> DataService:
    global _default
    if _default is None:
        _default = DataService()
    return _default
