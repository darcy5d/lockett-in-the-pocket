"""
Rugby League Data Service — parameterised by competition.

Teams, grounds, lineups, fixture for any competition (NRL, NSW Cup, QLD Cup, UK).
Uses competition_config and per-competition mapper modules.
"""

from __future__ import annotations

import functools
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from core.competition_config import (
    get_competition,
    get_competition_slugs,
    slug_matches_file,
    slug_matches_lineup_file,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "nrl_data" / "data"


def _load_mappers(competition_id: str, player_index_path: Optional[Path] = None):
    """Load TeamMapper, VenueMapper, PlayerMapper from competition's mapper_module."""
    cfg = get_competition(competition_id)
    if not cfg:
        raise ValueError(f"Unknown competition: {competition_id}")
    mod_name = cfg.get("mapper_module", "core.nrl_mappings")
    output_dir = Path(cfg.get("output_dir", str(_PROJECT_ROOT / "model" / "output" / competition_id)))
    pidx = player_index_path or (output_dir / "player_index.json")

    mod = importlib.import_module(mod_name)
    TeamMapper = getattr(mod, "TeamMapper", getattr(mod, "NRLTeamMapper", None))
    VenueMapper = getattr(mod, "VenueMapper", getattr(mod, "NRLVenueMapper", None))
    PlayerMapper = getattr(mod, "PlayerMapper", getattr(mod, "NRLPlayerMapper", None))
    if not all([TeamMapper, VenueMapper, PlayerMapper]):
        raise ValueError(f"Mapper module {mod_name} missing TeamMapper/VenueMapper/PlayerMapper")

    return (
        TeamMapper(),
        VenueMapper(),
        PlayerMapper(pidx),
    )


class RugbyDataService:
    """
    Rugby League data service parameterised by competition_id.
    """

    def __init__(
        self,
        competition_id: str,
        data_dir: Optional[Path] = None,
        player_index_path: Optional[Path] = None,
    ):
        self._competition_id = competition_id
        cfg = get_competition(competition_id)
        if not cfg:
            raise ValueError(f"Unknown competition: {competition_id}")

        self._data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self._lineup_dir = self._data_dir / "lineups"
        self._match_dir = self._data_dir / "matches"
        self._slugs = get_competition_slugs(competition_id)

        team_m, venue_m, player_m = _load_mappers(
            competition_id,
            player_index_path=Path(player_index_path) if player_index_path else None,
        )
        self._team_mapper = team_m
        self._venue_mapper = venue_m
        self._player_mapper = player_m

        self._fixture_filename = cfg.get("fixture_filename", "matches_2026.csv")
        self._fixture_path = self._match_dir / self._fixture_filename
        self._fixture_meta_path = self._match_dir / f"fixture_2026_meta_{competition_id}.json"
        self._teams: list[str] = self._load_teams()
        self._grounds: list[str] = self._load_grounds()
        self._fixture_df: Optional[pd.DataFrame] = None
        self._lineup_df: Optional[pd.DataFrame] = None

    def _slug_matches_match(self, name: str) -> bool:
        return slug_matches_file(name, self._slugs)

    def _slug_matches_lineup(self, name: str) -> bool:
        return slug_matches_lineup_file(name, self._slugs)

    def _load_teams(self) -> list[str]:
        teams = set()
        for path in self._match_dir.glob("matches_*.csv"):
            if "2026" in path.name or not self._slug_matches_match(path.name):
                continue
            try:
                df = pd.read_csv(path, usecols=["team_1_team_name", "team_2_team_name"])
                teams.update(df["team_1_team_name"].dropna().unique())
                teams.update(df["team_2_team_name"].dropna().unique())
            except Exception:
                pass
        return sorted(teams)

    def _load_grounds(self) -> list[str]:
        venues = set()
        for path in self._match_dir.glob("matches_*.csv"):
            if "2026" in path.name or not self._slug_matches_match(path.name):
                continue
            try:
                df = pd.read_csv(path, usecols=["venue"])
                venues.update(df["venue"].dropna().unique())
            except Exception:
                pass
        return sorted(venues)

    def get_teams(self) -> list[str]:
        return self._teams

    def get_team_display_names(self) -> list[dict]:
        return [
            {"value": t, "label": self._team_mapper.to_display(t)}
            for t in self._teams
        ]

    def get_grounds(self) -> list[str]:
        return self._grounds

    def get_grounds_display(self) -> list[dict]:
        return [
            {"value": g, "label": self._venue_mapper.to_display(g)}
            for g in self._grounds
        ]

    def _load_lineup_df(self) -> pd.DataFrame:
        if self._lineup_df is None:
            files = [
                f for f in sorted(self._lineup_dir.glob("lineup_details_*.csv"))
                if self._slug_matches_lineup(f.name)
            ]
            if not files:
                self._lineup_df = pd.DataFrame()
            else:
                self._lineup_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        return self._lineup_df

    def _get_fixture_slug(self) -> str:
        """Slug for 2026 fixture match_ids (matches lineup scraper)."""
        return {
            "nrl": "nrl",
            "uk-super-league": "super-league-uk",
            "nsw-cup": "nsw-cup",
            "qld-cup": "qld-cup",
            "uk-championship": "championship-uk",
        }.get(self._competition_id, self._competition_id)

    @functools.lru_cache(maxsize=64)
    def get_last_lineup(self, team_key: str) -> list[dict]:
        df = self._load_lineup_df()
        if df.empty or "team" not in df.columns:
            return []
        team_df = df[df["team"].astype(str).str.strip() == team_key.strip()]
        if team_df.empty:
            return []
        match_ids = team_df["match_id"].unique()
        matches_df = pd.DataFrame()
        for path in self._match_dir.glob("matches_*.csv"):
            if not self._slug_matches_match(path.name):
                continue
            try:
                m = pd.read_csv(path, usecols=["match_id", "year", "round_num"], dtype=str)
                matches_df = pd.concat([matches_df, m], ignore_index=True)
            except Exception:
                pass
        # Include 2026 fixture so scraped lineups (nrl_2026_r1_m0 etc) are found
        if self._fixture_path.exists():
            try:
                fix = pd.read_csv(self._fixture_path, dtype=str)
                if not fix.empty and "round_num" in fix.columns:
                    slug = self._get_fixture_slug()
                    year = "2026"
                    idx_per_round: dict[str, int] = {}
                    rows = []
                    for _, r in fix.iterrows():
                        rn = str(r.get("round_num", "")).strip()
                        if rn not in idx_per_round:
                            idx_per_round[rn] = 0
                        mid = f"{slug}_{year}_r{rn}_m{idx_per_round[rn]}"
                        idx_per_round[rn] += 1
                        rows.append({"match_id": mid, "year": year, "round_num": rn})
                    if rows:
                        matches_df = pd.concat([matches_df, pd.DataFrame(rows)], ignore_index=True)
            except Exception:
                pass
        if matches_df.empty:
            return []
        team_matches = matches_df[matches_df["match_id"].isin(match_ids)]
        if team_matches.empty:
            return []
        latest = team_matches.sort_values(["year", "round_num"], ascending=[False, False]).iloc[0]
        mid = latest["match_id"]
        lineup = team_df[team_df["match_id"] == mid]
        return [
            {
                "display_name": str(r.get("player_name", "") or self._player_mapper.to_display(str(r["player_id"]))),
                "player_id": str(r["player_id"]),
            }
            for _, r in lineup.iterrows()
        ]

    @functools.lru_cache(maxsize=64)
    def get_season_players(self, team_key: str) -> list[dict]:
        df = self._load_lineup_df()
        if df.empty:
            return []
        team_df = df[df["team"].astype(str).str.strip() == team_key.strip()]
        seen = set()
        players = []
        for _, row in team_df.iterrows():
            pid = str(row.get("player_id", ""))
            if pid and pid not in seen:
                seen.add(pid)
                name = str(row.get("player_name", "")) or self._player_mapper.to_display(pid)
                players.append({"display_name": name, "player_id": pid})
        return sorted(players, key=lambda p: p["display_name"].lower())

    def _load_fixture(self) -> pd.DataFrame:
        if self._fixture_df is None:
            if not self._fixture_path.exists():
                self._fixture_df = pd.DataFrame()
            else:
                self._fixture_df = pd.read_csv(self._fixture_path, dtype=str)
        return self._fixture_df

    def _invalidate_fixture_cache(self) -> None:
        self._fixture_df = None

    def _round_sort_key(self, r: str) -> tuple[int, str]:
        try:
            return (0, str(int(r)).zfill(3))
        except ValueError:
            return (1, str(r))

    def get_fixture_last_updated(self) -> Optional[str]:
        if not self._fixture_meta_path.exists():
            return None
        try:
            with open(self._fixture_meta_path) as f:
                meta = json.load(f)
            return meta.get("last_fetched")
        except Exception:
            return None

    def save_fixture_meta(self) -> None:
        meta = {"last_fetched": datetime.now(timezone.utc).isoformat()}
        self._fixture_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._fixture_meta_path, "w") as f:
            json.dump(meta, f)

    def get_fixture_rounds(self) -> list[str]:
        df = self._load_fixture()
        if df.empty or "round_num" not in df.columns:
            return []
        rounds = df["round_num"].dropna().unique().tolist()
        return sorted(rounds, key=self._round_sort_key)

    def get_fixture_round(self, round_num: str) -> list[dict]:
        df = self._load_fixture()
        if df.empty:
            return []
        mask = df["round_num"].astype(str).str.strip() == str(round_num).strip()
        subset = df[mask]
        matches = []
        for _, row in subset.iterrows():
            date_str = str(row.get("date", "")).strip()
            venue_internal = str(row.get("venue", "")).strip()
            home = str(row.get("team_1_team_name", "")).strip()
            away = str(row.get("team_2_team_name", "")).strip()
            matches.append({
                "round_num": round_num,
                "date": date_str,
                "time_confirmed": bool(date_str and " " in date_str),
                "venue": venue_internal,
                "venue_display": self._venue_mapper.to_display(venue_internal),
                "home_team_key": home,
                "home_team_display": self._team_mapper.to_display(home),
                "away_team_key": away,
                "away_team_display": self._team_mapper.to_display(away),
            })
        return matches

    def search_players(self, query: str, limit: int = 30) -> list[dict]:
        df = self._load_lineup_df()
        if df.empty or "player_name" not in df.columns:
            return []
        q = query.lower().strip()
        if len(q) < 2:
            return []
        mask = df["player_name"].astype(str).str.lower().str.contains(q, na=False)
        subset = df[mask][["player_id", "player_name"]].drop_duplicates()
        results = []
        seen = set()
        for _, row in subset.head(limit * 2).iterrows():
            pid = str(row["player_id"])
            name = str(row.get("player_name", "")) or self._player_mapper.to_display(pid)
            if pid not in seen:
                seen.add(pid)
                results.append({"player_id": pid, "display_name": name})
                if len(results) >= limit:
                    break
        return sorted(results, key=lambda x: x["display_name"].lower())
