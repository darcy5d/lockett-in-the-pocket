"""
Data access layer for AFL data: read from either CSV files (afl_data/data) or SQLite (afl.db).

Provides the same interface as file-based loading so FeatureEngine, DataService, and
training can use either source via a single abstraction.

Usage:
  store = AFLDataStore.from_path(afl_data/data)  # uses afl.db if present, else CSV
  store = AFLDataStore.from_sqlite(afl_data/data/afl.db)
  df = store.load_matches(2020, 2025)
  lineup_df = store.load_lineups()
  player_logs = store.load_player_stats()
"""

from __future__ import annotations

import glob
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Must match core/feature_engine.STAT_FIELDS for player_games
STAT_FIELDS = [
    "kicks", "marks", "handballs", "goals", "behinds",
    "hit_outs", "tackles", "rebound_50s", "inside_50s", "clearances",
    "clangers", "free_kicks_for", "free_kicks_against",
    "contested_possessions", "uncontested_possessions", "contested_marks",
    "marks_inside_50", "one_percenters", "bounces", "goal_assist",
    "percentage_of_game_played",
]

DB_FILENAME = "afl.db"


def _schema_sql() -> str:
    return """
CREATE TABLE IF NOT EXISTS matches (
    year INTEGER NOT NULL,
    round_num TEXT NOT NULL,
    team_1_team_name TEXT NOT NULL,
    team_2_team_name TEXT NOT NULL,
    date TEXT,
    venue TEXT,
    attendance TEXT,
    team_1_q1_goals INTEGER, team_1_q1_behinds INTEGER,
    team_1_q2_goals INTEGER, team_1_q2_behinds INTEGER,
    team_1_q3_goals INTEGER, team_1_q3_behinds INTEGER,
    team_1_final_goals INTEGER, team_1_final_behinds INTEGER,
    team_2_q1_goals INTEGER, team_2_q1_behinds INTEGER,
    team_2_q2_goals INTEGER, team_2_q2_behinds INTEGER,
    team_2_q3_goals INTEGER, team_2_q3_behinds INTEGER,
    team_2_final_goals INTEGER, team_2_final_behinds INTEGER,
    PRIMARY KEY (year, round_num, team_1_team_name, team_2_team_name)
);

CREATE TABLE IF NOT EXISTS lineups (
    year INTEGER NOT NULL,
    round_num TEXT NOT NULL,
    team_name TEXT NOT NULL,
    date TEXT,
    players TEXT NOT NULL,
    PRIMARY KEY (year, round_num, team_name)
);

CREATE TABLE IF NOT EXISTS player_games (
    player_id TEXT NOT NULL,
    year INTEGER NOT NULL,
    round_num TEXT,
    team TEXT,
    opponent TEXT,
    games_played INTEGER,
    result TEXT,
    jersey_num INTEGER,
    kicks REAL, marks REAL, handballs REAL, disposals REAL,
    goals REAL, behinds REAL, hit_outs REAL, tackles REAL,
    rebound_50s REAL, inside_50s REAL, clearances REAL, clangers REAL,
    free_kicks_for REAL, free_kicks_against REAL, brownlow_votes REAL,
    contested_possessions REAL, uncontested_possessions REAL,
    contested_marks REAL, marks_inside_50 REAL, one_percenters REAL,
    bounces REAL, goal_assist REAL, percentage_of_game_played REAL,
    PRIMARY KEY (player_id, year, games_played)
);
CREATE INDEX IF NOT EXISTS idx_player_games_id ON player_games(player_id);
"""


class AFLDataStore:
    """Single interface for loading matches, lineups, and player stats from CSV or SQLite."""

    def __init__(self, data_dir: Path, db_path: Optional[Path] = None):
        self._data_dir = Path(data_dir)
        self._db_path = Path(db_path) if db_path else self._data_dir / DB_FILENAME
        self._use_sqlite = self._db_path.exists()

    @classmethod
    def from_path(cls, data_dir: Path) -> "AFLDataStore":
        """Use data_dir; if afl.db exists inside it, read from SQLite, else CSV."""
        return cls(data_dir=data_dir)

    @classmethod
    def from_sqlite(cls, db_path: Path) -> "AFLDataStore":
        """Force use of SQLite at db_path. data_dir is parent of db_path."""
        return cls(data_dir=db_path.parent, db_path=db_path)

    def load_matches(self, year_from: int = 1897, year_to: int = 2025) -> pd.DataFrame:
        """Load match DataFrame for the given year range (same columns as matches_YYYY.csv)."""
        if self._use_sqlite:
            return self._load_matches_sqlite(year_from, year_to)
        return self._load_matches_csv(year_from, year_to)

    def _load_matches_sqlite(self, year_from: int, year_to: int) -> pd.DataFrame:
        conn = sqlite3.connect(self._db_path)
        try:
            df = pd.read_sql_query(
                "SELECT * FROM matches WHERE year >= ? AND year <= ? ORDER BY year, date",
                conn,
                params=(year_from, year_to),
            )
        finally:
            conn.close()
        if df.empty:
            return df
        df["round_num"] = df["round_num"].astype(str)
        for c in ["team_1_final_goals", "team_1_final_behinds", "team_2_final_goals", "team_2_final_behinds"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["team_1_final_goals", "team_2_final_goals"], how="all")
        df["score1"] = df["team_1_final_goals"] * 6 + df["team_1_final_behinds"]
        df["score2"] = df["team_2_final_goals"] * 6 + df["team_2_final_behinds"]
        return df

    def _load_matches_csv(self, year_from: int, year_to: int) -> pd.DataFrame:
        match_dir = self._data_dir / "matches"
        dfs = []
        for y in range(year_from, year_to + 1):
            p = match_dir / f"matches_{y}.csv"
            if p.exists():
                try:
                    dfs.append(pd.read_csv(p))
                except Exception:
                    pass
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        for c in ["team_1_final_goals", "team_2_final_goals"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["team_1_final_goals", "team_2_final_goals"], how="all")
        df["round_num"] = df["round_num"].astype(str)
        df["score1"] = df["team_1_final_goals"] * 6 + df["team_1_final_behinds"]
        df["score2"] = df["team_2_final_goals"] * 6 + df["team_2_final_behinds"]
        return df

    def load_lineups(self) -> pd.DataFrame:
        """Load lineup DataFrame (year, date, round_num, team_name, players)."""
        if self._use_sqlite:
            return self._load_lineups_sqlite()
        return self._load_lineups_csv()

    def _load_lineups_sqlite(self) -> pd.DataFrame:
        conn = sqlite3.connect(self._db_path)
        try:
            df = pd.read_sql_query("SELECT year, date, round_num, team_name, players FROM lineups ORDER BY year, date", conn)
        finally:
            conn.close()
        if not df.empty:
            df["round_num"] = df["round_num"].astype(str)
        return df

    def _load_lineups_csv(self) -> pd.DataFrame:
        lineup_dir = self._data_dir / "lineups"
        files = sorted(lineup_dir.glob("team_lineups_*.csv"))
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df["round_num"] = df["round_num"].astype(str)
        return df

    def load_player_stats(self) -> dict[str, list[dict]]:
        """
        Load per-player game logs: {player_id: [{kicks, marks, ..., year}, ...]}.
        Same shape as FeatureEngine._player_game_log.
        """
        if self._use_sqlite:
            return self._load_player_stats_sqlite()
        return self._load_player_stats_csv()

    def _load_player_stats_sqlite(self) -> dict[str, list[dict]]:
        conn = sqlite3.connect(self._db_path)
        out: dict[str, list[dict]] = {}
        try:
            df = pd.read_sql_query("SELECT * FROM player_games ORDER BY player_id, year, games_played", conn)
        finally:
            conn.close()
        if df.empty:
            return out
        for pid, group in df.groupby("player_id"):
            games = []
            for _, row in group.iterrows():
                g = {"year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0}
                for s in STAT_FIELDS:
                    v = row.get(s, 0)
                    g[s] = float(v) if pd.notna(v) else 0.0
                games.append(g)
            out[str(pid)] = games
        return out

    def _load_player_stats_csv(self) -> dict[str, list[dict]]:
        from core.feature_engine import STAT_FIELDS as FE_STAT_FIELDS

        player_dir = self._data_dir / "players"
        files = glob.glob(str(player_dir / "*_performance_details.csv"))
        out: dict[str, list[dict]] = {}
        for f in files:
            pid = Path(f).stem.replace("_performance_details", "")
            try:
                df = pd.read_csv(f)
                games = []
                for _, row in df.iterrows():
                    g = {"year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0}
                    for s in FE_STAT_FIELDS:
                        v = row.get(s, 0)
                        g[s] = float(v) if pd.notna(v) else 0.0
                    games.append(g)
                if games:
                    out[pid] = games
            except Exception:
                pass
        return out

    def get_teams_from_lineups(self) -> list[str]:
        """Return sorted list of team keys (e.g. 'adelaide', 'collingwood') from lineups."""
        if self._use_sqlite:
            conn = sqlite3.connect(self._db_path)
            try:
                df = pd.read_sql_query("SELECT DISTINCT team_name FROM lineups", conn)
                names = df["team_name"].dropna().unique().tolist()
            finally:
                conn.close()
            return sorted(str(n).replace(" ", "_").lower() for n in names)
        lineup_dir = self._data_dir / "lineups"
        files = sorted(lineup_dir.glob("team_lineups_*.csv"))
        return sorted(f.stem.replace("team_lineups_", "") for f in files)

    def get_grounds_from_matches(self) -> list[str]:
        """Return sorted list of venue names from matches."""
        df = self.load_matches(1897, 2030)
        if df.empty or "venue" not in df.columns:
            return []
        return sorted(df["venue"].dropna().unique().tolist())

    @property
    def use_sqlite(self) -> bool:
        return self._use_sqlite

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def db_path(self) -> Optional[Path]:
        return self._db_path if self._use_sqlite else None


def init_sqlite_schema(db_path: Path) -> None:
    """Create tables if they do not exist."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_schema_sql())
        conn.commit()
    finally:
        conn.close()
