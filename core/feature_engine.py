"""
Feature engineering engine — computes ELO ratings, form, venue stats,
head-to-head, and player-level form from raw AFL CSVs.

Used by both model/train.py (batch, historical) and
model/prediction_api.py (single-match, real-time).
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.mappings import PlayerMapper

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 21 stats tracked per player
STAT_FIELDS = [
    "kicks", "marks", "handballs", "goals", "behinds",
    "hit_outs", "tackles", "rebound_50s", "inside_50s", "clearances",
    "clangers", "free_kicks_for", "free_kicks_against",
    "contested_possessions", "uncontested_possessions", "contested_marks",
    "marks_inside_50", "one_percenters", "bounces", "goal_assist",
    "percentage_of_game_played",
]
NUM_STATS = len(STAT_FIELDS)

# ELO parameters
TEAM_ELO_K = 32
TEAM_ELO_START = 1500
PLAYER_ELO_K = 16
PLAYER_ELO_START = 1500

# Form parameters
FORM_WINDOW = 5          # last N team matches for recent form
PLAYER_FORM_WINDOW = 10  # last N player games for player form
PLAYER_FORM_DECAY = 0.85 # exponential decay factor per game (most recent = 1.0)

# Feature column names (for consistent ordering)
MATCH_FEATURE_COLS = [
    "year",
    "is_finals",
    "team1_elo", "team2_elo", "elo_diff",
    "team1_recent_win_pct", "team1_recent_avg_score", "team1_recent_avg_margin",
    "team2_recent_win_pct", "team2_recent_avg_score", "team2_recent_avg_margin",
    "venue_team1_win_pct", "venue_team2_win_pct",
    "h2h_team1_wins_last5", "h2h_avg_margin_last5",
    "team1_avg_player_elo", "team2_avg_player_elo",
    "team1_top5_player_elo", "team2_top5_player_elo",
]
NUM_MATCH_FEATURES = len(MATCH_FEATURE_COLS)


class FeatureEngine:
    """
    Computes all engineered features from raw AFL data.

    Call flow for training:
        engine = FeatureEngine()
        engine.load_matches(1990, 2025)
        engine.load_lineups()
        engine.load_player_stats()
        features_df, enhanced_t1, enhanced_t2 = engine.compute_training_features()

    Call flow for inference (single match):
        engine = FeatureEngine.load_from_state("model/output")
        feats = engine.get_prediction_features(home_team, away_team, venue, home_ids, away_ids)
    """

    def __init__(
        self,
        match_dir: Optional[Path] = None,
        lineup_dir: Optional[Path] = None,
        player_dir: Optional[Path] = None,
    ):
        self._match_dir = match_dir or (_PROJECT_ROOT / "afl_data/data/matches")
        self._lineup_dir = lineup_dir or (_PROJECT_ROOT / "afl_data/data/lineups")
        self._player_dir = player_dir or (_PROJECT_ROOT / "afl_data/data/players")
        self._mapper = PlayerMapper()

        self.match_data: Optional[pd.DataFrame] = None
        self.lineup_data: Optional[pd.DataFrame] = None

        # ELO state
        self.team_elos: dict[str, float] = defaultdict(lambda: TEAM_ELO_START)
        self.player_elos: dict[str, float] = defaultdict(lambda: PLAYER_ELO_START)

        # Per-player career game stats: {player_id: [{stat_field: val, ...}, ...]}
        self._player_game_log: dict[str, list[dict]] = {}

        # Venue history: {(team, venue): {"wins": int, "total": int}}
        self._venue_history: dict[tuple, dict] = defaultdict(lambda: {"wins": 0, "total": 0})

        # Head-to-head history: {(team1, team2): [margin_from_team1_perspective, ...]}
        self._h2h_history: dict[tuple, list[float]] = defaultdict(list)

        # Recent team results: {team: [{"score": int, "opp_score": int, "win": bool}, ...]}
        self._team_recent: dict[str, list[dict]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_matches(self, year_from: int = 1897, year_to: int = 2025) -> None:
        """Load and concatenate match CSVs for the given year range."""
        dfs = []
        for y in range(year_from, year_to + 1):
            p = self._match_dir / f"matches_{y}.csv"
            if p.exists():
                try:
                    dfs.append(pd.read_csv(p))
                except Exception:
                    pass
        if not dfs:
            raise RuntimeError(f"No match files found in {self._match_dir}")
        self.match_data = pd.concat(dfs, ignore_index=True)
        # Drop rows without scores (future fixtures)
        for c in ["team_1_final_goals", "team_1_final_behinds",
                   "team_2_final_goals", "team_2_final_behinds"]:
            self.match_data[c] = pd.to_numeric(self.match_data[c], errors="coerce")
        self.match_data = self.match_data.dropna(
            subset=["team_1_final_goals", "team_2_final_goals"]
        ).copy()
        self.match_data["round_num"] = self.match_data["round_num"].astype(str)
        # Compute total scores
        self.match_data["score1"] = (
            self.match_data["team_1_final_goals"] * 6
            + self.match_data["team_1_final_behinds"]
        )
        self.match_data["score2"] = (
            self.match_data["team_2_final_goals"] * 6
            + self.match_data["team_2_final_behinds"]
        )
        print(f"  Loaded {len(self.match_data)} matches ({year_from}–{year_to})")

    def load_lineups(self) -> None:
        """Load all lineup CSVs."""
        files = sorted(self._lineup_dir.glob("team_lineups_*.csv"))
        dfs = [pd.read_csv(f) for f in files]
        self.lineup_data = pd.concat(dfs, ignore_index=True)
        self.lineup_data["round_num"] = self.lineup_data["round_num"].astype(str)
        print(f"  Loaded {len(self.lineup_data)} lineup rows")

    def load_player_stats(self) -> None:
        """Load per-player game logs from performance CSVs."""
        files = glob.glob(str(self._player_dir / "*_performance_details.csv"))
        print(f"  Loading player game logs from {len(files)} files…")
        for f in files:
            pid = Path(f).stem.replace("_performance_details", "")
            try:
                df = pd.read_csv(f)
                games = []
                for _, row in df.iterrows():
                    g = {}
                    for s in STAT_FIELDS:
                        v = row.get(s, 0)
                        g[s] = float(v) if pd.notna(v) else 0.0
                    g["year"] = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
                    games.append(g)
                if games:
                    self._player_game_log[pid] = games
            except Exception:
                pass
        print(f"  Game logs for {len(self._player_game_log)} players")

    # ------------------------------------------------------------------
    # ELO computation
    # ------------------------------------------------------------------

    @staticmethod
    def _elo_expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _update_team_elo(self, team1: str, team2: str, score1: float, score2: float) -> None:
        r1 = self.team_elos[team1]
        r2 = self.team_elos[team2]
        e1 = self._elo_expected(r1, r2)
        if score1 > score2:
            actual = 1.0
        elif score1 < score2:
            actual = 0.0
        else:
            actual = 0.5
        self.team_elos[team1] = r1 + TEAM_ELO_K * (actual - e1)
        self.team_elos[team2] = r2 + TEAM_ELO_K * ((1.0 - actual) - (1.0 - e1))

    def _update_player_elos(
        self, team1_ids: list[str], team2_ids: list[str], score1: float, score2: float
    ) -> None:
        if not team1_ids or not team2_ids:
            return
        avg_t1 = np.mean([self.player_elos[p] for p in team1_ids]) if team1_ids else PLAYER_ELO_START
        avg_t2 = np.mean([self.player_elos[p] for p in team2_ids]) if team2_ids else PLAYER_ELO_START
        e1 = self._elo_expected(avg_t1, avg_t2)
        actual = 1.0 if score1 > score2 else (0.0 if score1 < score2 else 0.5)
        delta1 = PLAYER_ELO_K * (actual - e1)
        delta2 = PLAYER_ELO_K * ((1.0 - actual) - (1.0 - e1))
        for p in team1_ids:
            self.player_elos[p] += delta1
        for p in team2_ids:
            self.player_elos[p] += delta2

    # ------------------------------------------------------------------
    # Form, venue, H2H helpers
    # ------------------------------------------------------------------

    def _record_result(self, team: str, score: float, opp_score: float) -> None:
        self._team_recent[team].append({
            "score": score, "opp_score": opp_score, "win": score > opp_score
        })

    def _get_recent_form(self, team: str) -> dict:
        recent = self._team_recent[team][-FORM_WINDOW:]
        if not recent:
            return {"win_pct": 0.5, "avg_score": 80.0, "avg_margin": 0.0}
        wins = sum(1 for r in recent if r["win"])
        return {
            "win_pct": wins / len(recent),
            "avg_score": np.mean([r["score"] for r in recent]),
            "avg_margin": np.mean([r["score"] - r["opp_score"] for r in recent]),
        }

    def _record_venue(self, team: str, venue: str, won: bool) -> None:
        key = (team, venue)
        self._venue_history[key]["total"] += 1
        if won:
            self._venue_history[key]["wins"] += 1

    def _get_venue_win_pct(self, team: str, venue: str) -> float:
        h = self._venue_history.get((team, venue))
        if not h or h["total"] == 0:
            return 0.5
        return h["wins"] / h["total"]

    def _record_h2h(self, team1: str, team2: str, margin_from_t1: float) -> None:
        self._h2h_history[(team1, team2)].append(margin_from_t1)
        self._h2h_history[(team2, team1)].append(-margin_from_t1)

    def _get_h2h(self, team1: str, team2: str) -> dict:
        hist = self._h2h_history.get((team1, team2), [])
        last5 = hist[-5:]
        if not last5:
            return {"wins_last5": 2.5, "avg_margin_last5": 0.0}
        return {
            "wins_last5": sum(1 for m in last5 if m > 0),
            "avg_margin_last5": np.mean(last5),
        }

    # ------------------------------------------------------------------
    # Player form (exponential-decay weighted recent stats)
    # ------------------------------------------------------------------

    def _get_player_form_stats(self, player_id: str) -> dict[str, float]:
        """Weighted average of last PLAYER_FORM_WINDOW games for a player."""
        games = self._player_game_log.get(player_id, [])
        recent = games[-PLAYER_FORM_WINDOW:]
        if not recent:
            return {s: 0.0 for s in STAT_FIELDS}
        n = len(recent)
        weights = np.array([PLAYER_FORM_DECAY ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        result = {}
        for s in STAT_FIELDS:
            vals = np.array([g.get(s, 0.0) for g in recent])
            result[s] = float(np.dot(vals, weights))
        return result

    def _get_team_enhanced_stats(self, player_ids: list[str]) -> np.ndarray:
        """Compute form-weighted enhanced stats for a team lineup."""
        if not player_ids:
            return np.zeros(NUM_STATS)
        all_stats = [self._get_player_form_stats(pid) for pid in player_ids]
        team_avg = np.zeros(NUM_STATS)
        for j, s in enumerate(STAT_FIELDS):
            vals = [st[s] for st in all_stats]
            team_avg[j] = np.mean(vals) if vals else 0.0
        return team_avg

    # ------------------------------------------------------------------
    # Lineup resolution (name -> player_id)
    # ------------------------------------------------------------------

    def _resolve_lineup(self, team: str, year: int, round_num: str) -> list[str]:
        """Get player_ids for a lineup from lineup_data."""
        if self.lineup_data is None:
            return []
        sub = self.lineup_data[
            (self.lineup_data["year"] == year)
            & (self.lineup_data["round_num"] == round_num)
            & (self.lineup_data["team_name"] == team)
        ]
        if sub.empty:
            return []
        players_str = sub["players"].iloc[0]
        if pd.isna(players_str):
            return []
        names = [n.strip() for n in str(players_str).split(";") if n.strip()]
        return [self._mapper.to_player_id(n) or "unknown" for n in names]

    # ------------------------------------------------------------------
    # Batch feature computation (for training)
    # ------------------------------------------------------------------

    def compute_training_features(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Walk through all loaded matches chronologically.
        For each match: snapshot features BEFORE updating state, then update.

        Returns:
            features_df: DataFrame with MATCH_FEATURE_COLS, aligned to match_data
            enhanced_t1: (N, 21) form-weighted enhanced stats for team 1
            enhanced_t2: (N, 21) form-weighted enhanced stats for team 2
        """
        if self.match_data is None:
            raise RuntimeError("Call load_matches() first")

        print("\nComputing features for all matches…")
        n = len(self.match_data)
        feat_rows = []
        enh_t1 = np.zeros((n, NUM_STATS))
        enh_t2 = np.zeros((n, NUM_STATS))

        for i, (_, match) in enumerate(self.match_data.iterrows()):
            team1 = match["team_1_team_name"]
            team2 = match["team_2_team_name"]
            venue = str(match.get("venue", ""))
            year = int(match["year"])
            rnd = str(match["round_num"])
            s1 = float(match["score1"])
            s2 = float(match["score2"])
            is_finals = 1 if rnd in [
                "Qualifying Final", "Elimination Final",
                "Semi Final", "Preliminary Final", "Grand Final"
            ] else 0

            # --- Snapshot features BEFORE this match ---
            form1 = self._get_recent_form(team1)
            form2 = self._get_recent_form(team2)
            h2h = self._get_h2h(team1, team2)

            # Resolve lineups for player ELO and enhanced stats
            t1_ids = self._resolve_lineup(team1, year, rnd)
            t2_ids = self._resolve_lineup(team2, year, rnd)
            known1 = [p for p in t1_ids if p != "unknown"]
            known2 = [p for p in t2_ids if p != "unknown"]

            avg_p_elo_1 = np.mean([self.player_elos[p] for p in known1]) if known1 else PLAYER_ELO_START
            avg_p_elo_2 = np.mean([self.player_elos[p] for p in known2]) if known2 else PLAYER_ELO_START
            top5_1 = np.mean(sorted([self.player_elos[p] for p in known1], reverse=True)[:5]) if len(known1) >= 5 else avg_p_elo_1
            top5_2 = np.mean(sorted([self.player_elos[p] for p in known2], reverse=True)[:5]) if len(known2) >= 5 else avg_p_elo_2

            row = {
                "year": year,
                "is_finals": is_finals,
                "team1_elo": self.team_elos[team1],
                "team2_elo": self.team_elos[team2],
                "elo_diff": self.team_elos[team1] - self.team_elos[team2],
                "team1_recent_win_pct": form1["win_pct"],
                "team1_recent_avg_score": form1["avg_score"],
                "team1_recent_avg_margin": form1["avg_margin"],
                "team2_recent_win_pct": form2["win_pct"],
                "team2_recent_avg_score": form2["avg_score"],
                "team2_recent_avg_margin": form2["avg_margin"],
                "venue_team1_win_pct": self._get_venue_win_pct(team1, venue),
                "venue_team2_win_pct": self._get_venue_win_pct(team2, venue),
                "h2h_team1_wins_last5": h2h["wins_last5"],
                "h2h_avg_margin_last5": h2h["avg_margin_last5"],
                "team1_avg_player_elo": avg_p_elo_1,
                "team2_avg_player_elo": avg_p_elo_2,
                "team1_top5_player_elo": top5_1,
                "team2_top5_player_elo": top5_2,
            }
            feat_rows.append(row)

            # Enhanced stats (form-weighted)
            enh_t1[i] = self._get_team_enhanced_stats(known1)
            enh_t2[i] = self._get_team_enhanced_stats(known2)

            # --- Update state AFTER this match ---
            self._update_team_elo(team1, team2, s1, s2)
            self._update_player_elos(known1, known2, s1, s2)
            self._record_result(team1, s1, s2)
            self._record_result(team2, s2, s1)
            self._record_venue(team1, venue, s1 > s2)
            self._record_venue(team2, venue, s2 > s1)
            self._record_h2h(team1, team2, s1 - s2)

            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{n} matches processed…")

        features_df = pd.DataFrame(feat_rows, columns=MATCH_FEATURE_COLS)
        print(f"  Feature matrix: {features_df.shape}, enhanced: ({n}, {NUM_STATS}) per team")

        # Summary stats
        elo_vals = list(dict(self.team_elos).values())
        print(f"  Team ELO range: {min(elo_vals):.0f}–{max(elo_vals):.0f}")
        return features_df, enh_t1, enh_t2

    # ------------------------------------------------------------------
    # Single-match inference features
    # ------------------------------------------------------------------

    def get_prediction_features(
        self,
        home_team: str,
        away_team: str,
        venue: str,
        home_player_ids: list[str],
        away_player_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute feature vector for a single upcoming match.

        Returns:
            match_features: (1, NUM_MATCH_FEATURES) array
            enhanced_t1: (1, NUM_STATS) array
            enhanced_t2: (1, NUM_STATS) array
        """
        form1 = self._get_recent_form(home_team)
        form2 = self._get_recent_form(away_team)
        h2h = self._get_h2h(home_team, away_team)

        known1 = [p for p in home_player_ids if p != "unknown"]
        known2 = [p for p in away_player_ids if p != "unknown"]

        avg_p1 = np.mean([self.player_elos[p] for p in known1]) if known1 else PLAYER_ELO_START
        avg_p2 = np.mean([self.player_elos[p] for p in known2]) if known2 else PLAYER_ELO_START
        top5_1 = np.mean(sorted([self.player_elos[p] for p in known1], reverse=True)[:5]) if len(known1) >= 5 else avg_p1
        top5_2 = np.mean(sorted([self.player_elos[p] for p in known2], reverse=True)[:5]) if len(known2) >= 5 else avg_p2

        from datetime import datetime
        row = {
            "year": datetime.now().year,
            "is_finals": 0,
            "team1_elo": self.team_elos[home_team],
            "team2_elo": self.team_elos[away_team],
            "elo_diff": self.team_elos[home_team] - self.team_elos[away_team],
            "team1_recent_win_pct": form1["win_pct"],
            "team1_recent_avg_score": form1["avg_score"],
            "team1_recent_avg_margin": form1["avg_margin"],
            "team2_recent_win_pct": form2["win_pct"],
            "team2_recent_avg_score": form2["avg_score"],
            "team2_recent_avg_margin": form2["avg_margin"],
            "venue_team1_win_pct": self._get_venue_win_pct(home_team, venue),
            "venue_team2_win_pct": self._get_venue_win_pct(away_team, venue),
            "h2h_team1_wins_last5": h2h["wins_last5"],
            "h2h_avg_margin_last5": h2h["avg_margin_last5"],
            "team1_avg_player_elo": avg_p1,
            "team2_avg_player_elo": avg_p2,
            "team1_top5_player_elo": top5_1,
            "team2_top5_player_elo": top5_2,
        }

        match_features = np.array([[row[c] for c in MATCH_FEATURE_COLS]])
        enh1 = self._get_team_enhanced_stats(known1).reshape(1, -1)
        enh2 = self._get_team_enhanced_stats(known2).reshape(1, -1)
        return match_features, enh1, enh2

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, output_dir: Path) -> None:
        """Save ELO state + feature column list for inference."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "team_elos.json", "w") as f:
            json.dump(dict(self.team_elos), f)
        with open(output_dir / "player_elos.json", "w") as f:
            json.dump(dict(self.player_elos), f)
        with open(output_dir / "feature_cols.json", "w") as f:
            json.dump(MATCH_FEATURE_COLS, f)
        # Save venue and form state for inference
        venue_state = {f"{k[0]}|{k[1]}": v for k, v in self._venue_history.items()}
        with open(output_dir / "venue_history.json", "w") as f:
            json.dump(venue_state, f)
        h2h_state = {f"{k[0]}|{k[1]}": v[-10:] for k, v in self._h2h_history.items()}
        with open(output_dir / "h2h_history.json", "w") as f:
            json.dump(h2h_state, f)
        recent_state = {k: v[-FORM_WINDOW:] for k, v in self._team_recent.items()}
        with open(output_dir / "team_recent.json", "w") as f:
            json.dump(recent_state, f)
        print(f"  Engine state saved to {output_dir}")

    @classmethod
    def load_from_state(cls, state_dir: Path | str) -> "FeatureEngine":
        """Load a FeatureEngine with pre-computed ELO and form state (for inference)."""
        state_dir = Path(state_dir)
        engine = cls()
        # Team ELOs
        p = state_dir / "team_elos.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            for k, v in data.items():
                engine.team_elos[k] = v
        # Player ELOs
        p = state_dir / "player_elos.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            for k, v in data.items():
                engine.player_elos[k] = v
        # Venue history
        p = state_dir / "venue_history.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            for k, v in data.items():
                parts = k.split("|", 1)
                if len(parts) == 2:
                    engine._venue_history[(parts[0], parts[1])] = v
        # H2H history
        p = state_dir / "h2h_history.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            for k, v in data.items():
                parts = k.split("|", 1)
                if len(parts) == 2:
                    engine._h2h_history[(parts[0], parts[1])] = v
        # Team recent form
        p = state_dir / "team_recent.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            for k, v in data.items():
                engine._team_recent[k] = v
        # Load player game logs for form stats at inference
        engine.load_player_stats()
        print(f"  Engine state loaded from {state_dir}")
        return engine
