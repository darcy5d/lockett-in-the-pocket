"""
Rugby League Feature Engine — parameterised by competition.

ELO, form, venue, H2H, player presence, rest days, scoring components,
positional presence, out-of-position detection, lineup stability.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "nrl_data" / "data"

from core.competition_config import get_competition, get_competition_slugs, slug_matches_file, slug_matches_lineup_file

TEAM_ELO_K = 32
ELO_START = 1500
FORM_WINDOW = 5
PRESENCE_K = 16

SPINE_POSITIONS = frozenset({1, 6, 7, 9})
FORWARD_POSITIONS = frozenset({8, 10, 11, 12, 13})
BACK_POSITIONS = frozenset({2, 3, 4, 5})
DEFAULT_REST_DAYS = 7


def _expected_margin(elo1: float, elo2: float) -> float:
    return 20 * ((elo1 - elo2) / 400)


def _update_presence_elo(elo: float, actual: float, expected: float, k: float) -> float:
    return elo + k * (actual - expected)


def _round_sort_key(r: str) -> tuple[int, str]:
    """Numeric rounds first (0), then finals (1). For stable ordering."""
    r = str(r).strip()
    if "final" in r.lower() or "gf" in r.lower():
        return (1, r)
    try:
        return (0, str(int(r)).zfill(3))
    except ValueError:
        return (0, r)


def _parse_match_date(date_str, year) -> Optional[datetime]:
    """Parse 'Fri, 15th May' + year -> datetime."""
    if not date_str or (isinstance(date_str, float) and np.isnan(date_str)):
        return None
    s = str(date_str)
    parts = s.split(",", 1)
    date_part = parts[1].strip() if len(parts) == 2 else parts[0].strip()
    date_part = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_part)
    for fmt in ("%d %B", "%d %b"):
        try:
            dt = datetime.strptime(f"{date_part} {int(year)}", f"{fmt} %Y")
            return dt
        except ValueError:
            continue
    return None


RUGBY_FEATURE_COLS = [
    "year", "is_finals",
    "team1_elo", "team2_elo", "elo_diff",
    "team1_recent_win_pct", "team1_recent_avg_score", "team1_recent_avg_margin",
    "team2_recent_win_pct", "team2_recent_avg_score", "team2_recent_avg_margin",
    "venue_team1_win_pct", "venue_team2_win_pct",
    "h2h_team1_wins_last5", "h2h_avg_margin_last5",
    "team1_avg_presence_elo", "team2_avg_presence_elo",
    "team1_top5_presence_elo", "team2_top5_presence_elo",
    # Rest days
    "team1_days_rest", "team2_days_rest", "rest_diff",
    # Scoring components
    "team1_avg_tries_last5", "team2_avg_tries_last5",
    "team1_avg_tries_conceded_last5", "team2_avg_tries_conceded_last5",
    "team1_avg_kick_pts_last5", "team2_avg_kick_pts_last5",
    # Positional presence
    "team1_spine_presence_elo", "team2_spine_presence_elo",
    "team1_forwards_presence_elo", "team2_forwards_presence_elo",
    "team1_backs_presence_elo", "team2_backs_presence_elo",
    # Out of position
    "team1_out_of_position_count", "team2_out_of_position_count",
    # Lineup stability
    "team1_lineup_changes", "team2_lineup_changes",
    "team1_spine_changes", "team2_spine_changes",
]


class RugbyFeatureEngine:
    def __init__(
        self,
        competition_id: str,
        match_dir: Optional[Path] = None,
        lineup_dir: Optional[Path] = None,
        presence_path: Optional[Path] = None,
    ):
        self._competition_id = competition_id
        cfg = get_competition(competition_id)
        if not cfg:
            raise ValueError(f"Unknown competition: {competition_id}")

        self._slugs = get_competition_slugs(competition_id)
        self._match_dir = match_dir or (DATA_DIR / "matches")
        self._lineup_dir = lineup_dir or (DATA_DIR / "lineups")
        self._presence_path = presence_path or (DATA_DIR / f"player_presence_elo_{competition_id}.json")

        self.match_data: Optional[pd.DataFrame] = None
        self.lineup_data: Optional[pd.DataFrame] = None
        self.team_elos: dict[str, float] = defaultdict(lambda: ELO_START)
        self._presence_elos: dict[str, float] = {}
        self._venue_history: dict[tuple, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
        self._h2h_history: dict[tuple, list[float]] = defaultdict(list)
        self._team_recent: dict[str, list[dict]] = defaultdict(list)
        self._team_last_match_date: dict[str, datetime] = {}
        self._team_last_lineup: dict[str, set[str]] = {}
        self._team_last_spine: dict[str, set[str]] = {}
        self._player_position_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._load_presence()

    def _slug_matches_match(self, name: str) -> bool:
        return slug_matches_file(name, self._slugs)

    def _slug_matches_lineup(self, name: str) -> bool:
        return slug_matches_lineup_file(name, self._slugs)

    def _load_presence(self) -> None:
        if self._presence_path.exists():
            with open(self._presence_path) as f:
                self._presence_elos = json.load(f)

    def _presence(self, player_id: str) -> float:
        return self._presence_elos.get(str(player_id), float(ELO_START))

    def load_matches(self, year_from: int = 1990, year_to: int = 2025) -> None:
        files = sorted(self._match_dir.glob("matches_*.csv"))
        dfs = []
        for f in files:
            if "2026" in f.name or not self._slug_matches_match(f.name):
                continue
            try:
                df = pd.read_csv(f)
                df = df.dropna(subset=["team_1_score", "team_2_score"], how="all")
                dfs.append(df)
            except Exception:
                pass
        if not dfs:
            self.match_data = pd.DataFrame()
            return
        self.match_data = pd.concat(dfs, ignore_index=True)
        if "match_id" in self.match_data.columns:
            self.match_data = self.match_data.drop_duplicates(subset=["match_id"], keep="first")
        # Filter to requested year window for ELO warm-up
        if "year" in self.match_data.columns:
            self.match_data = self.match_data[
                (self.match_data["year"] >= year_from) & (self.match_data["year"] <= year_to)
            ]
        self.match_data["score1"] = pd.to_numeric(self.match_data["team_1_score"], errors="coerce")
        self.match_data["score2"] = pd.to_numeric(self.match_data["team_2_score"], errors="coerce")
        self.match_data = self.match_data.dropna(subset=["score1", "score2"])

        # Parse tries and compute kick_pts (score - tries*4)
        for prefix in ("team_1", "team_2"):
            col = f"{prefix}_tries"
            if col in self.match_data.columns:
                self.match_data[col] = pd.to_numeric(self.match_data[col], errors="coerce").fillna(0).astype(int)
            else:
                self.match_data[col] = 0

        # Parse date into datetime
        self.match_data["_parsed_date"] = self.match_data.apply(
            lambda r: _parse_match_date(r.get("date"), r.get("year")), axis=1
        )

        # Randomly swap team1/team2 for 50% of matches (source data is winner-first)
        self.match_data = self.match_data.reset_index(drop=True)
        rng = np.random.RandomState(42)
        swap_mask = rng.rand(len(self.match_data)) < 0.5
        for suffix in ["team_name", "score", "tries", "goals", "fg"]:
            c1, c2 = f"team_1_{suffix}", f"team_2_{suffix}"
            if c1 in self.match_data.columns and c2 in self.match_data.columns:
                v = self.match_data[[c1, c2]].values.copy()
                self.match_data.loc[swap_mask, c1] = v[swap_mask, 1]
                self.match_data.loc[swap_mask, c2] = v[swap_mask, 0]
        sv = self.match_data[["score1", "score2"]].values.copy()
        self.match_data.loc[swap_mask, "score1"] = sv[swap_mask, 1]
        self.match_data.loc[swap_mask, "score2"] = sv[swap_mask, 0]

        # Derived columns after swap
        self.match_data["tries1"] = self.match_data["team_1_tries"].astype(float)
        self.match_data["tries2"] = self.match_data["team_2_tries"].astype(float)
        self.match_data["kick_pts1"] = self.match_data["score1"] - self.match_data["tries1"] * 4
        self.match_data["kick_pts2"] = self.match_data["score2"] - self.match_data["tries2"] * 4

        self.match_data["_round_sort0"] = self.match_data["round_num"].astype(str).apply(lambda r: _round_sort_key(r)[0])
        self.match_data["_round_sort1"] = self.match_data["round_num"].astype(str).apply(lambda r: _round_sort_key(r)[1])
        self.match_data = self.match_data.sort_values(["year", "_round_sort0", "_round_sort1"]).drop(
            columns=["_round_sort0", "_round_sort1"]
        )

    def load_lineups(self) -> None:
        files = [
            f for f in sorted(self._lineup_dir.glob("lineup_details_*.csv"))
            if self._slug_matches_lineup(f.name)
        ]
        if not files:
            self.lineup_data = pd.DataFrame()
            return
        self.lineup_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    @staticmethod
    def _elo_expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    def _update_team_elo(self, t1: str, t2: str, s1: float, s2: float) -> None:
        e1 = self._elo_expected(self.team_elos[t1], self.team_elos[t2])
        actual = 1.0 if s1 > s2 else (0.5 if s1 == s2 else 0.0)
        self.team_elos[t1] += TEAM_ELO_K * (actual - e1)
        self.team_elos[t2] += TEAM_ELO_K * ((1 - actual) - (1 - e1))

    def _resolve_lineup(self, match_id: str, team: str) -> list[tuple[str, int]]:
        """Return [(player_id, jersey_position)] where position is 1-based."""
        if self.lineup_data is None or self.lineup_data.empty:
            return []
        sub = self.lineup_data[
            (self.lineup_data["match_id"] == match_id) & (self.lineup_data["team"] == team)
        ]
        return [(str(r["player_id"]), i + 1) for i, (_, r) in enumerate(sub.iterrows())]

    def _get_recent_form(self, team: str) -> dict:
        recent = self._team_recent[team][-FORM_WINDOW:]
        if not recent:
            return {
                "win_pct": 0.5, "avg_score": 25.0, "avg_margin": 0.0,
                "avg_tries": 3.0, "avg_tries_conceded": 3.0, "avg_kick_pts": 12.0,
            }
        wins = sum(1 for r in recent if r["win"])
        return {
            "win_pct": wins / len(recent),
            "avg_score": np.mean([r["score"] for r in recent]),
            "avg_margin": np.mean([r["score"] - r["opp_score"] for r in recent]),
            "avg_tries": np.mean([r["tries"] for r in recent]),
            "avg_tries_conceded": np.mean([r["opp_tries"] for r in recent]),
            "avg_kick_pts": np.mean([r["kick_pts"] for r in recent]),
        }

    def _get_venue_win_pct(self, team: str, venue: str) -> float:
        h = self._venue_history.get((team, venue))
        if not h or h["total"] == 0:
            return 0.5
        return h["wins"] / h["total"]

    def _get_h2h(self, t1: str, t2: str) -> dict:
        hist = self._h2h_history.get((t1, t2), [])
        last5 = hist[-5:]
        if not last5:
            return {"wins_last5": 2.5, "avg_margin_last5": 0.0}
        return {"wins_last5": sum(1 for m in last5 if m > 0), "avg_margin_last5": np.mean(last5)}

    def _get_days_rest(self, team: str, match_date) -> float:
        if match_date is None or team not in self._team_last_match_date:
            return float(DEFAULT_REST_DAYS)
        last = self._team_last_match_date[team]
        delta = (match_date - last).days
        return float(max(1, min(delta, 30)))

    def _get_usual_position(self, player_id: str) -> Optional[int]:
        counts = self._player_position_counts.get(player_id)
        if not counts:
            return None
        return max(counts, key=counts.get)

    def _compute_positional_presence(self, lineup: list[tuple[str, int]], player_elos: dict) -> dict:
        """Compute avg presence ELO for spine, forwards, backs from a lineup."""
        spine, forwards, backs = [], [], []
        for pid, pos in lineup:
            if pid == "unknown":
                continue
            elo = player_elos[pid]
            if pos in SPINE_POSITIONS:
                spine.append(elo)
            elif pos in FORWARD_POSITIONS:
                forwards.append(elo)
            elif pos in BACK_POSITIONS:
                backs.append(elo)
        return {
            "spine": np.mean(spine) if spine else ELO_START,
            "forwards": np.mean(forwards) if forwards else ELO_START,
            "backs": np.mean(backs) if backs else ELO_START,
        }

    def _count_out_of_position(self, lineup: list[tuple[str, int]]) -> int:
        count = 0
        for pid, pos in lineup:
            if pid == "unknown":
                continue
            usual = self._get_usual_position(pid)
            if usual is not None and usual != pos:
                count += 1
        return count

    def _count_lineup_changes(self, team: str, current_ids: set[str]) -> int:
        last = self._team_last_lineup.get(team)
        if last is None:
            return 0
        return len(current_ids - last) + len(last - current_ids)

    def _count_spine_changes(self, team: str, current_spine: set[str]) -> int:
        last = self._team_last_spine.get(team)
        if last is None:
            return 0
        return len(current_spine - last) + len(last - current_spine)

    def compute_training_features(self, verbose: bool = True) -> pd.DataFrame:
        """
        Build features using round-level snapshots: for all matches in round R,
        use state at end of round R-1. Updates state only after processing
        the entire round to avoid within-round leakage.
        """
        if self.match_data is None or self.match_data.empty:
            return pd.DataFrame(columns=RUGBY_FEATURE_COLS)
        if verbose:
            print("  Loading lineups…", flush=True)
        self.load_lineups()
        total = len(self.match_data)
        if verbose:
            print(f"  Computing round-level ELO, form, venue, H2H, presence for {total} matches…", flush=True)
        feat_rows = []
        t0 = time.perf_counter()
        last_report = 0
        report_interval = max(500, total // 20)
        player_elos: dict[str, float] = defaultdict(lambda: float(ELO_START))

        for (year, rnd), round_df in self.match_data.groupby(["year", "round_num"], sort=False):
            # Phase 1: Build features for all matches in this round using current state
            for _, match in round_df.iterrows():
                t1 = match["team_1_team_name"]
                t2 = match["team_2_team_name"]
                venue = str(match.get("venue", ""))
                rnd_str = str(match.get("round_num", ""))
                s1 = float(match["score1"])
                s2 = float(match["score2"])
                mid = str(match.get("match_id", ""))
                is_finals = 1 if "final" in rnd_str.lower() or "gf" in rnd_str.lower() else 0
                match_date = match.get("_parsed_date")

                lineup1 = self._resolve_lineup(mid, t1)
                lineup2 = self._resolve_lineup(mid, t2)
                known1 = [(p, pos) for p, pos in lineup1 if p and p != "unknown"]
                known2 = [(p, pos) for p, pos in lineup2 if p and p != "unknown"]
                ids1 = [p for p, _ in known1]
                ids2 = [p for p, _ in known2]

                # Overall presence
                avg_p1 = np.mean([player_elos[p] for p in ids1]) if ids1 else ELO_START
                avg_p2 = np.mean([player_elos[p] for p in ids2]) if ids2 else ELO_START
                top5_1 = np.mean(sorted([player_elos[p] for p in ids1], reverse=True)[:5]) if len(ids1) >= 5 else avg_p1
                top5_2 = np.mean(sorted([player_elos[p] for p in ids2], reverse=True)[:5]) if len(ids2) >= 5 else avg_p2

                # Positional presence
                pos_p1 = self._compute_positional_presence(known1, player_elos)
                pos_p2 = self._compute_positional_presence(known2, player_elos)

                # Out of position
                oop1 = self._count_out_of_position(known1)
                oop2 = self._count_out_of_position(known2)

                # Lineup stability
                set1 = set(ids1)
                set2 = set(ids2)
                lineup_chg1 = self._count_lineup_changes(t1, set1)
                lineup_chg2 = self._count_lineup_changes(t2, set2)
                spine1_ids = {p for p, pos in known1 if pos in SPINE_POSITIONS}
                spine2_ids = {p for p, pos in known2 if pos in SPINE_POSITIONS}
                spine_chg1 = self._count_spine_changes(t1, spine1_ids)
                spine_chg2 = self._count_spine_changes(t2, spine2_ids)

                form1 = self._get_recent_form(t1)
                form2 = self._get_recent_form(t2)
                h2h = self._get_h2h(t1, t2)

                # Rest days
                rest1 = self._get_days_rest(t1, match_date)
                rest2 = self._get_days_rest(t2, match_date)

                feat_rows.append({
                    "year": int(year),
                    "is_finals": is_finals,
                    "team1_elo": self.team_elos[t1],
                    "team2_elo": self.team_elos[t2],
                    "elo_diff": self.team_elos[t1] - self.team_elos[t2],
                    "team1_recent_win_pct": form1["win_pct"],
                    "team1_recent_avg_score": form1["avg_score"],
                    "team1_recent_avg_margin": form1["avg_margin"],
                    "team2_recent_win_pct": form2["win_pct"],
                    "team2_recent_avg_score": form2["avg_score"],
                    "team2_recent_avg_margin": form2["avg_margin"],
                    "venue_team1_win_pct": self._get_venue_win_pct(t1, venue),
                    "venue_team2_win_pct": self._get_venue_win_pct(t2, venue),
                    "h2h_team1_wins_last5": h2h["wins_last5"],
                    "h2h_avg_margin_last5": h2h["avg_margin_last5"],
                    "team1_avg_presence_elo": avg_p1,
                    "team2_avg_presence_elo": avg_p2,
                    "team1_top5_presence_elo": top5_1,
                    "team2_top5_presence_elo": top5_2,
                    # Rest days
                    "team1_days_rest": rest1,
                    "team2_days_rest": rest2,
                    "rest_diff": rest1 - rest2,
                    # Scoring components
                    "team1_avg_tries_last5": form1["avg_tries"],
                    "team2_avg_tries_last5": form2["avg_tries"],
                    "team1_avg_tries_conceded_last5": form1["avg_tries_conceded"],
                    "team2_avg_tries_conceded_last5": form2["avg_tries_conceded"],
                    "team1_avg_kick_pts_last5": form1["avg_kick_pts"],
                    "team2_avg_kick_pts_last5": form2["avg_kick_pts"],
                    # Positional presence
                    "team1_spine_presence_elo": pos_p1["spine"],
                    "team2_spine_presence_elo": pos_p2["spine"],
                    "team1_forwards_presence_elo": pos_p1["forwards"],
                    "team2_forwards_presence_elo": pos_p2["forwards"],
                    "team1_backs_presence_elo": pos_p1["backs"],
                    "team2_backs_presence_elo": pos_p2["backs"],
                    # Out of position
                    "team1_out_of_position_count": oop1,
                    "team2_out_of_position_count": oop2,
                    # Lineup stability
                    "team1_lineup_changes": lineup_chg1,
                    "team2_lineup_changes": lineup_chg2,
                    "team1_spine_changes": spine_chg1,
                    "team2_spine_changes": spine_chg2,
                })

            # Phase 2: Update state after processing entire round
            for _, match in round_df.iterrows():
                t1 = match["team_1_team_name"]
                t2 = match["team_2_team_name"]
                venue = str(match.get("venue", ""))
                s1 = float(match["score1"])
                s2 = float(match["score2"])
                mid = str(match.get("match_id", ""))
                match_date = match.get("_parsed_date")
                tries1 = float(match.get("tries1", 0))
                tries2 = float(match.get("tries2", 0))
                kick_pts1 = float(match.get("kick_pts1", s1 - tries1 * 4))
                kick_pts2 = float(match.get("kick_pts2", s2 - tries2 * 4))

                lineup1 = self._resolve_lineup(mid, t1)
                lineup2 = self._resolve_lineup(mid, t2)
                known1 = [(p, pos) for p, pos in lineup1 if p and p != "unknown"]
                known2 = [(p, pos) for p, pos in lineup2 if p and p != "unknown"]
                ids1 = [p for p, _ in known1]
                ids2 = [p for p, _ in known2]

                margin = s1 - s2
                exp_margin = _expected_margin(self.team_elos[t1], self.team_elos[t2])
                t1_residual = margin - exp_margin
                t2_residual = -t1_residual
                self._update_team_elo(t1, t2, s1, s2)
                for pid in ids1:
                    player_elos[pid] = _update_presence_elo(player_elos[pid], t1_residual, 0, PRESENCE_K)
                for pid in ids2:
                    player_elos[pid] = _update_presence_elo(player_elos[pid], t2_residual, 0, PRESENCE_K)

                self._team_recent[t1].append({
                    "score": s1, "opp_score": s2, "win": s1 > s2,
                    "tries": tries1, "opp_tries": tries2, "kick_pts": kick_pts1,
                })
                self._team_recent[t2].append({
                    "score": s2, "opp_score": s1, "win": s2 > s1,
                    "tries": tries2, "opp_tries": tries1, "kick_pts": kick_pts2,
                })

                self._venue_history[(t1, venue)]["total"] += 1
                self._venue_history[(t2, venue)]["total"] += 1
                if s1 > s2:
                    self._venue_history[(t1, venue)]["wins"] += 1
                else:
                    self._venue_history[(t2, venue)]["wins"] += 1
                self._h2h_history[(t1, t2)].append(s1 - s2)
                self._h2h_history[(t2, t1)].append(s2 - s1)

                # Rest days: update last match date
                if match_date is not None:
                    self._team_last_match_date[t1] = match_date
                    self._team_last_match_date[t2] = match_date

                # Lineup stability: update last lineup and spine
                self._team_last_lineup[t1] = set(ids1)
                self._team_last_lineup[t2] = set(ids2)
                self._team_last_spine[t1] = {p for p, pos in known1 if pos in SPINE_POSITIONS}
                self._team_last_spine[t2] = {p for p, pos in known2 if pos in SPINE_POSITIONS}

                # Position tracking for out-of-position detection
                for pid, pos in known1:
                    self._player_position_counts[pid][pos] += 1
                for pid, pos in known2:
                    self._player_position_counts[pid][pos] += 1

            rows_so_far = len(feat_rows)
            if verbose and rows_so_far - last_report >= report_interval:
                last_report = rows_so_far
                pct = 100 * rows_so_far / total
                elapsed = time.perf_counter() - t0
                rate = rows_so_far / elapsed if elapsed > 0 else 0
                remaining = (total - rows_so_far) / rate if rate > 0 else 0
                print(
                    f"  Features (round-level): {rows_so_far}/{total} ({pct:.0f}%) — ~{remaining:.0f}s remaining",
                    flush=True,
                )

        return pd.DataFrame(feat_rows, columns=RUGBY_FEATURE_COLS)

    def get_prediction_features(
        self,
        home_team: str,
        away_team: str,
        venue: str,
        home_player_ids: list[str],
        away_player_ids: list[str],
    ) -> np.ndarray:
        known1 = [p for p in home_player_ids if p and p != "unknown"]
        known2 = [p for p in away_player_ids if p and p != "unknown"]
        avg_p1 = np.mean([self._presence(p) for p in known1]) if known1 else ELO_START
        avg_p2 = np.mean([self._presence(p) for p in known2]) if known2 else ELO_START
        top5_1 = np.mean(sorted([self._presence(p) for p in known1], reverse=True)[:5]) if len(known1) >= 5 else avg_p1
        top5_2 = np.mean(sorted([self._presence(p) for p in known2], reverse=True)[:5]) if len(known2) >= 5 else avg_p2

        form1 = self._get_recent_form(home_team)
        form2 = self._get_recent_form(away_team)
        h2h = self._get_h2h(home_team, away_team)

        rest1 = float(DEFAULT_REST_DAYS)
        rest2 = float(DEFAULT_REST_DAYS)

        row = [
            pd.Timestamp.now().year, 0,
            self.team_elos[home_team], self.team_elos[away_team],
            self.team_elos[home_team] - self.team_elos[away_team],
            form1["win_pct"], form1["avg_score"], form1["avg_margin"],
            form2["win_pct"], form2["avg_score"], form2["avg_margin"],
            self._get_venue_win_pct(home_team, venue), self._get_venue_win_pct(away_team, venue),
            h2h["wins_last5"], h2h["avg_margin_last5"],
            avg_p1, avg_p2, top5_1, top5_2,
            # Rest days (default for predictions)
            rest1, rest2, rest1 - rest2,
            # Scoring components
            form1["avg_tries"], form2["avg_tries"],
            form1["avg_tries_conceded"], form2["avg_tries_conceded"],
            form1["avg_kick_pts"], form2["avg_kick_pts"],
            # Positional presence (use overall as proxy when no lineup positions)
            avg_p1, avg_p2, avg_p1, avg_p2, avg_p1, avg_p2,
            # Out of position (default 0)
            0, 0,
            # Lineup stability (default 0)
            0, 0, 0, 0,
        ]
        return np.array([row])

    def save_state(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "team_elos.json", "w") as f:
            json.dump(dict(self.team_elos), f)
        with open(output_dir / "feature_cols.json", "w") as f:
            json.dump(RUGBY_FEATURE_COLS, f)
