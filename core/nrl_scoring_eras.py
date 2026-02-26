"""
NRL scoring era normalization.

Source: nrl_competition_history.md § Scoring Eras; RLP plan.

| Era       | Try   | Conversion | Field Goal        |
| --------- | ----- | ---------- | ----------------- |
| 1908–1982 | 3 pts | 2 pts      | 2 pts (1908–1970) |
| 1983+     | 4 pts | 2 pts      | 1 pt (1971+)      |

Era-adjusted score: score = tries * try_pts + goals * 2 + fg * fg_pts
"""

from __future__ import annotations

from typing import Tuple


def get_scoring_era(year: int) -> Tuple[int, int, int]:
    """
    Return (try_pts, conversion_pts, fg_pts) for the given year.

    Parameters
    ----------
    year : int
        Season year (e.g. 1995).

    Returns
    -------
    tuple of (int, int, int)
        (try_pts, conversion_pts, fg_pts)
    """
    try_pts = 3 if year <= 1982 else 4
    conversion_pts = 2  # Always 2
    fg_pts = 2 if year <= 1970 else 1
    return (try_pts, conversion_pts, fg_pts)


def era_adjusted_score(tries: int, goals: int, fg: int, year: int) -> float:
    """
    Compute era-adjusted total score for cross-year comparison.

    Parameters
    ----------
    tries, goals, fg : int
        Raw counts from match data.
    year : int
        Season year.

    Returns
    -------
    float
        Era-normalized score (tries * try_pts + goals * 2 + fg * fg_pts).
    """
    try_pts, _, fg_pts = get_scoring_era(year)
    return tries * try_pts + goals * 2 + fg * fg_pts
