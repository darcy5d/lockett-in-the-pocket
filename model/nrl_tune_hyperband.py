#!/usr/bin/env python3
"""
NRL hyperparameter tuning — runs Hyperband search for Rugby League.
With --save-final, trains and saves the best model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from model.rugby_hyperband import run_search

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", type=str, default="nrl")
    ap.add_argument("--year-from", type=int, default=2020)
    ap.add_argument("--year-to", type=int, default=2025)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hyperband-iterations", type=int, default=2)
    ap.add_argument("--save-final", action="store_true")
    args = ap.parse_args()
    run_search(
        competition_id=args.competition,
        year_from=args.year_from,
        year_to=args.year_to,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hyperband_iterations=args.hyperband_iterations,
        save_final=args.save_final,
    )
