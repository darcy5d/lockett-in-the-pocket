#!/usr/bin/env python3
"""
NRL model training — delegates to rugby_train with competition=nrl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from model.rugby_train import train


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-from", type=int, default=2020)
    ap.add_argument("--year-to", type=int, default=2025)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    train(competition_id="nrl", year_from=args.year_from, year_to=args.year_to, epochs=args.epochs)
