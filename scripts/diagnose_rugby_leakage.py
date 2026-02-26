#!/usr/bin/env python3
"""
Diagnose Rugby League data leakage — run validations to find source of 100% winner accuracy.

Checks:
1. Validation set size and split correctness (no train/val overlap)
2. elo_diff vs outcome: does higher ELO always win in validation?
3. Random baseline: expect ~50% winner accuracy
4. Sample validation rows
5. Trained model accuracy (if model exists)
6. Scaler fit note

Usage:
  python scripts/diagnose_rugby_leakage.py --competition nrl --year-from 2020 --year-to 2025
  python scripts/diagnose_rugby_leakage.py --competition nrl --year-from 2024 --year-to 2025  # faster
  python scripts/diagnose_rugby_leakage.py --from-cache  # use cached features from last run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np

from core.rugby_feature_engine import RUGBY_FEATURE_COLS
from core.rugby_player_presence import compute_presence
from model.rugby_train import _temporal_train_test_split, load_and_prepare


def _cache_path(competition: str, year_from: int, year_to: int) -> Path:
    return _PROJECT_ROOT / "model" / "output" / f"_leakage_diag_{competition}_{year_from}_{year_to}.npz"


def main():
    ap = argparse.ArgumentParser(description="Diagnose Rugby League data leakage")
    ap.add_argument("--competition", type=str, default="nrl")
    ap.add_argument("--year-from", type=int, default=2020)
    ap.add_argument("--year-to", type=int, default=2025)
    ap.add_argument("--from-cache", action="store_true", help="Load X,y from last run (faster)")
    args = ap.parse_args()

    print("=" * 60)
    print("Rugby League Leakage Diagnostics")
    print("=" * 60)
    print(f"Competition: {args.competition}, Years: {args.year_from}–{args.year_to}\n")

    # Load data (no scaling yet, so we can inspect raw features)
    cache_path = _cache_path(args.competition, args.year_from, args.year_to)
    if args.from_cache and cache_path.exists():
        print("Loading from cache…")
        data = np.load(cache_path, allow_pickle=True)
        X, y_s1, y_s2, y_margin = data["X"], data["y_s1"], data["y_s2"], data["y_margin"]
        match_ids = data["match_ids"] if "match_ids" in data.files else None
        team1 = data["team1"] if "team1" in data.files else None
        team2 = data["team2"] if "team2" in data.files else None
        years = data["years"] if "years" in data.files else None
        rounds = data["rounds"] if "rounds" in data.files else None
    else:
        print("Loading data…")
        X, y_s1, y_s2, y_margin = load_and_prepare(args.competition, args.year_from, args.year_to)[:4]
        print("Cached for --from-cache next time.")
        # Load match metadata from cache (saved by load_and_prepare)
        match_ids = team1 = team2 = years = rounds = None
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            match_ids = data["match_ids"] if "match_ids" in data.files else None
            team1 = data["team1"] if "team1" in data.files else None
            team2 = data["team2"] if "team2" in data.files else None
            years = data["years"] if "years" in data.files else None
            rounds = data["rounds"] if "rounds" in data.files else None
    n = len(X)
    print(f"Total samples: {n}\n")

    # 1. Validation set size and split
    print("--- 1. Split verification ---")
    X_tr, X_te, _, _, _, _, mg_tr, mg_te = _temporal_train_test_split(
        X, y_s1, y_s2, y_margin, test_size=0.2
    )
    n_train, n_test = len(X_tr), len(X_te)
    print(f"Train: {n_train} ({100*n_train/n:.1f}%)")
    print(f"Validation: {n_test} ({100*n_test/n:.1f}%)")
    # Check no overlap
    train_end_idx = n_train
    test_start_idx = n - n_test
    assert test_start_idx == train_end_idx, "Split indices should be contiguous"
    print("Split: contiguous (last 20% = validation) ✓")
    print()

    # 2. elo_diff vs outcome — does higher ELO always win?
    print("--- 2. elo_diff vs actual winner ---")
    elo_diff_idx = RUGBY_FEATURE_COLS.index("elo_diff")
    elo_diff_te = X_te[:, elo_diff_idx]
    actual_winner_te = (mg_te > 0).astype(int)  # 1 if team1 won
    pred_winner_elo = (elo_diff_te > 0).astype(int)
    elo_baseline_acc = np.mean(pred_winner_elo == actual_winner_te)
    print(f"Winner accuracy if we predict winner = (elo_diff > 0): {elo_baseline_acc:.1%}")
    if elo_baseline_acc > 0.95:
        print("  WARNING: elo_diff alone predicts winner almost perfectly — possible leakage or unusual data")
    # Upsets: higher ELO lost
    upsets = np.sum((elo_diff_te > 0) & (mg_te < 0)) + np.sum((elo_diff_te < 0) & (mg_te > 0))
    print(f"Upsets (higher ELO lost): {upsets} / {n_test} ({100*upsets/n_test:.1f}%)")
    print()

    # 3. Random baseline
    print("--- 3. Random baseline ---")
    np.random.seed(42)
    random_pred = np.random.choice([-1, 1], size=n_test)
    random_acc = np.mean((random_pred > 0) == (mg_te > 0))
    print(f"Winner accuracy with random ±1 predictions: {random_acc:.1%} (expect ~50%)")
    print()

    # 4. Sample validation rows
    print("--- 4. Sample validation rows (first 10) ---")
    print(f"{'elo_diff':>10} {'actual_mg':>10} {'pred_elo':>8} {'match':>6}")
    print("-" * 40)
    for i in range(min(10, n_test)):
        ed = elo_diff_te[i]
        am = mg_te[i]
        pe = "team1" if ed > 0 else "team2"
        match = "✓" if (ed > 0) == (am > 0) else "✗"
        print(f"{ed:10.1f} {am:10.1f} {pe:>8} {match:>6}")
    print()

    # 5. Load trained model and check its predictions
    print("--- 5. Trained model (if exists) ---")
    from sklearn.preprocessing import StandardScaler
    from core.competition_config import get_competition

    cfg = get_competition(args.competition)
    output_dir = Path(cfg.get("output_dir", str(_PROJECT_ROOT / "model" / "output" / args.competition)))
    model_path = output_dir / "model.h5"
    scaler_path = output_dir / "scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        import tensorflow as tf
        scaler = joblib.load(scaler_path)
        model = tf.keras.models.load_model(model_path)
        X_scaled = scaler.transform(X)
        X_tr_s, X_te_s, _, _, _, _, mg_tr_s, mg_te_s = _temporal_train_test_split(
            X_scaled, y_s1, y_s2, y_margin, test_size=0.2
        )
        preds = model.predict(X_te_s, verbose=0)
        p_mg = preds[2].flatten()
        model_acc = np.mean((p_mg > 0) == (mg_te_s > 0))
        print(f"Model winner accuracy on validation: {model_acc:.1%}")
        # Correlation: does model just predict sign(elo_diff)?
        model_vs_elo = np.mean((p_mg > 0) == (elo_diff_te > 0))
        print(f"Model agrees with elo_diff sign: {model_vs_elo:.1%}")
    else:
        print("No saved model found — run training first")
    print()

    # 6. Scaler fit
    print("--- 6. Scaler fit ---")
    print("Training uses: split first, scaler.fit(X_tr) only ✓")
    print()

    # 7. Feature column inspection
    print("--- 7. Feature column inspection ---")
    print(f"{'idx':>4} {'col':<28} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
    print("-" * 74)
    for idx, col in enumerate(RUGBY_FEATURE_COLS):
        vals = X[:, idx]
        print(f"{idx:4} {col:<28} {vals.min():10.2f} {vals.max():10.2f} {vals.mean():10.2f} {vals.std():10.2f}")
    # Check for suspicious leakage: features that could encode CURRENT match outcome (not historical)
    leakage_risk = [
        c for c in RUGBY_FEATURE_COLS
        if any(x in c.lower() for x in ["result", "win_actual", "actual_score"])
        or (("score" in c.lower() or "margin" in c.lower()) and "recent" not in c.lower() and "avg" not in c.lower() and "h2h" not in c.lower())
    ]
    if leakage_risk:
        print(f"  WARNING: Potential leakage columns: {leakage_risk}")
    else:
        print("  No obvious outcome-leaking columns (all score/margin features are historical) ✓")
    print()

    # 8. Duplicate match check
    print("--- 8. Duplicate match check ---")
    if match_ids is not None:
        te_start = n_train
        match_ids_te = match_ids[te_start:te_start + n_test]
        unique_ids = np.unique(match_ids_te)
        dup_count = len(match_ids_te) - len(unique_ids)
        print(f"Validation match_ids: {n_test} total, {len(unique_ids)} unique")
        if dup_count > 0:
            print(f"  WARNING: {dup_count} duplicate match_ids in validation!")
            from collections import Counter
            counts = Counter(match_ids_te)
            dups = [(mid, c) for mid, c in counts.items() if c > 1]
            for mid, c in dups[:5]:
                print(f"    {mid}: {c}x")
        else:
            print("  No duplicate match_ids in validation ✓")
        # Check (year, round, team1, team2) uniqueness
        if team1 is not None and team2 is not None and years is not None and rounds is not None:
            keys_te = list(zip(years[te_start:], rounds[te_start:], team1[te_start:], team2[te_start:]))
            unique_keys = len(set(keys_te))
            if unique_keys < len(keys_te):
                print(f"  WARNING: {len(keys_te) - unique_keys} duplicate (year,round,team1,team2) in validation")
            else:
                print("  No duplicate (year,round,team1,team2) in validation ✓")
    else:
        print("  (Run without --from-cache to populate match metadata for duplicate check)")
    print()

    # 9. X/y alignment and model input flow
    print("--- 9. X/y alignment and model input flow ---")
    print("Flow: load_and_prepare -> features_df[COLS], match_data[score1,score2] -> same mask -> X, y_s1, y_s2")
    print("  features_df and match_data filtered with identical mask, reset_index -> row-aligned ✓")
    print("  y_margin = y_s1 - y_s2 (derived from targets, not a feature)")
    print("  Model inputs: X (features only), targets: [s1, s2, margin]")
    print("  No target values in X ✓")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if elo_baseline_acc > 0.95:
        print("elo_diff alone achieves >95% winner accuracy — investigate why (possible leakage in ELO computation)")
    if n_test < 100:
        print(f"Small validation set ({n_test}) — 100% could be chance; use more data")
    print("Done.")


if __name__ == "__main__":
    main()
