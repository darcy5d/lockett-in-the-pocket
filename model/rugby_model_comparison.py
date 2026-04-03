#!/usr/bin/env python3
"""
Rugby League model comparison — terminal script.

Compares model types and targets:
- XGBoost: binary (winner), margin (regression)
- LightGBM: binary (winner), margin (regression)
- NN (Keras): multi-output (scores/margin)
- Ensembles: avg margin, avg binary prob, combined

Same data, same temporal split. Prints comparison table.
Use before deploying to GUI. Run: python model/rugby_model_comparison.py --competition qld-cup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.competition_config import get_competition
from core.rugby_feature_engine import RUGBY_FEATURE_COLS, RugbyFeatureEngine
from core.rugby_player_presence import compute_presence

# Optional imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def _temporal_split(X, y_margin, y_winner, test_size: float = 0.2):
    """Split by time: last test_size fraction for validation."""
    n = len(X)
    n_test = max(1, int(n * test_size))
    n_train = n - n_test
    idx = np.arange(n)
    tr, te = idx[:n_train], idx[n_train:]
    return (
        X[tr], X[te],
        y_margin[tr], y_margin[te],
        y_winner[tr], y_winner[te],
    )


def _eval(y_margin_te, y_winner_te, p_margin, p_winner):
    """Compute metrics. p_winner: probabilities (use >0.5) or binary. p_margin: regression output or None."""
    if p_margin is not None:
        p_winner_final = (np.asarray(p_margin) > 0).astype(int)
    else:
        # Binary model: p_winner is prob or 0/1
        pw = np.asarray(p_winner)
        p_winner_final = (pw > 0.5).astype(int) if np.any((pw > 0) & (pw < 1)) else pw.astype(int)

    winner_acc = np.mean(p_winner_final == y_winner_te)
    rmse_mg = np.sqrt(np.mean((np.asarray(p_margin) - y_margin_te) ** 2)) if p_margin is not None else float("nan")
    mae_mg = np.mean(np.abs(np.asarray(p_margin) - y_margin_te)) if p_margin is not None else float("nan")
    bias = np.mean(np.asarray(p_margin) - y_margin_te) if p_margin is not None else float("nan")
    return winner_acc, rmse_mg, mae_mg, bias


def _sigmoid(x: np.ndarray, scale: float = 20.0) -> np.ndarray:
    """Convert margin to probability-like [0,1]. scale controls steepness."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x) / scale))


def _train_xgb_binary(X_tr, X_te, y_winner_tr, y_winner_te, y_margin_te):
    if not HAS_XGB:
        return None, None, "XGB"
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_tr, y_winner_tr, eval_set=[(X_te, y_winner_te)], verbose=False)
    p_prob = model.predict_proba(X_te)[:, 1]
    p_winner = p_prob
    p_margin = None  # no margin from binary
    return p_margin, p_winner, None


def _train_xgb_margin(X_tr, X_te, y_margin_tr, y_margin_te, y_winner_te):
    if not HAS_XGB:
        return None, None, "XGB"
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_tr, y_margin_tr, eval_set=[(X_te, y_margin_te)], verbose=False)
    p_margin = model.predict(X_te)
    p_winner = (p_margin > 0).astype(int)
    return p_margin, p_winner, None


def _train_lgb_binary(X_tr, X_te, y_winner_tr, y_winner_te, y_margin_te):
    if not HAS_LGB:
        return None, None, "LGB"
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_winner_tr, eval_set=[(X_te, y_winner_te)])
    p_prob = model.predict_proba(X_te)[:, 1]
    p_winner = p_prob
    p_margin = None
    return p_margin, p_winner, None


def _train_lgb_margin(X_tr, X_te, y_margin_tr, y_margin_te, y_winner_te):
    if not HAS_LGB:
        return None, None, "LGB"
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_margin_tr, eval_set=[(X_te, y_margin_te)])
    p_margin = model.predict(X_te)
    p_winner = (p_margin > 0).astype(int)
    return p_margin, p_winner, None


def run_comparison(
    competition_id: str = "qld-cup",
    year_from: int = 2000,
    year_to: int = 2025,
    test_size: float = 0.2,
) -> None:
    from model.rugby_train import load_and_prepare

    print(f"Loading data for {competition_id} ({year_from}–{year_to})…", flush=True)
    X, y_t1, y_t2, y_k1, y_k2, y_s1, y_s2, y_margin, _ = load_and_prepare(
        competition_id, year_from, year_to
    )
    y_winner = (y_s1 > y_s2).astype(int)
    n = len(X)
    print(f"Total samples: {n}", flush=True)

    X_tr, X_te, y_margin_tr, y_margin_te, y_winner_tr, y_winner_te = _temporal_split(
        X, y_margin, y_winner, test_size
    )
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}", flush=True)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    # Use DataFrames for LGB to avoid feature-names warning
    X_tr_df = pd.DataFrame(X_tr_s, columns=RUGBY_FEATURE_COLS)
    X_te_df = pd.DataFrame(X_te_s, columns=RUGBY_FEATURE_COLS)

    results = []
    preds_store = {"margin": [], "binary_prob": []}  # for ensembles

    # XGB binary (numpy)
    if HAS_XGB:
        print("Training XGBoost binary…", flush=True)
        p_mg, p_w, _ = _train_xgb_binary(X_tr_s, X_te_s, y_winner_tr, y_winner_te, y_margin_te)
        if p_w is not None:
            acc, rmse, mae, bias = _eval(y_margin_te, y_winner_te, None, p_w)
            results.append(("XGBoost", "binary", acc, rmse, mae, bias))
            preds_store["binary_prob"].append(("XGB", p_w))

        print("Training XGBoost margin…", flush=True)
        p_mg, p_w, _ = _train_xgb_margin(X_tr_s, X_te_s, y_margin_tr, y_margin_te, y_winner_te)
        if p_mg is not None:
            acc, rmse, mae, bias = _eval(y_margin_te, y_winner_te, p_mg, p_w)
            results.append(("XGBoost", "margin", acc, rmse, mae, bias))
            preds_store["margin"].append(("XGB", p_mg))

    # LGB binary (DataFrame for feature names)
    if HAS_LGB:
        print("Training LightGBM binary…", flush=True)
        p_mg, p_w, _ = _train_lgb_binary(X_tr_df, X_te_df, y_winner_tr, y_winner_te, y_margin_te)
        if p_w is not None:
            acc, rmse, mae, bias = _eval(y_margin_te, y_winner_te, None, p_w)
            results.append(("LightGBM", "binary", acc, rmse, mae, bias))
            preds_store["binary_prob"].append(("LGB", p_w))

        print("Training LightGBM margin…", flush=True)
        p_mg, p_w, _ = _train_lgb_margin(X_tr_df, X_te_df, y_margin_tr, y_margin_te, y_winner_te)
        if p_mg is not None:
            acc, rmse, mae, bias = _eval(y_margin_te, y_winner_te, p_mg, p_w)
            results.append(("LightGBM", "margin", acc, rmse, mae, bias))
            preds_store["margin"].append(("LGB", p_mg))

    # NN
    if HAS_TF:
        print("Training NN (Keras)…", flush=True)
        from model.rugby_train import build_model, _configure_tensorflow

        _configure_tensorflow()
        n_train = len(X_tr)
        tr_idx = np.arange(n_train)

        tries_mean = float(np.mean(np.concatenate([y_t1, y_t2])))
        kick_mean = float(np.mean(np.concatenate([y_k1, y_k2])))

        model = build_model(X.shape[1], tries_bias=tries_mean, kick_pts_bias=kick_mean)
        model.fit(
            X_tr_s,
            [
                y_t1[tr_idx], y_t2[tr_idx],
                y_k1[tr_idx], y_k2[tr_idx],
                y_s1[tr_idx], y_s2[tr_idx],
                y_margin[tr_idx],
            ],
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )
        preds = model.predict(X_te_s, verbose=0)
        p_mg = preds[6].flatten()
        p_winner = (p_mg > 0).astype(int)
        acc, rmse, mae, bias = _eval(y_margin_te, y_winner_te, p_mg, p_winner)
        results.append(("NN (Keras)", "multi-output", acc, rmse, mae, bias))
        preds_store["margin"].append(("NN", p_mg))

    # Ensembles
    print("Computing ensembles…", flush=True)
    margins = [p for _, p in preds_store["margin"]]
    probs = [p for _, p in preds_store["binary_prob"]]

    if len(margins) >= 2:
        avg_margin = np.mean(margins, axis=0)
        acc, rmse, mae, bias = _eval(y_margin_te, y_winner_te, avg_margin, (avg_margin > 0).astype(int))
        names = "+".join(n for n, _ in preds_store["margin"])
        results.append((f"Ensemble ({names})", "avg margin", acc, rmse, mae, bias))

    if len(probs) >= 2:
        avg_prob = np.mean(probs, axis=0)
        acc, rmse, mae, bias = _eval(y_margin_te, y_winner_te, None, avg_prob)
        names = "+".join(n for n, _ in preds_store["binary_prob"])
        results.append((f"Ensemble ({names})", "avg binary", acc, rmse, mae, bias))

    if margins and probs:
        # Combined: margin sigmoid + binary probs, avg
        margin_probs = [_sigmoid(m) for m in margins]
        combined = np.mean(margin_probs + probs, axis=0)
        acc, rmse, mae, bias = _eval(y_margin_te, y_winner_te, None, combined)
        results.append(("Ensemble (all)", "margin+binary", acc, rmse, mae, bias))

    # Print table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<24} {'Target':<14} {'Winner Acc':>12} {'RMSE margin':>12} {'MAE margin':>12} {'Bias':>8}")
    print("-" * 70)
    for model_name, target, acc, rmse, mae, bias in results:
        rmse_s = f"{rmse:.2f}" if not np.isnan(rmse) else "—"
        mae_s = f"{mae:.2f}" if not np.isnan(mae) else "—"
        bias_s = f"{bias:.2f}" if not np.isnan(bias) else "—"
        print(f"{model_name:<24} {target:<14} {acc:>11.1%} {rmse_s:>12} {mae_s:>12} {bias_s:>8}")
    print("=" * 70)

    best = max(results, key=lambda r: r[2])
    print(f"\nBest winner accuracy: {best[0]} {best[1]} @ {best[2]:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compare rugby model types (XGB, LGB, NN). Terminal-only; no GUI."
    )
    ap.add_argument("--competition", type=str, default="qld-cup")
    ap.add_argument("--competitions", type=str, default=None, help="Comma-separated, e.g. qld-cup,nsw-cup")
    ap.add_argument("--year-from", type=int, default=2000)
    ap.add_argument("--year-to", type=int, default=2025)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    comps = [c.strip() for c in args.competitions.split(",")] if args.competitions else [args.competition]
    for comp in comps:
        if len(comps) > 1:
            print(f"\n{'#' * 70}\n# {comp.upper()}\n{'#' * 70}")
        run_comparison(comp, args.year_from, args.year_to, args.test_size)
