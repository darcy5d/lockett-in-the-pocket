#!/usr/bin/env python3
"""
model/tune_hyperband.py — Hyperband hyperparameter search for AFL model.

Optimises winner accuracy (derived from predicted margin). Saves best
hyperparameters to model/output/best_hyperparams.json. With --save-final,
trains the best model and saves all artefacts for prediction.

Usage:
    python model/tune_hyperband.py --year-from 1990 --year-to 2025 --save-final
    python model/tune_hyperband.py --max-epochs 30 --trials 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.train import (
    NUM_MATCH_FEATURES,
    load_and_prepare,
    build_model,
    OUTPUT_DIR,
)


def winner_accuracy_metric(y_true, y_pred):
    """Accuracy of (pred > 0) == (true > 0) for margin output."""
    pred_wins = tf.cast(y_pred > 0, tf.float32)
    actual_wins = tf.cast(y_true > 0, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(pred_wins, actual_wins), tf.float32))


def build_model_hp(hp, feature_size: int, num_players: int, goals_bias: float, behinds_bias: float):
    """Build model with tunable hyperparameters for keras-tuner."""
    embedding_dim = hp.Int("embedding_dim", min_value=16, max_value=64, step=16)
    dense_1 = hp.Int("dense_1", min_value=128, max_value=512, step=128)
    dense_2 = hp.Int("dense_2", min_value=64, max_value=256, step=64)
    dense_3 = hp.Int("dense_3", min_value=32, max_value=128, step=32)
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.3, step=0.05)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-3, sampling="log")
    l2 = hp.Float("l2", min_value=1e-5, max_value=1e-2, sampling="log")

    hyperparams = {
        "embedding_dim": embedding_dim,
        "dense_1": dense_1,
        "dense_2": dense_2,
        "dense_3": dense_3,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": hp.Choice("batch_size", values=[16, 32, 64]),
        "l2": float(l2),
    }

    model = build_model(
        feature_size, num_players, goals_bias, behinds_bias, hyperparams=hyperparams
    )

    huber = tf.keras.losses.Huber(delta=5.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "team1_goals": huber,
            "team1_behinds": huber,
            "team2_goals": huber,
            "team2_behinds": huber,
            "calculated_margin": huber,
        },
        loss_weights={
            "team1_goals": 5.0, "team1_behinds": 5.0,
            "team2_goals": 5.0, "team2_behinds": 5.0,
            "calculated_margin": 2.0,
        },
        metrics={
            "team1_goals": tf.keras.metrics.MeanAbsoluteError(name="mae"),
            "team1_behinds": tf.keras.metrics.MeanAbsoluteError(name="mae"),
            "team2_goals": tf.keras.metrics.MeanAbsoluteError(name="mae"),
            "team2_behinds": tf.keras.metrics.MeanAbsoluteError(name="mae"),
            "calculated_margin": [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanMetricWrapper(winner_accuracy_metric, name="winner_accuracy"),
            ],
        },
    )
    return model


def run_search(
    year_from: int,
    year_to: int,
    max_epochs: int = 100,
    patience: int = 15,
    hyperband_iterations: int = 2,
    save_final: bool = False,
) -> dict:
    """Run Hyperband search and optionally train best model."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (X_match, t1_idx, t2_idx,
     enh_t1_raw, enh_t2_raw,
     y_winner, y_margin,
     y_t1g, y_t1b, y_t2g, y_t2b,
     feature_cols, num_players, engine) = load_and_prepare(year_from, year_to)

    match_scaler = StandardScaler()
    X_match_scaled = match_scaler.fit_transform(X_match)

    enh_scaler = StandardScaler()
    both_enh = np.vstack([enh_t1_raw, enh_t2_raw])
    enh_scaler.fit(both_enh)
    enh_t1 = enh_scaler.transform(enh_t1_raw)
    enh_t2 = enh_scaler.transform(enh_t2_raw)

    goals_mean = float(np.mean(np.concatenate([y_t1g, y_t2g])))
    behinds_mean = float(np.mean(np.concatenate([y_t1b, y_t2b])))

    (Xm_tr, Xm_te, t1_tr, t1_te, t2_tr, t2_te,
     e1_tr, e1_te, e2_tr, e2_te,
     ymg_tr, ymg_te,
     yt1g_tr, yt1g_te, yt1b_tr, yt1b_te,
     yt2g_tr, yt2g_te, yt2b_tr, yt2b_te,
     yw_tr, yw_te) = train_test_split(
        X_match_scaled, t1_idx, t2_idx, enh_t1, enh_t2,
        y_margin, y_t1g, y_t1b, y_t2g, y_t2b, y_winner,
        test_size=0.2, random_state=42,
    )

    def model_builder(hp):
        return build_model_hp(
            hp, NUM_MATCH_FEATURES, num_players, goals_mean, behinds_mean
        )

    tuner = kt.Hyperband(
        model_builder,
        objective=kt.Objective("val_calculated_margin_winner_accuracy", direction="max"),
        max_epochs=max_epochs,
        factor=3,
        hyperband_iterations=hyperband_iterations,
        overwrite=True,
        directory=str(OUTPUT_DIR / "tuner"),
        project_name="afl_hyperband",
    )

    print(f"\n=== Hyperband search (max_epochs={max_epochs}, patience={patience}, "
          f"iterations={hyperband_iterations}) ===")
    tuner.search(
        [Xm_tr, t1_tr, t2_tr, e1_tr, e2_tr],
        {"team1_goals": yt1g_tr, "team1_behinds": yt1b_tr,
         "team2_goals": yt2g_tr, "team2_behinds": yt2b_tr, "calculated_margin": ymg_tr},
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_calculated_margin_winner_accuracy",
                mode="max",
                patience=patience,
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    try:
        l2_val = float(best_hp.get("l2"))
    except (KeyError, TypeError, ValueError, AttributeError):
        l2_val = 0.0
    best_hyperparams = {
        "embedding_dim": best_hp.get("embedding_dim"),
        "dense_1": best_hp.get("dense_1"),
        "dense_2": best_hp.get("dense_2"),
        "dense_3": best_hp.get("dense_3"),
        "dropout": best_hp.get("dropout"),
        "learning_rate": float(best_hp.get("learning_rate")),
        "batch_size": best_hp.get("batch_size"),
        "l2": l2_val,
    }

    best_hyperparams_path = OUTPUT_DIR / "best_hyperparams.json"
    with open(best_hyperparams_path, "w") as f:
        json.dump(best_hyperparams, f, indent=2)
    print(f"\nBest hyperparams saved to {best_hyperparams_path}")
    for k, v in best_hyperparams.items():
        print(f"  {k}: {v}")

    if save_final:
        print("\n=== Training final model with best hyperparams ===")
        from model.train import train
        train(year_from, year_to, max_epochs, hyperparams=best_hyperparams, patience=patience)

    return best_hyperparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperband hyperparameter search for AFL model")
    parser.add_argument("--year-from", type=int, default=1990)
    parser.add_argument("--year-to", type=int, default=2025)
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--hyperband-iterations", type=int, default=2, help="Hyperband algorithm iterations")
    parser.add_argument("--save-final", action="store_true", help="Train and save best model after search")
    args = parser.parse_args()
    run_search(
        args.year_from,
        args.year_to,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hyperband_iterations=args.hyperband_iterations,
        save_final=args.save_final,
    )
