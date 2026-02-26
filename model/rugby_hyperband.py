#!/usr/bin/env python3
"""
Rugby League Hyperband hyperparameter search.

Optimises winner accuracy (derived from predicted margin). Saves best
hyperparameters to model/output/{competition_id}/best_hyperparams.json.
With --save-final, trains the best model and saves all artefacts.

Usage:
    python model/rugby_hyperband.py --competition nrl --year-from 2020 --year-to 2025 --save-final
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    message="Skipping variable loading for optimizer",
)
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler

from model.rugby_train import (
    build_model,
    get_output_dir,
    load_and_prepare,
    _configure_tensorflow,
    _temporal_train_test_split,
)


def winner_accuracy_metric(y_true, y_pred):
    """Accuracy of (pred > 0) == (true > 0) for margin output."""
    pred_wins = tf.cast(y_pred > 0, tf.float32)
    actual_wins = tf.cast(y_true > 0, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(pred_wins, actual_wins), tf.float32))


def build_model_hp(hp, feature_size: int, tries_bias: float, kick_pts_bias: float):
    """Build Rugby model with tunable hyperparameters for keras-tuner."""
    dense_1 = hp.Int("dense_1", min_value=64, max_value=256, step=64)
    dense_2 = hp.Int("dense_2", min_value=32, max_value=128, step=32)
    dense_3 = hp.Int("dense_3", min_value=16, max_value=64, step=16)
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.3, step=0.05)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-3, sampling="log")
    l2 = hp.Float("l2", min_value=1e-5, max_value=1e-2, sampling="log")

    hyperparams = {
        "dense_1": dense_1,
        "dense_2": dense_2,
        "dense_3": dense_3,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": hp.Choice("batch_size", values=[16, 32, 64]),
        "l2": float(l2),
    }

    model = build_model(feature_size, tries_bias=tries_bias, kick_pts_bias=kick_pts_bias, hyperparams=hyperparams)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "team1_tries": tf.keras.losses.Huber(delta=2.0),
            "team2_tries": tf.keras.losses.Huber(delta=2.0),
            "team1_kick_pts": tf.keras.losses.Huber(delta=3.0),
            "team2_kick_pts": tf.keras.losses.Huber(delta=3.0),
            "team1_score": tf.keras.losses.Huber(delta=5.0),
            "team2_score": tf.keras.losses.Huber(delta=5.0),
            "calculated_margin": tf.keras.losses.Huber(delta=5.0),
        },
        loss_weights={
            "team1_tries": 1.0, "team2_tries": 1.0,
            "team1_kick_pts": 1.0, "team2_kick_pts": 1.0,
            "team1_score": 0.5, "team2_score": 0.5,
            "calculated_margin": 1.0,
        },
        metrics={
            "team1_score": ["mae"],
            "team2_score": ["mae"],
            "calculated_margin": [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanMetricWrapper(winner_accuracy_metric, name="winner_accuracy"),
            ],
        },
    )
    return model


def run_search(
    competition_id: str = "nrl",
    year_from: int = 2020,
    year_to: int = 2025,
    max_epochs: int = 100,
    patience: int = 15,
    hyperband_iterations: int = 2,
    save_final: bool = False,
) -> dict:
    """Run Hyperband search and optionally train best model."""
    _configure_tensorflow()
    output_dir = get_output_dir(competition_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y_t1, y_t2, y_k1, y_k2, y_s1, y_s2, y_margin, _ = load_and_prepare(competition_id, year_from, year_to)
    print(f"Training samples: {len(X)}", flush=True)

    (X_tr, X_te,
     t1_tr, t1_te, t2_tr, t2_te,
     k1_tr, k1_te, k2_tr, k2_te,
     s1_tr, s1_te, s2_tr, s2_te,
     mg_tr, mg_te) = _temporal_train_test_split(
        X, y_t1, y_t2, y_k1, y_k2, y_s1, y_s2, y_margin, test_size=0.2
    )

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    tries_mean = float(np.mean(np.concatenate([y_t1, y_t2])))
    kick_mean = float(np.mean(np.concatenate([y_k1, y_k2])))

    def model_builder(hp):
        return build_model_hp(hp, X.shape[1], tries_bias=tries_mean, kick_pts_bias=kick_mean)

    tuner_dir = output_dir / "tuner"
    tuner = kt.Hyperband(
        model_builder,
        objective=kt.Objective("val_calculated_margin_winner_accuracy", direction="max"),
        max_epochs=max_epochs,
        factor=3,
        hyperband_iterations=hyperband_iterations,
        overwrite=True,
        directory=str(tuner_dir),
        project_name=f"{competition_id}_hyperband",
    )

    print(
        f"\n=== Rugby Hyperband ({competition_id}) {year_from}–{year_to} "
        f"max_epochs={max_epochs}, patience={patience}, iterations={hyperband_iterations} ===",
        flush=True,
    )

    tuner.search(
        X_tr,
        [t1_tr, t2_tr, k1_tr, k2_tr, s1_tr, s2_tr, mg_tr],
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
        "dense_1": best_hp.get("dense_1"),
        "dense_2": best_hp.get("dense_2"),
        "dense_3": best_hp.get("dense_3"),
        "dropout": best_hp.get("dropout"),
        "learning_rate": float(best_hp.get("learning_rate")),
        "batch_size": best_hp.get("batch_size"),
        "l2": l2_val,
    }

    best_hyperparams_path = output_dir / "best_hyperparams.json"
    with open(best_hyperparams_path, "w") as f:
        json.dump(best_hyperparams, f, indent=2)
    print(f"\nBest hyperparams saved to {best_hyperparams_path}", flush=True)
    for k, v in best_hyperparams.items():
        print(f"  {k}: {v}", flush=True)

    if save_final:
        print("\n=== Training final model with best hyperparams ===", flush=True)
        from model.rugby_train import train

        train(
            competition_id=competition_id,
            year_from=year_from,
            year_to=year_to,
            epochs=max_epochs,
            patience=patience,
            hyperparams=best_hyperparams,
        )

    return best_hyperparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperband hyperparameter search for Rugby League")
    parser.add_argument("--competition", type=str, default="nrl")
    parser.add_argument("--year-from", type=int, default=2020)
    parser.add_argument("--year-to", type=int, default=2025)
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--hyperband-iterations", type=int, default=2, help="Hyperband algorithm iterations")
    parser.add_argument("--save-final", action="store_true", help="Train and save best model after search")
    args = parser.parse_args()
    run_search(
        competition_id=args.competition,
        year_from=args.year_from,
        year_to=args.year_to,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hyperband_iterations=args.hyperband_iterations,
        save_final=args.save_final,
    )
