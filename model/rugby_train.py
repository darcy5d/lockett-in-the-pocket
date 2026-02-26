#!/usr/bin/env python3
"""
Rugby League model training — parameterised by competition.

Predicts tries and kick_pts per team; derives scores and margin.
Uses RugbyFeatureEngine and rugby_player_presence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from pathlib import Path

from core.competition_config import get_competition
from core.rugby_feature_engine import RUGBY_FEATURE_COLS, RugbyFeatureEngine
from core.rugby_player_presence import compute_presence


def _temporal_train_test_split(X, y_t1, y_t2, y_k1, y_k2, y_s1, y_s2, y_margin, test_size: float = 0.2):
    """Split by time: last test_size fraction for validation (no future leakage)."""
    n = len(X)
    n_test = max(1, int(n * test_size))
    n_train = n - n_test
    idx = np.arange(n)
    tr, te = idx[:n_train], idx[n_train:]
    return (
        X[tr], X[te],
        y_t1[tr], y_t1[te],
        y_t2[tr], y_t2[te],
        y_k1[tr], y_k1[te],
        y_k2[tr], y_k2[te],
        y_s1[tr], y_s1[te],
        y_s2[tr], y_s2[te],
        y_margin[tr], y_margin[te],
    )


def _configure_tensorflow():
    """Apple Silicon M2: Metal GPU if available; memory growth for 16GB; thread limits."""
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
    except RuntimeError:
        pass
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def get_output_dir(competition_id: str) -> Path:
    cfg = get_competition(competition_id)
    if not cfg:
        raise ValueError(f"Unknown competition: {competition_id}")
    return Path(cfg.get("output_dir", str(_PROJECT_ROOT / "model" / "output" / competition_id)))


def load_and_prepare(
    competition_id: str,
    year_from: int,
    year_to: int,
) -> tuple:
    """Load matches, compute presence, build features. Returns (X, y_t1, y_t2, y_k1, y_k2, y_s1, y_s2, y_margin, engine)."""
    print("Computing player presence…", flush=True)
    compute_presence(competition_id)
    print("Building features…", flush=True)
    engine = RugbyFeatureEngine(competition_id)
    # Load from 10 years before training start for ELO warm-up (not full history)
    elo_warmup_from = max(1908, year_from - 10)
    engine.load_matches(year_from=elo_warmup_from, year_to=year_to)
    features_df = engine.compute_training_features()
    if features_df.empty:
        raise RuntimeError(f"No training data for {competition_id}")
    match_data = engine.match_data
    mask = ((match_data["year"] >= year_from) & (match_data["year"] <= year_to)).values
    features_df = features_df.iloc[mask].reset_index(drop=True)
    match_data = match_data.iloc[mask].reset_index(drop=True)
    if features_df.empty:
        raise RuntimeError(f"No training data for {competition_id} in {year_from}–{year_to}")
    X = features_df[RUGBY_FEATURE_COLS].values.astype(np.float32)
    y_t1 = match_data["tries1"].values.astype(np.float32)
    y_t2 = match_data["tries2"].values.astype(np.float32)
    y_k1 = match_data["kick_pts1"].values.astype(np.float32)
    y_k2 = match_data["kick_pts2"].values.astype(np.float32)
    y_s1 = match_data["score1"].values.astype(np.float32)
    y_s2 = match_data["score2"].values.astype(np.float32)
    y_margin = (y_s1 - y_s2).astype(np.float32)
    return X, y_t1, y_t2, y_k1, y_k2, y_s1, y_s2, y_margin, engine


def build_model(
    feature_size: int,
    tries_bias: float = 3.5,
    kick_pts_bias: float = 10.0,
    hyperparams: dict | None = None,
) -> tf.keras.Model:
    hp = hyperparams or {}
    d1 = hp.get("dense_1", 128)
    d2 = hp.get("dense_2", 64)
    d3 = hp.get("dense_3", 32)
    dropout = hp.get("dropout", 0.1)
    l2_val = hp.get("l2", 0.0)
    lr = hp.get("learning_rate", 0.001)
    reg = tf.keras.regularizers.L2(l2_val) if l2_val else None
    inp = tf.keras.Input(shape=(feature_size,), name="match_input")
    x = tf.keras.layers.Dense(d1, activation="relu", kernel_regularizer=reg)(inp)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(d2, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(d3, activation="relu", kernel_regularizer=reg)(x)

    tries_init = np.log(np.exp(tries_bias) - 1) if tries_bias > 0 else 0.0
    kick_init = np.log(np.exp(kick_pts_bias) - 1) if kick_pts_bias > 0 else 0.0

    t1_tries = tf.keras.layers.Dense(
        1, activation="softplus",
        bias_initializer=tf.keras.initializers.Constant(tries_init),
        name="team1_tries",
    )(x)
    t2_tries = tf.keras.layers.Dense(
        1, activation="softplus",
        bias_initializer=tf.keras.initializers.Constant(tries_init),
        name="team2_tries",
    )(x)
    t1_kick = tf.keras.layers.Dense(
        1, activation="softplus",
        bias_initializer=tf.keras.initializers.Constant(kick_init),
        name="team1_kick_pts",
    )(x)
    t2_kick = tf.keras.layers.Dense(
        1, activation="softplus",
        bias_initializer=tf.keras.initializers.Constant(kick_init),
        name="team2_kick_pts",
    )(x)

    t1_score = tf.keras.layers.Lambda(
        lambda v: v[0] * 4 + v[1], name="team1_score"
    )([t1_tries, t1_kick])
    t2_score = tf.keras.layers.Lambda(
        lambda v: v[0] * 4 + v[1], name="team2_score"
    )([t2_tries, t2_kick])
    margin = tf.keras.layers.Lambda(
        lambda v: v[0] - v[1], name="calculated_margin"
    )([t1_score, t2_score])

    model = tf.keras.Model(
        inputs=inp,
        outputs=[t1_tries, t2_tries, t1_kick, t2_kick, t1_score, t2_score, margin],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
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
            "calculated_margin": ["mae"],
        },
    )
    return model


def _epoch_callback(epochs_total: int):
    def on_epoch_end(epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss", 0)
        loss = logs.get("loss", 0)
        print(f"Epoch {epoch + 1}/{epochs_total} — loss: {loss:.2f} val_loss: {val_loss:.2f}", flush=True)
    return tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)


def train(
    competition_id: str = "nrl",
    year_from: int = 2020,
    year_to: int = 2025,
    epochs: int = 30,
    patience: int = 10,
    hyperparams: dict | None = None,
) -> None:
    _configure_tensorflow()
    output_dir = get_output_dir(competition_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data for {competition_id} ({year_from}–{year_to})…", flush=True)
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
    batch_size = (hyperparams or {}).get("batch_size", 32)
    model = build_model(X.shape[1], tries_bias=tries_mean, kick_pts_bias=kick_mean, hyperparams=hyperparams)

    print(f"Training for up to {epochs} epochs (patience={patience})…", flush=True)
    model.fit(
        X_tr,
        [t1_tr, t2_tr, k1_tr, k2_tr, s1_tr, s2_tr, mg_tr],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            _epoch_callback(epochs),
        ],
    )

    print("Evaluating…", flush=True)
    preds = model.predict(X_te, verbose=0)
    p_s1 = preds[4].flatten()
    p_s2 = preds[5].flatten()
    p_mg = preds[6].flatten()
    mae_s1 = np.mean(np.abs(p_s1 - s1_te))
    mae_s2 = np.mean(np.abs(p_s2 - s2_te))
    mae_mg = np.mean(np.abs(p_mg - mg_te))
    rmse_mg = np.sqrt(np.mean((p_mg - mg_te) ** 2))
    winner_correct = np.sum((p_mg > 0) == (mg_te > 0))
    winner_acc = winner_correct / len(mg_te) if len(mg_te) > 0 else 0.0
    margin_bias = np.mean(p_mg - mg_te)
    print(f"MAE team1: {mae_s1:.2f} team2: {mae_s2:.2f} margin: {mae_mg:.2f}", flush=True)
    print(f"Winner acc: {winner_acc:.1%} | RMSE margin: {rmse_mg:.2f} | Margin bias: {margin_bias:.2f}", flush=True)
    joblib.dump(scaler, output_dir / "scaler.joblib")
    model.save(output_dir / "model.h5")
    with open(output_dir / "feature_cols.json", "w") as f:
        json.dump(RUGBY_FEATURE_COLS, f)
    with open(output_dir / "evaluation_metrics.txt", "w") as f:
        f.write(
            f"MAE team1_score: {mae_s1:.2f}\n"
            f"MAE team2_score: {mae_s2:.2f}\n"
            f"MAE margin: {mae_mg:.2f}\n"
            f"RMSE margin: {rmse_mg:.2f}\n"
            f"Winner accuracy: {winner_acc:.1%}\n"
            f"Margin bias: {margin_bias:.2f}\n"
        )
    print(f"Saved to {output_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", type=str, default="nrl")
    ap.add_argument("--year-from", type=int, default=2020)
    ap.add_argument("--year-to", type=int, default=2025)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    train(args.competition, args.year_from, args.year_to, args.epochs)
