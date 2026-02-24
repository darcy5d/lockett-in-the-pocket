#!/usr/bin/env python3
"""
model/train.py — AFL model training with full feature engineering.

Features:
  - Team ELO and player ELO ratings (computed chronologically)
  - Recent form (last 5 matches: win%, avg score, avg margin)
  - Venue win rates per team
  - Head-to-head history (last 5 meetings)
  - Player form-weighted enhanced stats (exp decay over last 10 games)
  - Player embeddings (22 per team, 32-dim)

Architecture:
  - Softplus activation for score branches (no dying ReLU)
  - Bias initialisation to training mean
  - Huber loss for regression outputs
  - Dropout 0.1

Usage:
    python model/train.py
    python model/train.py --year-from 1990 --year-to 2025 --epochs 50
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
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Input, Lambda
from tensorflow.keras.models import Model
from core.feature_engine import (
    MATCH_FEATURE_COLS,
    NUM_MATCH_FEATURES,
    NUM_STATS,
    FeatureEngine,
)
from core.mappings import PlayerMapper
from model.bounded_output_layer import BoundedOutputLayer
from model.custom_loss import mae, mse

EMBEDDING_DIM = 32
MAX_PLAYERS_PER_TEAM = 22
OUTPUT_DIR = _PROJECT_ROOT / "model" / "output"


# ---------------------------------------------------------------------------
# Custom layers
# ---------------------------------------------------------------------------

class MeanPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"output_dim": self.output_dim})
        return cfg


tf.keras.utils.get_custom_objects()["MeanPoolingLayer"] = MeanPoolingLayer
tf.keras.utils.get_custom_objects()["BoundedOutputLayer"] = BoundedOutputLayer


# ---------------------------------------------------------------------------
# Player index + lineup parsing
# ---------------------------------------------------------------------------

def create_player_index() -> dict[str, int]:
    import glob as _glob
    files = _glob.glob(str(_PROJECT_ROOT / "afl_data/data/players/*_performance_details.csv"))
    ids = sorted({Path(f).stem.replace("_performance_details", "") for f in files})
    index = {pid: i + 1 for i, pid in enumerate(ids)}
    index["unknown"] = 0
    print(f"  Player index: {len(index)} entries")
    return index


def lineup_to_indices(players_str: str, player_index: dict[str, int], mapper: PlayerMapper) -> list[int]:
    if not players_str or (isinstance(players_str, float) and np.isnan(players_str)):
        return [0] * MAX_PLAYERS_PER_TEAM
    names = [n.strip() for n in str(players_str).split(";") if n.strip()]
    indices = [player_index.get(mapper.to_player_id(n) or "unknown", 0) for n in names]
    if len(indices) < MAX_PLAYERS_PER_TEAM:
        indices.extend([0] * (MAX_PLAYERS_PER_TEAM - len(indices)))
    return indices[:MAX_PLAYERS_PER_TEAM]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare(year_from: int, year_to: int) -> tuple:
    """
    Load data and compute all features using FeatureEngine.

    Returns:
        (match_features, player_indices_t1, player_indices_t2,
         enhanced_t1, enhanced_t2,
         y_winner, y_margin, y_t1g, y_t1b, y_t2g, y_t2b,
         feature_cols, num_players, engine)
    """
    print(f"\n{'='*60}")
    print(f"Loading data and computing features ({year_from}–{year_to})")
    print(f"{'='*60}")

    # Build player index and mapper
    mapper = PlayerMapper()
    player_index = create_player_index()

    # Feature engine: compute ELO, form, venue, H2H, player form
    engine = FeatureEngine()
    engine.load_matches(year_from, year_to)
    engine.load_lineups()
    engine.load_player_stats()

    features_df, enh_t1_raw, enh_t2_raw = engine.compute_training_features()

    # Build player index arrays (for embeddings)
    match_data = engine.match_data
    print("\nBuilding player index arrays…")
    t1_indices_list, t2_indices_list = [], []
    for _, row in match_data.iterrows():
        team1 = row["team_1_team_name"]
        team2 = row["team_2_team_name"]
        year = int(row["year"])
        rnd = str(row["round_num"])

        # Find lineup
        if engine.lineup_data is not None:
            sub1 = engine.lineup_data[
                (engine.lineup_data["year"] == year)
                & (engine.lineup_data["round_num"] == rnd)
                & (engine.lineup_data["team_name"] == team1)
            ]
            sub2 = engine.lineup_data[
                (engine.lineup_data["year"] == year)
                & (engine.lineup_data["round_num"] == rnd)
                & (engine.lineup_data["team_name"] == team2)
            ]
            ps1 = sub1["players"].iloc[0] if not sub1.empty else ""
            ps2 = sub2["players"].iloc[0] if not sub2.empty else ""
        else:
            ps1, ps2 = "", ""

        t1_indices_list.append(lineup_to_indices(ps1, player_index, mapper))
        t2_indices_list.append(lineup_to_indices(ps2, player_index, mapper))

    t1_idx = np.array(t1_indices_list)
    t2_idx = np.array(t2_indices_list)

    nz1 = np.sum(t1_idx > 0, axis=1).mean()
    nz2 = np.sum(t2_idx > 0, axis=1).mean()
    print(f"  Avg known players: team1={nz1:.1f}, team2={nz2:.1f}")

    # Targets
    y_winner = match_data["score1"].gt(match_data["score2"]).astype(float).values
    y_winner[match_data["score1"].values == match_data["score2"].values] = 0.5
    y_margin = (match_data["score1"] - match_data["score2"]).values
    y_t1g = np.maximum(0, match_data["team_1_final_goals"].values.astype(float))
    y_t1b = np.maximum(0, match_data["team_1_final_behinds"].values.astype(float))
    y_t2g = np.maximum(0, match_data["team_2_final_goals"].values.astype(float))
    y_t2b = np.maximum(0, match_data["team_2_final_behinds"].values.astype(float))

    # Sanity: print feature columns
    print(f"\nMatch feature columns ({NUM_MATCH_FEATURES}):")
    for c in MATCH_FEATURE_COLS:
        print(f"  {c}")
    for c in MATCH_FEATURE_COLS:
        if "goal" in c.lower() or "behind" in c.lower() or "score" in c.lower() and "avg" not in c.lower():
            if c not in ("team1_recent_avg_score", "team2_recent_avg_score"):
                raise ValueError(f"DATA LEAKAGE: '{c}' looks like a score column!")

    X_match = features_df.values
    print(f"\nData shapes: X_match={X_match.shape}, player_idx={t1_idx.shape}, enhanced={enh_t1_raw.shape}")

    return (
        X_match, t1_idx, t2_idx,
        enh_t1_raw, enh_t2_raw,
        y_winner, y_margin, y_t1g, y_t1b, y_t2g, y_t2b,
        MATCH_FEATURE_COLS, len(player_index), engine,
    )


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def _default_hyperparams() -> dict:
    """Default hyperparameters when none are provided."""
    return {
        "embedding_dim": 32,
        "dense_1": 256,
        "dense_2": 128,
        "dense_3": 64,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "l2": 0.0,
    }


def build_model(
    feature_size: int,
    num_players: int,
    goals_bias: float = 14.0,
    behinds_bias: float = 12.0,
    hyperparams: dict | None = None,
) -> tf.keras.Model:
    hp = hyperparams or _default_hyperparams()
    emb_dim = hp.get("embedding_dim", 32)
    dense_1 = hp.get("dense_1", 256)
    dense_2 = hp.get("dense_2", 128)
    dense_3 = hp.get("dense_3", 64)
    dropout = hp.get("dropout", 0.1)
    lr = hp.get("learning_rate", 0.001)
    l2 = hp.get("l2", 0.0)
    reg = tf.keras.regularizers.L2(l2) if l2 > 0 else None

    match_input = Input(shape=(feature_size,), name="match_input")
    t1_input = Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team1_players_input")
    t2_input = Input(shape=(MAX_PLAYERS_PER_TEAM,), name="team2_players_input")
    t1_enhanced = Input(shape=(NUM_STATS,), name="team1_enhanced_input")
    t2_enhanced = Input(shape=(NUM_STATS,), name="team2_enhanced_input")

    embedding = Embedding(
        input_dim=num_players,
        output_dim=emb_dim,
        name="player_embedding",
        embeddings_regularizer=reg,
    )
    t1_emb = MeanPoolingLayer(output_dim=emb_dim, name="team1_aggregated")(embedding(t1_input))
    t2_emb = MeanPoolingLayer(output_dim=emb_dim, name="team2_aggregated")(embedding(t2_input))

    combined = Concatenate()([match_input, t1_emb, t2_emb, t1_enhanced, t2_enhanced])
    x = Dense(dense_1, activation="relu", kernel_regularizer=reg)(combined)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = Dense(dense_2, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = Dense(dense_3, activation="relu", kernel_regularizer=reg)(x)

    # Softplus: smooth, never-zero, avoids dying ReLU
    # Bias init: softplus(b) ~ b for large b, so init to mean target value
    def score_branch(x_shared, enhanced, name, bias_init):
        b = Concatenate()([x_shared, enhanced])
        b = Dense(64, activation="relu", kernel_regularizer=reg)(b)
        b = Dense(32, activation="relu", kernel_regularizer=reg)(b)
        return Dense(
            1,
            activation="softplus",
            bias_initializer=tf.keras.initializers.Constant(np.log(np.exp(bias_init) - 1) if bias_init > 0 else 0.0),
            name=name,
        )(b)

    t1_goals = score_branch(x, t1_enhanced, "team1_goals", goals_bias)
    t1_behinds = score_branch(x, t1_enhanced, "team1_behinds", behinds_bias)
    t2_goals = score_branch(x, t2_enhanced, "team2_goals", goals_bias)
    t2_behinds = score_branch(x, t2_enhanced, "team2_behinds", behinds_bias)

    t1_score = Lambda(lambda v: v[0] * 6 + v[1], name="team1_score")([t1_goals, t1_behinds])
    t2_score = Lambda(lambda v: v[0] * 6 + v[1], name="team2_score")([t2_goals, t2_behinds])
    margin = Lambda(lambda v: v[0] - v[1], name="calculated_margin")([t1_score, t2_score])

    # No softmax winner head — win probability is derived from predicted margin
    # at inference time via calibrated logistic (Goddard 2005)

    model = Model(
        inputs=[match_input, t1_input, t2_input, t1_enhanced, t2_enhanced],
        outputs=[t1_goals, t1_behinds, t2_goals, t2_behinds, margin],
    )

    huber = tf.keras.losses.Huber(delta=5.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
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
            "calculated_margin": tf.keras.metrics.MeanAbsoluteError(name="mae"),
        },
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    year_from: int = 1990,
    year_to: int = 2025,
    epochs: int = 50,
    hyperparams: dict | None = None,
    patience: int = 10,
) -> None:
    print(f"=== AFL Model Training ===")
    print(f"Years: {year_from}–{year_to} | Epochs: {epochs}")
    print(f"Output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (X_match, t1_idx, t2_idx,
     enh_t1_raw, enh_t2_raw,
     y_winner, y_margin,
     y_t1g, y_t1b, y_t2g, y_t2b,
     feature_cols, num_players, engine) = load_and_prepare(year_from, year_to)

    # Scale features
    match_scaler = StandardScaler()
    X_match_scaled = match_scaler.fit_transform(X_match)

    enh_scaler = StandardScaler()
    both_enh = np.vstack([enh_t1_raw, enh_t2_raw])
    enh_scaler.fit(both_enh)
    enh_t1 = enh_scaler.transform(enh_t1_raw)
    enh_t2 = enh_scaler.transform(enh_t2_raw)

    # Compute bias init from training means
    goals_mean = float(np.mean(np.concatenate([y_t1g, y_t2g])))
    behinds_mean = float(np.mean(np.concatenate([y_t1b, y_t2b])))
    margin_sigma = float(np.std(y_margin))
    draw_rate = float(np.mean(y_winner == 0.5))
    print(f"\nTarget means: goals={goals_mean:.1f}, behinds={behinds_mean:.1f}")
    print(f"Margin sigma: {margin_sigma:.1f}, draw rate: {draw_rate:.3f}")

    # Train/test split (no winner one-hot needed)
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

    hp = hyperparams or _default_hyperparams()
    batch_size = hp.get("batch_size", 32)
    model = build_model(NUM_MATCH_FEATURES, num_players, goals_mean, behinds_mean, hyperparams=hyperparams)
    model.summary()

    print(f"\nTraining for up to {epochs} epochs…")
    model.fit(
        [Xm_tr, t1_tr, t2_tr, e1_tr, e2_tr],
        {"team1_goals": yt1g_tr, "team1_behinds": yt1b_tr,
         "team2_goals": yt2g_tr, "team2_behinds": yt2b_tr, "calculated_margin": ymg_tr},
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(OUTPUT_DIR / "best_model.h5"), monitor="val_loss", save_best_only=True, verbose=1
            ),
        ],
    )

    # Evaluate
    print("\nEvaluating…")
    preds = model.predict([Xm_te, t1_te, t2_te, e1_te, e2_te])
    # Model outputs: [t1_goals, t1_behinds, t2_goals, t2_behinds, margin]
    pred_t1g = preds[0].flatten()
    pred_t1b = preds[1].flatten()
    pred_t2g = preds[2].flatten()
    pred_t2b = preds[3].flatten()
    pred_margins = preds[4].flatten()

    def rmse(a, b):
        return float(np.sqrt(np.mean(np.square(np.array(a).flatten() - np.array(b).flatten()))))

    eval_res = model.evaluate(
        [Xm_te, t1_te, t2_te, e1_te, e2_te],
        {"team1_goals": yt1g_te, "team1_behinds": yt1b_te,
         "team2_goals": yt2g_te, "team2_behinds": yt2b_te, "calculated_margin": ymg_te},
        return_dict=True,
    )
    eval_res["rmse_margin"] = rmse(ymg_te, pred_margins)
    eval_res["rmse_team1_goals"] = rmse(yt1g_te, pred_t1g)
    eval_res["rmse_team1_behinds"] = rmse(yt1b_te, pred_t1b)
    eval_res["rmse_team2_goals"] = rmse(yt2g_te, pred_t2g)
    eval_res["rmse_team2_behinds"] = rmse(yt2b_te, pred_t2b)

    # Derive winner accuracy from predicted margin vs actual outcome
    pred_home_wins = pred_margins > 0
    actual_home_wins = ymg_te > 0
    winner_accuracy = float(np.mean(pred_home_wins == actual_home_wins))
    eval_res["winner_accuracy_from_margin"] = winner_accuracy
    print(f"\nWinner accuracy (margin-derived): {winner_accuracy*100:.1f}%")

    # Calibrate logistic k: find k that minimizes log-loss of sigmoid(margin/k) vs actual
    from scipy.optimize import minimize_scalar
    actual_wins_binary = (ymg_te > 0).astype(float)
    actual_wins_binary[ymg_te == 0] = 0.5

    def neg_log_loss(k_val):
        if k_val <= 0:
            return 1e10
        p = 1.0 / (1.0 + np.exp(-pred_margins / k_val))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -float(np.mean(actual_wins_binary * np.log(p) + (1 - actual_wins_binary) * np.log(1 - p)))

    cal_result = minimize_scalar(neg_log_loss, bounds=(10, 80), method="bounded")
    calibration_k = float(cal_result.x)
    print(f"Calibrated k: {calibration_k:.2f} (log-loss: {cal_result.fun:.4f})")

    # Show calibration table
    print("\nCalibration table:")
    for m in [-40, -20, -10, 0, 10, 20, 40]:
        p = 1.0 / (1.0 + np.exp(-m / calibration_k))
        print(f"  Margin {m:+4d} pts -> P(home win) = {p*100:.1f}%")

    # Sample predictions with derived probabilities
    print("\n=== Sample predictions (first 5 test matches) ===")
    for i in range(min(5, len(pred_t1g))):
        pg1, pb1, pg2, pb2 = pred_t1g[i], pred_t1b[i], pred_t2g[i], pred_t2b[i]
        ag1, ab1, ag2, ab2 = yt1g_te[i], yt1b_te[i], yt2g_te[i], yt2b_te[i]
        pm = pred_margins[i]
        p_home = 1.0 / (1.0 + np.exp(-pm / calibration_k))
        print(f"  Match {i+1}: pred {pg1:.1f}.{pb1:.1f} ({pg1*6+pb1:.0f}) vs {pg2:.1f}.{pb2:.1f} ({pg2*6+pb2:.0f})"
              f"  margin={pm:+.0f}  P(home)={p_home*100:.0f}%"
              f"  | actual {ag1:.0f}.{ab1:.0f} ({ag1*6+ab1:.0f}) vs {ag2:.0f}.{ab2:.0f} ({ag2*6+ab2:.0f})")

    print("\n=== Evaluation ===")
    for k, v in eval_res.items():
        print(f"  {k}: {v}")

    # Save artefacts
    print("\nSaving model and state…")
    model.save(str(OUTPUT_DIR / "model.h5"))
    joblib.dump(match_scaler, str(OUTPUT_DIR / "scaler.joblib"))
    joblib.dump(enh_scaler, str(OUTPUT_DIR / "enhanced_scaler.joblib"))

    player_index = create_player_index()
    with open(OUTPUT_DIR / "player_index.json", "w") as f:
        json.dump(player_index, f)
    with open(OUTPUT_DIR / "feature_cols.json", "w") as f:
        json.dump(MATCH_FEATURE_COLS, f)

    # Save calibration parameters
    calibration = {
        "k": calibration_k,
        "sigma": margin_sigma,
        "draw_base_rate": draw_rate,
    }
    with open(OUTPUT_DIR / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"  Calibration saved: k={calibration_k:.2f}, sigma={margin_sigma:.1f}, draw_rate={draw_rate:.3f}")

    # Save FeatureEngine state (ELO, form, venue, H2H)
    engine.save_state(OUTPUT_DIR)

    with open(OUTPUT_DIR / "evaluation_metrics.txt", "w") as f:
        for k, v in eval_res.items():
            f.write(f"{k}: {v}\n")

    print(f"\nAll artefacts saved to {OUTPUT_DIR}")
    print("=== Training complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AFL match prediction model")
    parser.add_argument("--year-from", type=int, default=1990)
    parser.add_argument("--year-to", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--hyperparams", type=str, default=None, help="Path to best_hyperparams.json")
    args = parser.parse_args()
    hyperparams = None
    if args.hyperparams:
        with open(args.hyperparams) as f:
            hyperparams = json.load(f)
        print(f"Using hyperparams from {args.hyperparams}")
    train(args.year_from, args.year_to, args.epochs, hyperparams=hyperparams, patience=args.patience)
