# Rugby League Data Leakage Investigation Plan

## Objective

Find and fix the source of 100% winner accuracy on the validation set, which indicates data leakage.

## Diagnostic Script

Run: `python scripts/diagnose_rugby_leakage.py --competition nrl --year-from 2020 --year-to 2025`

### Validations Performed

| # | Check | What it reveals |
|---|-------|-----------------|
| 1 | **Split verification** | Train/val sizes, confirm last 20% = validation, no overlap |
| 2 | **elo_diff vs outcome** | If predicting winner = (elo_diff > 0) gives >95% accuracy, ELO may be leaking or data is unusual |
| 3 | **Random baseline** | Expect ~50%; if not, split or evaluation may be wrong |
| 4 | **Sample rows** | Inspect elo_diff, actual margin, whether they align |
| 5 | **Trained model** | Load saved model, run on validation, compare to elo_diff agreement |
| 6 | **Scaler fit** | Document current behaviour (fit on train only) |
| 7 | **Feature columns** | Inspect all features for outcome leakage |
| 8 | **Duplicate matches** | Check for duplicate match_ids in validation |
| 9 | **X/y alignment** | Verify features and targets stay row-aligned |

## Root Cause Found: Duplicate Matches

**Problem**: Same matches appeared in multiple overlapping CSV files (e.g. `matches_nrl_1998_2025.csv` and `matches_nrl_2024_2025.csv` both contain 2024–2025). `load_matches` concatenated them without deduplication, producing 1530 rows but only 1131 unique matches. Validation had 306 rows but only 154 unique matches — 152 duplicates.

**Effect**: When evaluating winner accuracy, the model predicted the same match multiple times. Correct predictions on duplicates inflated accuracy toward 100%.

**Fix**: `core/rugby_feature_engine.py` — `load_matches()` now calls `drop_duplicates(subset=["match_id"], keep="first")` after concat.

## Hypotheses to Test

1. **elo_diff is a perfect predictor** — If higher ELO always wins in validation, either ELO is leaking or validation set is anomalous
2. **Evaluation bug** — Accidentally evaluating on training data
3. **Temporal order wrong** — Data not in chronological order, so we're validating on "past" that was trained on "future"
4. **groupby order** — Rounds processed in wrong order, using future round state for features
5. **Scaler** — Minor leak (fit on holdout); unlikely to cause 100% but should fix

## Fixes Applied

- **Scaler** ([model/rugby_train.py](model/rugby_train.py), [model/rugby_hyperband.py](model/rugby_hyperband.py)): Split first, then `scaler.fit(X_tr)` only; transform train and validation separately. Prevents holdout stats from influencing scaling.
- **Duplicate matches** ([core/rugby_feature_engine.py](core/rugby_feature_engine.py)): `load_matches()` deduplicates by `match_id` after concatenating overlapping CSV files.
- **Diagnostic script** ([scripts/diagnose_rugby_leakage.py](scripts/diagnose_rugby_leakage.py)): Run validations; use `--from-cache` for fast re-runs after first completion. Default years 2020–2025. Sections 7–9: feature inspection, duplicate check, X/y alignment.

## Next Steps if 100% Persists

1. Add logging to feature engine: print (year, round) when building features for each round
2. Verify groupby order: assert rounds are processed chronologically
3. Check for multi-competition mixing: (year, round_num) may group ARL + Super League 1997
4. Consider excluding player presence entirely as an experiment to isolate
