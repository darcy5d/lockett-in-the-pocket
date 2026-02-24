# AFL Match Predictor

Neural network model for predicting AFL match outcomes. Ready for the **2026 season**.

---

## Quick Start

```bash
# 1. Activate the Python 3.9 venv
source afl_venv_3_9/bin/activate

# 2. Run the Flask web app
python server/app.py
# Then open http://localhost:5000
```

---

## Setup from scratch

```bash
python -m venv afl_venv_3_9
source afl_venv_3_9/bin/activate
pip install -r requirements.txt
```

---

## Running the Flask App

```bash
python server/app.py
# or
flask --app server.app run --port 5000
```

Open **http://localhost:5000** in your browser.

### What the UI does
1. Select home and away teams
2. Last-known lineup loads automatically; players appear as chips
3. Move players between **Game Day Lineup** and **Season Squad** by clicking
4. Search the full player database to add players
5. Click **Predict Match Outcome** — results appear instantly

---

## 2026 Season

### Fetch the 2026 fixture

```bash
python datafetch/fetch_2026_fixture.py
```

This scrapes the 2026 fixture from fixturedownload.com and writes:
`afl_data/data/matches/matches_2026.csv`

The file contains all 207 matches with team and venue names normalised to match the existing historical data schema.

### Update historical data

All historical data is scraped directly from [AFL Tables](https://afltables.com/afl/afl_index.html) — the canonical source for AFL match and player data.

**Gap update** (scrape new rounds and append to existing data):

```bash
python datafetch/update_from_afl_tables.py --lineups --player-stats
```

**Full rebuild** (re-scrape all seasons, round-by-round with resume):

```bash
python datafetch/rebuild_from_afl_tables.py --year-from 1990 --year-to 2025 --lineups --player-stats
```

**One round at a time** (small bites, merges with existing output):

```bash
python datafetch/afl_tables_scraper.py --year-from 2025 --year-to 2025 --round 1 --lineups --player-stats
```

All data is written to `afl_data/data/` with three data types:
- `matches/matches_YYYY.csv` — match scores and quarter breakdowns
- `lineups/team_lineups_<team>.csv` — per-round team lineups
- `players/<player_id>_performance_details.csv` — per-player per-game stats (kicks, marks, etc.)

**Populate SQLite** (optional — for SQLite-backed app mode):

```bash
python datafetch/populate_sqlite_from_csv.py --source afl_data --output afl_data/data/afl.db --players
USE_AFL_SQLITE=1 python server/app.py
```

See `datafetch/AFL_TABLES_REFERENCE.md` for URL layout and schema mapping.

---

## Validate mappings

Always run this after modifying `core/mappings.py`:

```bash
python scripts/validate_mappings.py
```

Checks team names, venue names, and player ID round-trips all pass.

---

## Project structure

```
AFL/
├── server/                     Flask web application
│   ├── app.py                  Routes and API endpoints
│   ├── templates/              Jinja2 HTML templates
│   └── static/                 CSS and JS
├── core/                       Shared, league-agnostic layer
│   ├── mappings.py             Team / venue / player ID mapping (map-first)
│   ├── data_service.py         DataService: teams, grounds, lineups, search
│   └── league_config.py        (Phase B) League abstraction
├── model/                      Neural network model
│   ├── neural_network_with_embeddings.py   Training script
│   ├── prediction_api.py       predict_match_outcome()
│   ├── player_stats_api.py     Player stat helpers
│   └── output/                 Trained model artefacts
│       ├── model.h5
│       ├── scaler.joblib
│       ├── player_index.json
│       └── feature_cols.json
├── afl_data/                   AFL data (scraped from AFL Tables)
│   └── data/
│       ├── matches/            matches_YYYY.csv (1897–2026)
│       ├── lineups/            team_lineups_<team>.csv
│       └── players/            <player_id>_performance_details.csv
├── core/
│   ├── afl_data_store.py       Data access layer (CSV or SQLite)
│   └── ...
├── datafetch/                  Data pipeline scripts
│   ├── afl_tables_scraper.py   Scrape AFL Tables → matches, lineups, player stats
│   ├── rebuild_from_afl_tables.py   Full scrape (round-by-round with resume)
│   ├── update_from_afl_tables.py    Gap update (new rounds/seasons)
│   ├── fetch_2026_fixture.py   Scrape 2026 fixture
│   ├── populate_sqlite_from_csv.py  CSV → afl.db
│   └── load_afl_data.py        AFLDataLoader class
└── scripts/
    ├── compare_data_sources.py Data validation
    └── validate_mappings.py    Mapping validation (run after changes)
```

---

## Model Explanation

This section walks through exactly how the model produces a prediction, using **Sydney vs Carlton at the SCG (Opening Round 2026)** as a concrete example.

### Prediction pipeline overview

When you click **Predict Whole Round**, five things happen for each match:

1. The 2026 fixture provides the matchup (home team, away team, venue)
2. Each team's most recent known lineup is loaded (~22 players per team)
3. The `FeatureEngine` computes 19 match-level features from historical state (ELO ratings, recent form, venue history, head-to-head, player ELOs)
4. Player embeddings and form-weighted performance stats are computed for each lineup
5. All five inputs are fed through the neural network, which outputs win probabilities, predicted goals, behinds, and margin

### The 5 model inputs

The model receives five separate input tensors per match:

**Input 1 -- Match Features (19 numbers)**

Computed by `FeatureEngine`, which walks through every match from 1990-2025 chronologically, updating ELO, form, and venue state after each game. At prediction time it snapshots the current state.

| Feature | Example (Syd vs Carl) | Description |
|---------|-----------------------|-------------|
| `year` | 2026 | Current year |
| `is_finals` | 0 | 1 if finals match, 0 if regular season |
| `team1_elo` | 1645 | Sydney's ELO rating going into the match |
| `team2_elo` | 1497 | Carlton's ELO rating |
| `elo_diff` | +148 | Sydney is 148 ELO points stronger |
| `team1_recent_win_pct` | 0.6 | Sydney won 3 of last 5 games |
| `team1_recent_avg_score` | 92 | Sydney averaged 92 points in last 5 |
| `team1_recent_avg_margin` | +12 | Sydney won by 12 points on average |
| `team2_recent_win_pct` | 0.4 | Carlton won 2 of last 5 |
| `team2_recent_avg_score` | 78 | Carlton averaged 78 points |
| `team2_recent_avg_margin` | -8 | Carlton lost by 8 on average |
| `venue_team1_win_pct` | 0.62 | Sydney wins 62% of games at the SCG |
| `venue_team2_win_pct` | 0.35 | Carlton wins 35% at the SCG |
| `h2h_team1_wins_last5` | 3 | Sydney won 3 of last 5 meetings |
| `h2h_avg_margin_last5` | +15 | Sydney won by 15 pts avg in those 5 |
| `team1_avg_player_elo` | 1580 | Mean ELO of Sydney's 22 players |
| `team2_avg_player_elo` | 1520 | Mean ELO of Carlton's 22 players |
| `team1_top5_player_elo` | 1720 | Mean ELO of Sydney's best 5 players |
| `team2_top5_player_elo` | 1650 | Mean ELO of Carlton's best 5 players |

All 19 features are StandardScaler-normalised (zero mean, unit variance).

**Inputs 2 & 3 -- Player Embeddings (22 integers per team)**

Each player in the lineup is mapped to an index in `player_index.json` (13,207 players). The model has a learned 32-dimensional embedding for each player. The 22 embeddings per team are mean-pooled into a single 32-dim vector representing team composition.

**Inputs 4 & 5 -- Enhanced Player Stats (21 numbers per team)**

For each team's lineup, the `FeatureEngine` computes form-weighted averages of 21 performance metrics using exponential decay (0.85 per game) over each player's last 10 games:

kicks, marks, handballs, goals, behinds, hit_outs, tackles, rebound_50s, inside_50s, clearances, clangers, free_kicks_for, free_kicks_against, contested_possessions, uncontested_possessions, contested_marks, marks_inside_50, one_percenters, bounces, goal_assist, percentage_of_game_played

These are averaged across the 22 players and StandardScaler-normalised.

### Team ELO rankings (end of 2025 season)

ELO ratings are computed using the standard algorithm (K=32, start=1500) walked chronologically through all matches from 1990-2025:

| Rank | Team | ELO |
|------|------|-----|
| 1 | Brisbane Lions | 1747 |
| 2 | Collingwood | 1654 |
| 3 | Sydney | 1645 |
| 4 | Hawthorn | 1616 |
| 5 | Greater Western Sydney | 1616 |
| 6 | Port Adelaide | 1608 |
| 7 | Geelong | 1596 |
| 8 | Western Bulldogs | 1591 |
| 9 | St Kilda | 1537 |
| 10 | Fremantle | 1516 |
| 11 | Adelaide | 1500 |
| 12 | Carlton | 1497 |
| 13 | Melbourne | 1484 |
| 14 | Gold Coast | 1474 |
| 15 | Essendon | 1467 |
| 16 | Richmond | 1399 |
| 17 | West Coast | 1367 |
| 18 | North Melbourne | 1350 |

Player ELO ratings (K=16, start=1500) are also computed per-player, with credit/debit split across the lineup after each match.

### Neural network architecture

```
Match Features (19) ──┐
Player Emb T1 (32) ───┤
Player Emb T2 (32) ───┼── Concatenate (135) ── Dense(256) ── Dense(128) ── Dense(64)
Enhanced T1 (21) ─────┤                                                       |
Enhanced T2 (21) ─────┘                                                       |
                                                                              |
                        ┌─ Winner branch ────── Softmax(3) ── [home / draw / away]
                        ├─ Goals T1 branch ──── Softplus(1) ── predicted home goals
                        ├─ Behinds T1 branch ── Softplus(1) ── predicted home behinds
                        ├─ Goals T2 branch ──── Softplus(1) ── predicted away goals
                        ├─ Behinds T2 branch ── Softplus(1) ── predicted away behinds
                        └─ Margin ───────────── Calculated from (goals * 6 + behinds)
```

Each score branch receives the shared trunk output concatenated with that team's enhanced stats, then Dense(64) - Dense(32) - Softplus(1). Softplus activation ensures scores are always positive. Bias is initialised to the training data mean (~14 goals, ~12 behinds) so the network starts in a realistic range.

The model is trained with Huber loss (delta=5.0) for regression outputs and categorical crossentropy for the winner. Score loss weights (5x) are higher than winner (1x) to ensure the model learns realistic scores rather than just picking a winner.

### Current model performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Winner accuracy | 64.7% | Correctly picks the winner in ~65% of matches (random = 50%, best published AFL models ~70-72%) |
| F1 score | 0.42 | Macro-averaged across win/draw/loss classes |
| Goals MAE | 3.3-3.4 | Predictions off by ~3 goals on average (mean goals per team ~14) |
| Behinds MAE | 2.9-3.0 | Predictions off by ~3 behinds on average |
| RMSE margin | 39 pts | Predicted margin off by ~39 points on average (AFL margins are volatile) |

### What drives differentiation between matches

When Sydney vs Carlton produces a different prediction from Gold Coast vs Geelong, it is primarily because of:

1. **ELO difference** -- Sydney (1645) vs Carlton (1497) is a +148 gap; Gold Coast (1474) vs Geelong (1596) is -122 the other way
2. **Venue advantage** -- Sydney at the SCG has a 62% win rate; Gold Coast at Carrara is more neutral
3. **Player quality** -- the actual players in each lineup contribute via learned embeddings and form-weighted stats
4. **Recent form** -- a team on a winning streak vs one in a slump produces different form features
5. **Head-to-head history** -- some teams historically dominate specific opponents

---

## Data sources

| Source | Used for |
|--------|----------|
| [AFL Tables](https://afltables.com/afl/afl_index.html) | Historical matches (1990–2025), lineups, player stats — scraped directly via `datafetch/afl_tables_scraper.py` |
| [fixturedownload.com](https://fixturedownload.com/results/afl-2026) | 2026 AFL fixture |

AFL Tables is the canonical source for AFL match and player data (complete to end of season 2025). Our scraper writes data in CSV format with the same schema: team names (Sydney, Greater Western Sydney, etc.), venue names (S.C.G., M.C.G., Docklands), and match scores in goals.behinds form. See `datafetch/AFL_TABLES_REFERENCE.md` for URL layout and schema mapping.

---

## Multi-league roadmap (Phase B)

- **AFLW**: same data structure; separate model, player index, and data directory (`aflw_data/`)
- **NRL**: different scoring (tries, goals); placeholder at `nrl_data/`; data from Zyla NRL API or Champion Data

---

## Prerequisites

- Python 3.9+
- Dependencies: `pip install -r requirements.txt`
  - Flask, TensorFlow, pandas, numpy, scikit-learn, joblib, requests, beautifulsoup4
