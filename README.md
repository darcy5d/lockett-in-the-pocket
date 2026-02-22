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

```bash
python datafetch/fetch_afl_data.py
```

Downloads the latest data from [akareen/AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) — historical matches, lineups, player performance stats.

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
├── afl_data/                   AFL data
│   └── data/
│       ├── matches/            matches_YYYY.csv (1897–2026)
│       ├── lineups/            team_lineups_<team>.csv
│       └── players/            *_performance_details.csv
├── datafetch/                  Data pipeline scripts
│   ├── fetch_afl_data.py       Pull historical data from GitHub
│   ├── fetch_2026_fixture.py   Scrape 2026 fixture
│   └── load_afl_data.py        AFLDataLoader class
└── scripts/
    └── validate_mappings.py    Mapping validation (run after changes)
```

---

## Model details

The neural network uses:
- **Player embeddings** (32-dim) for each player in the lineup
- **Team-level enhanced stats** (21 metrics, weighted by games played)
- **Match features** (year, match type)

Outputs:
- Match winner probabilities (home win / draw / away win)
- Predicted goals and behinds per team
- Calculated margin

### Known limitation
The current model's `feature_cols.json` does not include ground/venue. The venue selector in the UI is recorded for future model retraining.

---

## Data sources

| Source | Used for |
|--------|----------|
| [akareen/AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) | Historical matches (1897–2025), lineups, player stats |
| [fixturedownload.com](https://fixturedownload.com/results/afl-2026) | 2026 AFL fixture |

---

## Multi-league roadmap (Phase B)

- **AFLW**: same data structure; separate model, player index, and data directory (`aflw_data/`)
- **NRL**: different scoring (tries, goals); placeholder at `nrl_data/`; data from Zyla NRL API or Champion Data

---

## Prerequisites

- Python 3.9+
- Dependencies: `pip install -r requirements.txt`
  - Flask, TensorFlow, pandas, numpy, scikit-learn, joblib, requests, beautifulsoup4
