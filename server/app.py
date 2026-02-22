"""
AFL Match Predictor — Flask server.

Entry point:
    python server/app.py
    or: flask --app server.app run --port 5000

Routes
------
GET  /                              Main prediction page

Fixture
GET  /api/fixture/rounds            ["Opening Round", "1", ..., "24"]
GET  /api/fixture/<round_num>       [{date, time_confirmed, venue, ...}] matches in round
GET  /api/fixture/meta              {last_fetched} fixture freshness metadata
POST /api/fixture/refresh           re-scrape fixturedownload.com, update matches_2026.csv
POST /api/predict/round             predict all matches in a round at once

Data & Training
GET  /api/data/status               data file counts and freshness
GET  /api/data/model-metrics        parsed evaluation_metrics.txt
POST /api/data/fetch-historical     download latest data from akareen GitHub
POST /api/train                     start model retraining subprocess
GET  /api/train/status?job_id=      poll training job (running, log lines, exit_code)
"""

import subprocess
import sys
import threading
import uuid
from pathlib import Path

# Ensure project root is on the path when running as `python server/app.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request

from core.data_service import DataService
from model.prediction_api import predict_match_outcome, clear_model_cache

# In-memory store for training jobs  {job_id: {running, lines, exit_code}}
_training_jobs: dict = {}
_training_lock = threading.Lock()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Module-level service — loaded once at startup
_service = DataService()


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        teams=_service.get_team_display_names(),
        grounds=_service.get_grounds_display(),
    )


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/teams")
def api_teams():
    return jsonify(_service.get_team_display_names())


@app.route("/api/grounds")
def api_grounds():
    return jsonify(_service.get_grounds_display())


@app.route("/api/lineup/<team_key>")
def api_lineup(team_key: str):
    lineup = _service.get_last_lineup(team_key)
    return jsonify(lineup)


@app.route("/api/season/<team_key>")
def api_season(team_key: str):
    players = _service.get_season_players(team_key)
    return jsonify(players)


@app.route("/api/players/search")
def api_players_search():
    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return jsonify([])
    results = _service.search_players(query, limit=30)
    return jsonify(results)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")
    ground = data.get("ground", "")
    home_player_ids: list = data.get("home_player_ids", [])
    away_player_ids: list = data.get("away_player_ids", [])

    if not home_player_ids or not away_player_ids:
        return jsonify({"error": "home_player_ids and away_player_ids are required"}), 400

    min_players = 18
    if len(home_player_ids) < min_players or len(away_player_ids) < min_players:
        return jsonify({
            "error": f"Each team must have at least {min_players} players. "
                     f"Got home={len(home_player_ids)}, away={len(away_player_ids)}."
        }), 400

    result = predict_match_outcome(home_player_ids, away_player_ids)
    if "error" in result:
        return jsonify(result), 500

    return jsonify({"home_team": home_team, "away_team": away_team, "ground": ground, **result})


# ---------------------------------------------------------------------------
# Fixture routes
# ---------------------------------------------------------------------------

@app.route("/api/fixture/meta")
def api_fixture_meta():
    last_fetched = _service.get_fixture_last_updated()
    return jsonify({"last_fetched": last_fetched})


@app.route("/api/fixture/refresh", methods=["POST"])
def api_fixture_refresh():
    """Re-scrape fixturedownload.com and update matches_2026.csv."""
    try:
        import sys
        _PROJECT_ROOT_STR = str(_PROJECT_ROOT)
        if _PROJECT_ROOT_STR not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT_STR)
        from datafetch.fetch_2026_fixture import fetch_and_save
        path = fetch_and_save()
        _service.save_fixture_meta()
        _service._invalidate_fixture_cache()
        return jsonify({"ok": True, "message": f"Fixture refreshed: {path.name}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/fixture/rounds")
def api_fixture_rounds():
    rounds = _service.get_fixture_rounds()
    return jsonify(rounds)


@app.route("/api/fixture/<path:round_num>")
def api_fixture_round(round_num: str):
    matches = _service.get_fixture_round(round_num)
    return jsonify(matches)


@app.route("/api/predict/round", methods=["POST"])
def api_predict_round():
    """
    Predict all matches in a round.

    Body: {matches: [{home_team_key, away_team_key, home_team_display,
                       away_team_display, venue, venue_display, date,
                       time_confirmed}]}

    For each match, auto-populates lineups using get_last_lineup().
    Returns list of prediction results, one per match.
    """
    data = request.get_json(silent=True)
    if not data or "matches" not in data:
        return jsonify({"error": "Body must contain {matches: [...]}"}), 400

    results = []
    for match in data["matches"]:
        home_key = match.get("home_team_key", "")
        away_key = match.get("away_team_key", "")

        home_lineup = _service.get_last_lineup(home_key)
        away_lineup = _service.get_last_lineup(away_key)

        home_ids = [p["player_id"] for p in home_lineup if p["player_id"] != "unknown"]
        away_ids = [p["player_id"] for p in away_lineup if p["player_id"] != "unknown"]

        entry = {
            "home_team": match.get("home_team_display", home_key),
            "away_team": match.get("away_team_display", away_key),
            "venue": match.get("venue_display", match.get("venue", "")),
            "date": match.get("date", ""),
            "time_confirmed": match.get("time_confirmed", False),
            "home_lineup_count": len(home_ids),
            "away_lineup_count": len(away_ids),
        }

        if len(home_ids) < 10 or len(away_ids) < 10:
            entry["prediction"] = None
            entry["error"] = "Insufficient lineup data"
        else:
            pred = predict_match_outcome(
                home_ids, away_ids,
                home_team=match.get("home_team_display", home_key),
                away_team=match.get("away_team_display", away_key),
                venue=match.get("venue", ""),
            )
            entry["prediction"] = pred if "error" not in pred else None
            if "error" in pred:
                entry["error"] = pred["error"]

        results.append(entry)

    return jsonify(results)


# ---------------------------------------------------------------------------
# Data & Training routes
# ---------------------------------------------------------------------------

@app.route("/api/data/status")
def api_data_status():
    """Return file counts and freshness for all data sources."""
    import glob as _glob
    match_files = sorted(_glob.glob(str(_PROJECT_ROOT / "afl_data/data/matches/matches_*.csv")))
    player_files = _glob.glob(str(_PROJECT_ROOT / "afl_data/data/players/*_performance_details.csv"))
    lineup_files = _glob.glob(str(_PROJECT_ROOT / "afl_data/data/lineups/team_lineups_*.csv"))

    # Year range from match filenames
    years = []
    for f in match_files:
        try:
            y = int(Path(f).stem.replace("matches_", ""))
            years.append(y)
        except ValueError:
            pass

    fixture_2026 = _PROJECT_ROOT / "afl_data/data/matches/matches_2026.csv"
    fixture_matches = 0
    if fixture_2026.exists():
        import pandas as pd
        try:
            df = pd.read_csv(fixture_2026, usecols=["round_num"])
            fixture_matches = len(df)
        except Exception:
            pass

    return jsonify({
        "match_files": len(match_files),
        "match_year_range": f"{min(years)}–{max(years)}" if years else "",
        "player_files": len(player_files),
        "lineup_files": len(lineup_files),
        "fixture_matches": fixture_matches,
        "fixture_last_updated": _service.get_fixture_last_updated(),
    })


@app.route("/api/data/model-metrics")
def api_model_metrics():
    """Parse and return the most recent training metrics."""
    metrics_path = _PROJECT_ROOT / "model/output/evaluation_metrics.txt"
    if not metrics_path.exists():
        return jsonify({"error": "No evaluation_metrics.txt found"})

    import os
    trained_date = None
    try:
        mtime = os.path.getmtime(metrics_path)
        from datetime import datetime
        trained_date = datetime.fromtimestamp(mtime).strftime("%d %b %Y %H:%M")
    except Exception:
        pass

    # Read only the FIRST block of metrics (before any "Categorized Results" duplication)
    metrics = {}
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Categorized"):
                break
            if ":" in line:
                k, v = line.split(":", 1)
                metrics[k.strip()] = v.strip()

    metrics["trained_date"] = trained_date
    return jsonify(metrics)


@app.route("/api/data/fetch-historical", methods=["POST"])
def api_fetch_historical():
    """Run the akareen data fetch script."""
    try:
        script = str(_PROJECT_ROOT / "datafetch/fetch_afl_data.py")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=600,
            cwd=str(_PROJECT_ROOT),
        )
        combined = (result.stdout or "") + (result.stderr or "")
        # Treat as success if the success message appears in output,
        # even if exit code is non-zero (e.g. temp-dir cleanup failure)
        if result.returncode == 0 or "Data fetching completed successfully" in combined:
            return jsonify({"ok": True, "message": "Historical data updated successfully"})
        snippet = combined[-600:] if combined else "Unknown error"
        return jsonify({"ok": False, "error": snippet}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Fetch timed out after 10 minutes"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _run_training(job_id: str, year_from: int, year_to: int, epochs: int) -> None:
    """Run model/train.py in a subprocess, streaming output to job store."""
    script = str(_PROJECT_ROOT / "model/train.py")
    cmd = [
        sys.executable, script,
        "--year-from", str(year_from),
        "--year-to", str(year_to),
        "--epochs", str(epochs),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_PROJECT_ROOT),
        )
        for line in proc.stdout:
            line = line.rstrip()
            with _training_lock:
                _training_jobs[job_id]["lines"].append(line)
        proc.wait()
        with _training_lock:
            _training_jobs[job_id]["running"] = False
            _training_jobs[job_id]["exit_code"] = proc.returncode
        # Clear stale model caches so next prediction loads new artefacts
        if proc.returncode == 0:
            clear_model_cache()
    except Exception as e:
        with _training_lock:
            _training_jobs[job_id]["lines"].append(f"ERROR: {e}")
            _training_jobs[job_id]["running"] = False
            _training_jobs[job_id]["exit_code"] = 1


@app.route("/api/train", methods=["POST"])
def api_train():
    """Start a model training job and return a job_id."""
    data = request.get_json(silent=True) or {}
    year_from = int(data.get("year_from", 1990))
    year_to = int(data.get("year_to", 2025))
    epochs = int(data.get("epochs", 20))

    # Only one job at a time
    with _training_lock:
        for jid, job in _training_jobs.items():
            if job["running"]:
                return jsonify({"error": "A training job is already running", "job_id": jid}), 409

    job_id = str(uuid.uuid4())[:8]
    with _training_lock:
        _training_jobs[job_id] = {"running": True, "lines": [], "exit_code": None, "seen": 0}

    t = threading.Thread(target=_run_training, args=(job_id, year_from, year_to, epochs), daemon=True)
    t.start()
    return jsonify({"job_id": job_id, "ok": True})


@app.route("/api/train/status")
def api_train_status():
    """Poll training job. Returns running flag, new log lines since last poll, exit_code."""
    job_id = request.args.get("job_id", "")
    with _training_lock:
        job = _training_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404

    with _training_lock:
        seen = job["seen"]
        all_lines = job["lines"]
        new_lines = all_lines[seen:]
        job["seen"] = len(all_lines)
        running = job["running"]
        exit_code = job["exit_code"]

    return jsonify({
        "running": running,
        "new_lines": new_lines,
        "exit_code": exit_code,
    })


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
