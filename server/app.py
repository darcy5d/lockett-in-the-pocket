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
POST /api/tune                      start Hyperband hyperparameter tuning (runs tune_hyperband.py --save-final)
GET  /api/tune/status?job_id=       poll tune job (running, log lines, exit_code)
"""

import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is on the path when running as `python server/app.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, jsonify, redirect, render_template, request, url_for

from core.competition_config import COMPETITIONS
from core.data_service import DataService
from core.nrl_data_service import NRLDataService
from core.rugby_data_service import RugbyDataService
from model.prediction_api import predict_match_outcome, clear_model_cache
from model.nrl_prediction_api import predict_nrl_match, clear_nrl_model_cache
from model.rugby_prediction_api import predict_rugby_match, clear_rugby_model_cache

# In-memory store for training jobs  {job_id: {running, lines, exit_code}}
_training_jobs: dict = {}
_training_lock = threading.Lock()

# In-memory store for rugby lineup build-cache jobs {job_id: {running, current, total, result, error}}
_build_cache_jobs: dict = {}
_build_cache_lock = threading.Lock()

# In-memory store for tune jobs (same shape)
_tune_jobs: dict = {}
_tune_lock = threading.Lock()

# In-memory store for NRL historical fetch jobs (same shape)
_fetch_jobs: dict = {}
_fetch_lock = threading.Lock()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Module-level services
_service = DataService()
_nrl_service = NRLDataService()


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def landing():
    """League selector landing page."""
    return render_template("landing.html")


@app.route("/afl")
def afl_index():
    """AFL Match Predictor UI."""
    return render_template(
        "index.html",
        teams=_service.get_team_display_names(),
        grounds=_service.get_grounds_display(),
    )


@app.route("/nrl")
def nrl_index():
    """Redirect to Rugby League Predictor with NRL selected."""
    return redirect(url_for("rugby_index", competition="nrl"))


@app.route("/rugby")
def rugby_index():
    """Rugby League Predictor UI — multi-competition."""
    competition = request.args.get("competition", "nrl")
    if competition not in COMPETITIONS:
        competition = "nrl"
    return render_template(
        "rugby_index.html",
        competitions=COMPETITIONS,
        default_competition=competition,
    )


# ---------------------------------------------------------------------------
# AFL API routes
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


@app.route("/api/fixture/update-scores", methods=["POST"])
def api_fixture_update_scores():
    """Update completed match scores from AFL Tables."""
    try:
        import sys
        _PROJECT_ROOT_STR = str(_PROJECT_ROOT)
        if _PROJECT_ROOT_STR not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT_STR)
        
        # Run AFL Tables scraper for 2026 to get completed match scores
        script = str(_PROJECT_ROOT / "datafetch/afl_tables_scraper.py")
        result = subprocess.run(
            [sys.executable, script, "--year-from", "2026", "--year-to", "2026"],
            capture_output=True, text=True, timeout=300,
            cwd=str(_PROJECT_ROOT),
        )
        
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            _service._invalidate_fixture_cache()  # Clear cache so new scores are loaded
            # Count how many matches have scores now
            matches_with_scores = 0
            try:
                import pandas as pd
                df = pd.read_csv(_PROJECT_ROOT / "afl_data/data/matches/matches_2026.csv")
                matches_with_scores = len(df[(pd.notna(df['team_1_final_goals'])) & (pd.notna(df['team_2_final_goals']))])
            except:
                pass
            return jsonify({"ok": True, "message": f"Scores updated from AFL Tables. {matches_with_scores} matches now have complete scores."})
        else:
            snippet = combined[-400:] if combined else "Unknown error"
            return jsonify({"ok": False, "error": f"AFL Tables scraper failed: {snippet}"}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "AFL Tables scraper timed out after 5 minutes"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/fixture/rounds")
def api_fixture_rounds():
    rounds = _service.get_fixture_rounds()
    return jsonify(rounds)


@app.route("/api/fixture/<path:round_num>")
def api_fixture_round(round_num: str):
    matches = _service.get_fixture_round(round_num)

    from datafetch.afl_lineup_scraper import get_lineup_meta
    meta = get_lineup_meta() or {}
    meta_round = meta.get("round_num", "")
    meta_updated = meta.get("afl_last_updated", "")
    meta_fetched = meta.get("last_fetched", "")

    for m in matches:
        home_key = m.get("home_team_key", "")
        away_key = m.get("away_team_key", "")
        home_lineup = _service.get_last_lineup(home_key)
        away_lineup = _service.get_last_lineup(away_key)
        home_known = [p for p in home_lineup if p.get("player_id") != "unknown"]
        away_known = [p for p in away_lineup if p.get("player_id") != "unknown"]

        home_year = _lineup_year(home_key)
        away_year = _lineup_year(away_key)

        m["home_lineup_count"] = len(home_lineup)
        m["away_lineup_count"] = len(away_lineup)
        m["home_lineup_known"] = len(home_known)
        m["away_lineup_known"] = len(away_known)
        m["home_lineup_fresh"] = home_year == 2026
        m["away_lineup_fresh"] = away_year == 2026
        m["lineup_meta_round"] = meta_round
        m["lineup_last_updated"] = meta_updated or meta_fetched

    return jsonify(matches)


def _lineup_year(team_key: str) -> int:
    """Return the year of the most recent lineup entry for a team, or 0."""
    df = _service._get_lineup_df_for_team(team_key)
    if df.empty or "year" not in df.columns:
        return 0
    return int(df["year"].max())


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

        if len(home_ids) < 8 or len(away_ids) < 8:
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
# AFL Lineup scraper routes
# ---------------------------------------------------------------------------

# In-memory store for lineup scrape jobs {job_id: {running, lines, exit_code}}
_lineup_jobs: dict = {}
_lineup_lock = threading.Lock()


def _run_lineup_scrape(job_id: str, round_num: Optional[str]) -> None:
    """Run afl_lineup_scraper in a subprocess, streaming output to job store."""
    script = str(_PROJECT_ROOT / "datafetch/afl_lineup_scraper.py")
    cmd = [sys.executable, script]
    if round_num:
        cmd.extend(["--round", round_num])
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
            with _lineup_lock:
                _lineup_jobs[job_id]["lines"].append(line)
        proc.wait()
        with _lineup_lock:
            _lineup_jobs[job_id]["running"] = False
            _lineup_jobs[job_id]["exit_code"] = proc.returncode
        if proc.returncode == 0:
            _service.invalidate_lineup_cache()
    except Exception as e:
        with _lineup_lock:
            _lineup_jobs[job_id]["lines"].append(f"ERROR: {e}")
            _lineup_jobs[job_id]["running"] = False
            _lineup_jobs[job_id]["exit_code"] = 1


@app.route("/api/afl/lineup/refresh", methods=["POST"])
def api_afl_lineup_refresh():
    """Start a background job to scrape AFL.com.au team lineups."""
    data = request.get_json(silent=True) or {}
    round_num = data.get("round_num")

    # Only one lineup scrape at a time
    with _lineup_lock:
        for job in _lineup_jobs.values():
            if job.get("running"):
                return jsonify({"error": "A lineup scrape is already running"}), 409

    job_id = str(uuid.uuid4())[:8]
    with _lineup_lock:
        _lineup_jobs[job_id] = {"running": True, "lines": [], "exit_code": None, "seen": 0}

    t = threading.Thread(target=_run_lineup_scrape, args=(job_id, round_num), daemon=True)
    t.start()
    return jsonify({"job_id": job_id, "ok": True})


@app.route("/api/afl/lineup/refresh/status")
def api_afl_lineup_refresh_status():
    """Poll lineup scrape job."""
    job_id = request.args.get("job_id", "")
    with _lineup_lock:
        job = _lineup_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404

    with _lineup_lock:
        seen = job["seen"]
        new_lines = job["lines"][seen:]
        job["seen"] = len(job["lines"])
        running = job["running"]
        exit_code = job["exit_code"]

    return jsonify({
        "running": running,
        "new_lines": new_lines,
        "exit_code": exit_code,
    })


@app.route("/api/afl/lineup/meta")
def api_afl_lineup_meta():
    """Return lineup freshness metadata."""
    from datafetch.afl_lineup_scraper import get_lineup_meta
    meta = get_lineup_meta()
    return jsonify(meta or {})


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


def _any_job_running() -> bool:
    """True if a train, tune, or fetch job is currently running."""
    with _training_lock:
        for job in _training_jobs.values():
            if job.get("running"):
                return True
    with _tune_lock:
        for job in _tune_jobs.values():
            if job.get("running"):
                return True
    with _fetch_lock:
        for job in _fetch_jobs.values():
            if job.get("running"):
                return True
    return False


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

    # Only one job at a time (train or tune)
    if _any_job_running():
        return jsonify({"error": "A training or tuning job is already running"}), 409

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


def _run_tuning(
    job_id: str,
    year_from: int,
    year_to: int,
    max_epochs: int,
    patience: int = 15,
    hyperband_iterations: int = 2,
) -> None:
    """Run model/tune_hyperband.py with --save-final in a subprocess."""
    script = str(_PROJECT_ROOT / "model/tune_hyperband.py")
    cmd = [
        sys.executable, script,
        "--year-from", str(year_from),
        "--year-to", str(year_to),
        "--max-epochs", str(max_epochs),
        "--patience", str(patience),
        "--hyperband-iterations", str(hyperband_iterations),
        "--save-final",
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
            with _tune_lock:
                _tune_jobs[job_id]["lines"].append(line)
        proc.wait()
        with _tune_lock:
            _tune_jobs[job_id]["running"] = False
            _tune_jobs[job_id]["exit_code"] = proc.returncode
        if proc.returncode == 0:
            clear_model_cache()
    except Exception as e:
        with _tune_lock:
            _tune_jobs[job_id]["lines"].append(f"ERROR: {e}")
            _tune_jobs[job_id]["running"] = False
            _tune_jobs[job_id]["exit_code"] = 1


@app.route("/api/tune", methods=["POST"])
def api_tune():
    """Start a Hyperband tuning job (runs tune_hyperband.py --save-final)."""
    data = request.get_json(silent=True) or {}
    year_from = int(data.get("year_from", 1990))
    year_to = int(data.get("year_to", 2025))
    max_epochs = int(data.get("max_epochs", 100))
    patience = int(data.get("patience", 15))
    hyperband_iterations = int(data.get("hyperband_iterations", 2))

    if _any_job_running():
        return jsonify({"error": "A training or tuning job is already running"}), 409

    job_id = str(uuid.uuid4())[:8]
    with _tune_lock:
        _tune_jobs[job_id] = {"running": True, "lines": [], "exit_code": None, "seen": 0}

    t = threading.Thread(
        target=_run_tuning,
        args=(job_id, year_from, year_to, max_epochs, patience, hyperband_iterations),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": job_id, "ok": True})


@app.route("/api/tune/status")
def api_tune_status():
    """Poll tune job. Returns running, new_lines, exit_code."""
    job_id = request.args.get("job_id", "")
    with _tune_lock:
        job = _tune_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404

    with _tune_lock:
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


@app.route("/api/model/retrain", methods=["POST"])
def api_model_retrain():
    """Retrain AFL model with latest data up to current year."""
    try:
        current_year = datetime.now().year
        
        # Run training script as subprocess
        result = subprocess.run([
            sys.executable, 
            str(_PROJECT_ROOT / "model" / "train.py"),
            "--year-to", str(current_year),
            "--epochs", "100"
        ], 
        capture_output=True, 
        text=True, 
        timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            return jsonify({"ok": True, "message": "Model retrained successfully"})
        else:
            return jsonify({"ok": False, "error": result.stderr}), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Training timeout - please use manual method"}), 408
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Rugby League API routes (multi-competition)
# ---------------------------------------------------------------------------

def _rugby_service(competition_id: str) -> RugbyDataService:
    if competition_id not in COMPETITIONS:
        raise ValueError(f"Unknown competition: {competition_id}")
    return RugbyDataService(competition_id)


@app.route("/api/rugby/<competition_id>/teams")
def api_rugby_teams(competition_id: str):
    try:
        return jsonify(_rugby_service(competition_id).get_team_display_names())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rugby/<competition_id>/grounds")
def api_rugby_grounds(competition_id: str):
    try:
        return jsonify(_rugby_service(competition_id).get_grounds_display())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rugby/<competition_id>/lineup/<team_key>")
def api_rugby_lineup(competition_id: str, team_key: str):
    try:
        return jsonify(_rugby_service(competition_id).get_last_lineup(team_key))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rugby/<competition_id>/season/<team_key>")
def api_rugby_season(competition_id: str, team_key: str):
    try:
        return jsonify(_rugby_service(competition_id).get_season_players(team_key))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rugby/<competition_id>/players/search")
def api_rugby_players_search(competition_id: str):
    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return jsonify([])
    try:
        return jsonify(_rugby_service(competition_id).search_players(query, limit=30))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rugby/<competition_id>/predict", methods=["POST"])
def api_rugby_predict(competition_id: str):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400
    home_player_ids = data.get("home_player_ids", [])
    away_player_ids = data.get("away_player_ids", [])
    if not home_player_ids or not away_player_ids:
        return jsonify({"error": "home_player_ids and away_player_ids required"}), 400
    min_players = 10
    if len(home_player_ids) < min_players or len(away_player_ids) < min_players:
        return jsonify({"error": f"Each team needs at least {min_players} players"}), 400
    try:
        result = predict_rugby_match(
            competition_id,
            home_player_ids, away_player_ids,
            home_team=data.get("home_team", ""),
            away_team=data.get("away_team", ""),
            venue=data.get("ground", ""),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    if "error" in result:
        return jsonify(result), 500
    return jsonify({
        "home_team": data.get("home_team", ""),
        "away_team": data.get("away_team", ""),
        "ground": data.get("ground", ""),
        **result,
    })


@app.route("/api/rugby/<competition_id>/fixture/meta")
def api_rugby_fixture_meta(competition_id: str):
    try:
        return jsonify({"last_fetched": _rugby_service(competition_id).get_fixture_last_updated()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


_FIXTURE_FETCH_MODULES = {
    "nrl": "datafetch.fetch_2026_nrl_fixture",
    "uk-super-league": "datafetch.fetch_2026_super_league_fixture",
    "nsw-cup": "datafetch.fetch_2026_nsw_cup_fixture",
    "qld-cup": "datafetch.fetch_2026_qld_cup_fixture",
    "uk-championship": "datafetch.fetch_2026_championship_fixture",
}


@app.route("/api/rugby/<competition_id>/fixture/refresh", methods=["POST"])
def api_rugby_fixture_refresh(competition_id: str):
    try:
        mod_name = _FIXTURE_FETCH_MODULES.get(competition_id)
        if not mod_name:
            return jsonify({"ok": False, "error": f"2026 fixture fetch not yet implemented for {competition_id}"}), 501
        mod = __import__(mod_name, fromlist=["fetch_and_save"])
        path = mod.fetch_and_save()
        svc = _rugby_service(competition_id)
        svc.save_fixture_meta()
        svc._invalidate_fixture_cache()
        return jsonify({"ok": True, "message": f"Fixture refreshed: {path.name}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _competition_to_cache_slug(competition_id: str) -> str:
    """Map API competition_id to build_cache competition slug."""
    return {
        "uk-super-league": "super-league-uk",
        "uk-championship": "championship-uk",
    }.get(competition_id, competition_id)


def _run_build_cache(job_id: str, competition_id: str, rebuild: bool = False) -> None:
    """Run build_cache in thread, updating job progress."""
    slug = _competition_to_cache_slug(competition_id)

    def progress_cb(current: int, total: int) -> None:
        with _build_cache_lock:
            if job_id in _build_cache_jobs:
                _build_cache_jobs[job_id]["current"] = current
                _build_cache_jobs[job_id]["total"] = total

    try:
        from scripts.build_rlp_player_dob_cache import build_cache
        cache = build_cache(competition=slug, delay=0.3, progress_callback=progress_cb, rebuild=rebuild)
        with _build_cache_lock:
            if job_id in _build_cache_jobs:
                _build_cache_jobs[job_id]["running"] = False
                _build_cache_jobs[job_id]["result"] = {"ok": True, "message": f"Player cache built: {len(cache)} entries"}
    except Exception as e:
        with _build_cache_lock:
            if job_id in _build_cache_jobs:
                _build_cache_jobs[job_id]["running"] = False
                _build_cache_jobs[job_id]["error"] = str(e)


@app.route("/api/rugby/<competition_id>/lineup/build-cache", methods=["POST"])
def api_rugby_lineup_build_cache(competition_id: str):
    """Start build cache job. Returns job_id for polling status."""
    if competition_id not in _LINEUP_SUPPORTED:
        return jsonify({"ok": False, "error": f"Cache build not implemented for {competition_id}"}), 501
    data = request.get_json(silent=True) or {}
    rebuild = bool(data.get("rebuild", False))
    job_id = str(uuid.uuid4())[:8]
    with _build_cache_lock:
        _build_cache_jobs[job_id] = {"running": True, "current": 0, "total": 0, "result": None, "error": None}
    threading.Thread(target=_run_build_cache, args=(job_id, competition_id), kwargs={"rebuild": rebuild}, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/rugby/<competition_id>/lineup/build-cache/status")
def api_rugby_lineup_build_cache_status(competition_id: str):
    """Poll build cache job. Returns running, current, total, result, error."""
    job_id = request.args.get("job_id", "")
    with _build_cache_lock:
        job = _build_cache_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify({
        "running": job["running"],
        "current": job["current"],
        "total": job["total"],
        "result": job.get("result"),
        "error": job.get("error"),
    })


_LINEUP_SUPPORTED = frozenset(("nrl", "uk-super-league", "nsw-cup", "qld-cup", "uk-championship"))


@app.route("/api/rugby/<competition_id>/lineup/refresh", methods=["POST"])
def api_rugby_lineup_refresh(competition_id: str):
    """Scrape League Unlimited Teams pages for fixture matches and update lineup CSV."""
    try:
        if competition_id not in _LINEUP_SUPPORTED:
            return jsonify({"ok": False, "error": f"Lineup fetch not implemented for {competition_id}"}), 501
        data = request.get_json(silent=True) or {}
        round_filter = data.get("round_num")
        from datafetch.league_unlimited_lineup_scraper import scrape_lineups
        path = scrape_lineups(
            competition_id=competition_id,
            year=2026,
            round_filter=round_filter,
            delay=0.3,
        )
        return jsonify({"ok": True, "message": f"Lineups refreshed: {path.name}"})
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/rugby/<competition_id>/fixture/rounds")
def api_rugby_fixture_rounds(competition_id: str):
    try:
        return jsonify(_rugby_service(competition_id).get_fixture_rounds())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rugby/<competition_id>/fixture/<path:round_num>")
def api_rugby_fixture_round(competition_id: str, round_num: str):
    try:
        svc = _rugby_service(competition_id)
        matches = svc.get_fixture_round(round_num)

        for m in matches:
            home_key = m.get("home_team_key", "")
            away_key = m.get("away_team_key", "")
            home_lineup = svc.get_last_lineup(home_key)
            away_lineup = svc.get_last_lineup(away_key)
            home_known = [p for p in home_lineup if p.get("player_id") != "unknown"]
            away_known = [p for p in away_lineup if p.get("player_id") != "unknown"]
            m["home_lineup_count"] = len(home_lineup)
            m["away_lineup_count"] = len(away_lineup)
            m["home_lineup_known"] = len(home_known)
            m["away_lineup_known"] = len(away_known)

        return jsonify(matches)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/rugby/<competition_id>/predict/round", methods=["POST"])
def api_rugby_predict_round(competition_id: str):
    data = request.get_json(silent=True)
    if not data or "matches" not in data:
        return jsonify({"error": "Body must contain {matches: [...]}"}), 400
    try:
        svc = _rugby_service(competition_id)
        results = []
        for match in data["matches"]:
            home_key = match.get("home_team_key", "")
            away_key = match.get("away_team_key", "")
            home_lineup = svc.get_last_lineup(home_key)
            away_lineup = svc.get_last_lineup(away_key)
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
            if len(home_ids) < 8 or len(away_ids) < 8:
                entry["prediction"] = None
                entry["error"] = "Insufficient lineup data"
            else:
                pred = predict_rugby_match(
                    competition_id,
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rugby/<competition_id>/data/status")
def api_rugby_data_status(competition_id: str):
    try:
        import glob as _glob
        from core.competition_config import get_competition_slugs, slug_matches_file, slug_matches_lineup_file
        slugs = get_competition_slugs(competition_id)
        match_dir = _PROJECT_ROOT / "nrl_data" / "data" / "matches"
        lineup_dir = _PROJECT_ROOT / "nrl_data" / "data" / "lineups"
        match_files = [f for f in _glob.glob(str(match_dir / "matches_*.csv")) if "2026" not in f and slug_matches_file(Path(f).name, slugs)]
        lineup_files = [f for f in _glob.glob(str(lineup_dir / "lineup_details_*.csv")) if slug_matches_lineup_file(Path(f).name, slugs)]
        cfg = COMPETITIONS.get(competition_id, {})
        fixture_path = match_dir / cfg.get("fixture_filename", "matches_2026.csv")
        fixture_matches = 0
        if fixture_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(fixture_path, usecols=["round_num"])
                fixture_matches = len(df)
            except Exception:
                pass
        years = []
        for f in match_files:
            try:
                stem = Path(f).stem
                for part in stem.replace("matches_", "").split("_"):
                    if part.isdigit() and len(part) == 4:
                        years.append(int(part))
            except Exception:
                pass
        match_year_range = f"{min(years)}–{max(years)}" if years else ""
        svc = _rugby_service(competition_id)
        return jsonify({
            "match_files": len(match_files),
            "lineup_files": len(lineup_files),
            "match_year_range": match_year_range,
            "fixture_matches": fixture_matches,
            "fixture_last_updated": svc.get_fixture_last_updated(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _run_rugby_fetch(job_id: str, competition_id: str, year_from: Optional[int], year_to: Optional[int]) -> None:
    script = str(_PROJECT_ROOT / "datafetch/rebuild_nrl_from_rlp.py")
    cmd = [sys.executable, script, "--competition", competition_id]
    if year_from is not None:
        cmd.extend(["--year-from", str(year_from)])
    if year_to is not None:
        cmd.extend(["--year-to", str(year_to)])
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
            with _fetch_lock:
                _fetch_jobs[job_id]["lines"].append(line.rstrip())
        proc.wait()
        with _fetch_lock:
            _fetch_jobs[job_id]["running"] = False
            _fetch_jobs[job_id]["exit_code"] = proc.returncode
    except Exception as e:
        with _fetch_lock:
            _fetch_jobs[job_id]["lines"].append(f"ERROR: {e}")
            _fetch_jobs[job_id]["running"] = False
            _fetch_jobs[job_id]["exit_code"] = 1


@app.route("/api/rugby/<competition_id>/data/fetch-historical", methods=["POST"])
def api_rugby_fetch_historical(competition_id: str):
    data = request.get_json(silent=True) or {}
    year_from = data.get("year_from")
    year_to = data.get("year_to")
    if year_from is not None:
        year_from = int(year_from)
    if year_to is not None:
        year_to = int(year_to)
    if _any_job_running():
        return jsonify({"error": "A training, tuning, or fetch job is already running"}), 409
    job_id = str(uuid.uuid4())[:8]
    with _fetch_lock:
        _fetch_jobs[job_id] = {"running": True, "lines": [], "exit_code": None, "seen": 0}
    threading.Thread(
        target=_run_rugby_fetch,
        args=(job_id, competition_id, year_from, year_to),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "ok": True})


@app.route("/api/rugby/<competition_id>/data/fetch-historical/status")
def api_rugby_fetch_historical_status(competition_id: str):
    job_id = request.args.get("job_id", "")
    with _fetch_lock:
        job = _fetch_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    with _fetch_lock:
        seen = job["seen"]
        new_lines = job["lines"][seen:]
        job["seen"] = len(job["lines"])
    return jsonify({
        "running": job["running"],
        "new_lines": new_lines,
        "exit_code": job["exit_code"],
    })


@app.route("/api/rugby/<competition_id>/data/model-metrics")
def api_rugby_model_metrics(competition_id: str):
    cfg = COMPETITIONS.get(competition_id, {})
    output_dir = Path(cfg.get("output_dir", str(_PROJECT_ROOT / "model" / "output" / competition_id)))
    metrics_path = output_dir / "evaluation_metrics.txt"
    if not metrics_path.exists():
        return jsonify({"error": "No evaluation_metrics.txt found"})
    metrics = {}
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            metrics[k.strip()] = v.strip()
    return jsonify(metrics)


def _run_rugby_training(job_id: str, competition_id: str, year_from: int, year_to: int, epochs: int) -> None:
    script = str(_PROJECT_ROOT / "model/rugby_train.py")
    cmd = [sys.executable, script, "--competition", competition_id, "--year-from", str(year_from), "--year-to", str(year_to), "--epochs", str(epochs)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(_PROJECT_ROOT))
        for line in proc.stdout:
            with _training_lock:
                _training_jobs[job_id]["lines"].append(line.rstrip())
        proc.wait()
        with _training_lock:
            _training_jobs[job_id]["running"] = False
            _training_jobs[job_id]["exit_code"] = proc.returncode
        if proc.returncode == 0:
            clear_rugby_model_cache(competition_id)
    except Exception as e:
        with _training_lock:
            _training_jobs[job_id]["lines"].append(f"ERROR: {e}")
            _training_jobs[job_id]["running"] = False
            _training_jobs[job_id]["exit_code"] = 1


@app.route("/api/rugby/<competition_id>/train", methods=["POST"])
def api_rugby_train(competition_id: str):
    data = request.get_json(silent=True) or {}
    year_from = int(data.get("year_from", 2020))
    year_to = int(data.get("year_to", 2025))
    epochs = int(data.get("epochs", 30))
    if _any_job_running():
        return jsonify({"error": "A training or tuning job is already running"}), 409
    job_id = str(uuid.uuid4())[:8]
    with _training_lock:
        _training_jobs[job_id] = {"running": True, "lines": [], "exit_code": None, "seen": 0}
    threading.Thread(target=_run_rugby_training, args=(job_id, competition_id, year_from, year_to, epochs), daemon=True).start()
    return jsonify({"job_id": job_id, "ok": True})


@app.route("/api/rugby/<competition_id>/train/status")
def api_rugby_train_status(competition_id: str):
    job_id = request.args.get("job_id", "")
    with _training_lock:
        job = _training_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    with _training_lock:
        seen = job["seen"]
        new_lines = job["lines"][seen:]
        job["seen"] = len(job["lines"])
    return jsonify({"running": job["running"], "new_lines": new_lines, "exit_code": job["exit_code"]})


def _run_rugby_tuning(job_id: str, competition_id: str, year_from: int, year_to: int, max_epochs: int) -> None:
    script = str(_PROJECT_ROOT / "model/nrl_tune_hyperband.py")
    cmd = [sys.executable, script, "--competition", competition_id, "--year-from", str(year_from), "--year-to", str(year_to), "--max-epochs", str(max_epochs), "--save-final"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(_PROJECT_ROOT))
        for line in proc.stdout:
            with _tune_lock:
                _tune_jobs[job_id]["lines"].append(line.rstrip())
        proc.wait()
        with _tune_lock:
            _tune_jobs[job_id]["running"] = False
            _tune_jobs[job_id]["exit_code"] = proc.returncode
        if proc.returncode == 0:
            clear_rugby_model_cache(competition_id)
    except Exception as e:
        with _tune_lock:
            _tune_jobs[job_id]["lines"].append(f"ERROR: {e}")
            _tune_jobs[job_id]["running"] = False
            _tune_jobs[job_id]["exit_code"] = 1


@app.route("/api/rugby/<competition_id>/tune", methods=["POST"])
def api_rugby_tune(competition_id: str):
    data = request.get_json(silent=True) or {}
    year_from = int(data.get("year_from", 2020))
    year_to = int(data.get("year_to", 2025))
    max_epochs = int(data.get("max_epochs", 100))
    if _any_job_running():
        return jsonify({"error": "A training or tuning job is already running"}), 409
    job_id = str(uuid.uuid4())[:8]
    with _tune_lock:
        _tune_jobs[job_id] = {"running": True, "lines": [], "exit_code": None, "seen": 0}
    threading.Thread(target=_run_rugby_tuning, args=(job_id, competition_id, year_from, year_to, max_epochs), daemon=True).start()
    return jsonify({"job_id": job_id, "ok": True})


@app.route("/api/rugby/<competition_id>/tune/status")
def api_rugby_tune_status(competition_id: str):
    job_id = request.args.get("job_id", "")
    with _tune_lock:
        job = _tune_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    with _tune_lock:
        seen = job["seen"]
        new_lines = job["lines"][seen:]
        job["seen"] = len(job["lines"])
    return jsonify({"running": job["running"], "new_lines": new_lines, "exit_code": job["exit_code"]})


# ---------------------------------------------------------------------------
# NRL API routes (backward compat — delegate to rugby)
# ---------------------------------------------------------------------------

@app.route("/api/nrl/teams")
def api_nrl_teams():
    return jsonify(_nrl_service.get_team_display_names())


@app.route("/api/nrl/grounds")
def api_nrl_grounds():
    return jsonify(_nrl_service.get_grounds_display())


@app.route("/api/nrl/lineup/<team_key>")
def api_nrl_lineup(team_key: str):
    return jsonify(_nrl_service.get_last_lineup(team_key))


@app.route("/api/nrl/season/<team_key>")
def api_nrl_season(team_key: str):
    return jsonify(_nrl_service.get_season_players(team_key))


@app.route("/api/nrl/players/search")
def api_nrl_players_search():
    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return jsonify([])
    return jsonify(_nrl_service.search_players(query, limit=30))


@app.route("/api/nrl/predict", methods=["POST"])
def api_nrl_predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400
    home_player_ids = data.get("home_player_ids", [])
    away_player_ids = data.get("away_player_ids", [])
    if not home_player_ids or not away_player_ids:
        return jsonify({"error": "home_player_ids and away_player_ids required"}), 400
    min_players = 10
    if len(home_player_ids) < min_players or len(away_player_ids) < min_players:
        return jsonify({"error": f"Each team needs at least {min_players} players"}), 400
    result = predict_nrl_match(
        home_player_ids, away_player_ids,
        home_team=data.get("home_team", ""),
        away_team=data.get("away_team", ""),
        venue=data.get("ground", ""),
    )
    if "error" in result:
        return jsonify(result), 500
    return jsonify({
        "home_team": data.get("home_team", ""),
        "away_team": data.get("away_team", ""),
        "ground": data.get("ground", ""),
        **result,
    })


@app.route("/api/nrl/fixture/meta")
def api_nrl_fixture_meta():
    return jsonify({"last_fetched": _nrl_service.get_fixture_last_updated()})


@app.route("/api/nrl/fixture/refresh", methods=["POST"])
def api_nrl_fixture_refresh():
    try:
        from datafetch.fetch_2026_nrl_fixture import fetch_and_save
        path = fetch_and_save()
        _nrl_service.save_fixture_meta()
        _nrl_service._invalidate_fixture_cache()
        return jsonify({"ok": True, "message": f"Fixture refreshed: {path.name}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/nrl/fixture/rounds")
def api_nrl_fixture_rounds():
    return jsonify(_nrl_service.get_fixture_rounds())


@app.route("/api/nrl/fixture/<path:round_num>")
def api_nrl_fixture_round(round_num: str):
    return jsonify(_nrl_service.get_fixture_round(round_num))


@app.route("/api/nrl/predict/round", methods=["POST"])
def api_nrl_predict_round():
    data = request.get_json(silent=True)
    if not data or "matches" not in data:
        return jsonify({"error": "Body must contain {matches: [...]}"}), 400
    results = []
    for match in data["matches"]:
        home_key = match.get("home_team_key", "")
        away_key = match.get("away_team_key", "")
        home_lineup = _nrl_service.get_last_lineup(home_key)
        away_lineup = _nrl_service.get_last_lineup(away_key)
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
        if len(home_ids) < 8 or len(away_ids) < 8:
            entry["prediction"] = None
            entry["error"] = "Insufficient lineup data"
        else:
            pred = predict_nrl_match(
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


@app.route("/api/nrl/data/status")
def api_nrl_data_status():
    import glob as _glob
    match_files = _glob.glob(str(_PROJECT_ROOT / "nrl_data/data/matches/matches_*.csv"))
    lineup_files = _glob.glob(str(_PROJECT_ROOT / "nrl_data/data/lineups/lineup_details_*.csv"))
    fixture_2026 = _PROJECT_ROOT / "nrl_data/data/matches/matches_2026.csv"
    fixture_matches = 0
    if fixture_2026.exists():
        try:
            import pandas as pd
            df = pd.read_csv(fixture_2026, usecols=["round_num"])
            fixture_matches = len(df)
        except Exception:
            pass
    years = []
    for f in match_files:
        if "2026" in f:
            continue
        try:
            stem = Path(f).stem
            for part in stem.replace("matches_", "").split("_"):
                if part.isdigit() and len(part) == 4:
                    years.append(int(part))
        except Exception:
            pass
    match_year_range = f"{min(years)}–{max(years)}" if years else ""
    return jsonify({
        "match_files": len([f for f in match_files if "2026" not in f]),
        "lineup_files": len(lineup_files),
        "match_year_range": match_year_range,
        "fixture_matches": fixture_matches,
        "fixture_last_updated": _nrl_service.get_fixture_last_updated(),
    })


def _run_nrl_fetch(job_id: str, year_from: Optional[int], year_to: Optional[int]) -> None:
    """Run rebuild_nrl_from_rlp.py in a subprocess, streaming output to job store."""
    script = str(_PROJECT_ROOT / "datafetch/rebuild_nrl_from_rlp.py")
    cmd = [sys.executable, script]
    if year_from is not None:
        cmd.extend(["--year-from", str(year_from)])
    if year_to is not None:
        cmd.extend(["--year-to", str(year_to)])
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
            with _fetch_lock:
                _fetch_jobs[job_id]["lines"].append(line.rstrip())
        proc.wait()
        with _fetch_lock:
            _fetch_jobs[job_id]["running"] = False
            _fetch_jobs[job_id]["exit_code"] = proc.returncode
    except Exception as e:
        with _fetch_lock:
            _fetch_jobs[job_id]["lines"].append(f"ERROR: {e}")
            _fetch_jobs[job_id]["running"] = False
            _fetch_jobs[job_id]["exit_code"] = 1


@app.route("/api/nrl/data/fetch-historical", methods=["POST"])
def api_nrl_fetch_historical():
    """Start background job to scrape full NRL lineage from Rugby League Project."""
    data = request.get_json(silent=True) or {}
    year_from = data.get("year_from")  # None = full lineage
    year_to = data.get("year_to")
    if year_from is not None:
        year_from = int(year_from)
    if year_to is not None:
        year_to = int(year_to)
    if _any_job_running():
        return jsonify({"error": "A training, tuning, or fetch job is already running"}), 409
    job_id = str(uuid.uuid4())[:8]
    with _fetch_lock:
        _fetch_jobs[job_id] = {"running": True, "lines": [], "exit_code": None, "seen": 0}
    threading.Thread(
        target=_run_nrl_fetch,
        args=(job_id, year_from, year_to),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "ok": True})


@app.route("/api/nrl/data/fetch-historical/status")
def api_nrl_fetch_historical_status():
    job_id = request.args.get("job_id", "")
    with _fetch_lock:
        job = _fetch_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    with _fetch_lock:
        seen = job["seen"]
        new_lines = job["lines"][seen:]
        job["seen"] = len(job["lines"])
    return jsonify({
        "running": job["running"],
        "new_lines": new_lines,
        "exit_code": job["exit_code"],
    })


@app.route("/api/nrl/data/model-metrics")
def api_nrl_model_metrics():
    metrics_path = _PROJECT_ROOT / "model/output/nrl/evaluation_metrics.txt"
    if not metrics_path.exists():
        return jsonify({"error": "No evaluation_metrics.txt found"})
    metrics = {}
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            metrics[k.strip()] = v.strip()
    return jsonify(metrics)


def _run_nrl_training(job_id: str, year_from: int, year_to: int, epochs: int) -> None:
    script = str(_PROJECT_ROOT / "model/nrl_train.py")
    cmd = [sys.executable, script, "--year-from", str(year_from), "--year-to", str(year_to), "--epochs", str(epochs)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(_PROJECT_ROOT))
        for line in proc.stdout:
            with _training_lock:
                _training_jobs[job_id]["lines"].append(line.rstrip())
        proc.wait()
        with _training_lock:
            _training_jobs[job_id]["running"] = False
            _training_jobs[job_id]["exit_code"] = proc.returncode
        if proc.returncode == 0:
            clear_nrl_model_cache()
    except Exception as e:
        with _training_lock:
            _training_jobs[job_id]["lines"].append(f"ERROR: {e}")
            _training_jobs[job_id]["running"] = False
            _training_jobs[job_id]["exit_code"] = 1


@app.route("/api/nrl/train", methods=["POST"])
def api_nrl_train():
    data = request.get_json(silent=True) or {}
    year_from = int(data.get("year_from", 2020))
    year_to = int(data.get("year_to", 2025))
    epochs = int(data.get("epochs", 30))
    if _any_job_running():
        return jsonify({"error": "A training or tuning job is already running"}), 409
    job_id = str(uuid.uuid4())[:8]
    with _training_lock:
        _training_jobs[job_id] = {"running": True, "lines": [], "exit_code": None, "seen": 0}
    threading.Thread(target=_run_nrl_training, args=(job_id, year_from, year_to, epochs), daemon=True).start()
    return jsonify({"job_id": job_id, "ok": True})


@app.route("/api/nrl/train/status")
def api_nrl_train_status():
    job_id = request.args.get("job_id", "")
    with _training_lock:
        job = _training_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    with _training_lock:
        seen = job["seen"]
        new_lines = job["lines"][seen:]
        job["seen"] = len(job["lines"])
    return jsonify({"running": job["running"], "new_lines": new_lines, "exit_code": job["exit_code"]})


def _run_nrl_tuning(job_id: str, year_from: int, year_to: int, max_epochs: int) -> None:
    script = str(_PROJECT_ROOT / "model/nrl_tune_hyperband.py")
    cmd = [sys.executable, script, "--year-from", str(year_from), "--year-to", str(year_to), "--max-epochs", str(max_epochs), "--save-final"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(_PROJECT_ROOT))
        for line in proc.stdout:
            with _tune_lock:
                _tune_jobs[job_id]["lines"].append(line.rstrip())
        proc.wait()
        with _tune_lock:
            _tune_jobs[job_id]["running"] = False
            _tune_jobs[job_id]["exit_code"] = proc.returncode
        if proc.returncode == 0:
            clear_nrl_model_cache()
    except Exception as e:
        with _tune_lock:
            _tune_jobs[job_id]["lines"].append(f"ERROR: {e}")
            _tune_jobs[job_id]["running"] = False
            _tune_jobs[job_id]["exit_code"] = 1


@app.route("/api/nrl/tune", methods=["POST"])
def api_nrl_tune():
    data = request.get_json(silent=True) or {}
    year_from = int(data.get("year_from", 2020))
    year_to = int(data.get("year_to", 2025))
    max_epochs = int(data.get("max_epochs", 100))
    if _any_job_running():
        return jsonify({"error": "A training or tuning job is already running"}), 409
    job_id = str(uuid.uuid4())[:8]
    with _tune_lock:
        _tune_jobs[job_id] = {"running": True, "lines": [], "exit_code": None, "seen": 0}
    threading.Thread(target=_run_nrl_tuning, args=(job_id, year_from, year_to, max_epochs), daemon=True).start()
    return jsonify({"job_id": job_id, "ok": True})


@app.route("/api/nrl/tune/status")
def api_nrl_tune_status():
    job_id = request.args.get("job_id", "")
    with _tune_lock:
        job = _tune_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    with _tune_lock:
        seen = job["seen"]
        new_lines = job["lines"][seen:]
        job["seen"] = len(job["lines"])
    return jsonify({"running": job["running"], "new_lines": new_lines, "exit_code": job["exit_code"]})


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))  # 5001 default: macOS AirPlay uses 5000
    app.run(debug=True, port=port)
