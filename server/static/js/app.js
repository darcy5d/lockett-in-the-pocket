/**
 * AFL Match Predictor — 2026
 *
 * Tabs:
 *   1. 2026 Fixture  — round selector, match cards, whole-round prediction table
 *   2. Data & Training — data status, model metrics, retrain controls + live log
 */

"use strict";

// ---------------------------------------------------------------------------
// DOM helper
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

function initTabs() {
  const tabs = document.querySelectorAll("#main-tabs .nav-link");
  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabs.forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");
      const target = btn.dataset.tab;
      $("tab-fixture").style.display = target === "fixture" ? "" : "none";
      $("tab-training").style.display = target === "training" ? "" : "none";
      if (target === "fixture" && !_fixtureTabInitialised) initFixtureTab();
      if (target === "training" && !_trainingTabInitialised) initTrainingTab();
    });
  });
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { ok: res.ok, status: res.status, data: await res.json() };
}

// ===========================================================================
// TAB 1 — 2026 Fixture
// ===========================================================================

let _fixtureTabInitialised = false;
let _currentRoundMatches = [];

async function initFixtureTab() {
  _fixtureTabInitialised = true;
  await Promise.all([loadRounds(), loadFixtureFreshness()]);
}

// ── Freshness ────────────────────────────────────────────────────────────────

async function loadFixtureFreshness() {
  try {
    const meta = await fetchJSON("/api/fixture/meta");
    const el = $("fixture-freshness");
    if (meta.last_fetched) {
      const d = new Date(meta.last_fetched);
      el.textContent = `Updated ${d.toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" })}`;
    } else {
      el.textContent = "Never refreshed";
    }
  } catch (e) {
    console.error("Freshness check failed:", e);
  }
}

// ── Rounds ───────────────────────────────────────────────────────────────────

async function loadRounds() {
  try {
    const rounds = await fetchJSON("/api/fixture/rounds");
    const sel = $("round-select");
    sel.innerHTML = '<option value="">Select round…</option>';
    rounds.forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r;
      opt.textContent = r === "Opening Round" ? "Opening Round" : `Round ${r}`;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error("Failed to load rounds:", e);
  }
}

async function loadRound(roundNum) {
  const container = $("round-matches");
  container.innerHTML = '<div class="text-secondary ps-1">Loading…</div>';
  $("round-results-section").style.display = "none";
  $("predict-round-btn").disabled = true;

  try {
    const matches = await fetchJSON(`/api/fixture/${encodeURIComponent(roundNum)}`);
    _currentRoundMatches = matches;
    renderMatchCards(matches);
    $("predict-round-btn").disabled = matches.length === 0;
  } catch (e) {
    container.innerHTML = `<div class="text-danger">Failed to load round: ${e.message}</div>`;
  }
}

function renderMatchCards(matches) {
  const container = $("round-matches");
  container.innerHTML = "";
  if (matches.length === 0) {
    container.innerHTML = '<div class="text-secondary p-3">No matches found.</div>';
    return;
  }
  matches.forEach((m) => {
    const dateLabel = formatMatchDate(m.date);
    const tbcHtml = m.time_confirmed ? "" : '<span class="tbc-pill ms-1">TIME TBC</span>';
    const col = document.createElement("div");
    col.className = "col-sm-6 col-xl-4";
    col.innerHTML = `
      <div class="match-card">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <span class="meta">${dateLabel}${tbcHtml}</span>
          <span class="meta">${m.venue_display || m.venue}</span>
        </div>
        <div class="d-flex align-items-center justify-content-between gap-3">
          <span class="team-name text-warning text-end" style="flex:1">${m.home_team_display}</span>
          <span class="vs-badge">VS</span>
          <span class="team-name text-light" style="flex:1">${m.away_team_display}</span>
        </div>
      </div>`;
    container.appendChild(col);
  });
}

function formatMatchDate(dateStr, timeUnconfirmed = false) {
  if (!dateStr) return "TBC";
  const parts = dateStr.split(" ");
  const [day, month, year] = parts[0].split("/");
  const d = new Date(`${year}-${month}-${day}`);
  const label = d.toLocaleDateString("en-AU", { weekday: "short", day: "numeric", month: "short" });
  return timeUnconfirmed ? `${label} (TBC)` : label;
}

// ── Predict round ────────────────────────────────────────────────────────────

async function predictRound() {
  if (_currentRoundMatches.length === 0) return;
  const btn = $("predict-round-btn");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  const roundLabel = $("round-select").value;

  try {
    const { ok, data } = await postJSON("/api/predict/round", { matches: _currentRoundMatches });
    if (!ok) { alert("Round prediction failed."); console.error(data); return; }
    renderRoundResults(data, roundLabel);
  } catch (e) {
    alert(`Network error: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict Whole Round";
  }
}

function renderRoundResults(results, roundLabel) {
  const section = $("round-results-section");
  $("round-results-title").textContent =
    roundLabel === "Opening Round" ? "Opening Round Predictions" : `Round ${roundLabel} Predictions`;

  const tbody = $("round-results-body");
  tbody.innerHTML = "";

  results.forEach((r) => {
    const pred = r.prediction;
    const tr = document.createElement("tr");
    const tbcBadge = r.time_confirmed ? "" : `<span class="tbc-pill ms-1">TBC</span>`;

    if (!pred) {
      tr.innerHTML = `
        <td class="meta">${formatMatchDate(r.date)}${tbcBadge}</td>
        <td class="meta">${r.venue}</td>
        <td class="text-warning fw-semibold">${r.home_team}</td>
        <td colspan="3" class="text-center text-secondary fst-italic small">—</td>
        <td class="text-light">${r.away_team}</td>
        <td class="text-center text-secondary small fst-italic" colspan="2">${r.error || "No data"}</td>`;
    } else {
      const probs = pred.winner_probabilities;
      const t1 = pred.team1, t2 = pred.team2;
      const s1 = Math.round(t1.goals * 6 + t1.behinds);
      const s2 = Math.round(t2.goals * 6 + t2.behinds);
      const margin = Math.abs(pred.margin).toFixed(0);
      const homeWins = probs.team1_win >= probs.team2_win;
      const leader = homeWins ? r.home_team : r.away_team;

      tr.innerHTML = `
        <td class="meta">${formatMatchDate(r.date)}${tbcBadge}</td>
        <td class="meta">${r.venue}</td>
        <td class="${homeWins ? "text-warning fw-bold" : "text-secondary"}">${r.home_team}</td>
        <td class="text-center">
          <span class="prob-pill ${homeWins ? "winner" : "loser"}">${(probs.team1_win*100).toFixed(0)}%</span>
        </td>
        <td class="text-center"><span class="meta">${(probs.draw*100).toFixed(0)}%</span></td>
        <td class="text-center">
          <span class="prob-pill ${!homeWins ? "winner" : "loser"}">${(probs.team2_win*100).toFixed(0)}%</span>
        </td>
        <td class="${!homeWins ? "text-warning fw-bold" : "text-secondary"}">${r.away_team}</td>
        <td class="text-center">
          <span class="score-display ${homeWins ? "text-warning" : "text-light"}">${t1.goals.toFixed(0)}.${t1.behinds.toFixed(0)} (${s1})</span>
          <span class="meta mx-1">–</span>
          <span class="score-display ${!homeWins ? "text-warning" : "text-light"}">${t2.goals.toFixed(0)}.${t2.behinds.toFixed(0)} (${s2})</span>
        </td>
        <td class="text-center"><span class="margin-display">${leader} by ${margin}</span></td>`;
    }
    tbody.appendChild(tr);
  });

  section.style.display = "block";
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Refresh fixture ───────────────────────────────────────────────────────────

async function refreshFixture() {
  const btn = $("refresh-fixture-btn");
  btn.disabled = true;
  btn.textContent = "Refreshing…";
  try {
    const { ok, data } = await postJSON("/api/fixture/refresh", {});
    if (ok && data.ok) {
      await loadFixtureFreshness();
      const round = $("round-select").value;
      if (round) await loadRound(round);
      btn.textContent = "Done!";
    } else {
      alert(`Refresh failed: ${data.error || "unknown"}`);
      btn.textContent = "Refresh Fixture";
    }
  } catch (e) {
    alert(`Error: ${e.message}`);
    btn.textContent = "Refresh Fixture";
  } finally {
    setTimeout(() => { btn.disabled = false; btn.textContent = "Refresh Fixture"; }, 2000);
  }
}


// ===========================================================================
// TAB 2 — Data & Training
// ===========================================================================

let _trainingTabInitialised = false;
let _trainingJobId = null;
let _trainingPollTimer = null;
let _trainingStart = null;

async function initTrainingTab() {
  _trainingTabInitialised = true;
  await Promise.all([loadDataStatus(), loadModelMetrics()]);
}

// ── Data status ───────────────────────────────────────────────────────────────

async function loadDataStatus() {
  try {
    const d = await fetchJSON("/api/data/status");
    $("stat-match-files").textContent = d.match_files ?? "—";
    $("stat-match-range").textContent = d.match_year_range ?? "";
    $("stat-player-files").textContent = d.player_files ?? "—";
    $("stat-lineup-files").textContent = d.lineup_files ?? "—";
    $("stat-fixture-matches").textContent = d.fixture_matches ?? "—";
    $("stat-fixture-updated").textContent = d.fixture_last_updated
      ? `Updated ${new Date(d.fixture_last_updated).toLocaleDateString("en-AU", { day:"numeric", month:"short" })}`
      : "Not fetched";
  } catch (e) {
    console.error("Data status failed:", e);
  }
}

// ── Model metrics ─────────────────────────────────────────────────────────────

async function loadModelMetrics() {
  try {
    const m = await fetchJSON("/api/data/model-metrics");
    const el = $("model-metrics");
    if (m.error) {
      el.innerHTML = `<div class="text-secondary small">${m.error}</div>`;
      return;
    }

    const rmseMargin = parseFloat(m.rmse_margin ?? 0);
    const marginBad = rmseMargin > 20;
    const acc = parseFloat(m.match_winner_accuracy ?? 0);

    el.innerHTML = `
      <div class="metrics-grid">
        ${metricRow("Winner accuracy", `${(acc*100).toFixed(1)}%`, acc < 0.6 ? "bad" : acc < 0.8 ? "ok" : "good")}
        ${metricRow("F1 score", parseFloat(m.f1_score ?? 0).toFixed(3), parseFloat(m.f1_score??0) < 0.5 ? "bad" : "ok")}
        ${metricRow("RMSE margin", rmseMargin.toFixed(1) + " pts", marginBad ? "bad" : "ok")}
        ${metricRow("RMSE team1 goals", parseFloat(m.rmse_team1_goals ?? 0).toFixed(2))}
        ${metricRow("RMSE team2 goals", parseFloat(m.rmse_team2_goals ?? 0).toFixed(2))}
        ${metricRow("Trained", m.trained_date ?? "Unknown")}
      </div>
      ${marginBad ? `<div class="text-danger small mt-2">High RMSE margin (${rmseMargin.toFixed(0)}) indicates the player embedding bug. Retrain to fix.</div>` : ""}`;
  } catch (e) {
    $("model-metrics").innerHTML = `<div class="text-secondary small">Could not load metrics.</div>`;
  }
}

function metricRow(label, value, quality) {
  const cls = quality === "bad" ? "text-danger" : quality === "good" ? "text-success" : quality === "ok" ? "text-warning" : "text-secondary";
  return `<div class="metric-row">
    <span class="metric-label">${label}</span>
    <span class="metric-value ${cls}">${value}</span>
  </div>`;
}

// ── Fetch historical data ─────────────────────────────────────────────────────

async function fetchHistorical() {
  const status = $("fetch-status");
  const btn = $("fetch-historical-btn");
  btn.disabled = true;
  status.textContent = "Fetching from akareen/AFL-Data-Analysis…";
  try {
    const { ok, data } = await postJSON("/api/data/fetch-historical", {});
    status.textContent = ok && data.ok ? `Done: ${data.message}` : `Failed: ${data.error}`;
    if (ok && data.ok) await loadDataStatus();
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

async function fetchFixtureFromTraining() {
  const status = $("fetch-status");
  const btn = $("fetch-fixture-btn");
  btn.disabled = true;
  status.textContent = "Fetching 2026 fixture…";
  try {
    const { ok, data } = await postJSON("/api/fixture/refresh", {});
    status.textContent = ok && data.ok ? `Done: ${data.message}` : `Failed: ${data.error}`;
    if (ok && data.ok) await loadDataStatus();
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

// ── Retrain ───────────────────────────────────────────────────────────────────

async function startTraining() {
  const btn = $("train-btn");
  const logBox = $("train-log-box");
  const statusBar = $("train-status-bar");

  btn.disabled = true;
  logBox.innerHTML = "";
  logBox.style.display = "block";
  statusBar.style.display = "block";
  $("train-status-label").textContent = "Starting…";
  $("train-elapsed").textContent = "";
  _trainingStart = Date.now();

  const yearFrom = parseInt($("train-year-from").value);
  const yearTo = parseInt($("train-year-to").value);
  const epochs = parseInt($("train-epochs").value);

  try {
    const { ok, data } = await postJSON("/api/train", { year_from: yearFrom, year_to: yearTo, epochs });
    if (!ok || data.error) {
      appendLog(`ERROR: ${data.error || "Failed to start training"}`, "error");
      btn.disabled = false;
      return;
    }
    _trainingJobId = data.job_id;
    appendLog(`Training started (job ${_trainingJobId})`, "info");
    $("train-status-label").textContent = "Training…";
    pollTrainingLog();
  } catch (e) {
    appendLog(`Error: ${e.message}`, "error");
    btn.disabled = false;
  }
}

async function pollTrainingLog() {
  if (!_trainingJobId) return;
  try {
    const d = await fetchJSON(`/api/train/status?job_id=${_trainingJobId}`);
    const elapsed = Math.round((Date.now() - _trainingStart) / 1000);
    $("train-elapsed").textContent = `${elapsed}s`;

    // Render new log lines
    if (d.new_lines && d.new_lines.length) {
      d.new_lines.forEach((line) => appendLog(line));
    }

    if (d.running) {
      $("train-status-label").textContent = "Training…";
      _trainingPollTimer = setTimeout(pollTrainingLog, 2000);
    } else {
      $("train-status-label").textContent = d.exit_code === 0 ? "Completed" : "Failed";
      $("train-progress").classList.remove("progress-bar-animated");
      $("train-progress").style.width = "100%";
      $("train-progress").className = d.exit_code === 0
        ? "progress-bar bg-success"
        : "progress-bar bg-danger";
      $("train-btn").disabled = false;
      _trainingJobId = null;
      // Reload metrics
      await loadModelMetrics();
    }
  } catch (e) {
    appendLog(`Poll error: ${e.message}`, "error");
    _trainingPollTimer = setTimeout(pollTrainingLog, 4000);
  }
}

function appendLog(text, type) {
  const logBox = $("train-log-box");
  const line = document.createElement("div");
  line.className = "log-line" + (type === "error" ? " log-error" : type === "info" ? " log-info" : "");
  line.textContent = text;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}


// ===========================================================================
// Boot
// ===========================================================================

document.addEventListener("DOMContentLoaded", () => {
  initTabs();

  // Fixture tab is default — hide training tab, init fixture
  $("tab-training").style.display = "none";
  initFixtureTab();

  // Fixture tab wiring
  $("round-select").addEventListener("change", (e) => { if (e.target.value) loadRound(e.target.value); });
  $("predict-round-btn").addEventListener("click", predictRound);
  $("refresh-fixture-btn").addEventListener("click", refreshFixture);

  // Training tab wiring
  $("refresh-data-btn").addEventListener("click", loadDataStatus);
  $("fetch-historical-btn").addEventListener("click", fetchHistorical);
  $("fetch-fixture-btn").addEventListener("click", fetchFixtureFromTraining);
  $("train-btn").addEventListener("click", startTraining);
});
