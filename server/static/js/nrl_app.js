/**
 * NRL Match Predictor — 2026
 * Uses /api/nrl/ endpoints. Score display: points (no goals/behinds).
 */

"use strict";

const $ = (id) => document.getElementById(id);
const API = "/api/nrl";

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return { ok: res.ok, data: await res.json() };
}

function initTabs() {
  document.querySelectorAll("#main-tabs .nav-link").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#main-tabs .nav-link").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.dataset.tab;
      $("tab-fixture").style.display = target === "fixture" ? "" : "none";
      $("tab-training").style.display = target === "training" ? "" : "none";
      if (target === "fixture" && !_fixtureInit) initFixtureTab();
      if (target === "training" && !_trainingInit) initTrainingTab();
    });
  });
}

let _fixtureInit = false, _trainingInit = false, _currentRoundMatches = [];

async function initFixtureTab() {
  _fixtureInit = true;
  await Promise.all([loadRounds(), loadFixtureFreshness()]);
}

async function loadFixtureFreshness() {
  try {
    const meta = await fetchJSON(`${API}/fixture/meta`);
    $("fixture-freshness").textContent = meta.last_fetched
      ? `Updated ${new Date(meta.last_fetched).toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" })}`
      : "Never refreshed";
  } catch (e) { $("fixture-freshness").textContent = "Error"; }
}

async function loadRounds() {
  try {
    const rounds = await fetchJSON(`${API}/fixture/rounds`);
    const sel = $("round-select");
    sel.innerHTML = '<option value="">Select round…</option>';
    rounds.forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r;
      opt.textContent = `Round ${r}`;
      sel.appendChild(opt);
    });
  } catch (e) { console.error(e); }
}

async function loadRound(roundNum) {
  $("round-matches").innerHTML = '<div class="text-secondary ps-1">Loading…</div>';
  $("round-results-section").style.display = "none";
  $("predict-round-btn").disabled = true;
  try {
    const matches = await fetchJSON(`${API}/fixture/${encodeURIComponent(roundNum)}`);
    _currentRoundMatches = matches;
    $("round-matches").innerHTML = "";
    matches.forEach((m) => {
      const col = document.createElement("div");
      col.className = "col-sm-6 col-xl-4";
      col.innerHTML = `
        <div class="match-card">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <span class="meta">${m.date || "TBC"}</span>
            <span class="meta">${m.venue_display || m.venue}</span>
          </div>
          <div class="d-flex align-items-center justify-content-between gap-3">
            <span class="team-name text-warning text-end" style="flex:1">${m.home_team_display}</span>
            <span class="vs-badge">VS</span>
            <span class="team-name text-light" style="flex:1">${m.away_team_display}</span>
          </div>
        </div>`;
      $("round-matches").appendChild(col);
    });
    $("predict-round-btn").disabled = matches.length === 0;
  } catch (e) {
    $("round-matches").innerHTML = `<div class="text-danger">Failed: ${e.message}</div>`;
  }
}

async function predictRound() {
  if (_currentRoundMatches.length === 0) return;
  const btn = $("predict-round-btn");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  const roundLabel = $("round-select").value;
  try {
    const { ok, data } = await postJSON(`${API}/predict/round`, { matches: _currentRoundMatches });
    if (!ok) { alert("Round prediction failed."); return; }
    renderRoundResults(data, roundLabel);
  } catch (e) { alert(`Error: ${e.message}`); }
  finally { btn.disabled = false; btn.textContent = "Predict Whole Round"; }
}

function renderRoundResults(results, roundLabel) {
  $("round-results-title").textContent = `Round ${roundLabel} Predictions`;
  const tbody = $("round-results-body");
  tbody.innerHTML = "";
  results.forEach((r) => {
    const pred = r.prediction;
    const tr = document.createElement("tr");
    if (!pred) {
      tr.innerHTML = `
        <td class="meta">${r.date || "TBC"}</td>
        <td class="meta">${r.venue}</td>
        <td class="text-warning fw-semibold">${r.home_team}</td>
        <td colspan="3" class="text-center text-secondary fst-italic small">—</td>
        <td class="text-light">${r.away_team}</td>
        <td class="text-center text-secondary small fst-italic" colspan="2">${r.error || "No data"}</td>`;
    } else {
      const s1 = Math.round(pred.team1_score || 0);
      const s2 = Math.round(pred.team2_score || 0);
      const homeWins = (pred.margin || 0) > 0;
      const pHome = (pred.p_home_win || 0) * 100;
      const pAway = (pred.p_away_win || 0) * 100;
      const pDraw = (pred.p_draw || 0) * 100;
      tr.innerHTML = `
        <td class="meta">${r.date || "TBC"}</td>
        <td class="meta">${r.venue}</td>
        <td class="${homeWins ? "text-warning fw-bold" : "text-secondary"}">${r.home_team}</td>
        <td class="text-center"><span class="prob-pill ${homeWins ? "winner" : "loser"}">${pHome.toFixed(0)}%</span></td>
        <td class="text-center"><span class="meta">${pDraw.toFixed(0)}%</span></td>
        <td class="text-center"><span class="prob-pill ${!homeWins ? "winner" : "loser"}">${pAway.toFixed(0)}%</span></td>
        <td class="${!homeWins ? "text-warning fw-bold" : "text-secondary"}">${r.away_team}</td>
        <td class="text-center"><span class="score-display">${s1} – ${s2}</span></td>
        <td class="text-center"><span class="margin-display">${pred.margin === 0 ? "Draw" : `${Math.abs(pred.margin).toFixed(0)} pts`}</span></td>`;
    }
    tbody.appendChild(tr);
  });
  $("round-results-section").style.display = "block";
  $("round-results-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function refreshFixture() {
  const btn = $("refresh-fixture-btn");
  btn.disabled = true;
  btn.textContent = "Refreshing…";
  try {
    const { ok, data } = await postJSON(`${API}/fixture/refresh`, {});
    if (ok && data.ok) {
      await loadFixtureFreshness();
      const round = $("round-select").value;
      if (round) await loadRound(round);
    } else alert(`Failed: ${data.error || "unknown"}`);
  } catch (e) { alert(`Error: ${e.message}`); }
  finally { btn.disabled = false; btn.textContent = "Refresh Fixture"; }
}

async function initTrainingTab() {
  _trainingInit = true;
  await Promise.all([loadDataStatus(), loadModelMetrics()]);
}

async function loadDataStatus() {
  try {
    const d = await fetchJSON(`${API}/data/status`);
    $("stat-match-files").textContent = d.match_files ?? "—";
    $("stat-match-year-range").textContent = d.match_year_range ? d.match_year_range : "";
    $("stat-lineup-files").textContent = d.lineup_files ?? "—";
    $("stat-fixture-matches").textContent = d.fixture_matches ?? "—";
    $("stat-fixture-updated").textContent = d.fixture_last_updated
      ? `Updated ${new Date(d.fixture_last_updated).toLocaleDateString("en-AU", { day: "numeric", month: "short" })}`
      : "Not fetched";
  } catch (e) { console.error(e); }
}

async function loadModelMetrics() {
  try {
    const m = await fetchJSON(`${API}/data/model-metrics`);
    const el = $("model-metrics");
    if (m.error) { el.innerHTML = `<div class="text-secondary small">${m.error}</div>`; return; }
    el.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-row"><span class="metric-label">MAE team1_score</span><span class="metric-value">${m["MAE team1_score"] || "—"}</span></div>
        <div class="metric-row"><span class="metric-label">MAE team2_score</span><span class="metric-value">${m["MAE team2_score"] || "—"}</span></div>
        <div class="metric-row"><span class="metric-label">MAE margin</span><span class="metric-value">${m["MAE margin"] || "—"}</span></div>
      </div>`;
  } catch (e) { $("model-metrics").innerHTML = `<div class="text-secondary small">Could not load metrics.</div>`; }
}

async function fetchFixture() {
  const status = $("fetch-status");
  const btn = $("fetch-fixture-btn");
  btn.disabled = true;
  status.textContent = "Fetching 2026 fixture…";
  try {
    const { ok, data } = await postJSON(`${API}/fixture/refresh`, {});
    status.textContent = ok && data.ok ? `Done: ${data.message}` : `Failed: ${data.error}`;
    if (ok && data.ok) await loadDataStatus();
  } catch (e) { status.textContent = `Error: ${e.message}`; }
  finally { btn.disabled = false; }
}

let _fetchJobId = null;

function setFetchRunning(running) {
  $("fetch-historical-btn").disabled = running;
  $("fetch-fixture-btn").disabled = running;
}

function appendFetchLog(text, type) {
  const logBox = $("fetch-historical-log");
  if (!logBox) return;
  const line = document.createElement("div");
  line.className = "log-line" + (type === "error" ? " log-error" : type === "info" ? " log-info" : "");
  line.textContent = text;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

async function fetchHistorical() {
  const status = $("fetch-status");
  const logBox = $("fetch-historical-log");
  setFetchRunning(true);
  status.textContent = "Starting…";
  if (logBox) { logBox.innerHTML = ""; logBox.style.display = "block"; }
  try {
    const { ok, data } = await postJSON(`${API}/data/fetch-historical`, {});
    if (!ok || data.error) {
      status.textContent = `Failed: ${data.error}`;
      appendFetchLog(`ERROR: ${data.error}`, "error");
      setFetchRunning(false);
      return;
    }
    _fetchJobId = data.job_id;
    status.textContent = "Scraping RLP (full lineage)…";
    appendFetchLog(`Fetch started (job ${_fetchJobId})`, "info");
    pollFetchLog();
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    appendFetchLog(`Error: ${e.message}`, "error");
    setFetchRunning(false);
  }
}

async function pollFetchLog() {
  if (!_fetchJobId) return;
  try {
    const d = await fetchJSON(`${API}/data/fetch-historical/status?job_id=${_fetchJobId}`);
    if (d.new_lines) d.new_lines.forEach((l) => appendFetchLog(l));
    if (d.running) { setTimeout(pollFetchLog, 3000); return; }
    _fetchJobId = null;
    setFetchRunning(false);
    $("fetch-status").textContent = d.exit_code === 0 ? "Historical data updated" : "Fetch failed";
    await loadDataStatus();
  } catch (e) { setTimeout(pollFetchLog, 5000); }
}

let _trainingJobId = null, _tuneJobId = null, _trainingStart = null;

function setJobRunning(running) {
  $("train-btn").disabled = running;
  $("tune-btn").disabled = running;
  $("fetch-historical-btn").disabled = running;
  $("fetch-fixture-btn").disabled = running;
}

function appendLog(text, type) {
  const logBox = $("train-log-box");
  const line = document.createElement("div");
  line.className = "log-line" + (type === "error" ? " log-error" : type === "info" ? " log-info" : "");
  line.textContent = text;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

async function startTraining() {
  setJobRunning(true);
  $("train-log-box").innerHTML = "";
  $("train-log-box").style.display = "block";
  $("train-status-bar").style.display = "block";
  $("train-status-label").textContent = "Starting…";
  _trainingStart = Date.now();
  const yearFrom = parseInt($("train-year-from").value);
  const yearTo = parseInt($("train-year-to").value);
  const epochs = parseInt($("train-epochs").value);
  try {
    const { ok, data } = await postJSON(`${API}/train`, { year_from: yearFrom, year_to: yearTo, epochs });
    if (!ok || data.error) { appendLog(`ERROR: ${data.error || "Failed"}`, "error"); setJobRunning(false); return; }
    _trainingJobId = data.job_id;
    appendLog(`Training started (job ${_trainingJobId})`, "info");
    pollTrainingLog();
  } catch (e) { appendLog(`Error: ${e.message}`, "error"); setJobRunning(false); }
}

async function pollTrainingLog() {
  if (!_trainingJobId) return;
  try {
    const d = await fetchJSON(`${API}/train/status?job_id=${_trainingJobId}`);
    $("train-elapsed").textContent = `${Math.round((Date.now() - _trainingStart) / 1000)}s`;
    if (d.new_lines) d.new_lines.forEach((l) => appendLog(l));
    if (d.running) { setTimeout(pollTrainingLog, 2000); return; }
    $("train-status-label").textContent = d.exit_code === 0 ? "Completed" : "Failed";
    setJobRunning(false);
    _trainingJobId = null;
    await loadModelMetrics();
  } catch (e) { setTimeout(pollTrainingLog, 4000); }
}

async function startTuning() {
  setJobRunning(true);
  $("train-log-box").innerHTML = "";
  $("train-log-box").style.display = "block";
  $("train-status-bar").style.display = "block";
  $("train-status-label").textContent = "Starting…";
  _trainingStart = Date.now();
  const yearFrom = parseInt($("train-year-from").value);
  const yearTo = parseInt($("train-year-to").value);
  const maxEpochs = parseInt($("train-epochs").value);
  try {
    const { ok, data } = await postJSON(`${API}/tune`, { year_from: yearFrom, year_to: yearTo, max_epochs: maxEpochs });
    if (!ok || data.error) { appendLog(`ERROR: ${data.error || "Failed"}`, "error"); setJobRunning(false); return; }
    _tuneJobId = data.job_id;
    appendLog(`Tuning started (job ${_tuneJobId})`, "info");
    pollTuningLog();
  } catch (e) { appendLog(`Error: ${e.message}`, "error"); setJobRunning(false); }
}

async function pollTuningLog() {
  if (!_tuneJobId) return;
  try {
    const d = await fetchJSON(`${API}/tune/status?job_id=${_tuneJobId}`);
    $("train-elapsed").textContent = `${Math.round((Date.now() - _trainingStart) / 1000)}s`;
    if (d.new_lines) d.new_lines.forEach((l) => appendLog(l));
    if (d.running) { setTimeout(pollTuningLog, 2000); return; }
    $("train-status-label").textContent = d.exit_code === 0 ? "Completed" : "Failed";
    setJobRunning(false);
    _tuneJobId = null;
    await loadModelMetrics();
  } catch (e) { setTimeout(pollTuningLog, 4000); }
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  $("tab-training").style.display = "none";
  initFixtureTab();
  $("round-select").addEventListener("change", (e) => { if (e.target.value) loadRound(e.target.value); });
  $("predict-round-btn").addEventListener("click", predictRound);
  $("refresh-fixture-btn").addEventListener("click", refreshFixture);
  $("refresh-data-btn").addEventListener("click", loadDataStatus);
  $("fetch-historical-btn").addEventListener("click", fetchHistorical);
  $("fetch-fixture-btn").addEventListener("click", fetchFixture);
  $("train-btn").addEventListener("click", startTraining);
  $("tune-btn").addEventListener("click", startTuning);
});
