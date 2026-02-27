/**
 * Rugby League Match Predictor — multi-competition.
 * Uses /api/rugby/<competition_id>/ endpoints.
 */

"use strict";

const $ = (id) => document.getElementById(id);

function getCompetitionId() {
  const sel = document.getElementById("competition-select");
  return sel ? sel.value || "nrl" : "nrl";
}

function apiBase() {
  return `/api/rugby/${getCompetitionId()}`;
}

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

function updateFixtureTitle() {
  const compId = getCompetitionId();
  const comps = window.RUGBY_COMPETITIONS || {};
  const display = comps[compId]?.display || compId;
  const title = $("fixture-title");
  if (title) title.textContent = `2026 ${display} Fixture`;
}

let _fixtureInit = false, _trainingInit = false, _currentRoundMatches = [];

async function initFixtureTab() {
  _fixtureInit = true;
  updateFixtureTitle();
  await Promise.all([loadRounds(), loadFixtureFreshness()]);
}

async function loadFixtureFreshness() {
  try {
    const meta = await fetchJSON(`${apiBase()}/fixture/meta`);
    $("fixture-freshness").textContent = meta.last_fetched
      ? `Updated ${new Date(meta.last_fetched).toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" })}`
      : "Never refreshed";
  } catch (e) { $("fixture-freshness").textContent = "Error"; }
}

async function loadRounds() {
  try {
    const rounds = await fetchJSON(`${apiBase()}/fixture/rounds`);
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
    const matches = await fetchJSON(`${apiBase()}/fixture/${encodeURIComponent(roundNum)}`);
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
    const { ok, data } = await postJSON(`${apiBase()}/predict/round`, { matches: _currentRoundMatches });
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
      const displayMargin = Math.abs(s1 - s2);
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
        <td class="text-center"><span class="margin-display">${displayMargin === 0 ? "Draw" : `${displayMargin} pts`}</span></td>`;
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
    const { ok, data } = await postJSON(`${apiBase()}/fixture/refresh`, {});
    if (ok && data.ok) {
      await loadFixtureFreshness();
      const round = $("round-select").value;
      if (round) await loadRound(round);
    } else alert(`Failed: ${data.error || "unknown"}`);
  } catch (e) { alert(`Error: ${e.message}`); }
  finally { btn.disabled = false; btn.textContent = "Fetch 2026 Fixture"; }
}

async function fetchLineups() {
  const btn = $("fetch-lineups-btn");
  if (!btn) return;
  btn.disabled = true;
  btn.textContent = "Fetching…";
  try {
    const round = $("round-select")?.value || null;
    const body = round ? { round_num: round } : {};
    const { ok, data } = await postJSON(`${apiBase()}/lineup/refresh`, body);
    if (ok && data.ok) {
      alert(`Done: ${data.message}`);
      const r = $("round-select")?.value;
      if (r) await loadRound(r);
    } else alert(`Failed: ${data.error || "unknown"}`);
  } catch (e) { alert(`Error: ${e.message}`); }
  finally { btn.disabled = false; btn.textContent = "Fetch Round Lineups"; }
}

function resetBuildCacheUI(btn, statusEl) {
  if (btn) { btn.disabled = false; btn.textContent = "Build Player Cache"; }
  if (statusEl) statusEl.style.display = "none";
}

async function buildPlayerCache() {
  const btn = $("build-cache-btn");
  const statusEl = $("build-cache-status");
  const statusText = $("build-cache-status-text");
  if (!btn) return;
  const rebuild = !!($("build-cache-rebuild")?.checked);
  btn.disabled = true;
  btn.textContent = "Building…";
  if (statusEl && statusText) {
    statusEl.style.display = "block";
    statusText.textContent = rebuild
      ? "Full rebuild… (re-fetching all players, may take longer)"
      : "Starting… (may take 2–5 min)";
  }
  try {
    const { ok, data } = await postJSON(`${apiBase()}/lineup/build-cache`, { rebuild });
    const jobId = data?.job_id;
    if (!ok || !data?.ok || !jobId) {
      resetBuildCacheUI(btn, statusEl);
      alert(`Failed: ${data?.error || "Could not start build"}`);
      return;
    }
    const poll = async () => {
      try {
        const s = await fetchJSON(`${apiBase()}/lineup/build-cache/status?job_id=${encodeURIComponent(jobId)}`);
        if (statusText) {
          if (s.running) {
            const { current, total } = s;
            statusText.textContent = total > 0
              ? `Fetching ${current}/${total} players from RLP…`
              : "Fetching DOB from RLP…";
          } else if (s.result) {
            statusText.textContent = s.result.message || "Done.";
          } else if (s.error) {
            statusText.textContent = `Error: ${s.error}`;
          }
        }
        if (!s.running) {
          btn.disabled = false;
          btn.textContent = "Build Player Cache";
          if (s.result) {
            if (statusText) statusText.textContent = s.result.message;
            if (statusEl) statusEl.style.display = "block";
            if (statusEl) statusEl.querySelector(".spinner-border")?.remove();
            setTimeout(() => { if (statusEl) statusEl.style.display = "none"; }, 5000);
          } else if (s.error) {
            if (statusText) statusText.textContent = `Error: ${s.error}`;
            if (statusEl) statusEl.style.display = "block";
            setTimeout(() => resetBuildCacheUI(btn, statusEl), 5000);
          } else {
            resetBuildCacheUI(btn, statusEl);
          }
          return;
        }
      } catch (e) {
        resetBuildCacheUI(btn, statusEl);
        const msg = e.message && e.message.includes("404")
          ? "Build status not found. Server may have restarted—try again."
          : e.message;
        alert(`Error: ${msg}`);
        return;
      }
      setTimeout(poll, 500);
    };
    poll();
  } catch (e) {
    resetBuildCacheUI(btn, statusEl);
    alert(`Error: ${e.message}`);
  }
}

async function initTrainingTab() {
  _trainingInit = true;
  await Promise.all([loadDataStatus(), loadModelMetrics()]);
}

async function loadDataStatus() {
  try {
    const d = await fetchJSON(`${apiBase()}/data/status`);
    $("stat-match-files").textContent = d.match_files ?? "—";
    $("stat-match-year-range").textContent = d.match_year_range ? d.match_year_range : "";
    $("stat-lineup-files").textContent = d.lineup_files ?? "—";
    $("stat-fixture-matches").textContent = d.fixture_matches ?? "—";
    $("stat-fixture-updated").textContent = d.fixture_last_updated
      ? `Updated ${new Date(d.fixture_last_updated).toLocaleDateString("en-AU", { day: "numeric", month: "short" })}`
      : "Not fetched";
  } catch (e) { console.error(e); }
}

function metricRow(label, value) {
  return `<div class="metric-row"><span class="metric-label">${label}</span><span class="metric-value">${value || "—"}</span></div>`;
}

async function loadModelMetrics() {
  try {
    const m = await fetchJSON(`${apiBase()}/data/model-metrics`);
    const el = $("model-metrics");
    if (m.error) { el.innerHTML = `<div class="metric-fallback">${m.error}</div>`; return; }
    el.innerHTML = `
      <div class="metrics-grid">
        ${metricRow("MAE team1_score", m["MAE team1_score"])}
        ${metricRow("MAE team2_score", m["MAE team2_score"])}
        ${metricRow("MAE margin", m["MAE margin"])}
        ${metricRow("RMSE team1_score", m["RMSE team1_score"])}
        ${metricRow("RMSE team2_score", m["RMSE team2_score"])}
        ${metricRow("RMSE margin", m["RMSE margin"])}
        ${metricRow("Winner accuracy", m["Winner accuracy"])}
        ${metricRow("Margin bias", m["Margin bias"])}
      </div>`;
  } catch (e) { $("model-metrics").innerHTML = `<div class="metric-fallback">Could not load metrics.</div>`; }
}

async function fetchFixture() {
  const status = $("fetch-status");
  const btn = $("fetch-fixture-btn");
  btn.disabled = true;
  status.textContent = "Fetching 2026 fixture…";
  try {
    const { ok, data } = await postJSON(`${apiBase()}/fixture/refresh`, {});
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
    const { ok, data } = await postJSON(`${apiBase()}/data/fetch-historical`, {});
    if (!ok || data.error) {
      status.textContent = `Failed: ${data.error}`;
      appendFetchLog(`ERROR: ${data.error}`, "error");
      setFetchRunning(false);
      return;
    }
    _fetchJobId = data.job_id;
    status.textContent = "Scraping RLP…";
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
    const d = await fetchJSON(`${apiBase()}/data/fetch-historical/status?job_id=${_fetchJobId}`);
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
    const { ok, data } = await postJSON(`${apiBase()}/train`, { year_from: yearFrom, year_to: yearTo, epochs });
    if (!ok || data.error) { appendLog(`ERROR: ${data.error || "Failed"}`, "error"); setJobRunning(false); return; }
    _trainingJobId = data.job_id;
    appendLog(`Training started (job ${_trainingJobId})`, "info");
    pollTrainingLog();
  } catch (e) { appendLog(`Error: ${e.message}`, "error"); setJobRunning(false); }
}

async function pollTrainingLog() {
  if (!_trainingJobId) return;
  try {
    const d = await fetchJSON(`${apiBase()}/train/status?job_id=${_trainingJobId}`);
    $("train-elapsed").textContent = `${Math.round((Date.now() - _trainingStart) / 1000)}s`;
    if (d.new_lines) d.new_lines.forEach((l) => appendLog(l));
    if (d.running) { setTimeout(pollTrainingLog, 2000); return; }
    $("train-status-label").textContent = d.exit_code === 0 ? "Completed" : "Failed";
    appendLog(d.exit_code === 0 ? "Training finished." : "Training failed.", d.exit_code === 0 ? "info" : "error");
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
    const { ok, data } = await postJSON(`${apiBase()}/tune`, { year_from: yearFrom, year_to: yearTo, max_epochs: maxEpochs });
    if (!ok || data.error) { appendLog(`ERROR: ${data.error || "Failed"}`, "error"); setJobRunning(false); return; }
    _tuneJobId = data.job_id;
    appendLog(`Tuning started (job ${_tuneJobId})`, "info");
    pollTuningLog();
  } catch (e) { appendLog(`Error: ${e.message}`, "error"); setJobRunning(false); }
}

async function pollTuningLog() {
  if (!_tuneJobId) return;
  try {
    const d = await fetchJSON(`${apiBase()}/tune/status?job_id=${_tuneJobId}`);
    $("train-elapsed").textContent = `${Math.round((Date.now() - _trainingStart) / 1000)}s`;
    if (d.new_lines) d.new_lines.forEach((l) => appendLog(l));
    if (d.running) { setTimeout(pollTuningLog, 2000); return; }
    $("train-status-label").textContent = d.exit_code === 0 ? "Completed" : "Failed";
    appendLog(d.exit_code === 0 ? "Hyperparameter tuning finished." : "Tuning failed.", d.exit_code === 0 ? "info" : "error");
    setJobRunning(false);
    _tuneJobId = null;
    await loadModelMetrics();
  } catch (e) { setTimeout(pollTuningLog, 4000); }
}

function onCompetitionChange() {
  _fixtureInit = false;
  _trainingInit = false;
  updateFixtureTitle();
  initFixtureTab();
  // Refresh Data & Training tab so it shows competition-specific data/metrics
  loadDataStatus();
  loadModelMetrics();
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  $("tab-training").style.display = "none";
  initFixtureTab();
  $("competition-select")?.addEventListener("change", onCompetitionChange);
  $("round-select").addEventListener("change", (e) => { if (e.target.value) loadRound(e.target.value); });
  $("predict-round-btn").addEventListener("click", predictRound);
  $("refresh-fixture-btn").addEventListener("click", refreshFixture);
  const buildCacheBtn = $("build-cache-btn");
  if (buildCacheBtn) buildCacheBtn.addEventListener("click", buildPlayerCache);
  $("fetch-lineups-btn")?.addEventListener("click", fetchLineups);
  $("refresh-data-btn").addEventListener("click", loadDataStatus);
  $("fetch-historical-btn").addEventListener("click", fetchHistorical);
  $("fetch-fixture-btn").addEventListener("click", fetchFixture);
  $("train-btn").addEventListener("click", startTraining);
  $("tune-btn").addEventListener("click", startTuning);
});
