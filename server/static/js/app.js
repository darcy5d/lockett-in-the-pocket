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
// Score rounding: cascade fractional goals into behinds
// ---------------------------------------------------------------------------

function cascadeScore(rawGoals, rawBehinds) {
  const wholeGoals = Math.floor(rawGoals);
  const fracGoals = rawGoals - wholeGoals;
  const extraBehinds = fracGoals * 6;
  const totalBehinds = Math.round(rawBehinds + extraBehinds);
  const score = wholeGoals * 6 + totalBehinds;
  return [wholeGoals, totalBehinds, score];
}

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
let _lineupJobId = null;
let _lineupPollTimer = null;

async function initFixtureTab() {
  _fixtureTabInitialised = true;
  await Promise.all([loadRounds(), loadFixtureFreshness(), loadLineupFreshness()]);
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
    const lineupHtml = buildLineupStatusHtml(m);
    
    // Build score display if match is completed
    let matchContentHtml = '';
    if (m.completed && m.team_1_final_goals !== null && m.team_2_final_goals !== null) {
      // Match is completed - show scores
      const team1Goals = parseInt(m.team_1_final_goals) || 0;
      const team1Behinds = parseInt(m.team_1_final_behinds) || 0;
      const team2Goals = parseInt(m.team_2_final_goals) || 0;
      const team2Behinds = parseInt(m.team_2_final_behinds) || 0;
      
      const team1Total = team1Goals * 6 + team1Behinds;
      const team2Total = team2Goals * 6 + team2Behinds;
      
      // Determine winner for styling
      const team1Won = team1Total > team2Total;
      const team2Won = team2Total > team1Total;
      const isDraw = team1Total === team2Total;
      
      const team1Style = team1Won ? 'text-success fw-bold' : team2Won ? 'text-muted' : 'text-warning';
      const team2Style = team2Won ? 'text-success fw-bold' : team1Won ? 'text-muted' : 'text-warning';
      
      matchContentHtml = `
        <div class="d-flex align-items-center justify-content-between gap-2 mb-2">
          <div class="text-end" style="flex:1">
            <div class="${team1Style}">${m.home_team_display}</div>
            <div class="score-display ${team1Style}">${team1Goals}.${team1Behinds} (${team1Total})</div>
          </div>
          <div class="text-center">
            <span class="result-badge ${isDraw ? 'draw' : 'completed'}">${isDraw ? 'DRAW' : 'FINAL'}</span>
          </div>
          <div class="text-start" style="flex:1">
            <div class="${team2Style}">${m.away_team_display}</div>
            <div class="score-display ${team2Style}">${team2Goals}.${team2Behinds} (${team2Total})</div>
          </div>
        </div>`;
    } else {
      // Match is upcoming - show vs format
      matchContentHtml = `
        <div class="d-flex align-items-center justify-content-between gap-3">
          <span class="team-name text-warning text-end" style="flex:1">${m.home_team_display}</span>
          <span class="vs-badge">VS</span>
          <span class="team-name text-light" style="flex:1">${m.away_team_display}</span>
        </div>`;
    }
    
    const col = document.createElement("div");
    col.className = "col-sm-6 col-xl-4";
    col.innerHTML = `
      <div class="match-card ${m.completed ? 'completed' : ''}">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <span class="meta">${dateLabel}${tbcHtml}</span>
          <span class="meta">${m.venue_display || m.venue}</span>
        </div>
        ${matchContentHtml}
        ${lineupHtml}
      </div>`;
    container.appendChild(col);
  });
}

function buildLineupStatusHtml(m) {
  const hFresh = m.home_lineup_fresh;
  const aFresh = m.away_lineup_fresh;
  const hCount = m.home_lineup_count || 0;
  const aCount = m.away_lineup_count || 0;
  const hKnown = m.home_lineup_known || 0;
  const aKnown = m.away_lineup_known || 0;
  const hUnknown = hCount - hKnown;
  const aUnknown = aCount - aKnown;
  const bothFresh = hFresh && aFresh;
  const eitherFresh = hFresh || aFresh;

  if (bothFresh) {
    const warnHtml = (hUnknown + aUnknown) > 0
      ? `<br><span class="lineup-warn">${playerMatchDetail(m, true)}</span>`
      : "";
    return `<div class="lineup-status lineup-fresh mt-2">
      <span class="lineup-dot fresh"></span>
      <span>2026 lineups loaded &middot; ${hKnown + aKnown}/${hCount + aCount} players matched to model${warnHtml}</span>
    </div>`;
  }
  if (eitherFresh) {
    const freshSide = hFresh ? "home" : "away";
    const freshTeam = hFresh ? m.home_team_display : m.away_team_display;
    const staleTeam = hFresh ? m.away_team_display : m.home_team_display;
    const fCount = hFresh ? hCount : aCount;
    const fKnown = hFresh ? hKnown : aKnown;
    return `<div class="lineup-status lineup-partial mt-2">
      <span class="lineup-dot partial"></span>
      <span>${freshTeam}: 2026 lineup (${fKnown}/${fCount} matched) &middot; ${staleTeam}: last known</span>
    </div>`;
  }
  if (hCount > 0 && aCount > 0) {
    return `<div class="lineup-status lineup-stale mt-2">
      <span class="lineup-dot stale"></span>
      Using last known lineups (${hKnown}/${hCount} + ${aKnown}/${aCount} matched)
    </div>`;
  }
  return `<div class="lineup-status lineup-none mt-2">
    <span class="lineup-dot none"></span>
    No lineup data available
  </div>`;
}

function playerMatchDetail(m, bothFresh) {
  const parts = [];
  const hUn = (m.home_lineup_count || 0) - (m.home_lineup_known || 0);
  const aUn = (m.away_lineup_count || 0) - (m.away_lineup_known || 0);
  if (hUn > 0) parts.push(`${m.home_team_display}: ${hUn} unrecognised`);
  if (aUn > 0) parts.push(`${m.away_team_display}: ${aUn} unrecognised`);
  return parts.join(" &middot; ");
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
      // Cascade rounding: floor goals, convert fractional goals to behinds, then round behinds once
      const [g1, b1, s1] = cascadeScore(t1.goals, t1.behinds);
      const [g2, b2, s2] = cascadeScore(t2.goals, t2.behinds);
      const displayMargin = Math.abs(s1 - s2);
      const homeWins = pred.margin > 0;
      const isDraw = s1 === s2;
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
          <span class="score-display ${homeWins ? "text-warning" : "text-light"}">${g1}.${b1} (${s1})</span>
          <span class="meta mx-1">–</span>
          <span class="score-display ${!homeWins ? "text-warning" : "text-light"}">${g2}.${b2} (${s2})</span>
        </td>
        <td class="text-center"><span class="margin-display">${isDraw ? "Draw" : `${leader} by ${displayMargin}`}</span></td>`;
    }
    tbody.appendChild(tr);
  });

  section.style.display = "block";
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Lineup freshness ──────────────────────────────────────────────────────────

async function loadLineupFreshness() {
  try {
    const meta = await fetchJSON("/api/afl/lineup/meta");
    const el = $("lineup-freshness");
    if (meta.last_fetched) {
      const d = new Date(meta.last_fetched);
      const rnd = meta.round_num || "";
      el.textContent = `Lineups: ${rnd} (${d.toLocaleDateString("en-AU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })})`;
    } else {
      el.textContent = "";
    }
  } catch (e) {
    // no meta yet — not an error
  }
}

// ── Refresh lineups from AFL.com.au ───────────────────────────────────────────

async function refreshLineups() {
  const btn = $("refresh-lineups-btn");
  const statusSection = $("lineup-status-section");
  const logBox = $("lineup-log-box");
  const statusLabel = $("lineup-status-label");
  const progress = $("lineup-progress");

  btn.disabled = true;
  btn.textContent = "Scraping…";
  statusSection.style.display = "";
  logBox.innerHTML = "";
  logBox.style.display = "block";
  statusLabel.textContent = "Starting lineup scrape…";
  progress.className = "progress-bar bg-warning progress-bar-striped progress-bar-animated";
  progress.style.width = "100%";

  const roundNum = $("round-select").value || null;
  const body = roundNum ? { round_num: roundNum } : {};

  try {
    const { ok, data } = await postJSON("/api/afl/lineup/refresh", body);
    if (!ok || data.error) {
      statusLabel.textContent = `Error: ${data.error || "Failed to start"}`;
      progress.className = "progress-bar bg-danger";
      btn.disabled = false;
      btn.textContent = "Refresh Lineups";
      return;
    }
    _lineupJobId = data.job_id;
    statusLabel.textContent = "Scraping lineups from AFL.com.au…";
    pollLineupStatus();
  } catch (e) {
    statusLabel.textContent = `Error: ${e.message}`;
    progress.className = "progress-bar bg-danger";
    btn.disabled = false;
    btn.textContent = "Refresh Lineups";
  }
}

async function pollLineupStatus() {
  if (!_lineupJobId) return;
  try {
    const d = await fetchJSON(`/api/afl/lineup/refresh/status?job_id=${_lineupJobId}`);
    const logBox = $("lineup-log-box");

    if (d.new_lines && d.new_lines.length) {
      d.new_lines.forEach((line) => {
        const div = document.createElement("div");
        div.className = "log-line";
        div.textContent = line;
        logBox.appendChild(div);
      });
      logBox.scrollTop = logBox.scrollHeight;
    }

    if (d.running) {
      _lineupPollTimer = setTimeout(pollLineupStatus, 2000);
    } else {
      const success = d.exit_code === 0;
      $("lineup-status-label").textContent = success ? "Lineups updated" : "Scrape failed";
      $("lineup-progress").classList.remove("progress-bar-animated");
      $("lineup-progress").className = success ? "progress-bar bg-success" : "progress-bar bg-danger";
      _lineupJobId = null;

      const btn = $("refresh-lineups-btn");
      btn.disabled = false;
      btn.textContent = "Refresh Lineups";

      if (success) {
        await loadLineupFreshness();
        const round = $("round-select").value;
        if (round) await loadRound(round);
      }

      setTimeout(() => {
        $("lineup-status-section").style.display = "none";
      }, 5000);
    }
  } catch (e) {
    _lineupPollTimer = setTimeout(pollLineupStatus, 4000);
  }
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

// ── Update match scores from AFL Tables ──────────────────────────────────────
async function updateScores() {
  const btn = $("update-scores-btn");
  btn.disabled = true;
  btn.textContent = "Updating…";
  try {
    const { ok, data } = await postJSON("/api/fixture/update-scores", {});
    if (ok && data.ok) {
      const round = $("round-select").value;
      if (round) await loadRound(round);
      btn.textContent = "Done!";
      alert(data.message);
    } else {
      alert(`Score update failed: ${data.error || "unknown"}`);
      btn.textContent = "Update Scores";
    }
  } catch (e) {
    alert(`Error: ${e.message}`);
    btn.textContent = "Update Scores";
  } finally {
    setTimeout(() => { btn.disabled = false; btn.textContent = "Update Scores"; }, 2000);
  }
}


// ===========================================================================
// TAB 2 — Data & Training
// ===========================================================================

let _trainingTabInitialised = false;
let _trainingJobId = null;
let _tuneJobId = null;
let _trainingPollTimer = null;
let _tunePollTimer = null;
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
    const acc = parseFloat(m.winner_accuracy_from_margin ?? m.match_winner_accuracy ?? 0);

    el.innerHTML = `
      <div class="metrics-grid">
        ${metricRow("Winner accuracy", `${(acc*100).toFixed(1)}%`, acc < 0.55 ? "bad" : acc < 0.65 ? "ok" : "good")}
        ${metricRow("RMSE margin", rmseMargin.toFixed(1) + " pts", rmseMargin > 50 ? "bad" : rmseMargin > 40 ? "ok" : "good")}
        ${metricRow("RMSE team1 goals", parseFloat(m.rmse_team1_goals ?? 0).toFixed(2))}
        ${metricRow("RMSE team2 goals", parseFloat(m.rmse_team2_goals ?? 0).toFixed(2))}
        ${metricRow("Goals MAE", parseFloat(m.team1_goals_mae ?? 0).toFixed(2))}
        ${metricRow("Trained", m.trained_date ?? "Unknown")}
      </div>`;
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

function setJobRunning(running) {
  $("train-btn").disabled = running;
  $("tune-btn").disabled = running;
  $("update-model-btn").disabled = running;
}

async function startTraining() {
  const logBox = $("train-log-box");
  const statusBar = $("train-status-bar");

  setJobRunning(true);
  logBox.innerHTML = "";
  logBox.style.display = "block";
  statusBar.style.display = "block";
  $("train-status-label").textContent = "Starting…";
  $("train-elapsed").textContent = "";
  $("train-progress").className = "progress-bar bg-warning progress-bar-striped progress-bar-animated";
  $("train-progress").style.width = "100%";
  _trainingStart = Date.now();

  const yearFrom = parseInt($("train-year-from").value);
  const yearTo = parseInt($("train-year-to").value);
  const epochs = parseInt($("train-epochs").value);

  try {
    const { ok, data } = await postJSON("/api/train", { year_from: yearFrom, year_to: yearTo, epochs });
    if (!ok || data.error) {
      appendLog(`ERROR: ${data.error || "Failed to start training"}`, "error");
      setJobRunning(false);
      return;
    }
    _trainingJobId = data.job_id;
    appendLog(`Training started (job ${_trainingJobId})`, "info");
    $("train-status-label").textContent = "Training…";
    pollTrainingLog();
  } catch (e) {
    appendLog(`Error: ${e.message}`, "error");
    setJobRunning(false);
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
      setJobRunning(false);
      _trainingJobId = null;
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

async function startTuning() {
  const logBox = $("train-log-box");
  const statusBar = $("train-status-bar");

  setJobRunning(true);
  logBox.innerHTML = "";
  logBox.style.display = "block";
  statusBar.style.display = "block";
  $("train-status-label").textContent = "Starting…";
  $("train-elapsed").textContent = "";
  $("train-progress").className = "progress-bar bg-warning progress-bar-striped progress-bar-animated";
  $("train-progress").style.width = "100%";
  _trainingStart = Date.now();

  const yearFrom = parseInt($("train-year-from").value);
  const yearTo = parseInt($("train-year-to").value);
  const maxEpochs = parseInt($("train-epochs").value);

  try {
    const { ok, data } = await postJSON("/api/tune", {
      year_from: yearFrom,
      year_to: yearTo,
      max_epochs: maxEpochs,
    });
    if (!ok || data.error) {
      appendLog(`ERROR: ${data.error || "Failed to start tuning"}`, "error");
      setJobRunning(false);
      return;
    }
    _tuneJobId = data.job_id;
    appendLog(`Tuning started (job ${_tuneJobId}) – 100 epochs, 2 iterations (1–2 hrs)`, "info");
    $("train-status-label").textContent = "Tuning…";
    pollTuningLog();
  } catch (e) {
    appendLog(`Error: ${e.message}`, "error");
    setJobRunning(false);
  }
}

async function updateModel() {
  const btn = $("update-model-btn");
  const originalContent = btn.innerHTML;
  
  btn.disabled = true;
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
  
  try {
    const response = await fetch('/api/model/retrain', { method: 'POST' });
    const result = await response.json();
    
    if (result.ok) {
      // Show success status
      btn.innerHTML = '<i class="fas fa-check text-success"></i> Model Updated';
      setTimeout(() => {
        btn.innerHTML = originalContent;
        btn.disabled = false;
      }, 3000);
      
      // Refresh model metrics to show updated data
      await loadModelMetrics();
    } else {
      // Show error status
      btn.innerHTML = '<i class="fas fa-exclamation-triangle text-danger"></i> Error';
      setTimeout(() => {
        btn.innerHTML = originalContent;
        btn.disabled = false;
      }, 3000);
      
      // Show error message if available
      if (result.error) {
        console.error('Model update error:', result.error);
        // Could add a toast notification here
      }
    }
  } catch (e) {
    console.error('Model update failed:', e);
    btn.innerHTML = '<i class="fas fa-exclamation-triangle text-danger"></i> Failed';
    setTimeout(() => {
      btn.innerHTML = originalContent;
      btn.disabled = false;
    }, 3000);
  }
}

async function pollTuningLog() {
  if (!_tuneJobId) return;
  try {
    const d = await fetchJSON(`/api/tune/status?job_id=${_tuneJobId}`);
    const elapsed = Math.round((Date.now() - _trainingStart) / 1000);
    $("train-elapsed").textContent = `${elapsed}s`;

    if (d.new_lines && d.new_lines.length) {
      d.new_lines.forEach((line) => appendLog(line));
    }

    if (d.running) {
      $("train-status-label").textContent = "Tuning…";
      _tunePollTimer = setTimeout(pollTuningLog, 2000);
    } else {
      $("train-status-label").textContent = d.exit_code === 0 ? "Completed" : "Failed";
      $("train-progress").classList.remove("progress-bar-animated");
      $("train-progress").style.width = "100%";
      $("train-progress").className = d.exit_code === 0
        ? "progress-bar bg-success"
        : "progress-bar bg-danger";
      setJobRunning(false);
      _tuneJobId = null;
      await loadModelMetrics();
    }
  } catch (e) {
    appendLog(`Poll error: ${e.message}`, "error");
    _tunePollTimer = setTimeout(pollTuningLog, 4000);
  }
}


// ===========================================================================
// Boot
// ===========================================================================

function initTrainingForm() {
  const currentYear = new Date().getFullYear();
  
  // Update training form to reflect current year and available data
  const yearToInput = $("train-year-to");
  const yearFromInput = $("train-year-from");
  
  if (yearToInput) {
    // Set max attribute to current year
    yearToInput.max = currentYear;
    
    // Update default value to current year if it's still the old default
    if (yearToInput.value === "2025" || parseInt(yearToInput.value) < currentYear) {
      yearToInput.value = currentYear;
    }
  }
  
  if (yearFromInput) {
    // Update max attribute for year-from as well
    yearFromInput.max = currentYear;
  }
  
  console.log(`Training form initialized for year range: 1990-${currentYear}`);
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();

  // Fixture tab is default — hide training tab, init fixture
  $("tab-training").style.display = "none";
  initFixtureTab();
  
  // Initialize dynamic training form values
  initTrainingForm();

  // Fixture tab wiring
  $("round-select").addEventListener("change", (e) => { if (e.target.value) loadRound(e.target.value); });
  $("predict-round-btn").addEventListener("click", predictRound);
  $("refresh-fixture-btn").addEventListener("click", refreshFixture);
  $("update-scores-btn").addEventListener("click", updateScores);
  $("refresh-lineups-btn").addEventListener("click", refreshLineups);

  // Training tab wiring
  $("refresh-data-btn").addEventListener("click", loadDataStatus);
  $("fetch-historical-btn").addEventListener("click", fetchHistorical);
  $("fetch-fixture-btn").addEventListener("click", fetchFixtureFromTraining);
  $("train-btn").addEventListener("click", startTraining);
  $("tune-btn").addEventListener("click", startTuning);
  $("update-model-btn").addEventListener("click", updateModel);
});
