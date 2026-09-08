/**
 * Turbofan Engine Health & RUL Monitoring Dashboard
 * Real-Time Industrial Telemetry & PyTorch Inference Client
 * Primary Workflow: Fleet Overview (Ranked Risk) -> Engine Detail View
 */

// ── Sensor Definitions & Physical Engineering Units ──────────────────────────
const SENSOR_META = {
  s1:  { name: "Fan Inlet Temperature", unit: "°R", desc: "Total temperature at fan inlet" },
  s2:  { name: "LPC Outlet Temperature", unit: "°R", desc: "Total temperature at LPC outlet" },
  s3:  { name: "HPC Outlet Temperature", unit: "°R", desc: "Total temperature at HPC outlet" },
  s4:  { name: "LPT Outlet Temperature", unit: "°R", desc: "Total temperature at LPT outlet" },
  s5:  { name: "Fan Inlet Pressure", unit: "psia", desc: "Pressure at fan inlet" },
  s6:  { name: "Bypass Duct Pressure", unit: "psia", desc: "Total pressure in bypass-duct" },
  s7:  { name: "HPC Outlet Pressure", unit: "psia", desc: "Total pressure at HPC outlet" },
  s8:  { name: "Physical Fan Speed", unit: "rpm", desc: "Physical fan rotational speed" },
  s9:  { name: "Physical Core Speed", unit: "rpm", desc: "Physical core rotational speed" },
  s10: { name: "Engine Pressure Ratio", unit: "--", desc: "EPR (P50/P2)" },
  s11: { name: "HPC Static Pressure", unit: "psia", desc: "Static pressure at HPC outlet" },
  s12: { name: "Fuel Flow Ratio", unit: "pps/psia", desc: "Ratio of fuel flow to Ps30" },
  s13: { name: "Corrected Fan Speed", unit: "rpm", desc: "Corrected fan speed" },
  s14: { name: "Corrected Core Speed", unit: "rpm", desc: "Corrected core speed" },
  s15: { name: "Bypass Ratio", unit: "--", desc: "Bypass ratio" },
  s16: { name: "Burner Fuel-Air Ratio", unit: "--", desc: "Burner fuel-air ratio" },
  s17: { name: "Bleed Enthalpy", unit: "--", desc: "Bleed enthalpy" },
  s18: { name: "Demanded Fan Speed", unit: "rpm", desc: "Demanded fan speed" },
  s19: { name: "Demanded Corr. Fan Speed", unit: "rpm", desc: "Demanded corrected fan speed" },
  s20: { name: "HPT Coolant Bleed", unit: "lbm/s", desc: "HPT coolant bleed" },
  s21: { name: "LPT Coolant Bleed", unit: "lbm/s", desc: "LPT coolant bleed" }
};

// ── Display Formatting Helpers (Whole Natural Number Cycles) ────────────────
function formatRUL(value) {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return "N/A";
  }
  return Math.round(Number(value));
}

function formatUncertainty(value) {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return "--";
  }
  return `±${Math.round(Number(value))} cycles`;
}

function formatCI(ci) {
  if (!ci || ci.lower === undefined || ci.upper === undefined || isNaN(Number(ci.lower))) return "--";
  return `[${Math.round(Number(ci.lower))} – ${Math.round(Number(ci.upper))}]`;
}

function formatGroundTruth(value) {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return "--";
  }
  return Math.round(Number(value));
}

// ── Application State ────────────────────────────────────────────────────────
const state = {
  // Navigation & Active Tab
  activeTab: "tab-fleet",

  // Fleet Overview State
  fleetDataset: "FD001",
  rawFleetData: [],
  filteredFleetData: [],
  fleetSearchQuery: "",
  healthFilter: "all",
  priorityFilter: "all",
  alertFilter: "all",
  sortBy: "critical_first",
  currentPage: 1,
  pageSize: 25,
  autoRefreshTimer: null,

  // Uploaded Ingested Fleet State
  uploadState: {
    rawUploadData: [],
    filteredUploadData: [],
    searchQuery: "",
    healthFilter: "all",
    priorityFilter: "all",
    alertFilter: "all",
    sortBy: "critical_first",
    currentPage: 1,
    pageSize: 25,
    datasetUsed: "FD001"
  },

  // Live Engine Detail State
  dataset: "FD001",
  engineId: 81,
  currentCycle: 213,
  maxCycle: 213,
  minCycle: 1,
  windowSize: 30,
  selectedSensors: [],
  telemetry: [],
  telemetryByCycle: {},
  predictionHistory: null,
  activeSensor: "s2",
  sensorFilter: "all",
  isPlaying: false,
  simTimer: null,
  simSpeed: 500,
  predictionCache: {},
  currentEngineRank: 1
};

// Chart instances
let rulTrendChart = null;
let sensorTrendChart = null;

// ── Application Initialization ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupEventListeners();
  initCharts();
  await loadDatasets();
  loadBenchmarks();

  // Automatically default and run FD001 dataset
  state.fleetDataset = "FD001";
  state.dataset = "FD001";

  const fleetDsSelect = document.getElementById("fleet-dataset-select");
  if (fleetDsSelect) fleetDsSelect.value = "FD001";

  const dsSelect = document.getElementById("dataset-select");
  if (dsSelect) dsSelect.value = "FD001";

  // Load Primary View: Fleet Overview with FD001
  await loadFleetOverview();

  // Pre-load FD001 for Live Engine Monitor
  await switchDataset("FD001");
});

// ── Tab Navigation ───────────────────────────────────────────────────────────
function setupTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      switchTab(targetId);
    });
  });

  // Back to Fleet Button in Engine Detail View
  const backBtn = document.getElementById("btn-back-to-fleet");
  if (backBtn) {
    backBtn.addEventListener("click", () => {
      switchTab("tab-fleet");
      // Re-render fleet in case simulated cycles progressed
      renderFleetTable();
    });
  }
}

function switchTab(targetId) {
  state.activeTab = targetId;

  // Update nav buttons
  document.querySelectorAll(".nav-tab").forEach(t => {
    t.classList.toggle("active", t.getAttribute("data-tab") === targetId);
  });

  // Update content sections
  document.querySelectorAll(".tab-content").forEach(c => {
    c.classList.toggle("active", c.id === targetId);
  });

  // Update model status text in header
  const statusEl = document.getElementById("model-status-text");
  if (statusEl) {
    if (targetId === "tab-fleet") {
      statusEl.innerText = `Fleet View: ${state.fleetDataset === 'all' ? 'All Datasets (707 Turbofans)' : state.fleetDataset}`;
    } else if (targetId === "tab-monitor") {
      statusEl.innerText = `Live Engine: ${state.dataset} Unit ${state.engineId}`;
    } else if (targetId === "tab-upload") {
      statusEl.innerText = "Custom Telemetry Ingestion";
    } else if (targetId === "tab-benchmarks") {
      statusEl.innerText = "Offline Model Verification";
    }
  }

  // Scroll to top
  window.scrollTo({ top: 0, behavior: "smooth" });

  // Handle Tab-specific chart resizes and re-renders
  if (targetId === "tab-monitor") {
    setTimeout(async () => {
      if (rulTrendChart) {
        rulTrendChart.resize();
      }
      if (state.predictionHistory) {
        renderRULTrendChart(state.predictionHistory);
      } else {
        await loadRULTrend();
      }
    }, 60);
  }
}

// ── Event Listeners ──────────────────────────────────────────────────────────
function setupEventListeners() {
  // Fleet Dataset Filter
  const fleetDsSelect = document.getElementById("fleet-dataset-select");
  if (fleetDsSelect) {
    fleetDsSelect.addEventListener("change", async (e) => {
      state.fleetDataset = e.target.value;
      state.currentPage = 1;
      await loadFleetOverview();
    });
  }

  // Refresh Fleet Button
  const refreshBtn = document.getElementById("btn-refresh-fleet");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", async () => {
      await loadFleetOverview(true);
    });
  }

  // Auto Refresh Toggle
  const autoRefreshToggle = document.getElementById("fleet-auto-refresh");
  const autoRefreshInterval = document.getElementById("fleet-refresh-interval");
  if (autoRefreshToggle) {
    autoRefreshToggle.addEventListener("change", (e) => {
      if (e.target.checked) {
        const ms = parseInt(autoRefreshInterval.value, 10) || 30000;
        state.autoRefreshTimer = setInterval(async () => {
          if (state.activeTab === "tab-fleet") {
            await loadFleetOverview(false);
          }
        }, ms);
      } else {
        clearInterval(state.autoRefreshTimer);
        state.autoRefreshTimer = null;
      }
    });
  }

  // Fleet Search Input
  const searchInput = document.getElementById("fleet-search-input");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      state.fleetSearchQuery = e.target.value.trim().toLowerCase();
      state.currentPage = 1;
      applyFleetFiltersAndRender();
    });
  }

  // Interactive Fleet Stat Boxes (Total, Critical, Warning, Healthy, High Uncertainty)
  document.querySelectorAll(".clickable-stat").forEach(card => {
    card.addEventListener("click", () => {
      const filterType = card.getAttribute("data-filter");
      selectFleetStatusBox(filterType);
    });
  });

  // Health Filter Pills
  document.querySelectorAll("#health-filter-pills .pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const val = pill.getAttribute("data-filter-val");
      selectFleetStatusBox(val);
    });
  });

  // Priority Filter Pills
  document.querySelectorAll("#prio-filter-pills .pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll("#prio-filter-pills .pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      state.priorityFilter = pill.getAttribute("data-filter-val");
      state.currentPage = 1;
      applyFleetFiltersAndRender();
    });
  });

  // Alert Filter Dropdown
  const alertFilter = document.getElementById("fleet-alert-filter");
  if (alertFilter) {
    alertFilter.addEventListener("change", (e) => {
      state.alertFilter = e.target.value;
      state.currentPage = 1;
      applyFleetFiltersAndRender();
    });
  }

  // Sort Dropdown
  const sortSelect = document.getElementById("fleet-sort-select");
  if (sortSelect) {
    sortSelect.addEventListener("change", (e) => {
      state.sortBy = e.target.value;
      applyFleetFiltersAndRender();
    });
  }

  // Page Size Select
  const pageSizeSelect = document.getElementById("page-size-select");
  if (pageSizeSelect) {
    pageSizeSelect.addEventListener("change", (e) => {
      state.pageSize = e.target.value === "all" ? 9999 : parseInt(e.target.value, 10);
      state.currentPage = 1;
      renderFleetTable();
    });
  }

  // Pagination Buttons
  const prevBtn = document.getElementById("btn-page-prev");
  const nextBtn = document.getElementById("btn-page-next");
  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (state.currentPage > 1) {
        state.currentPage--;
        renderFleetTable();
      }
    });
  }
  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      const maxPages = Math.ceil(state.filteredFleetData.length / state.pageSize) || 1;
      if (state.currentPage < maxPages) {
        state.currentPage++;
        renderFleetTable();
      }
    });
  }

  // ── Engine Detail Listeners ──
  const dsSelectElem = document.getElementById("dataset-select");
  if (dsSelectElem) {
    dsSelectElem.addEventListener("change", async (e) => {
      await switchDataset(e.target.value);
    });
  }

  const engSelectElem = document.getElementById("engine-select");
  if (engSelectElem) {
    engSelectElem.addEventListener("change", async (e) => {
      await switchEngine(parseInt(e.target.value, 10));
    });
  }

  const slider = document.getElementById("cycle-slider");
  if (slider) {
    slider.addEventListener("input", (e) => {
      pauseSimulation();
      setCycle(parseInt(e.target.value, 10));
    });
  }

  const playBtn = document.getElementById("btn-sim-play");
  if (playBtn) playBtn.addEventListener("click", startSimulation);

  const pauseBtn = document.getElementById("btn-sim-pause");
  if (pauseBtn) pauseBtn.addEventListener("click", pauseSimulation);

  const resetBtn = document.getElementById("btn-sim-reset");
  if (resetBtn) resetBtn.addEventListener("click", resetSimulation);

  const simSpeedElem = document.getElementById("sim-speed");
  if (simSpeedElem) {
    simSpeedElem.addEventListener("change", (e) => {
      state.simSpeed = parseInt(e.target.value, 10);
      if (state.isPlaying) {
        pauseSimulation();
        startSimulation();
      }
    });
  }

  const sensorSelectElem = document.getElementById("sensor-select");
  if (sensorSelectElem) {
    sensorSelectElem.addEventListener("change", (e) => {
      state.activeSensor = e.target.value;
      updateSensorTrendChart();
    });
  }

  document.querySelectorAll(".sensor-filter-pills .pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".sensor-filter-pills .pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      state.sensorFilter = pill.getAttribute("data-filter");
      renderSensorTable();
    });
  });

  const toggleGtElem = document.getElementById("toggle-ground-truth");
  if (toggleGtElem) {
    toggleGtElem.addEventListener("change", (e) => {
      if (rulTrendChart && rulTrendChart.data && rulTrendChart.data.datasets[1]) {
        rulTrendChart.data.datasets[1].hidden = !e.target.checked;
        rulTrendChart.update();
      }
    });
  }

  // CSV Upload
  const fileInput = document.getElementById("csv-file-input");
  const dropzone = document.getElementById("upload-dropzone");
  const submitUploadBtn = document.getElementById("btn-submit-upload");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("hover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("hover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("hover");
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelected();
      }
    });
  }

  if (fileInput) fileInput.addEventListener("change", handleFileSelected);
  if (submitUploadBtn) submitUploadBtn.addEventListener("click", submitCsvUpload);

  // Setup Ingested Fleet Table Event Listeners
  setupUploadEventListeners();
}

function handleFileSelected() {
  const fileInput = document.getElementById("csv-file-input");
  const submitUploadBtn = document.getElementById("btn-submit-upload");
  if (fileInput.files.length > 0) {
    const file = fileInput.files[0];
    document.querySelector("#upload-dropzone h3").innerText = `Selected: ${file.name}`;
    submitUploadBtn.disabled = false;
  }
}

async function loadSampleCsv(fileName, datasetHint, label) {
  const submitBtn = document.getElementById("btn-submit-upload");
  const datasetSelect = document.getElementById("upload-dataset-select");
  if (datasetSelect) datasetSelect.value = datasetHint;

  submitBtn.disabled = true;
  submitBtn.innerText = `Ingesting ${label}...`;
  document.querySelector("#upload-dropzone h3").innerText = `Loaded Sample: ${fileName}`;

  try {
    const fetchRes = await fetch(`/sample_data/${fileName}`);
    if (!fetchRes.ok) throw new Error("Could not fetch sample CSV file");
    const blob = await fetchRes.blob();
    const file = new File([blob], fileName, { type: "text/csv" });

    const formData = new FormData();
    formData.append("file", file);
    formData.append("dataset_hint", datasetHint);

    const res = await fetch("/api/upload-csv", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (!res.ok) {
      alert(`Upload Error: ${data.detail || "Validation failed"}`);
      return;
    }

    renderUploadResults(data);
  } catch (err) {
    console.error("Sample CSV upload failed:", err);
    alert(`Failed to process ${fileName}: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerText = "Process & Predict CSV";
  }
}

// ── Primary View: Fleet Overview Operations ─────────────────────────────────

async function loadFleetOverview(force = false) {
  const tableCounter = document.getElementById("fleet-table-counter");
  const activeBadge = document.getElementById("fleet-active-badge");
  const tbody = document.getElementById("fleet-tbody");

  if (activeBadge) {
    activeBadge.innerText = state.fleetDataset === 'all' ? 'All Datasets (FD001–FD004)' : `${state.fleetDataset} Fleet`;
  }
  if (tableCounter) tableCounter.innerText = "Evaluating PyTorch model fleet inference...";

  try {
    const res = await fetch(`/api/fleet-overview/${state.fleetDataset}?force=${force}`);
    const data = await res.json();

    state.rawFleetData = data.fleet || [];

    // Update Fleet Summary Cards
    document.getElementById("fleet-total-count").innerText = data.total_engines || state.rawFleetData.length;
    document.getElementById("fleet-critical-count").innerText = data.critical_count || 0;
    document.getElementById("fleet-warning-count").innerText = data.warning_count || 0;
    document.getElementById("fleet-healthy-count").innerText = data.healthy_count || 0;
    document.getElementById("fleet-uncertainty-count").innerText = data.high_uncertainty_count || 0;

    // Render Attention Required (Top 3-5 Critical Engines)
    renderAttentionSection(data.attention_required || []);

    // Filter, Sort, and Render Table
    applyFleetFiltersAndRender();

    // If first critical engine available, set as initial target for Live Monitor
    if (state.rawFleetData.length > 0) {
      const topCritical = state.rawFleetData[0];
      state.dataset = topCritical.dataset;
      state.engineId = topCritical.engine_id;
    }
  } catch (err) {
    console.error("Failed to load fleet overview:", err);
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="11" class="text-center" style="padding: 24px; color: var(--rose);">Failed to evaluate fleet predictions: ${err.message}</td></tr>`;
    }
  }
}

function renderAttentionSection(attentionList) {
  const container = document.getElementById("attention-cards-container");
  if (!container) return;

  container.innerHTML = "";

  if (!attentionList || attentionList.length === 0) {
    container.innerHTML = `
      <div class="attention-card" style="border-color: var(--emerald-border); background: rgba(16, 185, 129, 0.08);">
        <div style="font-size: 14px; font-weight: 700; color: var(--emerald);">ALL ENGINES NOMINAL</div>
        <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">No engines currently in critical failure threshold. Fleet health is stable.</p>
      </div>
    `;
    return;
  }

  // Display top 3-5 critical engines
  const topEngines = attentionList.slice(0, 5);

  topEngines.forEach((eng, idx) => {
    const card = document.createElement("div");
    card.className = "attention-card";

    const rulVal = eng.predicted_rul !== null ? formatRUL(eng.predicted_rul) : "--";
    const uncertVal = eng.uncertainty !== null ? `±${Math.round(eng.uncertainty)}` : "--";
    const actionText = eng.action || (eng.health_status === 'CRITICAL' ? 'Immediate Overhaul' : 'Schedule Inspection');

    card.innerHTML = `
      <div>
        <div class="attention-card-top">
          <div class="attention-card-title">
            <span class="rank-badge top-critical">#${eng.rank || (idx + 1)}</span>
            <span>Engine ${eng.engine_id}</span>
          </div>
          <span class="badge-tag selected">${eng.dataset}</span>
        </div>

        <div class="attention-card-rul">
          <span class="attention-card-number">${rulVal}</span>
          <span class="attention-card-unit">CYCLES RUL</span>
        </div>

        <div class="attention-card-meta">
          <span>Uncertainty: <strong style="color: var(--cyan);">${uncertVal}</strong></span>
          <span>Cycle: <strong style="color: #fff;">${eng.current_cycle}</strong></span>
          <span>Status: <strong style="color: var(--rose);">${eng.health_status}</strong></span>
        </div>
      </div>

      <button class="attention-card-action-btn">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
        <span>${actionText} &bull; Inspect</span>
      </button>
    `;

    card.addEventListener("click", () => {
      openEngineDetail(eng.dataset, eng.engine_id, eng.rank || (idx + 1));
    });

    container.appendChild(card);
  });
}

function selectFleetStatusBox(filterType) {
  // 1. Highlight the selected stat box
  document.querySelectorAll(".clickable-stat").forEach(c => c.classList.remove("active-filter"));
  const clickedCard = document.querySelector(`.clickable-stat[data-filter="${filterType}"]`);
  if (clickedCard) clickedCard.classList.add("active-filter");

  // 2. Clear any lingering search query so the full category is visible
  state.fleetSearchQuery = "";
  const searchInput = document.getElementById("fleet-search-input");
  if (searchInput) searchInput.value = "";

  // 3. Reset Priority filter to All
  state.priorityFilter = "all";
  document.querySelectorAll("#prio-filter-pills .pill").forEach(p => {
    p.classList.toggle("active", p.getAttribute("data-filter-val") === "all");
  });

  // 4. Set Health and Alert filter states
  const alertSelect = document.getElementById("fleet-alert-filter");
  if (filterType === "high_uncert") {
    state.healthFilter = "all";
    state.alertFilter = "HIGH_UNCERTAINTY";
    if (alertSelect) alertSelect.value = "HIGH_UNCERTAINTY";
    document.querySelectorAll("#health-filter-pills .pill").forEach(p => {
      p.classList.toggle("active", p.getAttribute("data-filter-val") === "all");
    });
  } else {
    state.alertFilter = "all";
    if (alertSelect) alertSelect.value = "all";
    state.healthFilter = filterType; // "all", "critical", "warning", "healthy"
    document.querySelectorAll("#health-filter-pills .pill").forEach(p => {
      p.classList.toggle("active", p.getAttribute("data-filter-val") === filterType);
    });
  }

  // 5. Default sort: lowest predicted RUL first
  state.sortBy = "critical_first";
  const sortSelect = document.getElementById("fleet-sort-select");
  if (sortSelect) sortSelect.value = "critical_first";

  // 6. Reset to first page
  state.currentPage = 1;

  // 7. Apply filters and render all matching engines in table rows
  applyFleetFiltersAndRender();

  // 8. Smoothly scroll table into view
  const tableCard = document.querySelector(".fleet-queue-card");
  if (tableCard) {
    tableCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function applyFleetFiltersAndRender() {
  let filtered = [...state.rawFleetData];

  // 1. Filter by Health Status
  if (state.healthFilter !== "all") {
    filtered = filtered.filter(e => (e.health_status || "").toLowerCase() === state.healthFilter);
  }

  // 2. Filter by Priority
  if (state.priorityFilter !== "all") {
    filtered = filtered.filter(e => (e.priority || "").toLowerCase() === state.priorityFilter);
  }

  // 3. Filter by Alert Type
  if (state.alertFilter === "HIGH_UNCERTAINTY") {
    filtered = filtered.filter(e => (e.uncertainty !== null && e.uncertainty >= 8.0) || e.alert_type === "HIGH_UNCERTAINTY");
  } else if (state.alertFilter !== "all") {
    filtered = filtered.filter(e => e.alert_type === state.alertFilter);
  }

  // 4. Search Filter (Engine ID or Dataset)
  if (state.fleetSearchQuery) {
    filtered = filtered.filter(e =>
      `engine ${e.engine_id}`.toLowerCase().includes(state.fleetSearchQuery) ||
      `${e.engine_id}`.includes(state.fleetSearchQuery) ||
      `${e.dataset}`.toLowerCase().includes(state.fleetSearchQuery)
    );
  }

  // 5. Sorting
  if (state.sortBy === "critical_first") {
    // Priority / Risk -> Predicted RUL ascending
    const prioOrder = { "CRITICAL": 0, "WARNING": 1, "HEALTHY": 2, "INSUFFICIENT_DATA": 3 };
    filtered.sort((a, b) => {
      const pDiff = (prioOrder[a.health_status] ?? 4) - (prioOrder[b.health_status] ?? 4);
      if (pDiff !== 0) return pDiff;
      return (a.predicted_rul ?? 9999) - (b.predicted_rul ?? 9999);
    });
  } else if (state.sortBy === "rul_asc") {
    filtered.sort((a, b) => (a.predicted_rul ?? 9999) - (b.predicted_rul ?? 9999));
  } else if (state.sortBy === "rul_desc") {
    filtered.sort((a, b) => (b.predicted_rul ?? -1) - (a.predicted_rul ?? -1));
  } else if (state.sortBy === "engine_asc") {
    filtered.sort((a, b) => a.engine_id - b.engine_id);
  } else if (state.sortBy === "dataset_asc") {
    filtered.sort((a, b) => a.dataset.localeCompare(b.dataset) || a.engine_id - b.engine_id);
  } else if (state.sortBy === "uncertainty_desc") {
    filtered.sort((a, b) => (b.uncertainty ?? -1) - (a.uncertainty ?? -1));
  } else if (state.sortBy === "cycle_desc") {
    filtered.sort((a, b) => b.current_cycle - a.current_cycle);
  } else if (state.sortBy === "health_status") {
    const prioOrder = { "CRITICAL": 0, "WARNING": 1, "HEALTHY": 2, "INSUFFICIENT_DATA": 3 };
    filtered.sort((a, b) => (prioOrder[a.health_status] ?? 4) - (prioOrder[b.health_status] ?? 4));
  }

  state.filteredFleetData = filtered;

  const tableCounter = document.getElementById("fleet-table-counter");
  if (tableCounter) {
    tableCounter.innerText = `${filtered.length} of ${state.rawFleetData.length} Turbofans Displayed`;
  }

  renderFleetTable();
}

function renderFleetTable() {
  const tbody = document.getElementById("fleet-tbody");
  if (!tbody) return;

  tbody.innerHTML = "";

  if (state.filteredFleetData.length === 0) {
    tbody.innerHTML = `<tr><td colspan="11" class="text-center" style="padding: 30px; color: var(--text-muted);">No engines match the current filters.</td></tr>`;
    updatePaginationUI(0, 0, 0);
    return;
  }

  // Pagination Slice
  const total = state.filteredFleetData.length;
  const startIdx = (state.currentPage - 1) * state.pageSize;
  const endIdx = Math.min(startIdx + state.pageSize, total);
  const pageItems = state.filteredFleetData.slice(startIdx, endIdx);

  pageItems.forEach(eng => {
    const tr = document.createElement("tr");
    const healthCls = (eng.health_status || "warning").toLowerCase();
    const prioCls = (eng.priority || "low").toLowerCase();
    const isTopCrit = (eng.rank || 99) <= 5 && healthCls === "critical";

    tr.innerHTML = `
      <td><span class="rank-badge ${isTopCrit ? 'top-critical' : ''}">#${eng.rank || '--'}</span></td>
      <td><strong style="color: #fff;">Engine ${eng.engine_id}</strong></td>
      <td><span class="badge-tag selected">${eng.dataset}</span></td>
      <td style="font-family: var(--font-mono);">${eng.current_cycle}</td>
      <td style="font-weight: 700; color: #fff; font-family: var(--font-mono); font-size: 13px;">
        ${eng.predicted_rul !== null ? formatRUL(eng.predicted_rul) + ' cycles' : '<span style="color: var(--amber);">N/A</span>'}
      </td>
      <td style="font-family: var(--font-mono); color: var(--cyan);">
        ${eng.uncertainty !== null ? '±' + Math.round(eng.uncertainty) + ' cycles' : '--'}
      </td>
      <td><span class="health-badge ${healthCls}">${eng.health_status}</span></td>
      <td><span class="priority-badge ${prioCls}">${eng.priority}</span></td>
      <td style="font-size: 11px;">${eng.alert_label || '🟢 NOMINAL'}</td>
      <td style="font-size: 11px; color: ${healthCls === 'critical' ? 'var(--rose)' : (healthCls === 'warning' ? 'var(--amber)' : 'var(--emerald)')};">
        ${eng.action || 'Routine Check'}
      </td>
      <td style="text-align: right;">
        <button class="btn btn-outline" style="padding: 4px 10px; font-size: 11px;">
          View Details &rarr;
        </button>
      </td>
    `;

    // Row click -> opens Engine Detail View
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      openEngineDetail(eng.dataset, eng.engine_id, eng.rank);
    });

    tbody.appendChild(tr);
  });

  updatePaginationUI(startIdx + 1, endIdx, total);
}

function updatePaginationUI(start, end, total) {
  const info = document.getElementById("pagination-info");
  const prevBtn = document.getElementById("btn-page-prev");
  const nextBtn = document.getElementById("btn-page-next");
  const numbersContainer = document.getElementById("page-numbers-container");

  if (info) {
    info.innerText = total > 0 ? `Showing ${start}–${end} of ${total} engines` : `0 engines`;
  }

  const maxPages = Math.ceil(total / state.pageSize) || 1;

  if (prevBtn) prevBtn.disabled = state.currentPage <= 1;
  if (nextBtn) nextBtn.disabled = state.currentPage >= maxPages;

  if (numbersContainer) {
    numbersContainer.innerHTML = "";
    // Display up to 5 page buttons around currentPage
    let startP = Math.max(1, state.currentPage - 2);
    let endP = Math.min(maxPages, startP + 4);
    if (endP - startP < 4) startP = Math.max(1, endP - 4);

    for (let p = startP; p <= endP; p++) {
      const btn = document.createElement("button");
      btn.className = `page-btn ${p === state.currentPage ? 'active' : ''}`;
      btn.innerText = p;
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        state.currentPage = p;
        renderFleetTable();
      });
      numbersContainer.appendChild(btn);
    }
  }
}

// ── Open Engine Detail View (Interactive One-Click Transition) ───────────────

async function openEngineDetail(dataset, engineId, rank = null) {
  state.dataset = dataset;
  state.engineId = engineId;
  state.currentEngineRank = rank;

  // 1. Switch active tab to Live Engine Monitor
  switchTab("tab-monitor");

  // 2. Update Header Strip
  document.getElementById("detail-engine-title").innerText = `ENGINE ${engineId}`;
  document.getElementById("detail-dataset-title").innerText = `${dataset}`;
  const rankChip = document.getElementById("detail-rank-chip");
  if (rankChip) {
    rankChip.innerText = rank ? `Rank #${rank} in Fleet Priority Queue` : `Selected Turbofan Unit`;
  }

  // 3. Update Dataset selector
  const dsSelect = document.getElementById("dataset-select");
  if (dsSelect) dsSelect.value = dataset;

  // 4. Load Engine List for dataset and select engine
  try {
    const res = await fetch(`/api/engines/${dataset}`);
    const data = await res.json();
    if (data.status === "success") {
      const engSelect = document.getElementById("engine-select");
      engSelect.innerHTML = "";
      data.engines.forEach(eng => {
        const opt = document.createElement("option");
        opt.value = eng.engine_id;
        opt.innerText = `Engine ${eng.engine_id} (${eng.total_cycles} cycles)`;
        engSelect.appendChild(opt);
      });
      engSelect.value = engineId;
    }
  } catch (err) {
    console.error("Failed to load engine list:", err);
  }

  // 5. Load full engine telemetry, RUL trend, and run model prediction
  await switchEngine(engineId);
}

// Global window alias for table inline click
window.inspectEngine = function(dataset, engineId) {
  openEngineDetail(dataset, engineId);
};

// ── Engine Detail View Operations ────────────────────────────────────────────

async function switchDataset(datasetId) {
  state.dataset = datasetId;
  pauseSimulation();
  document.getElementById("model-status-text").innerText = `Live Engine: ${datasetId}`;

  try {
    const res = await fetch(`/api/engines/${datasetId}`);
    const data = await res.json();
    if (data.status === "success") {
      const engSelect = document.getElementById("engine-select");
      engSelect.innerHTML = "";
      data.engines.forEach(eng => {
        const opt = document.createElement("option");
        opt.value = eng.engine_id;
        opt.innerText = `Engine ${eng.engine_id} (${eng.total_cycles} cycles)`;
        engSelect.appendChild(opt);
      });

      const firstId = data.engines[0]?.engine_id || 1;
      await switchEngine(firstId);
    }
  } catch (err) {
    console.error("Failed to switch dataset:", err);
  }
}

async function switchEngine(engineId) {
  state.engineId = engineId;
  pauseSimulation();

  try {
    // 1. Fetch full engine telemetry
    const telRes = await fetch(`/api/engine-telemetry/${state.dataset}/${engineId}`);
    const telData = await telRes.json();
    if (telData.status !== "success") return;

    state.telemetry = telData.telemetry;
    state.windowSize = telData.window_size;
    state.selectedSensors = telData.selected_sensors;
    state.minCycle = telData.min_cycle;
    state.maxCycle = telData.max_cycle;

    state.telemetryByCycle = {};
    state.telemetry.forEach(r => {
      state.telemetryByCycle[r.cycle] = r;
    });

    populateSensorSelector();

    // 2. Setup Cycle Slider
    const slider = document.getElementById("cycle-slider");
    slider.min = state.minCycle;
    slider.max = state.maxCycle;
    slider.value = state.maxCycle;
    document.getElementById("max-cycle-display").innerText = state.maxCycle;

    // 3. Load historical RUL trajectory curve
    loadRULTrend();

    // 4. Run PyTorch inference on latest cycle
    await setCycle(state.maxCycle);
  } catch (err) {
    console.error("Failed to switch engine:", err);
  }
}

function populateSensorSelector() {
  const sensorSelect = document.getElementById("sensor-select");
  if (!sensorSelect) return;
  sensorSelect.innerHTML = "";
  for (let i = 1; i <= 21; i++) {
    const sId = `s${i}`;
    const meta = SENSOR_META[sId];
    const isSelected = state.selectedSensors.includes(sId);
    const opt = document.createElement("option");
    opt.value = sId;
    opt.innerText = `${sId.toUpperCase()}: ${meta.name} ${isSelected ? '★ (Model Input)' : ''}`;
    if (sId === state.activeSensor) opt.selected = true;
    sensorSelect.appendChild(opt);
  }
}

async function loadRULTrend() {
  try {
    const res = await fetch(`/api/prediction-history/${state.dataset}/${state.engineId}`);
    const data = await res.json();
    state.predictionHistory = data;
    renderRULTrendChart(data);
  } catch (err) {
    console.error("Failed to load RUL trend:", err);
  }
}

async function setCycle(cycle) {
  state.currentCycle = cycle;
  document.getElementById("cycle-slider").value = cycle;
  document.getElementById("current-cycle-display").innerText = cycle;
  document.getElementById("substat-cur-cycle").innerText = cycle;
  document.getElementById("detail-cycle-stat").innerText = `Cycle ${cycle}`;

  // Operating conditions strip
  const curTel = state.telemetryByCycle[cycle] || {};
  document.getElementById("op-setting-1").innerText = curTel.op_1 !== undefined ? curTel.op_1.toFixed(3) : "--";
  document.getElementById("op-setting-2").innerText = curTel.op_2 !== undefined ? curTel.op_2.toFixed(3) : "--";
  document.getElementById("op-setting-3").innerText = curTel.op_3 !== undefined ? curTel.op_3.toFixed(1) : "--";

  const clusterId = curTel.condition_cluster !== undefined ? curTel.condition_cluster : 0;
  document.getElementById("op-cluster-id").innerText = `Cluster #${clusterId}`;

  // Window status
  const winLabel = document.getElementById("window-status-label");
  if (cycle < state.windowSize) {
    winLabel.innerText = `Insufficient (${cycle} / ${state.windowSize}c)`;
    winLabel.style.color = "var(--amber)";
  } else {
    winLabel.innerText = `Valid Window (W=${state.windowSize})`;
    winLabel.style.color = "var(--emerald)";
  }

  renderSensorTable();
  updateSensorTrendChart();

  await fetchAndDisplayPrediction(cycle);
}

async function fetchAndDisplayPrediction(cycle) {
  const insufficientBanner = document.getElementById("insufficient-history-alert");
  const heroRul = document.getElementById("hero-rul-value");
  const healthBadge = document.getElementById("health-badge");

  if (cycle < state.windowSize) {
    insufficientBanner.classList.remove("hidden");
    document.getElementById("insufficient-msg").innerText =
      `Insufficient history for RUL prediction. Minimum ${state.windowSize} cycles required.`;
    heroRul.innerText = "--";
    healthBadge.className = "health-badge warning";
    healthBadge.innerText = "INSUFFICIENT DATA";
    document.getElementById("substat-est-failure").innerText = "--";
    document.getElementById("substat-uncertainty").innerText = "--";
    const ciEl = document.getElementById("substat-ci");
    if (ciEl) ciEl.innerText = "--";
    document.getElementById("ground-truth-strip").classList.add("hidden");
    document.getElementById("detail-status-stat").innerText = "INSUFFICIENT DATA";
    document.getElementById("detail-rul-stat").innerText = "--";
    updateAlerts([]);
    return;
  }

  insufficientBanner.classList.add("hidden");

  // Check client cache
  const cacheKey = `${state.dataset}_${state.engineId}_${cycle}`;
  let pred = state.predictionCache[cacheKey];

  if (!pred) {
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: state.dataset,
          engine_id: state.engineId,
          cycle: cycle,
          mc_samples: 50
        })
      });
      pred = await res.json();
      state.predictionCache[cacheKey] = pred;
    } catch (err) {
      console.error("Prediction API failed:", err);
      return;
    }
  }

  if (pred.status !== "success") return;

  // Hero RUL Readout
  const roundedRul = formatRUL(pred.predicted_rul);
  heroRul.innerText = roundedRul;
  document.getElementById("substat-est-failure").innerText = Math.round(pred.current_cycle + pred.predicted_rul);
  document.getElementById("substat-uncertainty").innerText = formatUncertainty(pred.uncertainty);
  const ciElFilled = document.getElementById("substat-ci");
  if (ciElFilled) ciElFilled.innerText = formatCI(pred.confidence_interval);

  // Health Status Badges
  healthBadge.className = `health-badge ${pred.health_status.toLowerCase()}`;
  healthBadge.innerText = pred.health_status;

  // Update Detail Header Strip
  document.getElementById("detail-rul-stat").innerText = `${roundedRul} cycles`;
  const detailStatus = document.getElementById("detail-status-stat");
  detailStatus.innerText = pred.health_status;
  detailStatus.className = pred.health_status === 'HEALTHY' ? 'highlight-cyan' : (pred.health_status === 'WARNING' ? 'substat-val' : 'rank-badge top-critical');

  // Ground Truth Validation Strip
  const gtStrip = document.getElementById("ground-truth-strip");
  if (pred.ground_truth_rul !== undefined && pred.ground_truth_rul !== null) {
    gtStrip.classList.remove("hidden");
    document.getElementById("gt-rul-val").innerText = formatGroundTruth(pred.ground_truth_rul);
    const absErr = Math.round(Math.abs(pred.predicted_rul - pred.ground_truth_rul));
    document.getElementById("gt-err-val").innerText = absErr;
  } else {
    gtStrip.classList.add("hidden");
  }

  // Decision Support
  const rec = pred.maintenance_recommendation;
  document.getElementById("rec-action-title").innerText = rec.action;
  document.getElementById("rec-action-desc").innerText =
    pred.health_status === "CRITICAL"
      ? "Predicted RUL is critically low. Rapid degradation detected. Immediate hangar inspection recommended."
      : (pred.health_status === "WARNING"
          ? "RUL has crossed below safe operational threshold. Prepare maintenance work orders."
          : "Engine is operating within nominal health envelope. Continue routine monitoring.");
  document.getElementById("rec-window").innerText = rec.recommended_window;

  const prioBadge = document.getElementById("priority-badge");
  prioBadge.className = `priority-badge ${rec.priority.toLowerCase()}`;
  prioBadge.innerText = `${rec.priority} PRIORITY`;

  // Alerts
  updateAlerts(pred.alerts);

  // Synchronize state with raw fleet data if present
  const fleetEng = state.rawFleetData.find(e => e.dataset === state.dataset && e.engine_id === state.engineId);
  if (fleetEng) {
    fleetEng.current_cycle = cycle;
    fleetEng.predicted_rul = pred.predicted_rul;
    fleetEng.uncertainty = pred.uncertainty;
    fleetEng.health_status = pred.health_status;
    fleetEng.priority = rec.priority;
    fleetEng.estimated_failure_cycle = pred.estimated_failure_cycle;
  }
}

function updateAlerts(alerts) {
  const container = document.getElementById("alerts-container");
  const countBadge = document.getElementById("alert-count-badge");
  container.innerHTML = "";

  if (!alerts || alerts.length === 0) {
    countBadge.innerText = "0 Alerts";
    countBadge.style.color = "var(--emerald)";
    container.innerHTML = `
      <div class="alert-item ok">
        <span class="alert-dot green"></span>
        <span class="alert-text">All engine telemetry nominal. No threshold violations.</span>
      </div>
    `;
    return;
  }

  countBadge.innerText = `${alerts.length} Alert${alerts.length > 1 ? 's' : ''}`;
  countBadge.style.color = alerts.some(a => a.severity === 'critical') ? 'var(--rose)' : 'var(--amber)';

  alerts.forEach(a => {
    const item = document.createElement("div");
    item.className = `alert-item ${a.severity}`;
    const dotColor = a.severity === 'critical' ? 'red' : 'amber';
    item.innerHTML = `
      <span class="alert-dot ${dotColor}"></span>
      <div class="alert-body">
        <strong>${a.title}:</strong> ${a.message}
      </div>
    `;
    container.appendChild(item);
  });
}

// ── Streaming Simulation Controls ────────────────────────────────────────────

function startSimulation() {
  if (state.isPlaying) return;
  state.isPlaying = true;
  document.getElementById("btn-sim-play").disabled = true;
  document.getElementById("btn-sim-pause").disabled = false;

  state.simTimer = setInterval(async () => {
    let nextCycle = state.currentCycle + 1;
    if (nextCycle > state.maxCycle) {
      nextCycle = state.minCycle;
    }
    await setCycle(nextCycle);
  }, state.simSpeed);
}

function pauseSimulation() {
  if (!state.isPlaying) return;
  state.isPlaying = false;
  clearInterval(state.simTimer);
  state.simTimer = null;
  document.getElementById("btn-sim-play").disabled = false;
  document.getElementById("btn-sim-pause").disabled = true;
}

function resetSimulation() {
  pauseSimulation();
  setCycle(state.minCycle);
}

// ── Chart.js Visualizations ──────────────────────────────────────────────────

function initCharts() {
  // 1. RUL Degradation Trajectory Chart
  const ctxRul = document.getElementById("rulTrendChart").getContext("2d");
  rulTrendChart = new Chart(ctxRul, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Predicted RUL",
          data: [],
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.1)",
          fill: true,
          tension: 0.25,
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 6
        },
        {
          label: "Ground Truth RUL",
          data: [],
          borderColor: "#a855f7",
          borderDash: [4, 4],
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 5
        },
        {
          label: "Warning Threshold (55)",
          data: [],
          borderColor: "#f59e0b",
          borderDash: [6, 4],
          borderWidth: 1.5,
          pointRadius: 0
        },
        {
          label: "Critical Threshold (25)",
          data: [],
          borderColor: "#f43f5e",
          borderDash: [6, 4],
          borderWidth: 1.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#f8fafc",
          bodyColor: "#94a3b8",
          borderColor: "#334155",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y !== null ? Math.round(ctx.parsed.y) : '--'} cycles`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(51, 65, 85, 0.4)" },
          ticks: { color: "#64748b", font: { family: "'JetBrains Mono'" } },
          title: { display: true, text: "Operating Cycle", color: "#94a3b8" }
        },
        y: {
          grid: { color: "rgba(51, 65, 85, 0.4)" },
          ticks: { color: "#64748b", font: { family: "'JetBrains Mono'" } },
          title: { display: true, text: "RUL (Cycles)", color: "#94a3b8" }
        }
      }
    }
  });

  // 2. Sensor Trend Chart (if present)
  const sensorCanvas = document.getElementById("sensorTrendChart");
  if (sensorCanvas) {
    const ctxSensor = sensorCanvas.getContext("2d");
    sensorTrendChart = new Chart(ctxSensor, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Sensor Reading",
            data: [],
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59, 130, 246, 0.08)",
            fill: true,
            tension: 0.15,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0f172a",
            titleColor: "#f8fafc",
            bodyColor: "#94a3b8",
            borderColor: "#334155",
            borderWidth: 1,
            padding: 10
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(51, 65, 85, 0.4)" },
            ticks: { color: "#64748b", font: { family: "'JetBrains Mono'" } },
            title: { display: true, text: "Operating Cycle", color: "#94a3b8" }
          },
          y: {
            grid: { color: "rgba(51, 65, 85, 0.4)" },
            ticks: { color: "#64748b", font: { family: "'JetBrains Mono'" } },
            title: { display: true, text: "Sensor Value", color: "#94a3b8" }
          }
        }
      }
    });
  }
}

function renderRULTrendChart(historyData) {
  if (!rulTrendChart || !historyData || !historyData.trend) return;

  const points = historyData.trend;
  const labels = points.map(p => p.cycle);
  const predRul = points.map(p => p.predicted_rul);
  const gtRul = points.map(p => p.ground_truth_rul);
  const warnLine = points.map(() => 55);
  const critLine = points.map(() => 25);

  rulTrendChart.data.labels = labels;
  rulTrendChart.data.datasets[0].data = predRul;
  rulTrendChart.data.datasets[1].data = gtRul;
  rulTrendChart.data.datasets[2].data = warnLine;
  rulTrendChart.data.datasets[3].data = critLine;
  rulTrendChart.resize();
  rulTrendChart.update();
}

function updateSensorTrendChart() {
  if (!sensorTrendChart || state.telemetry.length === 0) return;

  const sId = state.activeSensor;
  const meta = SENSOR_META[sId] || { name: sId, unit: "" };

  const labels = state.telemetry.map(r => r.cycle);
  const vals = state.telemetry.map(r => r[sId]);

  sensorTrendChart.data.labels = labels;
  sensorTrendChart.data.datasets[0].data = vals;
  sensorTrendChart.data.datasets[0].label = `${sId.toUpperCase()}: ${meta.name}`;
  sensorTrendChart.options.scales.y.title.text = `${meta.name} (${meta.unit})`;
  sensorTrendChart.update();

  const curVal = state.telemetryByCycle[state.currentCycle]?.[sId];
  const descEl = document.getElementById("sensor-desc-label");
  if (descEl) descEl.innerText = `${sId.toUpperCase()}: ${meta.name} (${meta.unit})`;
  const valEl = document.getElementById("sensor-current-val");
  if (valEl) valEl.innerText = `Current: ${curVal !== undefined ? curVal.toFixed(2) : '--'}`;
}

// ── Sensor Matrix Table ──────────────────────────────────────────────────────

function renderSensorTable() {
  const tbody = document.getElementById("sensor-matrix-tbody");
  tbody.innerHTML = "";

  const curTel = state.telemetryByCycle[state.currentCycle] || {};

  for (let i = 1; i <= 21; i++) {
    const sId = `s${i}`;
    const meta = SENSOR_META[sId];
    const isSelected = state.selectedSensors.includes(sId);

    if (state.sensorFilter === "selected" && !isSelected) continue;
    if (state.sensorFilter === "constant" && isSelected) continue;

    const sVals = state.telemetry.map(r => r[sId]).filter(v => v !== undefined);
    const minVal = sVals.length ? Math.min(...sVals).toFixed(2) : "--";
    const maxVal = sVals.length ? Math.max(...sVals).toFixed(2) : "--";
    const curVal = curTel[sId] !== undefined ? curTel[sId].toFixed(2) : "--";

    let statusBadge = `<span style="color: var(--emerald);">Normal</span>`;
    if (isSelected && sVals.length > 5) {
      const rng = Math.max(...sVals) - Math.min(...sVals);
      if (rng > 0.001) {
        const pct = (curTel[sId] - Math.min(...sVals)) / rng;
        if (pct > 0.85 || pct < 0.15) {
          statusBadge = `<span style="color: var(--amber);">Degraded (Trend)</span>`;
        }
      }
    }

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${sId.toUpperCase()}</strong></td>
      <td>${meta.name}</td>
      <td>${meta.unit}</td>
      <td style="color: #fff; font-weight: 600;">${curVal}</td>
      <td>${minVal} / ${maxVal}</td>
      <td>${statusBadge}</td>
      <td>${isSelected ? '<span class="badge-tag selected">Model Input</span>' : '<span class="badge-tag ignored">Unused</span>'}</td>
    `;

    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      state.activeSensor = sId;
      document.getElementById("sensor-select").value = sId;
      updateSensorTrendChart();
    });

    tbody.appendChild(tr);
  }
}

// ── Model Evaluation Benchmarks ──────────────────────────────────────────────

async function loadBenchmarks() {
  try {
    const res = await fetch("/api/model-performance");
    const data = await res.json();
    const container = document.getElementById("benchmark-grid");
    if (!container) return;
    container.innerHTML = "";

    // Render Benchmark Cards
    data.metrics.forEach(m => {
      const card = document.createElement("div");
      card.className = "benchmark-card";
      card.innerHTML = `
        <div class="bench-header">
          <span class="bench-title">${m.dataset} Benchmark</span>
          <span class="badge-tag selected">${m.status}</span>
        </div>
        <div class="bench-metric-row">
          <span>Flight Operating Conditions:</span>
          <span class="metric-val" style="color: var(--cyan);">${m.conditions}</span>
        </div>
        <div class="bench-metric-row">
          <span>Operational Profile:</span>
          <span class="metric-val" style="font-size: 11px;">${m.conditions_summary || '0 kft | 0.00 M | 100% TRA'}</span>
        </div>
        <div class="bench-metric-row">
          <span>Normalization Strategy:</span>
          <span class="metric-val" style="color: #c084fc; font-size: 11px;">${m.normalization || 'Standard Scaler'}</span>
        </div>
        <div class="bench-metric-row">
          <span>Fault Mode:</span>
          <span class="metric-val">${m.faults}</span>
        </div>
        <div class="bench-metric-row">
          <span>Window Size (W):</span>
          <span class="metric-val">${m.window_size} cycles</span>
        </div>
        <div class="bench-metric-row highlight">
          <span>Root Mean Squared Error (RMSE):</span>
          <span class="metric-val">${m.rmse}</span>
        </div>
        <div class="bench-metric-row">
          <span>C-MAPSS Score (Asymmetric):</span>
          <span class="metric-val">${m.score}</span>
        </div>
        <div class="bench-metric-row">
          <span>Mean Absolute Error (MAE):</span>
          <span class="metric-val">${m.mae}</span>
        </div>
        <div class="bench-metric-row">
          <span>Avg Uncertainty (±1σ):</span>
          <span class="metric-val">${m.uncertainty_cycles}</span>
        </div>
      `;
      container.appendChild(card);
    });

    // Render 6 Operating Conditions Matrix Table
    const tbody = document.getElementById("op-regimes-tbody");
    if (tbody && data.operating_regimes_matrix) {
      tbody.innerHTML = "";
      data.operating_regimes_matrix.forEach(reg => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><span class="op-tag-badge">Regime ${reg.regime_id}</span></td>
          <td><strong style="color: #fff;">${reg.altitude}</strong></td>
          <td><span style="font-family: var(--font-mono); color: var(--cyan);">${reg.mach}</span></td>
          <td><span style="font-family: var(--font-mono); color: var(--amber);">${reg.throttle}</span></td>
          <td>${reg.phase}</td>
          <td><code style="color: #93c5fd;">${reg.datasets}</code></td>
          <td><span class="op-tag-cluster">${reg.clustering}</span></td>
        `;
        tbody.appendChild(row);
      });
    }
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}

// ── CSV Telemetry Upload ─────────────────────────────────────────────────────

async function submitCsvUpload() {
  const fileInput = document.getElementById("csv-file-input");
  const datasetSelect = document.getElementById("upload-dataset-select");
  const submitBtn = document.getElementById("btn-submit-upload");

  if (!fileInput.files.length) return;

  submitBtn.disabled = true;
  submitBtn.innerText = "Ingesting & Running Inference...";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("dataset_hint", datasetSelect.value);

  try {
    const res = await fetch("/api/upload-csv", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (!res.ok) {
      alert(`Upload Error: ${data.detail || "Validation failed"}`);
      return;
    }

    renderUploadResults(data);
  } catch (err) {
    console.error("CSV upload failed:", err);
    alert("CSV upload failed. Check server connection.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerText = "Process & Predict CSV";
  }
}

function setupUploadEventListeners() {
  // Stat box filters
  document.querySelectorAll("#upload-stats-summary .clickable-stat").forEach(card => {
    card.addEventListener("click", () => {
      const filterType = card.getAttribute("data-filter");
      selectUploadStatusBox(filterType);
    });
  });

  // Search input
  const searchInput = document.getElementById("upload-search-input");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      state.uploadState.searchQuery = e.target.value.trim().toLowerCase();
      state.uploadState.currentPage = 1;
      applyUploadFiltersAndRender();
    });
  }

  // Health Filter Pills
  document.querySelectorAll("#upload-health-filter-pills .pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll("#upload-health-filter-pills .pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const val = pill.getAttribute("data-filter-val");
      state.uploadState.healthFilter = val;
      state.uploadState.currentPage = 1;

      // Sync stat card highlight
      document.querySelectorAll("#upload-stats-summary .clickable-stat").forEach(c => c.classList.remove("active-filter"));
      const statCard = document.querySelector(`#upload-stats-summary .clickable-stat[data-filter="${val}"]`);
      if (statCard) statCard.classList.add("active-filter");

      applyUploadFiltersAndRender();
    });
  });

  // Priority Filter Pills
  document.querySelectorAll("#upload-prio-filter-pills .pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll("#upload-prio-filter-pills .pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      state.uploadState.priorityFilter = pill.getAttribute("data-filter-val");
      state.uploadState.currentPage = 1;
      applyUploadFiltersAndRender();
    });
  });

  // Alert Filter Dropdown
  const alertSelect = document.getElementById("upload-alert-filter");
  if (alertSelect) {
    alertSelect.addEventListener("change", (e) => {
      state.uploadState.alertFilter = e.target.value;
      state.uploadState.currentPage = 1;
      applyUploadFiltersAndRender();
    });
  }

  // Sort By Dropdown
  const sortSelect = document.getElementById("upload-sort-select");
  if (sortSelect) {
    sortSelect.addEventListener("change", (e) => {
      state.uploadState.sortBy = e.target.value;
      state.uploadState.currentPage = 1;
      applyUploadFiltersAndRender();
    });
  }

  // Pagination page size
  const pageSizeSelect = document.getElementById("upload-page-size-select");
  if (pageSizeSelect) {
    pageSizeSelect.addEventListener("change", (e) => {
      state.uploadState.pageSize = e.target.value === "all" ? 999999 : parseInt(e.target.value, 10);
      state.uploadState.currentPage = 1;
      renderUploadTable();
    });
  }

  // Prev / Next Page Buttons
  const btnPrev = document.getElementById("upload-btn-page-prev");
  const btnNext = document.getElementById("upload-btn-page-next");
  if (btnPrev) {
    btnPrev.addEventListener("click", () => {
      if (state.uploadState.currentPage > 1) {
        state.uploadState.currentPage--;
        renderUploadTable();
      }
    });
  }
  if (btnNext) {
    btnNext.addEventListener("click", () => {
      const maxPage = Math.ceil(state.uploadState.filteredUploadData.length / state.uploadState.pageSize) || 1;
      if (state.uploadState.currentPage < maxPage) {
        state.uploadState.currentPage++;
        renderUploadTable();
      }
    });
  }
}

function selectUploadStatusBox(filterType) {
  // Highlight stat box
  document.querySelectorAll("#upload-stats-summary .clickable-stat").forEach(c => c.classList.remove("active-filter"));
  const clickedCard = document.querySelector(`#upload-stats-summary .clickable-stat[data-filter="${filterType}"]`);
  if (clickedCard) clickedCard.classList.add("active-filter");

  // Clear search query
  state.uploadState.searchQuery = "";
  const searchInput = document.getElementById("upload-search-input");
  if (searchInput) searchInput.value = "";

  // Reset priority to all
  state.uploadState.priorityFilter = "all";
  document.querySelectorAll("#upload-prio-filter-pills .pill").forEach(p => {
    p.classList.toggle("active", p.getAttribute("data-filter-val") === "all");
  });

  const alertSelect = document.getElementById("upload-alert-filter");
  if (filterType === "high_uncert") {
    state.uploadState.healthFilter = "all";
    state.uploadState.alertFilter = "HIGH_UNCERTAINTY";
    if (alertSelect) alertSelect.value = "HIGH_UNCERTAINTY";
    document.querySelectorAll("#upload-health-filter-pills .pill").forEach(p => {
      p.classList.toggle("active", p.getAttribute("data-filter-val") === "all");
    });
  } else {
    state.uploadState.alertFilter = "all";
    if (alertSelect) alertSelect.value = "all";
    state.uploadState.healthFilter = filterType;
    document.querySelectorAll("#upload-health-filter-pills .pill").forEach(p => {
      p.classList.toggle("active", p.getAttribute("data-filter-val") === filterType);
    });
  }

  state.uploadState.currentPage = 1;
  applyUploadFiltersAndRender();
}

function applyUploadFiltersAndRender() {
  let filtered = [...state.uploadState.rawUploadData];

  // 1. Filter by Health Status
  if (state.uploadState.healthFilter !== "all") {
    filtered = filtered.filter(e => (e.health_status || "").toLowerCase() === state.uploadState.healthFilter);
  }

  // 2. Filter by Priority
  if (state.uploadState.priorityFilter !== "all") {
    filtered = filtered.filter(e => (e.priority || "").toLowerCase() === state.uploadState.priorityFilter);
  }

  // 3. Filter by Alert Type
  if (state.uploadState.alertFilter === "HIGH_UNCERTAINTY") {
    filtered = filtered.filter(e => (e.uncertainty !== null && e.uncertainty >= 8.0) || e.alert_type === "HIGH_UNCERTAINTY");
  } else if (state.uploadState.alertFilter !== "all") {
    filtered = filtered.filter(e => e.alert_type === state.uploadState.alertFilter);
  }

  // 4. Filter by Search Query
  if (state.uploadState.searchQuery) {
    const q = state.uploadState.searchQuery;
    filtered = filtered.filter(e =>
      String(e.engine_id).includes(q) ||
      String(e.dataset || "").toLowerCase().includes(q) ||
      String(e.health_status || "").toLowerCase().includes(q)
    );
  }

  // 5. Sort engines
  filtered.sort((a, b) => {
    switch (state.uploadState.sortBy) {
      case "critical_first": {
        const rulA = a.predicted_rul !== null ? a.predicted_rul : 99999;
        const rulB = b.predicted_rul !== null ? b.predicted_rul : 99999;
        return rulA - rulB;
      }
      case "rul_asc": {
        const rulA = a.predicted_rul !== null ? a.predicted_rul : 99999;
        const rulB = b.predicted_rul !== null ? b.predicted_rul : 99999;
        return rulA - rulB;
      }
      case "rul_desc": {
        const rulA = a.predicted_rul !== null ? a.predicted_rul : -1;
        const rulB = b.predicted_rul !== null ? b.predicted_rul : -1;
        return rulB - rulA;
      }
      case "engine_asc":
        return a.engine_id - b.engine_id;
      case "uncertainty_desc": {
        const unA = a.uncertainty !== null ? a.uncertainty : -1;
        const unB = b.uncertainty !== null ? b.uncertainty : -1;
        return unB - unA;
      }
      case "cycle_desc":
        return b.current_cycle - a.current_cycle;
      case "health_status": {
        const order = { "CRITICAL": 1, "WARNING": 2, "HEALTHY": 3, "INSUFFICIENT_DATA": 4 };
        return (order[a.health_status] || 5) - (order[b.health_status] || 5);
      }
      default:
        return 0;
    }
  });

  state.uploadState.filteredUploadData = filtered;
  renderUploadTable();
}

function renderUploadTable() {
  const tbody = document.getElementById("upload-fleet-tbody");
  const counter = document.getElementById("upload-table-counter");
  const paginationInfo = document.getElementById("upload-pagination-info");
  const pageNumbersContainer = document.getElementById("upload-page-numbers-container");
  const btnPrev = document.getElementById("upload-btn-page-prev");
  const btnNext = document.getElementById("upload-btn-page-next");

  if (!tbody) return;

  const total = state.uploadState.filteredUploadData.length;
  const rawTotal = state.uploadState.rawUploadData.length;

  if (counter) {
    counter.innerText = total === rawTotal ? `${total} Ingested Engines` : `${total} of ${rawTotal} Ingested Engines Filtered`;
  }

  if (total === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="11" class="text-center" style="padding: 30px; color: var(--text-muted);">
          No ingested engines match the selected filters.
        </td>
      </tr>
    `;
    if (paginationInfo) paginationInfo.innerText = "Showing 0 of 0 engines";
    if (pageNumbersContainer) pageNumbersContainer.innerHTML = "";
    if (btnPrev) btnPrev.disabled = true;
    if (btnNext) btnNext.disabled = true;
    return;
  }

  // Calculate Pagination
  const pageSize = state.uploadState.pageSize;
  const maxPage = Math.ceil(total / pageSize) || 1;
  state.uploadState.currentPage = Math.min(state.uploadState.currentPage, maxPage);
  const curPage = state.uploadState.currentPage;

  const startIdx = (curPage - 1) * pageSize;
  const endIdx = Math.min(startIdx + pageSize, total);
  const pageData = state.uploadState.filteredUploadData.slice(startIdx, endIdx);

  if (paginationInfo) {
    paginationInfo.innerText = `Showing ${startIdx + 1}–${endIdx} of ${total} engines`;
  }
  if (btnPrev) btnPrev.disabled = curPage <= 1;
  if (btnNext) btnNext.disabled = curPage >= maxPage;

  if (pageNumbersContainer) {
    pageNumbersContainer.innerHTML = "";
    const maxButtons = 5;
    let startPage = Math.max(1, curPage - Math.floor(maxButtons / 2));
    let endPage = Math.min(maxPage, startPage + maxButtons - 1);
    if (endPage - startPage + 1 < maxButtons) {
      startPage = Math.max(1, endPage - maxButtons + 1);
    }

    for (let p = startPage; p <= endPage; p++) {
      const btn = document.createElement("button");
      btn.className = `page-btn ${p === curPage ? "active" : ""}`;
      btn.innerText = p;
      btn.addEventListener("click", () => {
        state.uploadState.currentPage = p;
        renderUploadTable();
      });
      pageNumbersContainer.appendChild(btn);
    }
  }

  // Render Table Rows
  tbody.innerHTML = "";
  pageData.forEach((eng, pageOffset) => {
    const globalRank = startIdx + pageOffset + 1;
    const tr = document.createElement("tr");

    const healthCls = (eng.health_status || "warning").toLowerCase();
    const prioCls = (eng.priority || "low").toLowerCase();
    const alertCls = (eng.alert_severity || "nominal").toLowerCase();

    const displayRul = formatRUL(eng.predicted_rul);
    const rulFormatted = eng.predicted_rul !== null ? `${displayRul} <span class="unit-label">cycles</span>` : `<span style="color: var(--text-muted);">--</span>`;
    const uncertFormatted = eng.uncertainty !== null ? `&plusmn;${Math.round(eng.uncertainty)} cycles` : `<span style="color: var(--text-muted);">--</span>`;
    const ciFormatted = eng.confidence_interval ? formatCI(eng.confidence_interval) : "";
    const isTopCritical = globalRank <= 3 && eng.health_status === "CRITICAL";

    tr.innerHTML = `
      <td>
        <span class="rank-badge ${isTopCritical ? 'top-critical' : ''}">#${globalRank}</span>
      </td>
      <td>
        <div class="engine-cell">
          <span class="engine-id-text">Engine ${eng.engine_id}</span>
        </div>
      </td>
      <td>
        <span class="badge-tag">${eng.dataset || state.uploadState.datasetUsed}</span>
      </td>
      <td>
        <span class="cycle-cell">Cycle ${eng.current_cycle}</span>
        <span class="cycle-sub">${eng.total_cycles || eng.current_cycle} total</span>
      </td>
      <td>
        <span class="rul-val-badge ${healthCls}">${rulFormatted}</span>
      </td>
      <td>
        <div class="uncert-cell">
          <span class="uncert-main">${uncertFormatted}</span>
          ${ciFormatted ? `<span class="uncert-ci">${ciFormatted}</span>` : ''}
        </div>
      </td>
      <td>
        <span class="health-badge ${healthCls}">${eng.health_status || 'UNKNOWN'}</span>
      </td>
      <td>
        <span class="priority-badge ${prioCls}">${eng.priority || 'LOW'}</span>
      </td>
      <td>
        <span class="alert-badge ${alertCls}">${eng.alert_title || 'Nominal'}</span>
      </td>
      <td>
        <span class="action-text">${eng.action || 'Normal Operations'}</span>
      </td>
      <td style="text-align: right;">
        <button class="btn btn-secondary inspect-btn" title="Inspect Engine Telemetry">Inspect &rarr;</button>
      </td>
    `;

    tr.addEventListener("click", () => {
      inspectUploadedEngine(eng.dataset || state.uploadState.datasetUsed, eng.engine_id, eng.current_cycle);
    });

    tbody.appendChild(tr);
  });
}

function renderUploadResults(data) {
  const wrap = document.getElementById("upload-results-wrap");
  if (!wrap) return;

  wrap.classList.remove("hidden");

  // Transform raw predictions to match fleet table schema
  const datasetUsed = data.dataset_used || "FD001";
  state.uploadState.datasetUsed = datasetUsed;

  let uncertCount = 0;
  const transformed = (data.predictions || []).map((pred, idx) => {
    const alert = pred.alerts && pred.alerts.length > 0 ? pred.alerts[0] : null;
    const isHighUncert = pred.uncertainty !== null && pred.uncertainty >= 8.0;
    if (isHighUncert) uncertCount++;

    return {
      rank: idx + 1,
      engine_id: pred.engine_id,
      dataset: datasetUsed,
      current_cycle: pred.current_cycle,
      total_cycles: pred.total_cycles || pred.current_cycle,
      predicted_rul: pred.predicted_rul,
      uncertainty: pred.uncertainty,
      confidence_interval: pred.confidence_interval,
      health_status: pred.health_status,
      priority: pred.maintenance_recommendation?.priority || "LOW",
      action: pred.maintenance_recommendation?.action || pred.message || "Normal Operating Parameters",
      alert_title: alert ? alert.title : "Nominal",
      alert_type: alert ? alert.type : "NOMINAL",
      alert_severity: alert ? alert.severity : "nominal"
    };
  });

  state.uploadState.rawUploadData = transformed;

  // Update Summary KPI Boxes
  const sum = data.summary || {
    total: transformed.length,
    critical: transformed.filter(e => e.health_status === "CRITICAL").length,
    warning: transformed.filter(e => e.health_status === "WARNING").length,
    healthy: transformed.filter(e => e.health_status === "HEALTHY").length
  };

  const totalEl = document.getElementById("upload-total-count");
  const critEl = document.getElementById("upload-critical-count");
  const warnEl = document.getElementById("upload-warning-count");
  const hlthEl = document.getElementById("upload-healthy-count");
  const uncertEl = document.getElementById("upload-uncertainty-count");
  const badgeEl = document.getElementById("upload-active-badge");

  if (totalEl) totalEl.innerText = sum.total;
  if (critEl) critEl.innerText = sum.critical;
  if (warnEl) warnEl.innerText = sum.warning;
  if (hlthEl) hlthEl.innerText = sum.healthy;
  if (uncertEl) uncertEl.innerText = uncertCount;
  if (badgeEl) badgeEl.innerText = `${datasetUsed} Model Ingestion`;

  // Default filters
  state.uploadState.healthFilter = "all";
  state.uploadState.priorityFilter = "all";
  state.uploadState.alertFilter = "all";
  state.uploadState.sortBy = "critical_first";
  state.uploadState.currentPage = 1;

  // Apply filters and render full table
  applyUploadFiltersAndRender();

  // Smooth scroll to results
  wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function inspectUploadedEngine(dataset, engineId, cycle) {
  state.dataset = dataset;
  state.engineId = engineId;
  state.currentCycle = cycle;

  const datasetSelect = document.getElementById("dataset-select");
  if (datasetSelect) datasetSelect.value = dataset;

  switchTab("tab-monitor");
  loadEngineData(true);
}

async function loadDatasets() {
  try {
    const res = await fetch("/api/datasets");
    const data = await res.json();
    if (data.status === "success") {
      const select = document.getElementById("dataset-select");
      if (select) {
        select.innerHTML = "";
        data.datasets.forEach(ds => {
          const opt = document.createElement("option");
          opt.value = ds.id;
          opt.innerText = `${ds.name} (W=${ds.window_size})`;
          select.appendChild(opt);
        });
        select.value = state.dataset;
      }
    }
  } catch (err) {
    console.error("Failed to load datasets:", err);
  }
}
