// ==========================================================================
// SeaSentinel - Application Logic & Real-Time Ops Engine
// ==========================================================================

let currentMissionId = null;
let currentReport = null;
let rawSonarImg = null;
let offscreenCanvas = null;
let offscreenCtx = null;

let gisMap = null;
let trackLayer = null;
let markersLayer = null;
let mapMarkers = {};

let activeClassFilter = "all";
let confThreshold = 60;
let selectedDetectionId = null;
let sliderDebounceTimer = null;

// Canvas Pan & Zoom State
let canvas = null;
let ctx = null;
let zoomLevel = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let startDragX = 0;
let startDragY = 0;
let showBoxes = true;
let showShadows = true;

const CLASS_COLORS = {
  ghost_net: "#00E599",       // Emerald Teal
  shipwreck: "#F59E0B",       // Amber
  pipe_cylinder: "#38BDF8",   // Cyan
  container: "#A855F7",       // Purple
  tire: "#10B981",            // Green
  generic_debris: "#A855F7",  // Purple / Generic Anomaly
  unknown_anomaly: "#94A3B8"  // Slate Gray
};

document.addEventListener("DOMContentLoaded", () => {
  initNavigationTabs();
  initMap();
  initCanvas();
  initEventListeners();
  checkSystemStatus();
});

// --------------------------------------------------------------------------
// Navigation Tab Switching (Workspace / Reports / Health)
// --------------------------------------------------------------------------
function initNavigationTabs() {
  const navItems = document.querySelectorAll(".sidebar-nav .nav-item");
  const views = {
    workspace: document.getElementById("viewWorkspace"),
    reports: document.getElementById("viewReports"),
    health: document.getElementById("viewHealth")
  };

  navItems.forEach((btn) => {
    btn.addEventListener("click", () => {
      navItems.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const viewKey = btn.getAttribute("data-view");
      Object.keys(views).forEach(k => {
        if (views[k]) views[k].classList.remove("active");
      });
      if (views[viewKey]) {
        views[viewKey].classList.add("active");
        if (viewKey === "workspace" && gisMap) {
          setTimeout(() => { gisMap.invalidateSize(); renderCanvasFast(); }, 150);
        }
      }
    });
  });
}

// --------------------------------------------------------------------------
// Leaflet GIS Map Initialization (Clean Dark Ocean Map)
// --------------------------------------------------------------------------
function initMap() {
  gisMap = L.map("gisMap", {
    zoomControl: true,
    attributionControl: false
  }).setView([18.9220, 72.8346], 15);

  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    opacity: 0.85
  }).addTo(gisMap);

  trackLayer = L.layerGroup().addTo(gisMap);
  markersLayer = L.layerGroup().addTo(gisMap);
}

// --------------------------------------------------------------------------
// Canvas Initialization with Offscreen Buffering
// --------------------------------------------------------------------------
function initCanvas() {
  canvas = document.getElementById("sonarCanvas");
  ctx = canvas.getContext("2d", { alpha: false });

  offscreenCanvas = document.createElement("canvas");
  offscreenCtx = offscreenCanvas.getContext("2d", { alpha: false });

  const container = document.getElementById("canvasContainer");

  container.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    zoomLevel = Math.max(0.2, Math.min(zoomLevel * zoomFactor, 8.0));
    renderCanvasFast();
  }, { passive: false });

  container.addEventListener("mousedown", (e) => {
    isDragging = true;
    startDragX = e.clientX - panX;
    startDragY = e.clientY - panY;
  });

  window.addEventListener("mousemove", (e) => {
    if (isDragging) {
      panX = e.clientX - startDragX;
      panY = e.clientY - startDragY;
      renderCanvasFast();
    }
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });
}

// --------------------------------------------------------------------------
// Event Listeners & UI Controls
// --------------------------------------------------------------------------
function initEventListeners() {
  // Top Action: Load Demo Survey
  document.getElementById("btnLoadDemo").addEventListener("click", () => {
    const scenario = document.getElementById("scenarioSelect").value;
    runScenario(scenario);
  });

  // Action Bar: Run Analysis
  document.getElementById("btnRunAnalysis").addEventListener("click", () => {
    const sFile = document.getElementById("mainSonarInput").files[0];
    if (sFile) {
      uploadAndAnalyze();
    } else {
      const scenario = document.getElementById("scenarioSelect").value;
      runScenario(scenario);
    }
  });

  // File Upload Trigger
  const uploadBtn = document.getElementById("btnUploadTrigger");
  const sonarInput = document.getElementById("mainSonarInput");
  uploadBtn.addEventListener("click", () => sonarInput.click());
  sonarInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      uploadBtn.innerHTML = `<span>📁 ${e.target.files[0].name}</span>`;
      uploadAndAnalyze();
    }
  });

  // Nav CSV Chooser
  const navInput = document.getElementById("mainNavInput");
  navInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      document.getElementById("mainNavFileName").innerText = e.target.files[0].name;
    }
  });

  // Confidence Slider (Debounced 150ms)
  const slider = document.getElementById("confSlider");
  slider.addEventListener("input", (e) => {
    confThreshold = parseInt(e.target.value);
    document.getElementById("confPill").innerText = `${confThreshold}%`;

    clearTimeout(sliderDebounceTimer);
    sliderDebounceTimer = setTimeout(() => {
      updateFilteredViews();
    }, 150);
  });

  // Class Filter Dropdown
  document.getElementById("classDropdown").addEventListener("change", (e) => {
    activeClassFilter = e.target.value;
    updateFilteredViews();
  });

  // Mosaic Tool Buttons
  document.getElementById("btnZoomIn").addEventListener("click", () => {
    zoomLevel = Math.min(zoomLevel * 1.25, 8.0);
    renderCanvasFast();
  });

  document.getElementById("btnZoomOut").addEventListener("click", () => {
    zoomLevel = Math.max(zoomLevel * 0.8, 0.2);
    renderCanvasFast();
  });

  document.getElementById("btnResetView").addEventListener("click", () => {
    zoomLevel = 1.0;
    panX = 0;
    panY = 0;
    renderCanvasFast();
  });

  document.getElementById("btnToggleBoxes").addEventListener("click", (e) => {
    showBoxes = !showBoxes;
    e.target.classList.toggle("active", showBoxes);
    renderCanvasFast();
  });

  document.getElementById("btnToggleShadows").addEventListener("click", (e) => {
    showShadows = !showShadows;
    e.target.classList.toggle("active", showShadows);
    renderCanvasFast();
  });

  // Reset Queue
  document.getElementById("btnResetQueue").addEventListener("click", () => {
    if (currentReport) {
      currentReport.detections.forEach(d => { d.status = "high_confidence"; });
      updateFilteredViews();
    }
  });

  // Export Buttons
  document.getElementById("btnExportJSON").addEventListener("click", () => {
    if (currentMissionId) window.open(`/api/export/${currentMissionId}/json`, "_blank");
  });
  document.getElementById("btnExportCSV").addEventListener("click", () => {
    if (currentMissionId) window.open(`/api/export/${currentMissionId}/csv`, "_blank");
  });
  document.getElementById("btnExportGeoJSON").addEventListener("click", () => {
    if (currentMissionId) window.open(`/api/export/${currentMissionId}/geojson`, "_blank");
  });
  document.getElementById("btnExportZIP").addEventListener("click", () => {
    if (currentMissionId) window.open(`/api/export/${currentMissionId}/zip`, "_blank");
  });

  // Modal Controls
  document.getElementById("btnCloseInspector").addEventListener("click", () => {
    document.getElementById("inspectorModal").classList.remove("open");
  });

  document.getElementById("btnConfirmTarget").addEventListener("click", () => submitFeedback(true));
  document.getElementById("btnRejectTarget").addEventListener("click", () => submitFeedback(false));

  // Reports Search Input
  document.getElementById("reportSearch").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    document.querySelectorAll("#reportsTableBody tr").forEach((row) => {
      row.style.display = row.innerText.toLowerCase().includes(q) ? "" : "none";
    });
  });
}

// --------------------------------------------------------------------------
// System Telemetry
// --------------------------------------------------------------------------
async function checkSystemStatus() {
  try {
    const isCloud = window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1";
    const statusText = document.getElementById("edgeStatusText");
    const metaSub = document.getElementById("edgeMetaSub");

    if (isCloud) {
      if (statusText) statusText.innerText = "Cloud mode · Online";
      if (metaSub) metaSub.innerText = "Hosted inference active";
    } else {
      if (statusText) statusText.innerText = "Edge mode · On-device";
      if (metaSub) metaSub.innerText = "On-device inference ready";
    }

    const res = await fetch("/api/system-status");
    const data = await res.json();
    const healthDev = document.getElementById("healthDevice");
    if (healthDev) {
      healthDev.innerText = `${data.device} Node (${isCloud ? "Cloud Hosted" : "Offline Edge"})`;
    }
  } catch (e) {}
}

// --------------------------------------------------------------------------
// Scenario Execution & File Upload
// --------------------------------------------------------------------------
async function runScenario(scenario) {
  setLoadingState(true);
  try {
    const res = await fetch(`/api/generate-sample?scenario=${scenario}&conf_threshold=${confThreshold}`, { method: "POST" });
    if (!res.ok) {
      let errDetail = `Server Error (${res.status})`;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || errDetail;
      } catch (e) {
        errDetail = await res.text();
      }
      throw new Error(errDetail);
    }
    const data = await res.json();
    loadMissionData(data.mission_id, data.report);
  } catch (err) {
    alert("Error loading demo scenario: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

async function uploadAndAnalyze() {
  const sFile = document.getElementById("mainSonarInput").files[0];
  const nFile = document.getElementById("mainNavInput").files[0];
  if (!sFile) return;

  const formData = new FormData();
  formData.append("sonar_file", sFile);
  if (nFile) formData.append("nav_file", nFile);
  formData.append("conf_threshold", confThreshold.toString());
  formData.append("enable_tvg", "true");
  formData.append("enable_lee", "true");
  formData.append("enable_slant_range", "true");
  formData.append("enable_physics", "true");

  setLoadingState(true);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) {
      let errDetail = `Server Error (${res.status})`;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || errDetail;
      } catch (e) {
        errDetail = await res.text();
      }
      throw new Error(errDetail);
    }
    const data = await res.json();
    loadMissionData(data.mission_id, data.report);
  } catch (err) {
    alert("Error processing sonar upload: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  const btn = document.getElementById("btnRunAnalysis");
  const loadingOverlay = document.getElementById("mosaicLoading");

  if (isLoading) {
    btn.disabled = true;
    btn.innerText = "Analyzing...";
    loadingOverlay.classList.add("active");
    document.getElementById("statStatus").innerText = "Analyzing...";
    document.getElementById("statStatusDesc").innerText = "Executing deep neural inference";
  } else {
    btn.disabled = false;
    btn.innerText = "Run analysis";
    loadingOverlay.classList.remove("active");
  }
}

// --------------------------------------------------------------------------
// Load Mission Data & Render Everything
// --------------------------------------------------------------------------
function loadMissionData(missionId, report) {
  currentMissionId = missionId;
  currentReport = report;

  const procMs = (report.summary.processing_time_sec * 1000).toFixed(0);
  document.getElementById("statStatus").innerText = "Ready";
  document.getElementById("statStatusDesc").innerText = `Processed in ${procMs} ms`;
  document.getElementById("statHighConf").innerText = report.detections.length;
  document.getElementById("statAuditLog").innerText = report.audit_log.length;
  document.getElementById("healthLatency").innerText = `${procMs} ms`;
  document.getElementById("mosaicStatusTitle").innerText = `${report.detections.length} acoustic targets detected`;
  document.getElementById("mosaicPlaceholder").classList.add("hidden");

  // Cache base image on offscreen buffer for instant rendering
  rawSonarImg = new Image();
  rawSonarImg.src = `/api/images/${missionId}/raw?t=${Date.now()}`;
  rawSonarImg.onload = () => {
    canvas.width = rawSonarImg.width;
    canvas.height = rawSonarImg.height;
    offscreenCanvas.width = rawSonarImg.width;
    offscreenCanvas.height = rawSonarImg.height;

    // Draw base image onto buffer
    offscreenCtx.drawImage(rawSonarImg, 0, 0);

    // Draw Nadir Center Line
    offscreenCtx.strokeStyle = "rgba(0, 229, 153, 0.4)";
    offscreenCtx.lineWidth = 1;
    offscreenCtx.setLineDash([4, 4]);
    offscreenCtx.beginPath();
    offscreenCtx.moveTo(canvas.width / 2, 0);
    offscreenCtx.lineTo(canvas.width / 2, canvas.height);
    offscreenCtx.stroke();
    offscreenCtx.setLineDash([]);

    updateFilteredViews();
  };

  renderMap(report);
  updateFilteredViews();
}

// --------------------------------------------------------------------------
// Render Geolocation Map
// --------------------------------------------------------------------------
function renderMap(report) {
  trackLayer.clearLayers();
  markersLayer.clearLayers();
  mapMarkers = {};

  if (!report.nav_track || report.nav_track.length === 0) return;

  // Draw vessel survey line
  const latLngs = report.nav_track.map((p) => [p.latitude, p.longitude]);
  const polyline = L.polyline(latLngs, {
    color: "#00E599",
    weight: 2.5,
    opacity: 0.9,
    dashArray: "6, 6"
  }).addTo(trackLayer);

  if (latLngs.length > 0) {
    gisMap.fitBounds(polyline.getBounds(), { padding: [25, 25] });
    const mid = report.nav_track[0];
    document.getElementById("geoCoordinatesReadout").innerText =
      `${mid.latitude.toFixed(4)}° N, ${mid.longitude.toFixed(4)}° E · Survey Track Active`;
  }

  // Draw detection pins
  const allDets = [...report.detections, ...report.audit_log];
  allDets.forEach((d) => {
    const color = CLASS_COLORS[d.class_name] || "#00E599";
    const marker = L.circleMarker([d.geo_location.latitude, d.geo_location.longitude], {
      radius: 6,
      fillColor: color,
      color: "#050C0E",
      weight: 1.5,
      opacity: 1,
      fillOpacity: 0.95
    });

    const popupHtml = `
      <div style="font-family: 'Inter', sans-serif; color: #111; min-width: 170px;">
        <h4 style="margin: 0 0 4px 0; text-transform: capitalize; color: ${color}; font-size: 13px;">${d.class_name.replace("_", " ")}</h4>
        <p style="margin: 0; font-size: 11px;"><strong>ID:</strong> ${d.id}</p>
        <p style="margin: 0; font-size: 11px;"><strong>Confidence:</strong> ${d.confidence_percent}%</p>
        <p style="margin: 0; font-size: 11px;"><strong>Dimensions:</strong> ${d.dimensions.length_m}m × ${d.dimensions.width_m}m</p>
        <button onclick="openInspector('${d.id}')" style="margin-top: 6px; padding: 4px 8px; font-size: 11px; background: #00E599; color: #000; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-weight: 600;">
          Inspect Target
        </button>
      </div>
    `;
    marker.bindPopup(popupHtml);
    markersLayer.addLayer(marker);
    mapMarkers[d.id] = marker;
  });
}

// --------------------------------------------------------------------------
// Update Filtered Views (Canvas, Review Queue & Reports Table)
// --------------------------------------------------------------------------
function updateFilteredViews() {
  if (!currentReport) return;

  const filtered = currentReport.detections.filter((d) => {
    const conf = d.confidence_percent;
    const matchClass = activeClassFilter === "all" || activeClassFilter === d.class_name;
    const matchConf = conf >= confThreshold;
    return matchClass && matchConf;
  });

  document.getElementById("statHighConf").innerText = filtered.length;
  document.getElementById("reviewQueueCount").innerText = filtered.length;

  renderReviewQueue(filtered);
  renderReportsTable(filtered);
  renderCanvasFast(filtered);

  // Update map markers visibility
  Object.keys(mapMarkers).forEach((id) => {
    const isVisible = filtered.some((d) => d.id === id);
    const marker = mapMarkers[id];
    if (marker) {
      if (isVisible) {
        if (!markersLayer.hasLayer(marker)) markersLayer.addLayer(marker);
      } else {
        if (markersLayer.hasLayer(marker)) markersLayer.removeLayer(marker);
      }
    }
  });
}

// --------------------------------------------------------------------------
// Review Queue Component Rendering
// --------------------------------------------------------------------------
function renderReviewQueue(detections) {
  const container = document.getElementById("queueCardList");
  const emptyMsg = document.getElementById("emptyQueueMsg");

  if (!detections || detections.length === 0) {
    emptyMsg.style.display = "block";
    container.innerHTML = "";
    return;
  }

  emptyMsg.style.display = "none";
  container.innerHTML = detections.map((d) => {
    const cropImgSrc = d.thumbnail_url || `/api/images/${currentMissionId}/raw`;
    const statusClass = d.status === "analyst_confirmed" ? "confirmed" : (d.status === "analyst_rejected" ? "rejected" : "");

    return `
      <div class="candidate-card ${statusClass}" onclick="openInspector('${d.id}')">
        <div class="cand-left">
          <img class="cand-crop" src="${cropImgSrc}" alt="crop">
          <div class="cand-info">
            <span class="cand-class">${d.class_name.replace("_", " ")}</span>
            <span class="cand-sub">${d.dimensions.length_m}m × ${d.dimensions.width_m}m · <strong class="text-teal">${d.confidence_percent}%</strong></span>
          </div>
        </div>
        <div class="cand-actions" onclick="event.stopPropagation()">
          <button class="btn-mini btn-mini-confirm" onclick="quickConfirm('${d.id}')">✓</button>
          <button class="btn-mini btn-mini-reject" onclick="quickReject('${d.id}')">✗</button>
        </div>
      </div>
    `;
  }).join("");
}

// --------------------------------------------------------------------------
// Fast Canvas Blit (Zero Lag)
// --------------------------------------------------------------------------
function renderCanvasFast(detections = null) {
  if (!ctx || !offscreenCanvas) return;

  const currentDets = detections || (currentReport ? currentReport.detections : []);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();

  // Apply Pan & Zoom
  ctx.translate(panX, panY);
  ctx.scale(zoomLevel, zoomLevel);

  // Blit pre-cached base image
  ctx.drawImage(offscreenCanvas, 0, 0);

  // Draw Bounding Boxes
  if (showBoxes && currentDets) {
    currentDets.forEach((d) => {
      const isSelected = d.id === selectedDetectionId;
      const color = CLASS_COLORS[d.class_name] || "#00E599";
      const b = d.bbox;
      const bw = b.xmax - b.xmin;
      const bh = b.ymax - b.ymin;

      ctx.shadowColor = color;
      ctx.shadowBlur = (isSelected ? 10 : 5) / zoomLevel;

      ctx.strokeStyle = isSelected ? "#FFFFFF" : color;
      ctx.lineWidth = (isSelected ? 2.5 : 1.5) / zoomLevel;
      ctx.strokeRect(b.xmin, b.ymin, bw, bh);

      ctx.shadowBlur = 0;

      const label = `${d.class_name.replace("_", " ")} ${d.confidence_percent}%`;
      ctx.font = `600 ${Math.max(9, 10 / zoomLevel)}px Inter, sans-serif`;
      const textW = ctx.measureText(label).width + 8;
      ctx.fillStyle = color;
      ctx.fillRect(b.xmin, Math.max(0, b.ymin - (15 / zoomLevel)), textW, 15 / zoomLevel);

      ctx.fillStyle = "#050C0E";
      ctx.fillText(label, b.xmin + 4, Math.max(11 / zoomLevel, b.ymin - (3 / zoomLevel)));

      if (showShadows && d.channel) {
        const arrowDir = d.channel === "port" ? -1 : 1;
        const arrowX = b.xmin + bw / 2;
        const arrowY = b.ymin + bh / 2;
        const arrowLen = Math.min(25, bw * 0.7) * arrowDir;

        ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
        ctx.lineWidth = 1.2 / zoomLevel;
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(arrowX + arrowLen, arrowY);
        ctx.stroke();
      }
    });
  }

  ctx.restore();
}

// --------------------------------------------------------------------------
// Full Detection Reports Table (Tab 2)
// --------------------------------------------------------------------------
function renderReportsTable(detections) {
  const tbody = document.getElementById("reportsTableBody");
  if (!detections || detections.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty-table-msg">No detections matching filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = detections.map((d) => {
    const cropImgSrc = d.thumbnail_url || `/api/images/${currentMissionId}/raw`;
    return `
      <tr>
        <td><img class="cand-crop" src="${cropImgSrc}" alt="crop" onclick="openInspector('${d.id}')"></td>
        <td class="font-mono">${d.id}</td>
        <td style="text-transform: capitalize; font-weight: 500;">${d.class_name.replace("_", " ")}</td>
        <td><strong class="text-teal">${d.confidence_percent}%</strong></td>
        <td><span style="font-size: 11px; color: var(--teal-primary);">Score: ${(d.physics_score * 100).toFixed(0)}% (C:${d.acoustic_signature.contrast_ratio})</span></td>
        <td class="font-mono">${d.geo_location.latitude.toFixed(6)}, ${d.geo_location.longitude.toFixed(6)}</td>
        <td class="font-mono">${d.dimensions.length_m}m × ${d.dimensions.width_m}m (H:${d.dimensions.estimated_height_m}m)</td>
        <td>${d.channel.toUpperCase()} (${d.geo_location.ground_range_m}m)</td>
        <td><span style="font-size: 11px; text-transform: capitalize; color: var(--text-muted);">${d.status.replace("_", " ")}</span></td>
        <td>
          <button class="tool-btn" onclick="openInspector('${d.id}')">Inspect</button>
        </td>
      </tr>
    `;
  }).join("");
}

// --------------------------------------------------------------------------
// Inspector Modal & Active Learning
// --------------------------------------------------------------------------
window.openInspector = function(detectionId) {
  selectedDetectionId = detectionId;
  const allDets = [...(currentReport.detections || []), ...(currentReport.audit_log || [])];
  const target = allDets.find((d) => d.id === detectionId);
  if (!target) return;

  document.getElementById("inspectorTitle").innerText = `Target Analysis: [${target.id}] ${target.class_name.replace("_", " ").toUpperCase()}`;
  document.getElementById("inspectorCropImg").src = target.thumbnail_url || `/api/images/${currentMissionId}/raw`;
  document.getElementById("inspClass").innerText = target.class_name.replace("_", " ");
  document.getElementById("inspConfidence").innerText = `${target.confidence_percent}%`;
  document.getElementById("inspScores").innerText = `${(target.raw_model_score * 100).toFixed(1)}% / ${(target.physics_score * 100).toFixed(1)}%`;
  document.getElementById("inspContrast").innerText = `${target.acoustic_signature.contrast_ratio} (HL: ${target.acoustic_signature.highlight_mean} / SH: ${target.acoustic_signature.shadow_mean})`;
  document.getElementById("inspCoords").innerText = `${target.geo_location.latitude.toFixed(7)}, ${target.geo_location.longitude.toFixed(7)} (Depth: ${target.geo_location.depth_m}m)`;
  document.getElementById("inspDims").innerText = `${target.dimensions.length_m}m (L) × ${target.dimensions.width_m}m (W) × ${target.dimensions.estimated_height_m}m (H) [Area: ${target.dimensions.area_m2}m²]`;

  document.getElementById("inspectorModal").classList.add("open");

  const marker = mapMarkers[detectionId];
  if (marker) {
    gisMap.setView(marker.getLatLng(), 17);
    marker.openPopup();
  }

  renderCanvasFast();
};

window.quickConfirm = function(id) {
  selectedDetectionId = id;
  submitFeedback(true);
};

window.quickReject = function(id) {
  selectedDetectionId = id;
  submitFeedback(false);
};

async function submitFeedback(isConfirmed) {
  if (!selectedDetectionId || !currentMissionId) return;
  const allDets = [...(currentReport.detections || []), ...(currentReport.audit_log || [])];
  const target = allDets.find((d) => d.id === selectedDetectionId);
  if (!target) return;

  const notes = document.getElementById("alNotes").value;

  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mission_id: currentMissionId,
        detection_id: selectedDetectionId,
        class_name: target.class_name,
        is_confirmed: isConfirmed,
        notes: notes
      })
    });

    target.status = isConfirmed ? "analyst_confirmed" : "analyst_rejected";
    document.getElementById("inspectorModal").classList.remove("open");
    updateFilteredViews();
  } catch (e) {
    alert("Feedback submission failed.");
  }
}
