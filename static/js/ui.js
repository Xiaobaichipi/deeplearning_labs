/* UI Rendering Functions */

/* =============== Project System =============== */

function populateProjectGrid(projects) {
    const grid = document.getElementById("projectGrid");
    const empty = document.getElementById("projectEmpty");

    if (!projects.length) {
        grid.innerHTML = "";
        empty.classList.add("show");
        return;
    }
    empty.classList.remove("show");

    const cards = projects.map((p) => {
        const name = esc(p.name || "Untitled");
        const date = p.updated_at ? new Date(p.updated_at).toLocaleDateString() : "";
        const modelCount = p.model_count || 0;
        const fileInfo = p.original_filename ? `from ${esc(p.original_filename)}` : "No dataset";

        return `
            <div class="project-card" data-project-id="${p.id}" onclick="activateProject('${p.id}')">
                <div class="project-card-name">${name}</div>
                <div class="project-card-meta">
                    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"/></svg>
                    ${fileInfo}
                </div>
                <div class="project-card-meta">
                    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zM4 10h12v6H4v-6z"/></svg>
                    ${date || "—"}
                </div>
                <div class="project-card-footer">
                    <span class="project-card-badge">${modelCount} model${modelCount !== 1 ? "s" : ""}</span>
                    <button class="project-card-delete" onclick="event.stopPropagation(); deleteProject('${p.id}')" title="Delete">&#x2715;</button>
                </div>
            </div>
        `;
    });

    grid.innerHTML = cards.join("");

    // GSAP stagger-in animation
    if (typeof gsap !== "undefined") {
        gsap.from(".project-card", {
            opacity: 0,
            y: 24,
            scale: 0.96,
            duration: 0.4,
            stagger: 0.06,
            ease: "power3.out",
            clearProps: "transform",
        });
    }
}

function showNewProjectModal() {
    const modal = document.getElementById("newProjectModal");
    modal.classList.add("show");

    // Wire up file input change
    document.getElementById("projectFileInput").onchange = function () {
        const text = document.getElementById("projectUploadText");
        text.textContent = this.files.length ? this.files[0].name : "Click to select file";
    };
}

function hideNewProjectModal() {
    document.getElementById("newProjectModal").classList.remove("show");
}

function backToProjects() {
    _activeProjectId = null;
    document.getElementById("trainingFlow").style.display = "none";
    document.getElementById("backToProjectsBtn").style.display = "none";

    const list = document.getElementById("projectList");
    list.style.display = "block";
    list.classList.add("active");

    loadProjects();
}

function populateModelList(models) {
    const container = document.getElementById("modelList");
    const actionBar = document.getElementById("modelCompareAction");

    if (!models.length) {
        container.innerHTML = `<div class="empty-state show" style="padding:40px 0;"><p>No models yet. Train a model to see it here.</p></div>`;
        if (actionBar) actionBar.style.display = "none";
        return;
    }

    const cards = models.map((m) => {
        const mid = esc(m.id || "—");
        const mtype = esc(m.model_type || "—");
        const date = m.created_at ? new Date(m.created_at).toLocaleString() : "—";
        const target = esc(m.target_name || "—");
        const fm = m.final_metrics || {};
        const trainLoss = fm.train_loss != null ? fm.train_loss.toFixed(4) : "—";
        const valLoss = fm.val_loss != null ? fm.val_loss.toFixed(4) : "—";
        const epochs = fm.epochs != null ? fm.epochs : "—";
        const avgTime = fm.avg_epoch_time != null ? fm.avg_epoch_time.toFixed(2) + "s" : "—";

        return `
            <div class="model-export-card">
                <label class="model-compare-cb" title="Select for comparison">
                    <input type="checkbox" class="model-cb" value="${mid}">
                </label>
                <div class="model-export-info">
                    <div class="model-export-name">${mtype} <span class="badge badge-soft">${mid}</span></div>
                    <div class="model-export-meta">Target: ${target} &middot; Created: ${date}</div>
                    <div class="model-export-metrics">
                        <span class="chip">Epochs: ${epochs}</span>
                        <span class="chip">Train Loss: ${trainLoss}</span>
                        <span class="chip">Val Loss: ${valLoss}</span>
                        <span class="chip">Time/Epoch: ${avgTime}</span>
                    </div>
                </div>
                <button class="btn btn-secondary btn-sm" onclick="exportModel('${mid}')">Export</button>
            </div>
        `;
    });

    container.innerHTML = cards.join("");

    // Show action bar and wire checkbox toggle
    if (actionBar) actionBar.style.display = "block";
    document.getElementById("modelCompareResult").style.display = "none";
}

/* =============== Model Selector Dropdown =============== */

function populateModelDropdown(models) {
    const select = document.getElementById("modelSelect");
    if (!select) return;
    select.innerHTML = '<option value="">-- Select a model --</option>';
    models.forEach((m) => {
        const mid = esc(m.id || "");
        const mtype = esc(m.model_type || "unknown");
        const fm = m.final_metrics || {};
        const epochs = fm.epochs != null ? ` (${fm.epochs} epochs)` : "";
        const opt = document.createElement("option");
        opt.value = mid;
        opt.textContent = `${mtype} - ${mid}${epochs}`;
        select.appendChild(opt);
    });
    document.getElementById("modelSelector").style.display = "block";
}

async function onModelSelect(modelId) {
    const badge = document.getElementById("loadedModelBadge");
    if (!modelId) {
        if (badge) badge.style.display = "none";
        return;
    }
    await loadModelToSession(modelId);
}

/* =============== Header Dropdown =============== */

function toggleHeaderMenu() {
    document.getElementById("headerMenu").classList.toggle("open");
}

function goToStep(n) {
    document.querySelectorAll(".step-item").forEach((s) => s.classList.remove("active"));
    document.querySelector(`.step-item[data-step="${n}"]`).classList.add("active");
    document.querySelectorAll(".section").forEach((s) => s.classList.remove("active"));
    document.getElementById("step" + n).classList.add("active");
}

function populateStep2(info) {
    // Stats
    document.getElementById("dataStats").innerHTML = `
        <div class="stat-card"><div class="stat-value">${info.shape[1]}</div><div class="stat-label">Columns</div></div>
        <div class="stat-card"><div class="stat-value">${info.shape[0].toLocaleString()}</div><div class="stat-label">Rows</div></div>
        <div class="stat-card"><div class="stat-value">${Object.keys(info.null_counts).filter((c) => info.null_counts[c] > 0).length}</div><div class="stat-label">Columns with Nulls</div></div>
        <div class="stat-card"><div class="stat-value">${info.columns.length}</div><div class="stat-label">Features</div></div>
    `;

    // Preview table
    const cols = info.columns;
    let html = "<table><thead><tr>";
    cols.forEach((c) => { html += `<th>${esc(c)}</th>`; });
    html += "</tr></thead><tbody>";
    info.sample.forEach((row) => {
        html += "<tr>";
        cols.forEach((c) => { html += `<td>${row[c] !== null && row[c] !== undefined ? esc(String(row[c])) : '<span style="color:var(--danger)">NaN</span>'}</td>`; });
        html += "</tr>";
    });
    html += "</tbody></table>";
    document.getElementById("previewTable").innerHTML = html;

    // Info text
    let infoText = `Dataset Info\n`;
    infoText += `Shape: ${info.shape[0]} rows x ${info.shape[1]} columns\n\n`;
    infoText += `Column dtypes:\n`;
    Object.entries(info.dtypes).forEach(([col, dt]) => {
        const nulls = info.null_counts[col];
        const nullPct = info.null_pcts[col];
        infoText += `  ${col.padEnd(25)} ${dt.padEnd(20)} ${nulls} null (${nullPct}%)\n`;
    });
    document.getElementById("dataInfoText").textContent = infoText;

    // Describe table
    let descHtml = '<div class="table-wrap"><table><thead><tr><th>Metric</th>';
    Object.keys(info.describe).forEach((col) => { descHtml += `<th>${esc(col)}</th>`; });
    descHtml += "</tr></thead><tbody>";
    const metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "unique", "top", "freq"];
    metrics.forEach((m) => {
        descHtml += `<tr><td style="font-weight:500;color:var(--charcoal);">${m}</td>`;
        Object.keys(info.describe).forEach((col) => {
            const val = info.describe[col][m];
            descHtml += `<td>${val !== undefined && val !== null ? (typeof val === "number" ? (Number.isInteger(val) ? val.toLocaleString() : val.toFixed(4)) : esc(String(val))) : "-"}</td>`;
        });
        descHtml += "</tr>";
    });
    descHtml += "</tbody></table></div>";
    document.getElementById("describeTable").innerHTML = descHtml;

    // Viz images
    const vizDiv = document.getElementById("vizImages");
    vizDiv.innerHTML = "";
    if (info.distribution_images) {
        Object.entries(info.distribution_images).forEach(([key, img]) => {
            vizDiv.innerHTML += `<div class="image-card"><img src="data:image/png;base64,${img}" alt="${key}"><div class="caption">Data Distribution</div></div>`;
        });
    }
    if (info.correlation_image) {
        document.getElementById("corrImage").innerHTML = `<div class="image-card"><img src="data:image/png;base64,${info.correlation_image}" alt="Correlation"><div class="caption">Correlation Heatmap</div></div>`;
    }
}

function populateStep3Columns(columns) {
    const div = document.getElementById("columnCheckboxes");
    div.innerHTML = `<div style="margin-bottom:12px;"><label style="font-weight:500;font-size:14px;color:var(--ink);">Columns to drop (optional):</label></div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">`;
    columns.forEach((col) => {
        div.innerHTML += `<label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;padding:4px 10px;border-radius:9999px;background:var(--surface-soft);">
            <input type="checkbox" class="drop-cb" value="${esc(col)}"> ${esc(col)}
        </label>`;
    });
    div.innerHTML += "</div>";
}

function populateTargetCol(columns) {
    const sel = document.getElementById("targetCol");
    sel.innerHTML = '<option value="">-- Select target column --</option>';
    columns.forEach((col) => {
        sel.innerHTML += `<option value="${esc(col)}">${esc(col)}</option>`;
    });
}

function toggleModelParams() {
    const type = document.getElementById("modelType").value;
    document.getElementById("mlpParams").style.display = type === "mlp" ? "block" : "none";
    document.getElementById("cnnParams").style.display = type === "cnn" ? "block" : "none";
    const seqTypes = ["rnn", "lstm", "gru"];
    document.getElementById("seqParams").style.display = seqTypes.includes(type) ? "block" : "none";
    document.getElementById("transformerParams").style.display = type === "transformer" ? "block" : "none";
}

function esc(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

/* =============== Live Training Progress =============== */

/* ── Charts (Chart.js) ───────────────────────────────────── */

let _lossChart = null;
let _metricChart = null;
let _chartData = { epochs: [], trainLoss: [], valLoss: [], trainMetric: [], valMetric: [] };

function _initCharts() {
    const ctx1 = document.getElementById("lossChart");
    const ctx2 = document.getElementById("metricChart");
    if (!ctx1 || !ctx2) return;
    if (_lossChart) { _lossChart.destroy(); _lossChart = null; }
    if (_metricChart) { _metricChart.destroy(); _metricChart = null; }

    _lossChart = new Chart(ctx1, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Train Loss", data: [], borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.08)", fill: true, tension: 0.3, pointRadius: 0 },
                { label: "Val Loss",   data: [], borderColor: "#ff5f56", backgroundColor: "rgba(255,95,86,0.08)", fill: true, tension: 0.3, pointRadius: 0 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { position: "bottom", labels: { boxWidth: 12, padding: 12 } } },
            scales: {
                x: { title: { display: true, text: "Epoch" } },
                y: { title: { display: true, text: "Loss" }, beginAtZero: false },
            },
        },
    });

    _metricChart = new Chart(ctx2, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Train MAE", data: [], borderColor: "#27c93f", backgroundColor: "rgba(39,201,63,0.08)", fill: true, tension: 0.3, pointRadius: 0 },
                { label: "Val MAE",   data: [], borderColor: "#ffbd2e", backgroundColor: "rgba(255,189,46,0.08)", fill: true, tension: 0.3, pointRadius: 0 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { position: "bottom", labels: { boxWidth: 12, padding: 12 } } },
            scales: {
                x: { title: { display: true, text: "Epoch" } },
                y: { title: { display: true, text: "MAE" }, beginAtZero: false },
            },
        },
    });
}

function _appendChartPoint(metrics) {
    _chartData.epochs.push(metrics.epoch);
    _chartData.trainLoss.push(metrics.train_loss);
    _chartData.valLoss.push(metrics.val_loss);
    _chartData.trainMetric.push(metrics.train_metric);
    _chartData.valMetric.push(metrics.val_metric);
}

function _updateCharts() {
    if (!_lossChart || !_metricChart) return;
    _lossChart.data.labels = _chartData.epochs.slice();
    _lossChart.data.datasets[0].data = _chartData.trainLoss.slice();
    _lossChart.data.datasets[1].data = _chartData.valLoss.slice();
    _lossChart.update("none");

    _metricChart.data.labels = _chartData.epochs.slice();
    _metricChart.data.datasets[0].data = _chartData.trainMetric.slice();
    _metricChart.data.datasets[1].data = _chartData.valMetric.slice();
    _metricChart.update("none");
}

function destroyCharts() {
    if (_lossChart) { _lossChart.destroy(); _lossChart = null; }
    if (_metricChart) { _metricChart.destroy(); _metricChart = null; }
    _chartData = { epochs: [], trainLoss: [], valLoss: [], trainMetric: [], valMetric: [] };
}

/* ── Panel controls ─────────────────────────────────────── */

function initTrainingProgress(total) {
    const panel = document.getElementById("trainingProgress");
    panel.style.display = "block";

    // Show chart area
    const chartsDiv = document.getElementById("trainingCharts");
    if (chartsDiv) chartsDiv.style.display = "grid";

    document.getElementById("progressCurrentEpoch").textContent = "0";
    document.getElementById("progressTotalEpochs").textContent = total;
    document.getElementById("progressBar").style.width = "0%";
    updateProgressMetrics({
        train_loss: "—", val_loss: "—", train_metric: "—", val_metric: "—",
    });

    destroyCharts();
    _initCharts();
}

function updateTrainingProgress(metrics) {
    const pct = Math.round((metrics.epoch / metrics.total_epochs) * 100);
    document.getElementById("progressCurrentEpoch").textContent = metrics.epoch;
    document.getElementById("progressBar").style.width = Math.min(pct, 100) + "%";
    updateProgressMetrics(metrics);
    _appendChartPoint(metrics);
    _updateCharts();
}

function updateProgressMetrics(m) {
    document.getElementById("liveTrainLoss").textContent =
        typeof m.train_loss === "number" ? m.train_loss.toFixed(6) : m.train_loss;
    document.getElementById("liveValLoss").textContent =
        typeof m.val_loss === "number" ? m.val_loss.toFixed(6) : m.val_loss;
    document.getElementById("liveTrainMetric").textContent =
        typeof m.train_metric === "number" ? m.train_metric.toFixed(6) : m.train_metric;
    document.getElementById("liveValMetric").textContent =
        typeof m.val_metric === "number" ? m.val_metric.toFixed(6) : m.val_metric;
}

function hideTrainingProgress() {
    document.getElementById("trainingProgress").style.display = "none";
}

function showTrainError(msg) {
    const div = document.getElementById("trainError");
    div.textContent = msg;
    div.style.display = "block";
}
