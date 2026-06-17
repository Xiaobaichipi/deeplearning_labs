/* UI Rendering Functions — pure DOM manipulation, no API calls */

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

    if (typeof gsap !== "undefined") {
        gsap.from(".project-card", {
            opacity: 0, y: 24, scale: 0.96, duration: 0.4,
            stagger: 0.06, ease: "power3.out", clearProps: "transform",
        });
    }
}

function showNewProjectModal() {
    const modal = document.getElementById("newProjectModal");
    modal.classList.add("show");
    document.getElementById("projectFileInput").onchange = function () {
        const text = document.getElementById("projectUploadText");
        text.textContent = this.files.length ? this.files[0].name : "Click to select file";
    };
}

function hideNewProjectModal() {
    document.getElementById("newProjectModal").classList.remove("show");
}

function showProjectList() {
    document.getElementById("trainingFlow").style.display = "none";
    document.getElementById("backToProjectsBtn").style.display = "none";
    const list = document.getElementById("projectList");
    list.style.display = "block";
    list.classList.add("active");
}

function hideProjectList() {
    document.getElementById("projectList").classList.remove("active");
    document.getElementById("projectList").style.display = "none";
    document.getElementById("trainingFlow").style.display = "block";
    document.getElementById("backToProjectsBtn").style.display = "inline-flex";
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
    if (actionBar) actionBar.style.display = "block";
    document.getElementById("modelCompareResult").style.display = "none";
}

/* =============== Model Selector Dropdown =============== */

function populateModelDropdown(models, taskType) {
    const select = document.getElementById("modelSelect");
    if (!select) return [];
    const filtered = filterModelsByTask(models, taskType);
    select.innerHTML = '<option value="">-- Select a model --</option>';

    if (models.length === 0) {
        // No models in project at all → hide selector
        document.getElementById("modelSelector").style.display = "none";
        return [];
    }

    if (filtered.length === 0) {
        // Models exist but none match the current task type → show message
        const taskLabel = taskType === "time_series" ? "time series" : "general";
        const opt = document.createElement("option");
        opt.disabled = true;
        opt.textContent = `No ${taskLabel} models trained yet`;
        select.appendChild(opt);
        document.getElementById("modelSelector").style.display = "block";
        return [];
    }

    filtered.forEach((m) => {
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
    return filtered;
}

function showLoadedModelBadge(ok, text) {
    const badge = document.getElementById("loadedModelBadge");
    if (!badge) return;
    if (ok) {
        badge.textContent = "Loaded: " + text;
        badge.style.display = "inline-block";
    } else {
        badge.style.display = "none";
    }
}

/* =============== Header Dropdown =============== */

function toggleHeaderMenu() {
    document.getElementById("headerMenu").classList.toggle("open");
}

/* =============== Navigation =============== */

function goToStep(n) {
    document.querySelectorAll(".step-item").forEach((s) => s.classList.remove("active"));
    document.querySelector(`.step-item[data-step="${n}"]`).classList.add("active");
    document.querySelectorAll(".section").forEach((s) => s.classList.remove("active"));
    document.getElementById("step" + n).classList.add("active");
}

/* =============== Task Config (Time Series) =============== */

function onTaskTypeChange() {
    const type = document.getElementById("taskTypeSelect").value;
    const fields = document.getElementById("tsConfigFields");
    fields.style.display = type === "time_series" ? "flex" : "none";
    // When switching back to general, clear the badge
    if (type !== "time_series") {
        document.getElementById("taskConfigBadge").style.display = "none";
    }
}

function populateTimeColSelect(columns) {
    const sel = document.getElementById("timeColSelect");
    if (!sel) return;
    sel.innerHTML = '<option value="">-- Select --</option>';
    columns.forEach((col) => {
        sel.innerHTML += `<option value="${esc(col)}">${esc(col)}</option>`;
    });
}

function showTaskConfigSaved() {
    const badge = document.getElementById("taskConfigBadge");
    if (badge) {
        badge.style.display = "inline-block";
        badge.textContent = "Saved";
        setTimeout(() => { badge.style.display = "none"; }, 3000);
    }
}

/* =============== Step 2: Data Exploration =============== */

function populateStep2(info) {
    document.getElementById("dataStats").innerHTML = `
        <div class="stat-card"><div class="stat-value">${info.shape[1]}</div><div class="stat-label">Columns</div></div>
        <div class="stat-card"><div class="stat-value">${info.shape[0].toLocaleString()}</div><div class="stat-label">Rows</div></div>
        <div class="stat-card"><div class="stat-value">${Object.keys(info.null_counts).filter((c) => info.null_counts[c] > 0).length}</div><div class="stat-label">Columns with Nulls</div></div>
        <div class="stat-card"><div class="stat-value">${info.columns.length}</div><div class="stat-label">Features</div></div>
    `;

    const cols = info.columns;
    let html = "<table><thead><tr>";
    cols.forEach((c) => { html += `<th>${esc(c)}</th>`; });
    html += "</tr></thead><tbody>";
    info.sample.forEach((row) => {
        html += "<tr>";
        cols.forEach((c) => {
            html += `<td>${row[c] !== null && row[c] !== undefined ? esc(String(row[c])) : '<span style="color:var(--danger)">NaN</span>'}</td>`;
        });
        html += "</tr>";
    });
    html += "</tbody></table>";
    document.getElementById("previewTable").innerHTML = html;

    let infoText = "Dataset Info\n";
    infoText += `Shape: ${info.shape[0]} rows x ${info.shape[1]} columns\n\n`;
    infoText += "Column dtypes:\n";
    Object.entries(info.dtypes).forEach(([col, dt]) => {
        const nulls = info.null_counts[col];
        const nullPct = info.null_pcts[col];
        infoText += `  ${col.padEnd(25)} ${dt.padEnd(20)} ${nulls} null (${nullPct}%)\n`;
    });
    document.getElementById("dataInfoText").textContent = infoText;

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
    document.getElementById("autoformerParams").style.display = type === "autoformer" ? "block" : "none";
    document.getElementById("informerParams").style.display = type === "informer" ? "block" : "none";
    document.getElementById("crossformerParams").style.display = type === "crossformer" ? "block" : "none";
    document.getElementById("etsformerParams").style.display = type === "etsformer" ? "block" : "none";
    document.getElementById("fedformerParams").style.display = type === "fedformer" ? "block" : "none";
    document.getElementById("filmParams").style.display = type === "film" ? "block" : "none";
    document.getElementById("dlinearParams").style.display = type === "dlinear" ? "block" : "none";
}

function esc(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

/* =============== Result Display Functions =============== */

function showUploadResult(data) {
    const status = document.getElementById("uploadStatus");
    status.style.display = "block";
    status.innerHTML = `<div class="alert alert-success">Upload successful! File: ${data.filename} (${data.shape[0]} rows x ${data.shape[1]} columns)</div>`;
}

function showUploadError(msg) {
    const status = document.getElementById("uploadStatus");
    status.style.display = "block";
    status.innerHTML = `<div class="alert alert-error">${msg}</div>`;
}

function showUploadLoading() {
    const status = document.getElementById("uploadStatus");
    status.style.display = "block";
    status.innerHTML = '<div class="loading-overlay"><span class="spinner"></span> Uploading and processing...</div>';
}

function showCleanResult(data) {
    const reportDiv = document.getElementById("cleanReport");
    reportDiv.style.display = "block";
    reportDiv.innerHTML = `<div class="card"><h3 class="card-title">Cleaning Report</h3><ul class="report-list">${data.report.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`;

    document.getElementById("cleanStats").style.display = "grid";
    document.getElementById("cleanStats").innerHTML = `
        <div class="stat-card"><div class="stat-value">${data.data.shape[1]}</div><div class="stat-label">Columns</div></div>
        <div class="stat-card"><div class="stat-value">${data.data.shape[0].toLocaleString()}</div><div class="stat-label">Rows</div></div>
    `;

    const cols = data.data.columns;
    let html = '<div class="table-wrap"><table><thead><tr>';
    cols.forEach((c) => { html += `<th>${esc(c)}</th>`; });
    html += "</tr></thead><tbody>";
    data.data.sample.slice(0, 20).forEach((row) => {
        html += "<tr>";
        cols.forEach((c) => {
            html += `<td>${row[c] !== null && row[c] !== undefined ? esc(String(row[c])) : '<span style="color:var(--danger)">NaN</span>'}</td>`;
        });
        html += "</tr>";
    });
    html += "</tbody></table></div>";
    document.getElementById("cleanTable").style.display = "block";
    document.getElementById("cleanTable").innerHTML = html;
}

function showFillResult(data) {
    const reportDiv = document.getElementById("cleanReport");
    reportDiv.style.display = "block";
    reportDiv.innerHTML = `<div class="card"><h3 class="card-title">Fill Report</h3><ul class="report-list">${data.report.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`;

    document.getElementById("cleanStats").style.display = "grid";
    document.getElementById("cleanStats").innerHTML = `
        <div class="stat-card"><div class="stat-value">${data.data.shape[1]}</div><div class="stat-label">Columns</div></div>
        <div class="stat-card"><div class="stat-value">${data.data.shape[0].toLocaleString()}</div><div class="stat-label">Rows</div></div>
        <div class="stat-card"><div class="stat-value">${Object.values(data.data.null_counts).filter((n) => n > 0).length}</div><div class="stat-label">Columns with Nulls Remaining</div></div>
    `;
}

function showTrainingComplete(data) {
    document.getElementById("trainingInfo").innerHTML = `
        <div class="stat-card"><div class="stat-value">${data.final_metrics.epochs}</div><div class="stat-label">Epochs Completed</div></div>
        <div class="stat-card"><div class="stat-value">${data.final_metrics.train_loss.toFixed(4)}</div><div class="stat-label">Final Training Loss</div></div>
        <div class="stat-card"><div class="stat-value">${data.final_metrics.val_loss.toFixed(4)}</div><div class="stat-label">Final Validation Loss</div></div>
        <div class="stat-card"><div class="stat-value">${data.final_metrics.avg_epoch_time.toFixed(2)}s</div><div class="stat-label">Avg Time / Epoch</div></div>
        <div class="stat-card"><div class="stat-value">${data.train_size}</div><div class="stat-label">Training Samples</div></div>
        <div class="stat-card"><div class="stat-value">${data.test_size}</div><div class="stat-label">Test Samples</div></div>
        <div class="stat-card"><div class="stat-value">${data.task_type}</div><div class="stat-label">Task Type</div></div>
    `;
    document.getElementById("trainingSummary").style.display = "block";
    document.getElementById("summaryMetrics").innerHTML = `
        <div class="metric-card"><div class="metric-value">${data.final_metrics.train_loss.toFixed(4)}</div><div class="metric-label">Train Loss</div></div>
        <div class="metric-card"><div class="metric-value">${data.final_metrics.val_loss.toFixed(4)}</div><div class="metric-label">Val Loss</div></div>
        <div class="metric-card"><div class="metric-value">${data.final_metrics.epochs}</div><div class="metric-label">Epochs</div></div>
        <div class="metric-card"><div class="metric-value">${data.final_metrics.avg_epoch_time.toFixed(2)}s</div><div class="metric-label">Avg Time / Epoch</div></div>
        <div class="metric-card"><div class="metric-value">${data.task_type}</div><div class="metric-label">Task</div></div>
    `;
}

function showEvalResult(data) {
    const evalData = data.evaluation;
    const metricsDiv = document.getElementById("evalMetrics");
    metricsDiv.style.display = "grid";
    metricsDiv.innerHTML = "";

    if (evalData.task_type === "classification") {
        metricsDiv.innerHTML = `
            <div class="metric-card"><div class="metric-value">${(evalData.accuracy * 100).toFixed(2)}%</div><div class="metric-label">Accuracy</div></div>
            <div class="metric-card"><div class="metric-value">${evalData.precision.toFixed(4)}</div><div class="metric-label">Precision</div></div>
            <div class="metric-card"><div class="metric-value">${evalData.recall.toFixed(4)}</div><div class="metric-label">Recall</div></div>
            <div class="metric-card"><div class="metric-value">${evalData.f1_score.toFixed(4)}</div><div class="metric-label">F1 Score</div></div>
        `;
    } else {
        metricsDiv.innerHTML = `
            <div class="metric-card"><div class="metric-value">${evalData.mse.toFixed(4)}</div><div class="metric-label">MSE</div></div>
            <div class="metric-card"><div class="metric-value">${evalData.rmse.toFixed(4)}</div><div class="metric-label">RMSE</div></div>
            <div class="metric-card"><div class="metric-value">${evalData.mae.toFixed(4)}</div><div class="metric-label">MAE</div></div>
            <div class="metric-card"><div class="metric-value">${evalData.r2.toFixed(4)}</div><div class="metric-label">R² Score</div></div>
        `;
    }

    const imgDiv = document.getElementById("evalImages");
    imgDiv.innerHTML = "";
    if (evalData.images) {
        const label_map = {
            confusion_matrix: "Confusion Matrix",
            roc_curve: "ROC Curve",
            pred_vs_true: "Predictions vs True Values",
            residuals: "Residual Distribution",
        };
        Object.entries(evalData.images).forEach(([key, img]) => {
            imgDiv.innerHTML += `<div class="image-card"><img src="data:image/png;base64,${img}" alt="${key}"><div class="caption">${label_map[key] || key}</div></div>`;
        });
    }
}

function showCVResult(data) {
    const div = document.getElementById("cvResults");
    div.style.display = "block";
    div.innerHTML = `
        <div class="card">
            <h3 class="card-title">Cross-Validation Results (${data.n_splits}-fold)</h3>
            <div class="metrics-grid">
                <div class="metric-card"><div class="metric-value">${(data.mean_score * 100).toFixed(2)}%</div><div class="metric-label">Mean Score</div></div>
                <div class="metric-card"><div class="metric-value">${(data.std_score * 100).toFixed(2)}%</div><div class="metric-label">Std Dev</div></div>
            </div>
            <div style="margin-top:12px;font-size:14px;">
                <strong style="color:var(--ink);">Per-fold scores:</strong>
                <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;">
                    ${data.cv_scores.map((s, i) => `<span class="chip">Fold ${i + 1}: ${(s * 100).toFixed(2)}%</span>`).join("")}
                </div>
            </div>
        </div>
    `;
}

function showPredResult(data) {
    document.getElementById("predResults").style.display = "block";

    const chartDiv = document.getElementById("predChart");
    if (data.plot_image) {
        chartDiv.style.display = "grid";
        let html = `<div class="image-card"><img src="data:image/png;base64,${data.plot_image}" alt="Predictions vs True (Scatter)"><div class="caption">Scatter: Predictions vs True Values</div></div>`;
        if (data.line_plot_image) {
            html += `<div class="image-card"><img src="data:image/png;base64,${data.line_plot_image}" alt="Predictions vs True (Line)"><div class="caption">Line: Predictions vs True Values</div></div>`;
        }
        chartDiv.innerHTML = html;
    } else {
        chartDiv.style.display = "none";
    }

    document.getElementById("predDownload").style.display = "flex";
}

function resetAllUI() {
    document.querySelectorAll(".step-item").forEach((s) => s.classList.remove("active"));
    document.querySelector('.step-item[data-step="1"]').classList.add("active");
    document.querySelectorAll(".section").forEach((s) => s.classList.remove("active"));
    document.getElementById("step1").classList.add("active");
    document.getElementById("uploadStatus").style.display = "none";
    document.getElementById("cleanReport").style.display = "none";
    document.getElementById("trainingSummary").style.display = "none";
    document.getElementById("trainingProgress").style.display = "none";
    destroyCharts();
    document.getElementById("trainError").style.display = "none";
    document.getElementById("evalMetrics").style.display = "none";
    document.getElementById("predResults").style.display = "none";
    document.getElementById("predChart").style.display = "none";
    document.getElementById("predDownload").style.display = "none";
    document.getElementById("cvResults").style.display = "none";
    document.getElementById("dataStats").innerHTML = "";
    document.getElementById("previewTable").innerHTML = "";
    document.getElementById("evalImages").innerHTML = "";
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

/* =============== Window Exports (for Vitest) =============== */
window.populateProjectGrid = populateProjectGrid;
window.showNewProjectModal = showNewProjectModal;
window.hideNewProjectModal = hideNewProjectModal;
window.showProjectList = showProjectList;
window.hideProjectList = hideProjectList;
window.populateModelList = populateModelList;
window.populateModelDropdown = populateModelDropdown;
window.showLoadedModelBadge = showLoadedModelBadge;
window.toggleHeaderMenu = toggleHeaderMenu;
window.goToStep = goToStep;
window.onTaskTypeChange = onTaskTypeChange;
window.populateTimeColSelect = populateTimeColSelect;
window.showTaskConfigSaved = showTaskConfigSaved;
window.populateStep2 = populateStep2;
window.populateStep3Columns = populateStep3Columns;
window.populateTargetCol = populateTargetCol;
window.toggleModelParams = toggleModelParams;
window.esc = esc;
window.showUploadResult = showUploadResult;
window.showUploadError = showUploadError;
window.showUploadLoading = showUploadLoading;
window.showCleanResult = showCleanResult;
window.showFillResult = showFillResult;
window.showTrainingComplete = showTrainingComplete;
window.showEvalResult = showEvalResult;
window.showCVResult = showCVResult;
window.showPredResult = showPredResult;
window.resetAllUI = resetAllUI;
window._initCharts = _initCharts;
window._appendChartPoint = _appendChartPoint;
window._updateCharts = _updateCharts;
window.destroyCharts = destroyCharts;
window.initTrainingProgress = initTrainingProgress;
window.updateTrainingProgress = updateTrainingProgress;
window.updateProgressMetrics = updateProgressMetrics;
window.hideTrainingProgress = hideTrainingProgress;
window.showTrainError = showTrainError;
