/* UI Rendering Functions */

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
