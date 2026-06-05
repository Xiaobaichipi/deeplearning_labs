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

function initTrainingProgress(total) {
    document.getElementById("trainingProgress").style.display = "block";
    document.getElementById("progressCurrentEpoch").textContent = "0";
    document.getElementById("progressTotalEpochs").textContent = total;
    document.getElementById("progressBar").style.width = "0%";
    updateProgressMetrics({
        train_loss: "—", val_loss: "—", train_metric: "—", val_metric: "—",
    });
}

function updateTrainingProgress(metrics) {
    const pct = Math.round((metrics.epoch / metrics.total_epochs) * 100);
    document.getElementById("progressCurrentEpoch").textContent = metrics.epoch;
    document.getElementById("progressBar").style.width = Math.min(pct, 100) + "%";
    updateProgressMetrics(metrics);
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
