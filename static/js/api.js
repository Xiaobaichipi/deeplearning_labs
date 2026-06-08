/* API Call Functions */

async function handleUpload(file) {
    const status = document.getElementById("uploadStatus");
    status.style.display = "block";
    status.innerHTML = '<div class="loading-overlay"><span class="spinner"></span> Uploading and processing...</div>';

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) {
            status.innerHTML = `<div class="alert alert-error">${data.error}</div>`;
            return;
        }
        status.innerHTML = `<div class="alert alert-success">Upload successful! File: ${data.data.filename} (${data.data.shape[0]} rows x ${data.data.shape[1]} columns)</div>`;
        currentDataInfo = data.data;
        populateStep2(data.data);
        populateStep3Columns(data.data.columns);
        populateTargetCol(data.data.columns);
        goToStep(2);
    } catch (err) {
        status.innerHTML = `<div class="alert alert-error">Upload failed: ${err.message}</div>`;
    }
}

async function runClean() {
    const dropCols = Array.from(document.querySelectorAll(".drop-cb:checked")).map((cb) => cb.value);
    const params = {
        drop_duplicates: document.getElementById("toggleDedup").classList.contains("active"),
        drop_columns: dropCols.length ? dropCols : null,
        handle_outliers: document.getElementById("toggleOutliers").classList.contains("active"),
        outlier_factor: parseFloat(document.getElementById("outlierFactor").value) || 1.5,
    };

    try {
        const res = await fetch("/api/data/clean", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });
        const data = await res.json();
        if (data.error) { alert("Error: " + data.error); return; }

        currentDataInfo = data.data;
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
            cols.forEach((c) => { html += `<td>${row[c] !== null && row[c] !== undefined ? esc(String(row[c])) : '<span style="color:var(--danger)">NaN</span>'}</td>`; });
            html += "</tr>";
        });
        html += "</tbody></table></div>";
        document.getElementById("cleanTable").style.display = "block";
        document.getElementById("cleanTable").innerHTML = html;
    } catch (err) {
        alert("Error: " + err.message);
    }
}

async function runFill() {
    const strategy = document.getElementById("fillStrategy").value;
    const params = { strategy };
    if (strategy === "constant") {
        params.fill_value = document.getElementById("fillConstantValue").value || null;
    }

    try {
        const res = await fetch("/api/data/fill", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });
        const data = await res.json();
        if (data.error) { alert("Error: " + data.error); return; }

        currentDataInfo = data.data;
        const reportDiv = document.getElementById("cleanReport");
        reportDiv.style.display = "block";
        reportDiv.innerHTML = `<div class="card"><h3 class="card-title">Fill Report</h3><ul class="report-list">${data.report.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`;

        document.getElementById("cleanStats").style.display = "grid";
        document.getElementById("cleanStats").innerHTML = `
            <div class="stat-card"><div class="stat-value">${data.data.shape[1]}</div><div class="stat-label">Columns</div></div>
            <div class="stat-card"><div class="stat-value">${data.data.shape[0].toLocaleString()}</div><div class="stat-label">Rows</div></div>
            <div class="stat-card"><div class="stat-value">${Object.values(data.data.null_counts).filter((n) => n > 0).length}</div><div class="stat-label">Columns with Nulls Remaining</div></div>
        `;
    } catch (err) {
        alert("Error: " + err.message);
    }
}

async function startTraining() {
    const targetCol = document.getElementById("targetCol").value;
    if (!targetCol) { alert("Please select a target column"); return; }

    const btn = document.getElementById("trainBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Training...';

    const params = {
        target_col: targetCol,
        test_size: parseFloat(document.getElementById("testSize").value) || 0.2,
        model_type: document.getElementById("modelType").value,
        hidden_layers: document.getElementById("hiddenLayers").value,
        hidden_channels: parseInt(document.getElementById("hiddenChannels").value) || 64,
        kernel_size: parseInt(document.getElementById("kernelSize").value) || 3,
        hidden_size: parseInt(document.getElementById("rnnHiddenSize").value) || 64,
        num_layers: parseInt(document.getElementById("rnnNumLayers").value) || 2,
        bidirectional: document.getElementById("toggleBidirectional").classList.contains("active"),
        d_model: parseInt(document.getElementById("transDModel").value) || 64,
        nhead: parseInt(document.getElementById("transNhead").value) || 4,
        dim_feedforward: parseInt(document.getElementById("transDimFeedforward").value) || 256,
        learning_rate: parseFloat(document.getElementById("learningRate").value) || 0.001,
        batch_size: parseInt(document.getElementById("batchSize").value) || 32,
        epochs: parseInt(document.getElementById("epochs").value) || 50,
        dropout: parseFloat(document.getElementById("dropout").value) || 0.2,
        patience: parseInt(document.getElementById("patience").value) || 10,
        normalization: document.getElementById("normalization").value,
    };

    // Hide previous results, show progress panel
    document.getElementById("trainError").style.display = "none";

    document.getElementById("trainingSummary").style.display = "none";

    try {
        // Step 1: POST params to setup endpoint
        const setupRes = await fetch("/api/train/setup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });
        const setupData = await setupRes.json();
        if (setupData.error) { alert("Error: " + setupData.error); return; }

        // Step 2: Show live progress panel
        const totalEpochs = parseInt(document.getElementById("epochs").value) || 50;
        initTrainingProgress(totalEpochs);
        btn.innerHTML = '<span class="spinner"></span> Training...';

        // Step 3: Open SSE stream
        const evtSource = new EventSource("/api/train/stream");

        evtSource.addEventListener("progress", function (e) {
            const m = JSON.parse(e.data);
            updateTrainingProgress(m);
        });

        evtSource.addEventListener("complete", function (e) {
            evtSource.close();
            const data = JSON.parse(e.data);

            // Show training info
            document.getElementById("trainingInfo").innerHTML = `
                <div class="stat-card"><div class="stat-value">${data.final_metrics.epochs}</div><div class="stat-label">Epochs Completed</div></div>
                <div class="stat-card"><div class="stat-value">${data.final_metrics.train_loss.toFixed(4)}</div><div class="stat-label">Final Training Loss</div></div>
                <div class="stat-card"><div class="stat-value">${data.final_metrics.val_loss.toFixed(4)}</div><div class="stat-label">Final Validation Loss</div></div>
                <div class="stat-card"><div class="stat-value">${data.train_size}</div><div class="stat-label">Training Samples</div></div>
                <div class="stat-card"><div class="stat-value">${data.test_size}</div><div class="stat-label">Test Samples</div></div>
                <div class="stat-card"><div class="stat-value">${data.task_type}</div><div class="stat-label">Task Type</div></div>
            `;

            document.getElementById("trainingSummary").style.display = "block";
            document.getElementById("summaryMetrics").innerHTML = `
                <div class="metric-card"><div class="metric-value">${data.final_metrics.train_loss.toFixed(4)}</div><div class="metric-label">Train Loss</div></div>
                <div class="metric-card"><div class="metric-value">${data.final_metrics.val_loss.toFixed(4)}</div><div class="metric-label">Val Loss</div></div>
                <div class="metric-card"><div class="metric-value">${data.final_metrics.epochs}</div><div class="metric-label">Epochs</div></div>
                <div class="metric-card"><div class="metric-value">${data.task_type}</div><div class="metric-label">Task</div></div>
            `;

            goToStep(6);
            btn.disabled = false;
            btn.textContent = "Start Training";
        });

        evtSource.addEventListener("error", function (e) {
            evtSource.close();
            let msg = "Training connection lost";
            try { msg = JSON.parse(e.data).error || msg; } catch (_) {}
            showTrainError(msg);
            btn.disabled = false;
            btn.textContent = "Start Training";
        });
    } catch (err) {
        showTrainError(err.message);
        btn.disabled = false;
        btn.textContent = "Start Training";
    }
}

async function runEvaluation() {
    const btn = document.getElementById("evalBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Evaluating...';

    try {
        const res = await fetch("/api/evaluate", { method: "POST" });
        const data = await res.json();
        if (data.error) { alert("Error: " + data.error); return; }

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
            Object.entries(evalData.images).forEach(([key, img]) => {
                const label_map = {
                    confusion_matrix: "Confusion Matrix",
                    roc_curve: "ROC Curve",
                    pred_vs_true: "Predictions vs True Values",
                    residuals: "Residual Distribution",
                };
                imgDiv.innerHTML += `<div class="image-card"><img src="data:image/png;base64,${img}" alt="${key}"><div class="caption">${label_map[key] || key}</div></div>`;
            });
        }
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Run Evaluation";
    }
}

async function runValidation() {
    const btn = document.getElementById("validBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running...';

    try {
        const nSplits = parseInt(document.getElementById("cvFolds").value) || 5;
        const res = await fetch("/api/validate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ n_splits: nSplits }),
        });
        const data = await res.json();
        if (data.error) { alert("Error: " + data.error); return; }

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
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Run Cross Validation";
    }
}

async function runPredict() {
    const btn = document.getElementById("predBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Predicting...';

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ use_test: true }),
        });
        const data = await res.json();
        if (data.error) { alert("Error: " + data.error); return; }

        document.getElementById("predResults").style.display = "block";

        // Show chart if available
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

        // Show download buttons
        document.getElementById("predDownload").style.display = "flex";
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Generate Predictions";
    }
}

function downloadPredictions(format) {
    window.location.href = `/api/predict/download?source=test&format=${format}`;
}

function downloadHistory(format) {
    window.location.href = `/api/train/history/download?format=${format}`;
}

async function resetAll() {
    if (!confirm("Reset all data and start over?")) return;
    try {
        await fetch("/api/reset", { method: "POST" });
        currentDataInfo = null;
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
    } catch (err) {
        alert("Error: " + err.message);
    }
}
