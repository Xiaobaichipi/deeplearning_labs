/* DeepLearning Labs — App Initialization & Orchestration */

const DEFAULTS = JSON.parse(document.getElementById("config-data").textContent);

let currentDataInfo = null;
let _activeProjectId = null;

/* =============== Header Dropdown =============== */
document.addEventListener("click", function (e) {
    const menu = document.getElementById("headerMenu");
    const btn = document.getElementById("headerMenuBtn");
    if (menu.classList.contains("open") && !menu.contains(e.target) && e.target !== btn && !btn.contains(e.target)) {
        menu.classList.remove("open");
    }
});

/* =============== Navigation =============== */
document.querySelectorAll(".step-item").forEach((item) => {
    item.addEventListener("click", () => {
        const step = item.dataset.step;
        document.querySelectorAll(".step-item").forEach((s) => s.classList.remove("active"));
        item.classList.add("active");
        document.querySelectorAll(".section").forEach((s) => s.classList.remove("active"));
        document.getElementById("step" + step).classList.add("active");

        if (step === "6") {
            const sel = document.getElementById("modelSelector");
            const select = document.getElementById("modelSelect");
            if (sel && select && select.options.length > 1) sel.style.display = "block";
        }
    });
});

/* =============== Tabs =============== */
document.addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (!tab) return;
    const parent = tab.closest(".tabs");
    if (!parent) return;
    parent.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const tabId = tab.dataset.tab;
    const contentParent = parent.parentElement;
    contentParent.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    const target = contentParent.querySelector("#tab" + tabId.charAt(0).toUpperCase() + tabId.slice(1));
    if (target) target.classList.add("active");

    if (tabId === "models" && _activeProjectId) {
        loadProjectModels();
    }
});

/* =============== Project Init =============== */
document.addEventListener("DOMContentLoaded", function () {
    loadProjects();
});

/* =============== Upload =============== */
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
});
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleUpload(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleUpload(fileInput.files[0]);
});

/* =============== Step 3: Toggle Controls =============== */
document.getElementById("toggleOutliers").addEventListener("click", function () {
    document.getElementById("outlierOptions").style.display = this.classList.contains("active") ? "block" : "none";
});
document.getElementById("fillStrategy").addEventListener("change", function () {
    document.getElementById("fillConstantGroup").style.display = this.value === "constant" ? "block" : "none";
});

/* ================================================================
   Orchestration — chain API calls → UI rendering
   ================================================================ */

// ── Data ────────────────────────────────────────────────────────────

async function handleUpload(file) {
    showUploadLoading();
    try {
        const data = await _uploadFile(file);
        currentDataInfo = data;
        showUploadResult(data);
        populateStep2(data);
        populateStep3Columns(data.columns);
        populateTargetCol(data.columns);
        populateTimeColSelect(data.columns);
        await loadTaskConfig();
        goToStep(2);
    } catch (err) {
        showUploadError(err.message);
    }
}

// ── Task Config ─────────────────────────────────────────────────────

async function loadTaskConfig() {
    try {
        const config = await _getTaskConfig();
        const typeSelect = document.getElementById("taskTypeSelect");
        if (config.task_type) {
            typeSelect.value = config.task_type;
            onTaskTypeChange();
            if (config.task_type === "time_series") {
                if (config.time_col) document.getElementById("timeColSelect").value = config.time_col;
                if (config.seq_len) document.getElementById("seqLenInput").value = config.seq_len;
                if (config.pred_len) document.getElementById("predLenInput").value = config.pred_len;
                if (config.label_len !== undefined) document.getElementById("labelLenInput").value = config.label_len;
                if (config.time_granularity) document.getElementById("granularitySelect").value = config.time_granularity;
            }
        }
        updateModelOptions(config.task_type || "general");
    } catch (_) {}
}

async function applyTaskConfig() {
    const btn = document.getElementById("applyTaskConfigBtn");
    btn.disabled = true;
    btn.textContent = "Applying...";

    const config = {
        task_type: document.getElementById("taskTypeSelect").value,
        time_col: document.getElementById("timeColSelect").value,
        seq_len: parseInt(document.getElementById("seqLenInput").value) || 10,
        pred_len: parseInt(document.getElementById("predLenInput").value) || 1,
        label_len: parseInt(document.getElementById("labelLenInput").value) || 0,
        time_granularity: document.getElementById("granularitySelect").value || "auto",
    };

    try {
        await _setTaskConfig(config);
        showTaskConfigSaved();
        updateModelOptions(config.task_type);
        refreshModelDropdown();
        // Clear downstream UI state (split/model/history removed on server)
        document.getElementById("trainingSummary").style.display = "none";
        document.getElementById("trainingProgress").style.display = "none";
        document.getElementById("cvResults").style.display = "none";
        document.getElementById("predResults").style.display = "none";
        document.getElementById("evalMetrics").style.display = "none";
        destroyCharts();
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Apply & Refresh";
    }
}

function updateModelOptions(taskType) {
    const sel = document.getElementById("modelType");
    const isTimeSeries = taskType === "time_series";

    const allOptions = {
        "mlp": "MLP (Fully Connected)",
        "cnn": "CNN (1D Convolutional)",
        "rnn": "RNN (Vanilla RNN)",
        "lstm": "LSTM (Long Short-Term Memory)",
        "gru": "GRU (Gated Recurrent Unit)",
        "transformer": "Transformer (Encoder)",
        "autoformer": "Autoformer (Long-term Forecast)",
        "informer": "Informer (ProbSparse Attention)",
        "crossformer": "Crossformer (Two-Stage Attention)",
        "etsformer": "ETSformer (Exp Smoothing Transformer)",
        "fedformer": "FEDformer (Frequency Enhanced Decomp Transformer)",
        "film": "FiLM (Frequency-enhanced Legendre Memory)",
        "dlinear": "DLinear (Decomposition Linear)",
    };

    const tsModels = ["rnn", "lstm", "gru", "transformer", "autoformer", "informer", "crossformer", "etsformer", "fedformer", "film", "dlinear"];
    const generalModels = ["mlp", "cnn"];

    sel.innerHTML = "";
    const visible = isTimeSeries ? tsModels : generalModels;
    Object.entries(allOptions).forEach(([val, label]) => {
        if (visible.includes(val)) {
            const opt = document.createElement("option");
            opt.value = val;
            opt.textContent = label;
            sel.appendChild(opt);
        }
    });

    toggleModelParams();
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
        const data = await _cleanData(params);
        currentDataInfo = data.data;
        showCleanResult(data);
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
        const data = await _fillData(params);
        currentDataInfo = data.data;
        showFillResult(data);
    } catch (err) {
        alert("Error: " + err.message);
    }
}

// ── Training ────────────────────────────────────────────────────────

async function startTraining() {
    const targetCol = document.getElementById("targetCol").value;
    if (!targetCol) { alert("Please select a target column"); return; }

    const btn = document.getElementById("trainBtn");

    // Sync task config to server before training, so the server uses the
    // current task type (fixes model selector disappearing when user changes
    // task type and clicks Train without applying first).
    await _setTaskConfig({
        task_type: document.getElementById("taskTypeSelect").value,
        time_col: document.getElementById("timeColSelect").value,
        seq_len: parseInt(document.getElementById("seqLenInput").value) || 10,
        pred_len: parseInt(document.getElementById("predLenInput").value) || 1,
        label_len: parseInt(document.getElementById("labelLenInput").value) || 0,
        time_granularity: document.getElementById("granularitySelect").value || "auto",
    });
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Training...';

    const params = {
        target_col: targetCol,
        test_size: parseFloat(document.getElementById("testSize").value) || DEFAULTS.training.test_size,
        model_type: document.getElementById("modelType").value,
        hidden_layers: document.getElementById("hiddenLayers").value,
        hidden_channels: parseInt(document.getElementById("hiddenChannels").value) || DEFAULTS.model.cnn.hidden_channels,
        kernel_size: parseInt(document.getElementById("kernelSize").value) || DEFAULTS.model.cnn.kernel_size,
        hidden_size: parseInt(document.getElementById("rnnHiddenSize").value) || DEFAULTS.model.rnn.hidden_size,
        num_layers: document.getElementById("modelType").value === "transformer"
            ? parseInt(document.getElementById("transNumLayers").value) || DEFAULTS.model.transformer.num_layers
            : parseInt(document.getElementById("rnnNumLayers").value) || DEFAULTS.model.rnn.num_layers,
        bidirectional: document.getElementById("toggleBidirectional").classList.contains("active"),
        d_model: parseInt(document.getElementById("transDModel").value) || DEFAULTS.model.transformer.d_model,
        nhead: parseInt(document.getElementById("transNhead").value) || DEFAULTS.model.transformer.nhead,
        dim_feedforward: parseInt(document.getElementById("transDimFeedforward").value) || DEFAULTS.model.transformer.dim_feedforward,
        learning_rate: parseFloat(document.getElementById("learningRate").value) || DEFAULTS.training.learning_rate,
        batch_size: parseInt(document.getElementById("batchSize").value) || DEFAULTS.training.batch_size,
        epochs: parseInt(document.getElementById("epochs").value) || DEFAULTS.training.epochs,
        dropout: parseFloat(document.getElementById("dropout").value) || DEFAULTS.training.dropout,
        patience: parseInt(document.getElementById("patience").value) || DEFAULTS.training.patience,
        normalization: document.getElementById("normalization").value,
        device: document.getElementById("deviceSelect").value,
    };

    const mt = params.model_type;
    if (mt === "autoformer") {
        Object.assign(params, {
            d_model: parseInt(document.getElementById("autoDModel").value) || DEFAULTS.model.autoformer.d_model,
            n_heads: parseInt(document.getElementById("autoNHeads").value) || DEFAULTS.model.autoformer.n_heads,
            e_layers: parseInt(document.getElementById("autoELayers").value) || DEFAULTS.model.autoformer.e_layers,
            d_layers: parseInt(document.getElementById("autoDLayers").value) || DEFAULTS.model.autoformer.d_layers,
            d_ff: parseInt(document.getElementById("autoDFF").value) || DEFAULTS.model.autoformer.d_ff,
            moving_avg: parseInt(document.getElementById("autoMovingAvg").value) || DEFAULTS.model.autoformer.moving_avg,
            factor: parseInt(document.getElementById("autoFactor").value) || DEFAULTS.model.autoformer.factor,
            activation: document.getElementById("autoActivation").value,
        });
    } else if (mt === "informer") {
        Object.assign(params, {
            d_model: parseInt(document.getElementById("infoDModel").value) || DEFAULTS.model.informer.d_model,
            n_heads: parseInt(document.getElementById("infoNHeads").value) || DEFAULTS.model.informer.n_heads,
            e_layers: parseInt(document.getElementById("infoELayers").value) || DEFAULTS.model.informer.e_layers,
            d_layers: parseInt(document.getElementById("infoDLayers").value) || DEFAULTS.model.informer.d_layers,
            d_ff: parseInt(document.getElementById("infoDFF").value) || DEFAULTS.model.informer.d_ff,
            factor: parseInt(document.getElementById("infoFactor").value) || DEFAULTS.model.informer.factor,
            distil: document.getElementById("toggleDistil").classList.contains("active"),
            activation: document.getElementById("infoActivation").value,
        });
    } else if (mt === "crossformer") {
        Object.assign(params, {
            d_model: parseInt(document.getElementById("crossDModel").value) || DEFAULTS.model.crossformer.d_model,
            n_heads: parseInt(document.getElementById("crossNHeads").value) || DEFAULTS.model.crossformer.n_heads,
            e_layers: parseInt(document.getElementById("crossELayers").value) || DEFAULTS.model.crossformer.e_layers,
            d_ff: parseInt(document.getElementById("crossDFF").value) || DEFAULTS.model.crossformer.d_ff,
            factor: parseInt(document.getElementById("crossFactor").value) || DEFAULTS.model.crossformer.factor,
            seg_len: parseInt(document.getElementById("crossSegLen").value) || DEFAULTS.model.crossformer.seg_len,
            win_size: parseInt(document.getElementById("crossWinSize").value) || DEFAULTS.model.crossformer.win_size,
            activation: document.getElementById("crossActivation").value,
        });
    } else if (mt === "etsformer") {
        Object.assign(params, {
            d_model: parseInt(document.getElementById("etsDModel").value) || DEFAULTS.model.etsformer.d_model,
            n_heads: parseInt(document.getElementById("etsNHeads").value) || DEFAULTS.model.etsformer.n_heads,
            e_layers: parseInt(document.getElementById("etsELayers").value) || DEFAULTS.model.etsformer.e_layers,
            d_ff: parseInt(document.getElementById("etsDFF").value) || DEFAULTS.model.etsformer.d_ff,
            top_k: parseInt(document.getElementById("etsTopK").value) || DEFAULTS.model.etsformer.top_k,
            dropout: parseFloat(document.getElementById("etsDropout").value) || DEFAULTS.model.etsformer.dropout,
            activation: document.getElementById("etsActivation").value,
        });
    } else if (mt === "fedformer") {
        Object.assign(params, {
            d_model: parseInt(document.getElementById("fedDModel").value) || DEFAULTS.model.fedformer.d_model,
            n_heads: parseInt(document.getElementById("fedNHeads").value) || DEFAULTS.model.fedformer.n_heads,
            e_layers: parseInt(document.getElementById("fedELayers").value) || DEFAULTS.model.fedformer.e_layers,
            d_layers: parseInt(document.getElementById("fedDLayers").value) || DEFAULTS.model.fedformer.d_layers,
            d_ff: parseInt(document.getElementById("fedDFF").value) || DEFAULTS.model.fedformer.d_ff,
            moving_avg: parseInt(document.getElementById("fedMovingAvg").value) || DEFAULTS.model.fedformer.moving_avg,
            dropout: parseFloat(document.getElementById("fedDropout").value) || DEFAULTS.model.fedformer.dropout,
            modes: parseInt(document.getElementById("fedModes").value) || DEFAULTS.model.fedformer.modes,
            version: document.getElementById("fedVersion").value,
            mode_select: document.getElementById("fedModeSelect").value,
            activation: document.getElementById("fedActivation").value,
        });
    } else if (mt === "film") {
        Object.assign(params, {
            window_size: document.getElementById("filmWindowSize").value || DEFAULTS.model.film.window_size,
            multiscale: document.getElementById("filmMultiscale").value || DEFAULTS.model.film.multiscale,
            dropout: parseFloat(document.getElementById("filmDropout").value) || DEFAULTS.model.film.dropout,
        });
    } else if (mt === "dlinear") {
        Object.assign(params, {
            moving_avg: parseInt(document.getElementById("dlMovingAvg").value) || DEFAULTS.model.dlinear.moving_avg,
            individual: document.getElementById("toggleIndividual").classList.contains("active"),
        });
    }

    document.getElementById("trainError").style.display = "none";
    document.getElementById("trainingSummary").style.display = "none";

    try {
        const setupData = await _setupTraining(params);

        const totalEpochs = parseInt(document.getElementById("epochs").value) || DEFAULTS.training.epochs;
        initTrainingProgress(totalEpochs);
        btn.innerHTML = '<span class="spinner"></span> Training...';

        const evtSource = new EventSource("/api/train/stream");

        evtSource.addEventListener("progress", function (e) {
            const m = JSON.parse(e.data);
            updateTrainingProgress(m);
        });

        evtSource.addEventListener("complete", function (e) {
            evtSource.close();
            const data = JSON.parse(e.data);
            showTrainingComplete(data);
            goToStep(6);
            refreshModelDropdown();
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

// ── Evaluation ──────────────────────────────────────────────────────

async function runEvaluation() {
    const btn = document.getElementById("evalBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Evaluating...';

    try {
        const data = await _evaluateModel();
        showEvalResult(data);
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
        const nSplits = parseInt(document.getElementById("cvFolds").value) || DEFAULTS.cv.default_folds;
        const data = await _validateModel({ n_splits: nSplits });
        showCVResult(data);
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
        const data = await _predictModel();
        showPredResult(data);
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Generate Predictions";
    }
}

// ── Projects ────────────────────────────────────────────────────────

async function loadProjects() {
    try {
        const projects = await _loadProjects();
        populateProjectGrid(projects);
    } catch (err) {
        console.error("loadProjects:", err.message);
    }
}

async function createProject() {
    const name = document.getElementById("projectName").value.trim();
    if (!name) { alert("Please enter a project name"); return; }

    const btn = document.getElementById("createProjectBtn");
    btn.disabled = true;
    btn.textContent = "Creating...";

    const fileInput = document.getElementById("projectFileInput");
    const formData = new FormData();
    formData.append("name", name);
    if (fileInput.files.length) {
        formData.append("file", fileInput.files[0]);
    }

    try {
        await _createProject(formData);
        hideNewProjectModal();
        document.getElementById("projectName").value = "";
        document.getElementById("projectFileInput").value = "";
        document.getElementById("projectUploadText").textContent = "Click to select file";
        loadProjects();
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Create";
    }
}

async function activateProject(projectId) {
    try {
        const data = await _activateProject(projectId);
        _activeProjectId = projectId;
        currentDataInfo = data;

        hideProjectList();

        populateStep2(data);
        populateStep3Columns(data.columns);
        populateTargetCol(data.columns);
        populateTimeColSelect(data.columns);
        await loadTaskConfig();

        const models = data.models || [];
        if (models.length) {
            const taskType = document.getElementById("taskTypeSelect").value;
            const filtered = populateModelDropdown(models, taskType);
            if (filtered.length) {
                document.getElementById("modelSelect").value = filtered[0].id;
                await _loadModelToSession(projectId, filtered[0].id);
                showLoadedModelBadge(true, filtered[0].model_type);
            } else {
                showLoadedModelBadge(false);
            }
        } else {
            document.getElementById("modelSelector").style.display = "none";
        }

        goToStep(2);
    } catch (err) {
        alert("Error: " + err.message);
    }
}

async function deleteProject(projectId) {
    if (!confirm("Delete this project permanently?")) return;
    try {
        await _deleteProject(projectId);
        loadProjects();
    } catch (err) {
        alert("Error: " + err.message);
    }
}

async function loadProjectModels() {
    if (!_activeProjectId) return;
    try {
        const models = await _loadProjectModels(_activeProjectId);
        populateModelList(models);
    } catch (err) {
        console.error("loadProjectModels:", err.message);
    }
}

async function refreshModelDropdown() {
    if (!_activeProjectId) return;
    try {
        const models = await _loadProjectModels(_activeProjectId);
        const taskType = document.getElementById("taskTypeSelect").value;
        const filtered = populateModelDropdown(models, taskType);
        if (filtered.length) {
            document.getElementById("modelSelect").value = filtered[0].id;
            await _loadModelToSession(_activeProjectId, filtered[0].id);
            showLoadedModelBadge(true, filtered[0].model_type);
        }
    } catch (err) {
        console.error("refreshModelDropdown:", err.message);
    }
}

async function onModelSelect(modelId) {
    if (!modelId) {
        showLoadedModelBadge(false);
        return;
    }
    try {
        const model = await _loadModelToSession(_activeProjectId, modelId);
        showLoadedModelBadge(true, model.model_type);
    } catch (err) {
        showLoadedModelBadge(false);
    }
}

async function compareModels() {
    const cbs = document.querySelectorAll(".model-cb:checked");
    const modelIds = Array.from(cbs).map((cb) => cb.value);
    if (!modelIds.length) { alert("Select at least one model to compare."); return; }

    const btn = document.querySelector("#modelCompareAction .btn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Comparing...';

    try {
        const data = await _compareModels(_activeProjectId, modelIds);
        const resultDiv = document.getElementById("modelCompareResult");
        const chartDiv = document.getElementById("modelCompareChart");
        chartDiv.innerHTML = `<div class="image-card"><img src="data:image/png;base64,${data.plot_image}" alt="Model Comparison"><div class="caption">Model Predictions Comparison (${data.loaded_count} models)</div></div>`;
        resultDiv.style.display = "block";
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Compare Selected";
    }
}

async function exportModel(modelId) {
    const defaultName = `${modelId}.pt`;
    const name = prompt("Export model filename:", defaultName);
    if (!name) return;
    const safe = name.trim() || defaultName;
    window.location.href = `/api/projects/${_activeProjectId}/models/${modelId}/export?name=${encodeURIComponent(safe)}`;
}

async function resetAll() {
    if (!confirm("Reset all data and start over?")) return;
    try {
        await fetch("/api/reset", { method: "POST" });
    } catch (_) {}
    currentDataInfo = null;
    _activeProjectId = null;
    resetAllUI();
}

function backToProjects() {
    _activeProjectId = null;
    showProjectList();
    loadProjects();
}

/* =============== Window Exports (for Vitest) =============== */
window.handleUpload = handleUpload;
window.loadTaskConfig = loadTaskConfig;
window.applyTaskConfig = applyTaskConfig;
window.updateModelOptions = updateModelOptions;
window.runClean = runClean;
window.runFill = runFill;
window.startTraining = startTraining;
window.runEvaluation = runEvaluation;
window.runValidation = runValidation;
window.runPredict = runPredict;
window.loadProjects = loadProjects;
window.createProject = createProject;
window.activateProject = activateProject;
window.deleteProject = deleteProject;
window.loadProjectModels = loadProjectModels;
window.refreshModelDropdown = refreshModelDropdown;
window.onModelSelect = onModelSelect;
window.compareModels = compareModels;
window.exportModel = exportModel;
window.resetAll = resetAll;
window.backToProjects = backToProjects;
