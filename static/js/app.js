/* DeepLearning Labs — App Initialization & Orchestration */

const DEFAULTS = JSON.parse(document.getElementById("config-data").textContent);

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

    if (tabId === "models" && AppState.activeProjectId) {
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
        AppState.currentDataInfo = data;
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

AppState.canvasModels = [];  // dynamically registered canvas models

function registerCanvasModel(modelType, label) {
    // Add to the dynamic list so updateModelOptions picks it up
    if (!AppState.canvasModels.find(m => m.type === modelType)) {
        AppState.canvasModels.push({type: modelType, label: label});
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
        "transformer": "Transformer (Tabular)",
        "vanilla_transformer": "Vanilla Transformer",
        "autoformer": "Autoformer (Long-term Forecast)",
        "informer": "Informer (ProbSparse Attention)",
        "crossformer": "Crossformer (Two-Stage Attention)",
        "etsformer": "ETSformer (Exp Smoothing Transformer)",
        "fedformer": "FEDformer (Frequency Enhanced Decomp Transformer)",
        "film": "FiLM (Frequency-enhanced Legendre Memory)",
        "frets": "FreTS (Frequency-enhanced Time Series)",
        "itransformer": "iTransformer (Inverted Transformer)",
        "koopa": "Koopa (Koopman Forecasting)",
        "lightts": "LightTS (Light Time Series)",
        "dlinear": "DLinear (Decomposition Linear)",
        "random_forest_regressor": "Random Forest (Regression)",
        "random_forest_classifier": "Random Forest (Classification)",
        "xgboost_regressor": "XGBoost (Regression)",
        "xgboost_classifier": "XGBoost (Classification)",
        "lightgbm_regressor": "LightGBM (Regression)",
        "lightgbm_classifier": "LightGBM (Classification)",
        "decision_tree_regressor": "Decision Tree (Regression)",
        "decision_tree_classifier": "Decision Tree (Classification)",
    };

    const tsModels = ["rnn", "lstm", "gru", "autoformer", "informer", "crossformer", "etsformer", "fedformer", "film", "frets", "itransformer", "koopa", "lightts", "vanilla_transformer", "dlinear"];
    const generalModels = ["mlp", "cnn", "transformer", "random_forest_regressor", "random_forest_classifier", "xgboost_regressor", "xgboost_classifier", "lightgbm_regressor", "lightgbm_classifier", "decision_tree_regressor", "decision_tree_classifier"];

    // Canvas-generated models (large pipeline) always go to time-series list
    const canvasTSTypes = [];
    AppState.canvasModels.forEach(function(m) {
        allOptions[m.type] = m.label;
        canvasTSTypes.push(m.type);
    });

    sel.innerHTML = "";
    const visible = isTimeSeries ? tsModels.concat(canvasTSTypes) : generalModels;
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
        AppState.currentDataInfo = data.data;
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
        AppState.currentDataInfo = data.data;
        showFillResult(data);
    } catch (err) {
        alert("Error: " + err.message);
    }
}

// ── Training ────────────────────────────────────────────────────────

// ── Model-specific parameter readers ───────────────────────────
// Each function reads the DOM elements for its model type and
// returns an object of param overrides merged into the base params.
const MODEL_PARAM_READERS = {
    autoformer: function() { return {
        d_model:     $int("autoDModel", DEFAULTS.model.autoformer.d_model),
        n_heads:     $int("autoNHeads", DEFAULTS.model.autoformer.n_heads),
        e_layers:    $int("autoELayers", DEFAULTS.model.autoformer.e_layers),
        d_layers:    $int("autoDLayers", DEFAULTS.model.autoformer.d_layers),
        d_ff:        $int("autoDFF", DEFAULTS.model.autoformer.d_ff),
        moving_avg:  $int("autoMovingAvg", DEFAULTS.model.autoformer.moving_avg),
        factor:      $int("autoFactor", DEFAULTS.model.autoformer.factor),
        activation:  $val("autoActivation"),
    };},
    informer: function() { return {
        d_model:     $int("infoDModel", DEFAULTS.model.informer.d_model),
        n_heads:     $int("infoNHeads", DEFAULTS.model.informer.n_heads),
        e_layers:    $int("infoELayers", DEFAULTS.model.informer.e_layers),
        d_layers:    $int("infoDLayers", DEFAULTS.model.informer.d_layers),
        d_ff:        $int("infoDFF", DEFAULTS.model.informer.d_ff),
        factor:      $int("infoFactor", DEFAULTS.model.informer.factor),
        distil:      $bool("toggleDistil"),
        activation:  $val("infoActivation"),
    };},
    crossformer: function() { return {
        d_model:     $int("crossDModel", DEFAULTS.model.crossformer.d_model),
        n_heads:     $int("crossNHeads", DEFAULTS.model.crossformer.n_heads),
        e_layers:    $int("crossELayers", DEFAULTS.model.crossformer.e_layers),
        d_ff:        $int("crossDFF", DEFAULTS.model.crossformer.d_ff),
        factor:      $int("crossFactor", DEFAULTS.model.crossformer.factor),
        seg_len:     $int("crossSegLen", DEFAULTS.model.crossformer.seg_len),
        win_size:    $int("crossWinSize", DEFAULTS.model.crossformer.win_size),
        activation:  $val("crossActivation"),
    };},
    etsformer: function() { return {
        d_model:     $int("etsDModel", DEFAULTS.model.etsformer.d_model),
        n_heads:     $int("etsNHeads", DEFAULTS.model.etsformer.n_heads),
        e_layers:    $int("etsELayers", DEFAULTS.model.etsformer.e_layers),
        d_ff:        $int("etsDFF", DEFAULTS.model.etsformer.d_ff),
        top_k:       $int("etsTopK", DEFAULTS.model.etsformer.top_k),
        dropout:     $float("etsDropout", DEFAULTS.model.etsformer.dropout),
        activation:  $val("etsActivation"),
    };},
    fedformer: function() { return {
        d_model:     $int("fedDModel", DEFAULTS.model.fedformer.d_model),
        n_heads:     $int("fedNHeads", DEFAULTS.model.fedformer.n_heads),
        e_layers:    $int("fedELayers", DEFAULTS.model.fedformer.e_layers),
        d_layers:    $int("fedDLayers", DEFAULTS.model.fedformer.d_layers),
        d_ff:        $int("fedDFF", DEFAULTS.model.fedformer.d_ff),
        moving_avg:  $int("fedMovingAvg", DEFAULTS.model.fedformer.moving_avg),
        dropout:     $float("fedDropout", DEFAULTS.model.fedformer.dropout),
        modes:       $int("fedModes", DEFAULTS.model.fedformer.modes),
        version:     $val("fedVersion"),
        mode_select: $val("fedModeSelect"),
        activation:  $val("fedActivation"),
    };},
    film: function() { return {
        window_size: $val("filmWindowSize", DEFAULTS.model.film.window_size),
        multiscale:  $val("filmMultiscale", DEFAULTS.model.film.multiscale),
        dropout:     $float("filmDropout", DEFAULTS.model.film.dropout),
    };},
    frets: function() { return {
        channel_independence: parseInt($val("fretsChannelIndependence"), 10),
        embed_size:           $int("fretsEmbedSize", DEFAULTS.model.frets.embed_size),
        hidden_size:          $int("fretsHiddenSize", DEFAULTS.model.frets.hidden_size),
    };},
    itransformer: function() { return {
        d_model:     $int("itransDModel", DEFAULTS.model.itransformer.d_model),
        n_heads:     $int("itransNHeads", DEFAULTS.model.itransformer.n_heads),
        e_layers:    $int("itransELayers", DEFAULTS.model.itransformer.e_layers),
        d_ff:        $int("itransDFF", DEFAULTS.model.itransformer.d_ff),
        dropout:     $float("itransDropout", DEFAULTS.model.itransformer.dropout),
        activation:  $val("itransActivation"),
    };},
    koopa: function() { return {
        dynamic_dim:   $int("koopaDynamicDim", DEFAULTS.model.koopa.dynamic_dim),
        hidden_dim:    $int("koopaHiddenDim", DEFAULTS.model.koopa.hidden_dim),
        hidden_layers: $int("koopaHiddenLayers", DEFAULTS.model.koopa.hidden_layers),
        num_blocks:    $int("koopaNumBlocks", DEFAULTS.model.koopa.num_blocks),
        multistep:     $bool("toggleKoopaMultistep"),
    };},
    vanilla_transformer: function() { return {
    lightts: function() { return {
        d_model:     $int("lighttsDModel", DEFAULTS.model.lightts.d_model),
        chunk_size:  $int("lighttsChunkSize", DEFAULTS.model.lightts.chunk_size),
        dropout:     $float("lighttsDropout", DEFAULTS.model.lightts.dropout),
    };},
        d_model:     $int("vanillaDModel", DEFAULTS.model.vanilla_transformer.d_model),
        n_heads:     $int("vanillaNHeads", DEFAULTS.model.vanilla_transformer.n_heads),
        e_layers:    $int("vanillaELayers", DEFAULTS.model.vanilla_transformer.e_layers),
        d_layers:    $int("vanillaDLayers", DEFAULTS.model.vanilla_transformer.d_layers),
        d_ff:        $int("vanillaDFF", DEFAULTS.model.vanilla_transformer.d_ff),
        dropout:     $float("vanillaDropout", DEFAULTS.model.vanilla_transformer.dropout),
        activation:  $val("vanillaActivation"),
    };},
    dlinear: function() { return {
        moving_avg:  $int("dlMovingAvg", DEFAULTS.model.dlinear.moving_avg),
        individual:  $bool("toggleIndividual"),
    };},
};

// Classical ML models share the same param reader
var _classicalMlReader = function() { return {
    n_estimators:      $int("classicalNEstimators", 100),
    max_depth:         (function() { var v = document.getElementById("classicalMaxDepth").value; return v === "" || v === "None" ? null : parseInt(v); })(),
    min_samples_split: $int("classicalMinSamplesSplit", 2),
    min_samples_leaf:  $int("classicalMinSamplesLeaf", 1),
};};
["random_forest_regressor", "random_forest_classifier", "xgboost_regressor", "xgboost_classifier", "lightgbm_regressor", "lightgbm_classifier", "decision_tree_regressor", "decision_tree_classifier"].forEach(function(mt) {
    MODEL_PARAM_READERS[mt] = _classicalMlReader;
});

// DOM readers — small helpers used by the parameter readers above
function $int(id, fallback) { return parseInt(document.getElementById(id).value) || fallback; }
function $float(id, fallback) { return parseFloat(document.getElementById(id).value) || fallback; }
function $val(id, fallback) { var v = document.getElementById(id).value; return v || fallback; }
function $bool(id) { return document.getElementById(id).classList.contains("active"); }


async function startTraining() {
    const targetCol = document.getElementById("targetCol").value;
    if (!targetCol) { alert("请先选择目标列 (target column)"); return; }
    // Verify the selected column still exists in the data (re-fetch columns from activation data)
    const columns = (AppState.currentDataInfo && AppState.currentDataInfo.columns) || [];
    if (columns.length > 0 && columns.indexOf(targetCol) === -1) {
        alert(`目标列 '${targetCol}' 在当前数据中不存在。可用列: ${columns.join(', ')}`);
        return;
    }

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

    // Model-specific params — dispatch via MODEL_PARAM_READERS registry
    var reader = MODEL_PARAM_READERS[params.model_type];
    if (reader) { Object.assign(params, reader()); }

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
        AppState.activate(projectId, data);

        hideProjectList();

        populateStep2(data);
        populateStep3Columns(data.columns);
        populateTargetCol(data.columns);
        populateTimeColSelect(data.columns);

        // Register canvas models BEFORE loadTaskConfig triggers updateModelOptions
        const canvasModels = data.canvas_models || [];
        canvasModels.forEach(function(m) {
            registerCanvasModel(m.model_type, m.model_type);
        });

        await loadTaskConfig();

        const models = data.models || [];
        if (models.length) {
            const taskType = document.getElementById("taskTypeSelect").value;
            const filtered = populateModelDropdown(models, taskType);
            if (filtered.length) {
                document.getElementById("modelSelect").value = filtered[0].id;
                try {
                    await _loadModelToSession(projectId, filtered[0].id);
                    showLoadedModelBadge(true, filtered[0].model_type);
                } catch (_) {
                    // Model type no longer available (e.g., canvas model deleted) — skip silently
                    showLoadedModelBadge(false);
                }
            } else {
                showLoadedModelBadge(false);
            }
        } else {
            document.getElementById("modelSelector").style.display = "none";
        }

        // Init canvas with project data
        document.getElementById("canvasToggleBtn").style.display = "inline-flex";
        initCanvasForProject(projectId, data.canvas);

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
    if (!AppState.activeProjectId) return;
    try {
        const models = await _loadProjectModels(AppState.activeProjectId);
        // Show only trained models — canvas-only (untrained) models are managed
        // from the Model Architecture dropdown, not here.
        const trained = models.filter(function(m) { return !m.canvas_only; });
        const taskType = document.getElementById("taskTypeSelect").value;
        const filtered = filterModelsByTask(trained, taskType);
        populateModelList(filtered);
    } catch (err) {
        console.error("loadProjectModels:", err.message);
    }
}

async function refreshModelDropdown() {
    if (!AppState.activeProjectId) return;
    try {
        const models = await _loadProjectModels(AppState.activeProjectId);
        const taskType = document.getElementById("taskTypeSelect").value;
        const filtered = populateModelDropdown(models, taskType);
        if (filtered.length) {
            document.getElementById("modelSelect").value = filtered[0].id;
            await _loadModelToSession(AppState.activeProjectId, filtered[0].id);
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
        const model = await _loadModelToSession(AppState.activeProjectId, modelId);
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
        const data = await _compareModels(AppState.activeProjectId, modelIds);
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

async function deleteCanvasModel(modelType) {
    if (!AppState.activeProjectId) return;
    if (!confirm(`确定删除画布模型「${modelType}」？此操作不可撤销。`)) return;
    try {
        await _deleteCanvasModel(AppState.activeProjectId, modelType);
        // Remove from the in-memory registry so updateModelOptions won't include it
        AppState.canvasModels = AppState.canvasModels.filter(function(m) { return m.type !== modelType; });
        // Clear the model architecture dropdown (Step 4) and hide delete button
        document.getElementById("modelType").value = "";
        document.getElementById("deleteCanvasModelBtn").style.display = "none";
        // Rebuild the dropdown — the deleted type is now out of AppState.canvasModels
        if (typeof updateModelOptions === "function") {
            updateModelOptions(document.getElementById("taskTypeSelect").value);
        }
        // Refresh the Trained Models list and Step 6 selector
        loadProjectModels();
        if (typeof refreshModelDropdown === "function") refreshModelDropdown();
    } catch (err) {
        alert("删除失败: " + err.message);
    }
}

async function exportModel(modelId) {
    const defaultName = `${modelId}.pt`;
    const name = prompt("Export model filename:", defaultName);
    if (!name) return;
    const safe = name.trim() || defaultName;
    window.location.href = `/api/projects/${AppState.activeProjectId}/models/${modelId}/export?name=${encodeURIComponent(safe)}`;
}

async function resetAll() {
    if (!confirm("Reset all data and start over?")) return;
    try {
        await fetch("/api/reset", { method: "POST" });
    } catch (_) {}
    AppState.deactivate();
    resetAllUI();
}

function backToProjects() {
    AppState.deactivate();
    resetCanvas();
    document.getElementById("canvasToggleBtn").style.display = "none";
    toggleCanvasView(false);
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
window.registerCanvasModel = registerCanvasModel;
window.deleteCanvasModel = deleteCanvasModel;
window.onModelSelect = onModelSelect;
window.compareModels = compareModels;
window.exportModel = exportModel;
window.resetAll = resetAll;
window.backToProjects = backToProjects;
