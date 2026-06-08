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
        goToStep(2);
    } catch (err) {
        showUploadError(err.message);
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

        const models = data.models || [];
        if (models.length) {
            populateModelDropdown(models);
            document.getElementById("modelSelect").value = models[0].id;
            await _loadModelToSession(projectId, models[0].id);
            showLoadedModelBadge(true, models[0].model_type);
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
        if (models.length) {
            populateModelDropdown(models);
            const latest = models[0];
            document.getElementById("modelSelect").value = latest.id;
            await _loadModelToSession(_activeProjectId, latest.id);
            showLoadedModelBadge(true, latest.model_type);
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
