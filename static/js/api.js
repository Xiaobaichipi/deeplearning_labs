/* API Call Functions — pure data fetching, no DOM manipulation */

// ── Data ────────────────────────────────────────────────────────────

async function _uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.data;
}

async function _cleanData(params) {
    const res = await fetch("/api/data/clean", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
}

async function _fillData(params) {
    const res = await fetch("/api/data/fill", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
}

// ── Task Config ─────────────────────────────────────────────────────

async function _getTaskConfig() {
    const res = await fetch("/api/data/task-config");
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.task_config || {};
}

async function _setTaskConfig(config) {
    const res = await fetch("/api/data/task-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.task_config;
}

// ── Training ────────────────────────────────────────────────────────

async function _setupTraining(params) {
    const res = await fetch("/api/train/setup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
}

// ── Evaluation ──────────────────────────────────────────────────────

async function _evaluateModel() {
    const res = await fetch("/api/evaluate", { method: "POST" });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
}

async function _validateModel(params) {
    const res = await fetch("/api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
}

async function _predictModel() {
    const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_test: true }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
}

function downloadPredictions(format) {
    window.location.href = `/api/predict/download?source=test&format=${format}`;
}

function downloadHistory(format) {
    window.location.href = `/api/train/history/download?format=${format}`;
}

// ── Projects ────────────────────────────────────────────────────────

async function _loadProjects() {
    const res = await fetch("/api/projects");
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.projects || [];
}

async function _createProject(formData) {
    const res = await fetch("/api/projects", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.project;
}

async function _activateProject(projectId) {
    const res = await fetch(`/api/projects/${projectId}/activate`, { method: "POST" });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.data;
}

async function _deleteProject(projectId) {
    const res = await fetch(`/api/projects/${projectId}`, { method: "DELETE" });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
}

async function _loadProjectModels(projectId) {
    const res = await fetch(`/api/projects/${projectId}/models`);
    const data = await res.json();
    if (data.error) return [];
    return data.models || [];
}

async function _loadModelToSession(projectId, modelId) {
    const res = await fetch(`/api/projects/${projectId}/load-model/${modelId}`, { method: "POST" });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.model;
}

async function _compareModels(projectId, modelIds) {
    const res = await fetch(`/api/projects/${projectId}/models/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_ids: modelIds }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
}

// ── Canvas ───────────────────────────────────────────────────────

async function _generateCanvasModel(projectId) {
    const res = await fetch(`/api/projects/${projectId}/canvas/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
}

async function _saveCanvas(projectId, canvas) {
    const res = await fetch(`/api/projects/${projectId}/canvas/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(canvas),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
}

async function _loadCanvas(projectId) {
    const res = await fetch(`/api/projects/${projectId}/canvas/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.canvas;
}

/* =============== Window Exports (for Vitest) =============== */
window._uploadFile = _uploadFile;
window._cleanData = _cleanData;
window._fillData = _fillData;
window._getTaskConfig = _getTaskConfig;
window._setTaskConfig = _setTaskConfig;
window._setupTraining = _setupTraining;
window._evaluateModel = _evaluateModel;
window._validateModel = _validateModel;
window._predictModel = _predictModel;
window.downloadPredictions = downloadPredictions;
window.downloadHistory = downloadHistory;
window._loadProjects = _loadProjects;
window._createProject = _createProject;
window._activateProject = _activateProject;
window._deleteProject = _deleteProject;
window._loadProjectModels = _loadProjectModels;
window._generateCanvasModel = _generateCanvasModel;
window._loadModelToSession = _loadModelToSession;
window._compareModels = _compareModels;
