/*
 * AppState — lightweight state container.
 *
 * Groups state variables that share the same lifecycle (e.g. project
 * activation) so they are set and cleared together, eliminating
 * implicit ordering dependencies between canvas.js and app.js.
 *
 * Usage:  import via <script> tag before all other JS files.
 *         Read/write directly:  AppState.activeProjectId = "abc";
 *
 * Lifecycle:
 *
 *   PROJECT_ACTIVE:
 *     activeProjectId   — the project the user is currently working on
 *     currentDataInfo   — full activation response (columns, models, …)
 *     canvasProjectId   — mirrors activeProjectId (for canvas.js)
 *     pendingCanvasData — canvas JSON to render when canvas first shows
 *     canvasModels      — registered canvas-generated model types
 *
 *   PROJECT_INACTIVE (backToProjects / reset):
 *     All fields above are cleared to null / [].
 */

var AppState = {
    // ── Project lifecycle ──────────────────────────────────────────
    activeProjectId: null,
    currentDataInfo: null,
    canvasProjectId: null,
    pendingCanvasData: null,
    canvasModels: [],

    // ── Project lifecycle ──────────────────────────────────────────
    /** Call when a project is activated. */
    activate: function(projectId, data) {
        this.activeProjectId = projectId;
        this.currentDataInfo = data;
        this.canvasProjectId = projectId;
        this.pendingCanvasData = data.canvas || null;
        this.canvasModels = data.canvas_models || [];
    },

    /** Call when exiting a project. */
    deactivate: function() {
        this.activeProjectId = null;
        this.currentDataInfo = null;
        this.canvasProjectId = null;
        this.pendingCanvasData = null;
        this.canvasModels = [];
    },
};
