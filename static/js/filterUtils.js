/**
 * Pure-function utilities — no DOM dependencies.
 * Loaded before ui.js / app.js via <script> tag in index.html.
 * Used through global scope; unit-tested via Vitest.
 */

function filterModelsByTask(models, taskType) {
    const isTs = taskType === "time_series";
    return models.filter((m) => {
        // is_time_series undefined (older models) → treated as non-time-series
        const mTs = m.is_time_series === true;
        return isTs ? mTs : !mTs;
    });
}
