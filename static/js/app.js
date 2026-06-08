/* DeepLearning Labs - App Initialization & Event Listeners */

let currentDataInfo = null;

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

    // Load models when the Models tab is activated
    if (tabId === "models" && _activeProjectId) {
        loadProjectModels();
    }
});

/* =============== Project Init =============== */

// Load projects on page load
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
