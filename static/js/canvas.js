/* Canvas — Drawflow-based model architecture editor */

let canvasEditor = null;
let canvasInitialized = false;
let _canvasProjectId = null;
let _selectedNodeId = null;
let _pendingCanvasData = null;  // canvas data stored during project activation, rendered when canvas first shows

/* =============== Initialization =============== */

function initCanvas() {
  if (canvasInitialized) return;

  const container = document.getElementById("drawflow-container");
  if (!container) return;

  canvasEditor = new Drawflow(container);
  canvasEditor.start();
  canvasInitialized = true;

  // Remove default module tab — we only use "Home"
  const moduleTabs = container.querySelector(".drawflow-delete");
  if (moduleTabs) moduleTabs.style.display = "none";

  // Node selected → show config panel
  container.addEventListener("click", (e) => {
    const nodeEl = e.target.closest(".drawflow-node");
    if (nodeEl) {
      const nodeId = nodeEl.id.replace("node-", "");
      selectNode(nodeId);
    } else {
      deselectNode();
    }
  });

  // Keyboard: Delete key to remove selected node
  document.addEventListener("keydown", (e) => {
    if (e.key === "Delete" || e.key === "Backspace") {
      if (_selectedNodeId && canvasEditor) {
        const active = document.activeElement;
        if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA" || active.tagName === "SELECT")) return;
        canvasEditor.removeNodeId(_selectedNodeId);
        deselectNode();
        markCanvasDirty();
      }
    }
  });
}

function buildNodeHtml(nodeType, label, comp) {
  const color = comp.color;
  return `
    <div style="padding:10px 16px;border-left:4px solid ${color};min-width:140px;background:#fff;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
      <div style="font-size:13px;font-weight:600;color:#333;">${label}</div>
      <div style="font-size:11px;color:#999;margin-top:2px;">${nodeType}</div>
    </div>
  `;
}

/* =============== Component Panel =============== */

function renderComponentPanel() {
  const panel = document.getElementById("component-panel");
  if (!panel) return;

  panel.innerHTML = "";

  // Group by category
  const groups = {};
  Object.entries(COMPONENT_TYPES).forEach(([type, def]) => {
    const cat = def.category || "其他";
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push({ type, ...def });
  });

  Object.entries(groups).forEach(([category, items]) => {
    const section = document.createElement("div");
    section.className = "component-group";

    const header = document.createElement("div");
    header.className = "component-group-header";
    header.textContent = category;
    section.appendChild(header);

    items.forEach((item) => {
      const el = document.createElement("div");
      el.className = "component-item";
      el.draggable = true;
      el.dataset.type = item.type;
      el.innerHTML = `
        <span class="component-color" style="background:${item.color}"></span>
        <span class="component-label">${item.label}</span>
      `;

      el.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/plain", item.type);
        e.dataTransfer.effectAllowed = "copy";
      });

      section.appendChild(el);
    });

    panel.appendChild(section);
  });
}

/* =============== Drag & Drop onto Canvas =============== */

let _dropZoneReady = false;

function setupCanvasDropZone() {
  if (_dropZoneReady) return;
  const container = document.getElementById("drawflow-container");
  if (!container) return;

  container.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });

  container.addEventListener("drop", (e) => {
    e.preventDefault();
    const type = e.dataTransfer.getData("text/plain");
    if (!type || !COMPONENT_TYPES[type]) return;

    const rect = container.getBoundingClientRect();
    const posX = e.clientX - rect.left - 80;
    const posY = e.clientY - rect.top - 30;

    addCanvasNode(type, posX, posY);
    markCanvasDirty();
  });

  _dropZoneReady = true;
}

function addCanvasNode(type, posX, posY) {
  const comp = COMPONENT_TYPES[type];
  const label = comp.label;

  const nodeId = canvasEditor.addNode(
    type,
    comp.inputs,
    comp.outputs,
    posX,
    posY,
    `node-${type}`,
    { label, config: { ...comp.defaults }, ports: JSON.parse(JSON.stringify(comp.ports)) },
    buildNodeHtml(type, label, comp),
  );

  return nodeId;
}

/* =============== Node Selection & Config Panel =============== */

function selectNode(nodeId) {
  _selectedNodeId = nodeId;

  // Get node data from Drawflow
  const nodeInfo = getNodeData(nodeId);
  if (!nodeInfo) return;

  // Highlight selected node
  document.querySelectorAll(".drawflow-node.selected").forEach((el) => el.classList.remove("selected"));
  const nodeEl = document.getElementById(`node-${nodeId}`);
  if (nodeEl) nodeEl.classList.add("selected");

  // Show config panel
  const panel = document.getElementById("config-panel");
  if (!panel) return;
  panel.style.display = "block";

  const comp = COMPONENT_TYPES[nodeInfo.name];
  // Merge saved config with defaults so missing fields never show as empty
  const savedConfig = nodeInfo.data.config || {};
  const defaultConfig = comp ? { ...comp.defaults } : {};
  const config = { ...defaultConfig, ...savedConfig };

  let html = `<div class="config-panel-header">
    <span class="config-panel-title" style="border-left:3px solid ${comp ? comp.color : '#999'};padding-left:8px;">${nodeInfo.data.label}</span>
    <span class="config-panel-close" onclick="deselectNode()">&times;</span>
  </div>`;

  html += `<div class="config-panel-body">`;
  html += `<div class="config-field">
    <label>节点名称</label>
    <input type="text" value="${nodeInfo.data.label}" data-key="label" onchange="updateNodeConfig(${nodeId}, this)">
  </div>`;

  Object.keys(config).forEach((key) => {
    const val = config[key];
    const isNum = typeof val === "number";
    html += `<div class="config-field">
      <label>${key}</label>
      <input type="${isNum ? "number" : "text"}" value="${val}" data-key="${key}" step="${isNum && Number.isInteger(val) ? "1" : "0.01"}" onchange="updateNodeConfig(${nodeId}, this)">
    </div>`;
  });

  html += `</div>`;
  panel.innerHTML = html;
}

function deselectNode() {
  _selectedNodeId = null;
  document.querySelectorAll(".drawflow-node.selected").forEach((el) => el.classList.remove("selected"));
  const panel = document.getElementById("config-panel");
  if (panel) panel.style.display = "none";
}

function updateNodeConfig(nodeId, input) {
  const key = input.dataset.key;
  const val = input.type === "number" ? parseFloat(input.value) : input.value;

  // Read current node data, modify, write back.
  // Do NOT call updateNodeDataFromId with {} — it REPLACES the entire data.
  const info = getNodeData(nodeId);
  if (!info) return;
  info.data = info.data || {};

  if (key === "label") {
    info.data.label = val;
    canvasEditor.updateNodeDataFromId(nodeId, info.data);
    const comp = COMPONENT_TYPES[info.name];
    if (comp) {
      canvasEditor.updateNodeHtmlFromId(nodeId, buildNodeHtml(info.name, val, comp));
    }
  } else {
    info.data.config = info.data.config || {};
    info.data.config[key] = val;
    canvasEditor.updateNodeDataFromId(nodeId, info.data);
  }

  markCanvasDirty();
}

function getNodeData(nodeId) {
  try {
    const exported = canvasEditor.export();
    const node = exported.drawflow.Home.data[nodeId];
    return node || null;
  } catch (_) {
    return null;
  }
}

/* =============== Canvas Save / Load =============== */

function markCanvasDirty() {
  // Visual indicator that canvas has unsaved changes
  const saveBtn = document.getElementById("canvas-save-btn");
  if (saveBtn) saveBtn.textContent = "● 保存";
}

function collectCanvasData() {
  const exported = canvasEditor.export();
  const dfData = exported.drawflow.Home.data;

  const nodes = [];
  const edges = [];
  const edgeSet = new Set();

  Object.keys(dfData).forEach((nodeId) => {
    const n = dfData[nodeId];
    // Merge saved config with component defaults to fill any missing fields
    const comp = COMPONENT_TYPES[n.name];
    const savedConfig = n.data.config || {};
    const defaultConfig = comp ? { ...comp.defaults } : {};
    const mergedConfig = { ...defaultConfig, ...savedConfig };

    nodes.push({
      id: nodeId,
      type: n.name,
      label: n.data.label || n.name,
      config: mergedConfig,
      ports: n.data.ports || {},
      position: { x: n.pos_x, y: n.pos_y },
    });

    Object.keys(n.inputs).forEach((inputKey) => {
      (n.inputs[inputKey].connections || []).forEach((conn) => {
        const edgeId = `e-${conn.node}-${nodeId}`;
        if (!edgeSet.has(edgeId)) {
          edgeSet.add(edgeId);
          edges.push({
            id: edgeId,
            from: conn.node,
            from_port: conn.output,
            to: nodeId,
            to_port: inputKey,
          });
        }
      });
    });
  });

  return {
    nodes,
    edges,
    metadata: { version: "0.1", description: "", created_at: new Date().toISOString() },
  };
}

function renderCanvasFromData(canvasData) {
  if (!canvasData || !canvasData.nodes || !canvasData.nodes.length) return;

  const drawflowData = { drawflow: { Home: { data: {} } } };
  const dfData = drawflowData.drawflow.Home.data;

  canvasData.nodes.forEach((node) => {
    const comp = COMPONENT_TYPES[node.type];
    const inputs = {};
    const outputs = {};

    if (comp) {
      for (let i = 0; i < comp.inputs; i++) inputs[`input_${i + 1}`] = { connections: [] };
      for (let i = 0; i < comp.outputs; i++) outputs[`output_${i + 1}`] = { connections: [] };
    } else {
      inputs["input_1"] = { connections: [] };
      outputs["output_1"] = { connections: [] };
    }

    dfData[node.id] = {
      id: node.id,
      name: node.type,
      data: { label: node.label, config: node.config, ports: node.ports },
      class: `node-${node.type}`,
      html: buildNodeHtml(node.type, node.label, comp || { color: "#999", label: node.type }),
      typenode: false,
      inputs,
      outputs,
      pos_x: node.position.x,
      pos_y: node.position.y,
    };
  });

  // Add edges — register on BOTH source outputs and target inputs
  // IMPORTANT: Drawflow's addNodeImport reads connections[i].input (not .output)
  // when rendering SVG paths during load(). Use "input" key for the port name.
  canvasData.edges.forEach((edge) => {
    const fromNode = dfData[edge.from];
    const toNode = dfData[edge.to];
    if (fromNode && toNode) {
      const outputKey = Object.keys(fromNode.outputs)[0];
      const inputKey = Object.keys(toNode.inputs)[0];
      if (outputKey && inputKey) {
        fromNode.outputs[outputKey].connections.push({
          node: edge.to,
        });
        toNode.inputs[inputKey].connections.push({
          node: edge.from,
          input: edge.from_port || "output_1",
        });
      }
    }
  });

  canvasEditor.import(drawflowData);
}

/* =============== Model Generation =============== */

async function generateModel() {
  if (!_canvasProjectId) return;
  const btn = document.getElementById("canvas-generate-btn");
  btn.disabled = true;
  btn.textContent = "生成中...";

  // Read user-defined model name (auto-generated default shown as placeholder)
  const nameInput = document.getElementById("canvas-model-name");
  const modelName = nameInput ? nameInput.value.trim() : "";

  try {
    // Auto-save canvas first
    const canvasData = collectCanvasData();
    await _saveCanvas(_canvasProjectId, canvasData);

    // Call generation API with optional model name
    const result = await _generateCanvasModel(_canvasProjectId, modelName);
    btn.textContent = "✓ 已生成";
    btn.style.background = "#059669";

    // Register in training dropdown
    if (typeof registerCanvasModel === "function") {
      registerCanvasModel(result.model_type, result.model_type);
    }

    // Show success + redirect option
    const msg = `模型「${result.model_type}」生成成功！\n现在可以前往 Step 4 训练此模型。`;
    if (confirm(msg + "\n\n跳转到训练配置？")) {
      toggleCanvasView(false);
      goToStep(4);
      // Refresh training model dropdown
      if (typeof updateModelOptions === "function") {
        const taskType = document.getElementById("taskTypeSelect").value;
        updateModelOptions(taskType);
      }
    }

    setTimeout(() => {
      btn.textContent = "生成模型";
      btn.style.background = "#7C3AED";
      btn.disabled = false;
    }, 5000);
  } catch (err) {
    btn.textContent = "✗ 失败";
    btn.style.background = "#DC2626";
    alert("模型生成失败: " + err.message);
    setTimeout(() => {
      btn.textContent = "生成模型";
      btn.style.background = "#7C3AED";
      btn.disabled = false;
    }, 3000);
  }
}

/* =============== API Integration =============== */

async function saveCanvas() {
  if (!_canvasProjectId) return;
  const saveBtn = document.getElementById("canvas-save-btn");
  saveBtn.disabled = true;
  saveBtn.textContent = "保存中...";

  try {
    const canvasData = collectCanvasData();
    await _saveCanvas(_canvasProjectId, canvasData);
    saveBtn.textContent = "✓ 已保存";
    setTimeout(() => { saveBtn.textContent = "保存"; saveBtn.disabled = false; }, 2000);
  } catch (err) {
    saveBtn.textContent = "✗ 保存失败";
    saveBtn.disabled = false;
    console.error("saveCanvas error:", err);
  }
}

async function loadCanvas(projectId) {
  try {
    const canvas = await _loadCanvas(projectId);
    if (canvas && canvas.nodes && canvas.nodes.length) {
      renderCanvasFromData(canvas);
    }
  } catch (err) {
    console.error("loadCanvas error:", err);
  }
}

function resetCanvas() {
  _dropZoneReady = false;
  _selectedNodeId = null;
  _canvasProjectId = null;
  _pendingCanvasData = null;
  deselectNode();
  if (canvasEditor) {
    canvasEditor.import({ drawflow: { Home: { data: {} } } });
  }
}

function clearCanvas() {
  if (!canvasEditor) return;
  if (canvasEditor.export().drawflow.Home.data && Object.keys(canvasEditor.export().drawflow.Home.data).length > 0) {
    if (!confirm("清空画布？")) return;
  }
  canvasEditor.import({ drawflow: { Home: { data: {} } } });
  deselectNode();
  markCanvasDirty();
}

/* =============== View Toggle =============== */

function toggleCanvasView(show) {
  const canvasSection = document.getElementById("canvas-section");
  const trainingFlow = document.getElementById("trainingFlow");
  const stepsNav = document.getElementById("stepsNav");
  const canvasToggle = document.getElementById("canvasToggleBtn");

  if (show) {
    canvasSection.style.display = "block";
    trainingFlow.style.display = "none";
    stepsNav.style.display = "none";
    if (canvasToggle) canvasToggle.textContent = "Training";
    // Lazy init — container is now visible (display:block), Drawflow can measure dimensions
    initCanvas();
    setupCanvasDropZone();
    renderComponentPanel();
    // Render any canvas data stored during project activation
    if (_pendingCanvasData) {
      renderCanvasFromData(_pendingCanvasData);
      _pendingCanvasData = null;
    }
  } else {
    canvasSection.style.display = "none";
    trainingFlow.style.display = "block";
    stepsNav.style.display = "flex";
    if (canvasToggle) canvasToggle.textContent = "Canvas";
  }
}

/* =============== Init on project activate =============== */

function initCanvasForProject(projectId, canvasData) {
  _canvasProjectId = projectId;
  _pendingCanvasData = canvasData || null;
  // Defer Drawflow init to toggleCanvasView(true) — container is hidden during project activation
  // initCanvas(), setupCanvasDropZone(), renderComponentPanel() are called there
}
