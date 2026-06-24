/* Vitest global setup — loads frontend JS sources into jsdom
   so tests can access functions through window.xxx exports. */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/* ── Minimal DOM skeleton matching index.html ── */
document.body.innerHTML = `
<script id="config-data" type="application/json">${JSON.stringify({
  training: { test_size: 0.2, learning_rate: 0.001, batch_size: 32, epochs: 10, dropout: 0.2, patience: 10 },
  model: {
    mlp: { hidden_layers: "128,64,32" },
    cnn: { hidden_channels: 64, kernel_size: 3 },
    rnn: { hidden_size: 128, num_layers: 2 },
    transformer: { d_model: 128, nhead: 4, dim_feedforward: 256, num_layers: 2 },
    autoformer: { d_model: 256, n_heads: 8, e_layers: 3, d_layers: 2, d_ff: 32, moving_avg: 25, factor: 3, activation: "gelu" },
    informer: { d_model: 256, n_heads: 8, e_layers: 3, d_layers: 2, d_ff: 32, factor: 3, activation: "gelu" },
    crossformer: { d_model: 256, n_heads: 8, e_layers: 3, d_ff: 32, factor: 3, seg_len: 12, win_size: 2, activation: "gelu" },
    etsformer: { d_model: 256, n_heads: 8, e_layers: 2, d_ff: 32, top_k: 5, dropout: 0.1, activation: "sigmoid" },
    fedformer: { d_model: 256, n_heads: 8, e_layers: 3, d_layers: 3, d_ff: 32, moving_avg: 25, dropout: 0.1, modes: 32, version: "Fourier", mode_select: "random", activation: "gelu" },
    film: { window_size: "256", multiscale: "1,2,4", dropout: 0.1 },
    vanilla_transformer: { d_model: 256, n_heads: 8, e_layers: 3, d_layers: 3, d_ff: 32, dropout: 0.1, activation: "gelu" },
    dlinear: { moving_avg: 25, individual: false },
    random_forest_regressor: { n_estimators: 100, max_depth: null, min_samples_split: 2, min_samples_leaf: 1 },
    random_forest_classifier: { n_estimators: 100, max_depth: null, min_samples_split: 2, min_samples_leaf: 1 },
    xgboost_regressor: { n_estimators: 100, max_depth: null, min_samples_split: 2, min_samples_leaf: 1 },
    xgboost_classifier: { n_estimators: 100, max_depth: null, min_samples_split: 2, min_samples_leaf: 1 },
    lightgbm_regressor: { n_estimators: 100, max_depth: null, min_samples_split: 2, min_samples_leaf: 1 },
    lightgbm_classifier: { n_estimators: 100, max_depth: null, min_samples_split: 2, min_samples_leaf: 1 },
    decision_tree_regressor: { max_depth: null, min_samples_split: 2, min_samples_leaf: 1 },
    decision_tree_classifier: { max_depth: null, min_samples_split: 2, min_samples_leaf: 1 },
  },
  cv: { default_folds: 5 },
  devices: ["cpu"],
})}</script>
<div id="mlpParams" style="display:block"></div>
<div id="cnnParams" style="display:none"></div>
<div id="seqParams" style="display:none"></div>
<div id="transformerParams" style="display:none"></div>
<div id="autoformerParams" style="display:none"></div>
<div id="informerParams" style="display:none"></div>
<div id="crossformerParams" style="display:none"></div>
<div id="etsformerParams" style="display:none"></div>
<div id="fedformerParams" style="display:none"></div>
<div id="filmParams" style="display:none"></div>
<div id="vanillaTransformerParams" style="display:none">
  <input id="vanillaDModel" value="256">
  <input id="vanillaNHeads" value="8">
  <input id="vanillaELayers" value="3">
  <input id="vanillaDLayers" value="3">
  <input id="vanillaDFF" value="32">
  <input id="vanillaDropout" value="0.1">
  <select id="vanillaActivation"><option value="gelu">GELU</option><option value="relu">ReLU</option></select>
</div>
<div id="dlinearParams" style="display:none"></div>
<div id="classicalMlParams" style="display:none">
  <div id="classicalNEstimatorsRow">
    <input id="classicalNEstimators" value="100">
  </div>
  <input id="classicalMaxDepth" value="">
  <input id="classicalMinSamplesSplit" value="2">
  <input id="classicalMinSamplesLeaf" value="1">
</div>
<select id="modelType">
  <option value="mlp">MLP</option>
  <option value="cnn">CNN</option>
  <option value="rnn">RNN</option>
  <option value="lstm">LSTM</option>
  <option value="gru">GRU</option>
  <option value="transformer">Transformer</option>
  <option value="autoformer">Autoformer</option>
  <option value="informer">Informer</option>
  <option value="crossformer">Crossformer</option>
  <option value="etsformer">ETSformer</option>
  <option value="fedformer">FEDformer</option>
  <option value="film">FiLM</option>
  <option value="vanilla_transformer">Vanilla Transformer</option>
  <option value="dlinear">DLinear</option>
  <option value="random_forest_regressor">Random Forest Regressor</option>
  <option value="random_forest_classifier">Random Forest Classifier</option>
  <option value="xgboost_regressor">XGBoost Regressor</option>
  <option value="xgboost_classifier">XGBoost Classifier</option>
  <option value="lightgbm_regressor">LightGBM Regressor</option>
  <option value="lightgbm_classifier">LightGBM Classifier</option>
  <option value="decision_tree_regressor">Decision Tree Regressor</option>
  <option value="decision_tree_classifier">Decision Tree Classifier</option>
</select>
<div id="modelSelector" style="display:none">
  <select id="modelSelect"><option value="">-- Select --</option></select>
</div>
<div id="loadedModelBadge" style="display:none"></div>
<div id="headerMenu" class="header-dropdown-menu"></div>
<select id="taskTypeSelect"><option value="general">General</option><option value="time_series">Time Series</option></select>
<div id="tsConfigFields" style="display:none"></div>
<select id="timeColSelect"><option value="">-- Select --</option><option value="date">date</option></select>
<div class="section" id="step3" data-step="3"></div>
<div class="section" id="step4" data-step="4"></div>
<input id="seqLenInput" value="10">
<input id="predLenInput" value="1">
<input id="labelLenInput" value="0">
<select id="granularitySelect"><option value="auto">Auto</option><option value="hour">Hour</option></select>
<div id="taskConfigBadge" style="display:none"></div>
<div id="trainingSummary" style="display:none"></div>
<div id="trainingProgress" style="display:none">
  <div id="trainingCharts"></div>
  <div id="progressCurrentEpoch">0</div>
  <div id="progressTotalEpochs">--</div>
  <div id="progressBar"></div>
  <div id="liveTrainLoss">--</div>
  <div id="liveValLoss">--</div>
  <div id="liveTrainMetric">--</div>
  <div id="liveValMetric">--</div>
</div>
<div id="cvResults" style="display:none"></div>
<div id="predResults" style="display:none">
  <div id="predChart" style="display:none"></div>
  <div id="predDownload" style="display:none"></div>
</div>
<div id="evalMetrics" style="display:none"></div>
<select id="targetCol"><option value="">-- Select --</option><option value="target">target</option></select>
<input id="testSize" value="0.2">
<input id="hiddenLayers" value="128,64,32">
<input id="hiddenChannels" value="64">
<input id="kernelSize" value="3">
<input id="rnnHiddenSize" value="128">
<input id="rnnNumLayers" value="2">
<input id="transNumLayers" value="2">
<div class="toggle" id="toggleBidirectional"></div>
<input id="transDModel" value="128">
<input id="transNhead" value="4">
<input id="transDimFeedforward" value="256">
<input id="learningRate" value="0.001">
<div id="trainingHyperparams">
<input id="batchSize" value="32">
<input id="epochs" value="10">
<input id="dropout" value="0.2">
<input id="patience" value="10">
<select id="normalization"><option value="none">None</option><option value="minmax">Min-Max</option><option value="mean">Mean</option></select>
<select id="deviceSelect"><option value="cpu">cpu</option></select>
</div>
<button id="trainBtn"></button>
<button id="applyTaskConfigBtn">Apply</button>
<div id="trainError" style="display:none"></div>
<input id="autoDModel" value="256">
<input id="autoNHeads" value="8">
<input id="autoELayers" value="3">
<input id="autoDLayers" value="2">
<input id="autoDFF" value="32">
<input id="autoMovingAvg" value="25">
<input id="autoFactor" value="3">
<select id="autoActivation"><option value="gelu">GELU</option><option value="relu">ReLU</option></select>
<input id="infoDModel" value="256">
<input id="infoNHeads" value="8">
<input id="infoELayers" value="3">
<input id="infoDLayers" value="2">
<input id="infoDFF" value="32">
<input id="infoFactor" value="3">
<div class="toggle active" id="toggleDistil"></div>
<select id="infoActivation"><option value="gelu">GELU</option><option value="relu">ReLU</option></select>
<input id="crossDModel" value="256">
<input id="crossNHeads" value="8">
<input id="crossELayers" value="3">
<input id="crossDFF" value="32">
<input id="crossFactor" value="3">
<input id="crossSegLen" value="12">
<input id="crossWinSize" value="2">
<input id="etsDModel" value="256">
<input id="etsNHeads" value="8">
<input id="etsELayers" value="2">
<input id="etsDFF" value="32">
<input id="etsTopK" value="5">
<input id="etsDropout" value="0.1">
<select id="etsActivation"><option value="sigmoid">Sigmoid</option><option value="gelu">GELU</option></select>
<input id="fedDModel" value="256">
<input id="fedNHeads" value="8">
<input id="fedELayers" value="3">
<input id="fedDLayers" value="3">
<input id="fedDFF" value="32">
<input id="fedMovingAvg" value="25">
<input id="fedDropout" value="0.1">
<input id="fedModes" value="32">
<select id="fedVersion"><option value="Fourier">Fourier</option><option value="Wavelets">Wavelets</option></select>
<select id="fedModeSelect"><option value="random">Random</option><option value="low">Low</option></select>
<select id="fedActivation"><option value="gelu">GELU</option><option value="relu">ReLU</option></select>
<input id="filmWindowSize" value="256">
<input id="filmMultiscale" value="1,2,4">
<input id="filmDropout" value="0.1">
<select id="crossActivation"><option value="gelu">GELU</option><option value="relu">ReLU</option></select>
<input id="dlMovingAvg" value="25">
<div class="toggle" id="toggleIndividual"></div>
<input type="hidden" id="dlIndividual" value="false">
<canvas id="lossChart"></canvas>
<canvas id="metricChart"></canvas>
<div id="dataStats"></div>
<div id="previewTable"></div>
<div id="dataInfoText"></div>
<div id="describeTable"></div>
<div id="vizImages"></div>
<div id="corrImage"></div>
<div id="uploadStatus" style="display:none"></div>
<div id="cleanReport" style="display:none"></div>
<div id="cleanStats" style="display:none"></div>
<div id="cleanTable" style="display:none"></div>
<div id="evalImages"></div>
<div id="modelList"></div>
<div id="modelCompareAction" style="display:none"></div>
<div id="modelCompareResult" style="display:none"></div>
<div id="modelCompareChart"></div>
<div id="outlierOptions" style="display:none"></div>
<input id="outlierFactor" value="1.5">
<select id="fillStrategy"><option value="mean">Mean</option></select>
<div id="fillConstantGroup" style="display:none"></div>
<input id="fillConstantValue">
<div id="projectGrid"></div>
<div id="projectEmpty"></div>
<div id="newProjectModal"></div>
<input id="projectName">
<input id="projectFileInput" type="file">
<div id="projectUploadText">Click to select file</div>
<button id="createProjectBtn"></button>
<div class="toggle" id="toggleDedup"></div>
<div class="toggle" id="toggleOutliers"></div>
<button id="evalBtn"></button>
<button id="validBtn"></button>
<button id="predBtn"></button>
<input id="cvFolds" value="5">
<div id="summaryMetrics"></div>
<div id="trainingInfo"></div>
<div id="uploadZone"><input id="fileInput" type="file" style="display:none"></div>
<span class="toggle" id="toggleOutliers"></span>
<select id="fillStrategy"><option value="mean">Mean</option></select>
<div class="step-item" data-step="1"></div>
<div class="step-item" data-step="2"></div>
<div class="step-item" data-step="3"></div>
<div class="step-item" data-step="4"></div>
<div class="step-item" data-step="5"></div>
<div class="step-item" data-step="6"></div>
<div class="tabs"><div class="tab" data-tab="clean"></div></div>
`;

/* ── Load source files into global scope via indirect eval ── */
const jsFiles = ["state.js", "filterUtils.js", "ui.js", "api.js", "canvas.js", "app.js"];
for (const file of jsFiles) {
  const code = fs.readFileSync(
    path.resolve(__dirname, "static/js", file),
    "utf-8",
  );
  (0, globalThis.eval)(code);
}
