import { describe, it, expect, beforeEach, vi } from "vitest";

/* ============ updateModelOptions ============ */

describe("updateModelOptions", () => {
  beforeEach(() => {
    const sel = document.getElementById("modelType");
    sel.innerHTML = "";
  });

  it("shows only general models for general task", () => {
    window.updateModelOptions("general");
    const sel = document.getElementById("modelType");
    const values = Array.from(sel.options).map((o) => o.value);
    expect(values).toContain("mlp");
    expect(values).toContain("cnn");
    expect(values).not.toContain("autoformer");
    expect(values).not.toContain("fedformer");
    expect(values).not.toContain("dlinear");
    expect(values).not.toContain("rnn");
  });

  it("filters to TS models for time_series", () => {
    window.updateModelOptions("time_series");
    const sel = document.getElementById("modelType");
    const values = Array.from(sel.options).map((o) => o.value);
    expect(values).toContain("rnn");
    expect(values).toContain("lstm");
    expect(values).toContain("autoformer");
    expect(values).toContain("fedformer");
    expect(values).toContain("dlinear");
    expect(values).not.toContain("cnn");
  });
});

/* ============ applyTaskConfig ============ */

describe("applyTaskConfig", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ task_config: {}, error: null }),
      }),
    );
  });

  it("collects task config from DOM and sends to server", async () => {
    document.getElementById("taskTypeSelect").value = "time_series";
    document.getElementById("timeColSelect").value = "date";
    document.getElementById("seqLenInput").value = "48";
    document.getElementById("predLenInput").value = "12";
    document.getElementById("labelLenInput").value = "6";
    document.getElementById("granularitySelect").value = "hour";

    await window.applyTaskConfig();

    expect(global.fetch).toHaveBeenCalledWith(
      "/api/data/task-config",
      expect.objectContaining({ method: "POST" }),
    );
    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.task_type).toBe("time_series");
    expect(body.seq_len).toBe(48);
    expect(body.pred_len).toBe(12);
    expect(body.label_len).toBe(6);
    expect(body.time_col).toBe("date");
    expect(body.time_granularity).toBe("hour");
  });

  it("falls back to defaults when inputs are empty", async () => {
    document.getElementById("taskTypeSelect").value = "general";
    document.getElementById("seqLenInput").value = "";
    document.getElementById("predLenInput").value = "";
    document.getElementById("labelLenInput").value = "";

    await window.applyTaskConfig();

    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.seq_len).toBe(10);
    expect(body.pred_len).toBe(1);
    expect(body.label_len).toBe(0);
  });
});

/* ============ startTraining — parameter collection ============ */

describe("startTraining — parameter collection", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    // Mock fetch for task config sync + training setup
    global.fetch = vi.fn((url, opts) => {
      if (url === "/api/train/setup") {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ success: true, error: null }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ task_config: {}, error: null }),
      });
    });

    // Restore full modelType options (may have been filtered by updateModelOptions tests)
    document.getElementById("modelType").innerHTML = `
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
      <option value="dlinear">DLinear</option>
    `;

    // Reset standard input values
    document.getElementById("testSize").value = "0.2";
    document.getElementById("modelType").value = "mlp";
    document.getElementById("hiddenLayers").value = "64,32";
    document.getElementById("learningRate").value = "0.001";
    document.getElementById("batchSize").value = "32";
    document.getElementById("epochs").value = "10";
    document.getElementById("dropout").value = "0.2";
    document.getElementById("patience").value = "10";
    document.getElementById("normalization").value = "mean";
    document.getElementById("targetCol").value = "target";
  });

  it("fails early when no target column is selected", async () => {
    document.getElementById("targetCol").value = "";

    // Suppress alert
    const spy = vi.spyOn(window, "alert").mockImplementation(() => {});

    await window.startTraining();

    expect(spy).toHaveBeenCalled();
    // Should not have called fetch at all
    expect(global.fetch).not.toHaveBeenCalledWith(
      "/api/train/setup",
      expect.anything(),
    );
  });

  it("collects MLP params correctly", async () => {
    // Suppress alert
    vi.spyOn(window, "alert").mockImplementation(() => {});
    // Mock EventSource for the SSE stream
    global.EventSource = vi.fn(() => ({
      addEventListener: vi.fn(),
      close: vi.fn(),
    }));

    await window.startTraining();

    // Find the /api/train/setup call
    const setupCalls = global.fetch.mock.calls.filter(
      (c) => c[0] === "/api/train/setup",
    );
    expect(setupCalls).toHaveLength(1);
    const body = JSON.parse(setupCalls[0][1].body);
    expect(body.model_type).toBe("mlp");
    expect(body.target_col).toBe("target");
    expect(body.hidden_layers).toBe("64,32");
    expect(body.learning_rate).toBe(0.001);
    expect(body.batch_size).toBe(32);
    expect(body.epochs).toBe(10);
    expect(body.dropout).toBe(0.2);
    expect(body.normalization).toBe("mean");
  });

  it("collects autoformer-specific params", async () => {
    vi.spyOn(window, "alert").mockImplementation(() => {});
    global.EventSource = vi.fn(() => ({
      addEventListener: vi.fn(),
      close: vi.fn(),
    }));
    document.getElementById("modelType").value = "autoformer";
    // Trigger toggle to show autoformer panel
    window.toggleModelParams();
    document.getElementById("autoDModel").value = "128";
    document.getElementById("autoNHeads").value = "4";
    document.getElementById("autoELayers").value = "2";
    document.getElementById("autoActivation").value = "relu";

    await window.startTraining();

    const setupCalls = global.fetch.mock.calls.filter(
      (c) => c[0] === "/api/train/setup",
    );
    expect(setupCalls).toHaveLength(1);
    const body = JSON.parse(setupCalls[0][1].body);
    expect(body.model_type).toBe("autoformer");
    expect(body.d_model).toBe(128);
    expect(body.n_heads).toBe(4);
    expect(body.e_layers).toBe(2);
    expect(body.activation).toBe("relu");
    // Should NOT have crossformer-specific fields
    expect(body.seg_len).toBeUndefined();
  });

  it("collects crossformer-specific params", async () => {
    vi.spyOn(window, "alert").mockImplementation(() => {});
    global.EventSource = vi.fn(() => ({
      addEventListener: vi.fn(),
      close: vi.fn(),
    }));
    document.getElementById("modelType").value = "crossformer";
    window.toggleModelParams();
    document.getElementById("crossSegLen").value = "24";
    document.getElementById("crossWinSize").value = "3";

    await window.startTraining();

    const setupCalls = global.fetch.mock.calls.filter(
      (c) => c[0] === "/api/train/setup",
    );
    expect(setupCalls).toHaveLength(1);
    const body = JSON.parse(setupCalls[0][1].body);
    expect(body.model_type).toBe("crossformer");
    expect(body.seg_len).toBe(24);
    expect(body.win_size).toBe(3);
    // Should NOT have autoformer-specific fields
    expect(body.moving_avg).toBeUndefined();
  });

  it("collects dlinear params", async () => {
    vi.spyOn(window, "alert").mockImplementation(() => {});
    global.EventSource = vi.fn(() => ({
      addEventListener: vi.fn(),
      close: vi.fn(),
    }));
    document.getElementById("modelType").value = "dlinear";
    window.toggleModelParams();
    document.getElementById("dlMovingAvg").value = "35";

    await window.startTraining();

    const setupCalls = global.fetch.mock.calls.filter(
      (c) => c[0] === "/api/train/setup",
    );
    expect(setupCalls).toHaveLength(1);
    const body = JSON.parse(setupCalls[0][1].body);
    expect(body.model_type).toBe("dlinear");
    expect(body.moving_avg).toBe(35);
  });

  it("collects etsformer-specific params", async () => {
    vi.spyOn(window, "alert").mockImplementation(() => {});
    global.EventSource = vi.fn(() => ({
      addEventListener: vi.fn(),
      close: vi.fn(),
    }));
    document.getElementById("modelType").value = "etsformer";
    window.toggleModelParams();
    document.getElementById("etsDModel").value = "128";
    document.getElementById("etsNHeads").value = "4";
    document.getElementById("etsELayers").value = "3";
    document.getElementById("etsTopK").value = "7";
    document.getElementById("etsActivation").value = "gelu";

    await window.startTraining();

    const setupCalls = global.fetch.mock.calls.filter(
      (c) => c[0] === "/api/train/setup",
    );
    expect(setupCalls).toHaveLength(1);
    const body = JSON.parse(setupCalls[0][1].body);
    expect(body.model_type).toBe("etsformer");
    expect(body.d_model).toBe(128);
    expect(body.n_heads).toBe(4);
    expect(body.e_layers).toBe(3);
    expect(body.top_k).toBe(7);
    expect(body.activation).toBe("gelu");
    // Should NOT have dlinear/autoformer-specific fields
    expect(body.moving_avg).toBeUndefined();
    expect(body.seg_len).toBeUndefined();
  });

  it("collects fedformer-specific params", async () => {
    vi.spyOn(window, "alert").mockImplementation(() => {});
    global.EventSource = vi.fn(() => ({
      addEventListener: vi.fn(),
      close: vi.fn(),
    }));
    document.getElementById("modelType").value = "fedformer";
    window.toggleModelParams();
    document.getElementById("fedDModel").value = "128";
    document.getElementById("fedNHeads").value = "4";
    document.getElementById("fedELayers").value = "2";
    document.getElementById("fedDLayers").value = "2";
    document.getElementById("fedDFF").value = "64";
    document.getElementById("fedMovingAvg").value = "15";
    document.getElementById("fedDropout").value = "0.2";
    document.getElementById("fedModes").value = "16";
    document.getElementById("fedVersion").value = "Wavelets";
    document.getElementById("fedModeSelect").value = "low";
    document.getElementById("fedActivation").value = "relu";

    await window.startTraining();

    const setupCalls = global.fetch.mock.calls.filter(
      (c) => c[0] === "/api/train/setup",
    );
    expect(setupCalls).toHaveLength(1);
    const body = JSON.parse(setupCalls[0][1].body);
    expect(body.model_type).toBe("fedformer");
    expect(body.d_model).toBe(128);
    expect(body.n_heads).toBe(4);
    expect(body.e_layers).toBe(2);
    expect(body.d_layers).toBe(2);
    expect(body.d_ff).toBe(64);
    expect(body.moving_avg).toBe(15);
    expect(body.dropout).toBe(0.2);
    expect(body.modes).toBe(16);
    expect(body.version).toBe("Wavelets");
    expect(body.mode_select).toBe("low");
    expect(body.activation).toBe("relu");
    // Should NOT have etsformer-specific fields
    expect(body.top_k).toBeUndefined();
  });

  it("collects film-specific params", async () => {
    vi.spyOn(window, "alert").mockImplementation(() => {});
    global.EventSource = vi.fn(() => ({
      addEventListener: vi.fn(),
      close: vi.fn(),
    }));
    document.getElementById("modelType").value = "film";
    window.toggleModelParams();
    document.getElementById("filmWindowSize").value = "128";
    document.getElementById("filmMultiscale").value = "1,2";
    document.getElementById("filmDropout").value = "0.2";

    await window.startTraining();

    const setupCalls = global.fetch.mock.calls.filter(
      (c) => c[0] === "/api/train/setup",
    );
    expect(setupCalls).toHaveLength(1);
    const body = JSON.parse(setupCalls[0][1].body);
    expect(body.model_type).toBe("film");
    expect(body.window_size).toBe("128");
    expect(body.multiscale).toBe("1,2");
    expect(body.dropout).toBe(0.2);
    // Should NOT have fedformer-specific fields
    expect(body.modes).toBeUndefined();
  });
});
