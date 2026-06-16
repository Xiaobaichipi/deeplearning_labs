import { describe, it, expect, beforeEach, vi } from "vitest";

/* ============ toggleModelParams ============ */

describe("toggleModelParams", () => {
  beforeEach(() => {
    // Reset all param panels to hidden
    document.getElementById("mlpParams").style.display = "none";
    document.getElementById("cnnParams").style.display = "none";
    document.getElementById("seqParams").style.display = "none";
    document.getElementById("transformerParams").style.display = "none";
    document.getElementById("autoformerParams").style.display = "none";
    document.getElementById("informerParams").style.display = "none";
    document.getElementById("crossformerParams").style.display = "none";
    document.getElementById("dlinearParams").style.display = "none";
  });

  it("shows mlpParams for mlp, hides all others", () => {
    document.getElementById("modelType").value = "mlp";
    window.toggleModelParams();
    expect(document.getElementById("mlpParams").style.display).toBe("block");
    expect(document.getElementById("cnnParams").style.display).toBe("none");
    expect(document.getElementById("autoformerParams").style.display).toBe("none");
  });

  it("shows seqParams for rnn/lstm/gru", () => {
    for (const t of ["rnn", "lstm", "gru"]) {
      document.getElementById("modelType").value = t;
      window.toggleModelParams();
      expect(document.getElementById("seqParams").style.display).toBe("block");
      expect(document.getElementById("transformerParams").style.display).toBe("none");
      expect(document.getElementById("mlpParams").style.display).toBe("none");
    }
  });

  it("shows transformerParams only for transformer", () => {
    document.getElementById("modelType").value = "transformer";
    window.toggleModelParams();
    expect(document.getElementById("transformerParams").style.display).toBe("block");
    expect(document.getElementById("seqParams").style.display).toBe("none");
  });

  it("shows autoformerParams only for autoformer", () => {
    document.getElementById("modelType").value = "autoformer";
    window.toggleModelParams();
    expect(document.getElementById("autoformerParams").style.display).toBe("block");
  });

  it("shows informerParams only for informer", () => {
    document.getElementById("modelType").value = "informer";
    window.toggleModelParams();
    expect(document.getElementById("informerParams").style.display).toBe("block");
  });

  it("shows crossformerParams only for crossformer", () => {
    document.getElementById("modelType").value = "crossformer";
    window.toggleModelParams();
    expect(document.getElementById("crossformerParams").style.display).toBe("block");
  });

  it("shows dlinearParams only for dlinear", () => {
    document.getElementById("modelType").value = "dlinear";
    window.toggleModelParams();
    expect(document.getElementById("dlinearParams").style.display).toBe("block");
  });
});

/* ============ populateModelDropdown ============ */

describe("populateModelDropdown", () => {
  const models = [
    { id: "m1", model_type: "lstm", is_time_series: true, final_metrics: { epochs: 10 } },
    { id: "m2", model_type: "mlp", is_time_series: false, final_metrics: { epochs: 5 } },
    { id: "m3", model_type: "gru", is_time_series: true, final_metrics: {} },
    { id: "m4", model_type: "cnn", is_time_series: false, final_metrics: null },
  ];

  beforeEach(() => {
    const sel = document.getElementById("modelSelect");
    sel.innerHTML = '<option value="">-- Select --</option>';
    document.getElementById("modelSelector").style.display = "none";
  });

  it("filters to time-series models in TS mode", () => {
    const result = window.populateModelDropdown(models, "time_series");
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe("m1");
    expect(result[1].id).toBe("m3");
    const sel = document.getElementById("modelSelect");
    expect(sel.options.length).toBe(3); // placeholder + 2 models
  });

  it("filters to general models in general mode", () => {
    const result = window.populateModelDropdown(models, "regression");
    expect(result).toHaveLength(2);
    expect(result[0].model_type).toBe("mlp");
    expect(result[1].model_type).toBe("cnn");
  });

  it("shows placeholder when no models exist", () => {
    const result = window.populateModelDropdown([], "time_series");
    expect(result).toHaveLength(0);
    expect(document.getElementById("modelSelector").style.display).toBe("none");
  });

  it("shows disabled option when no models match task type", () => {
    const tsOnly = [{ id: "m1", model_type: "lstm", is_time_series: true }];
    const result = window.populateModelDropdown(tsOnly, "regression");
    expect(result).toHaveLength(0);
    const sel = document.getElementById("modelSelect");
    expect(sel.options[1].disabled).toBe(true);
    expect(sel.options[1].textContent).toContain("general");
  });

  it("returns empty array when select element is missing", () => {
    const orig = document.getElementById("modelSelect");
    orig.parentNode.removeChild(orig);
    const result = window.populateModelDropdown(models, "time_series");
    expect(result).toHaveLength(0);
    // restore
    document.getElementById("modelSelector").appendChild(orig);
  });
});

/* ============ esc (XSS protection) ============ */

describe("esc", () => {
  it("escapes HTML special characters", () => {
    expect(window.esc('<script>alert("xss")</script>')).toBe(
      '&lt;script&gt;alert("xss")&lt;/script&gt;',
    );
  });

  it("returns empty string for empty input", () => {
    expect(window.esc("")).toBe("");
  });

  it("preserves normal text", () => {
    expect(window.esc("hello world")).toBe("hello world");
  });
});

/* ============ showLoadedModelBadge ============ */

describe("showLoadedModelBadge", () => {
  beforeEach(() => {
    const badge = document.getElementById("loadedModelBadge");
    badge.style.display = "none";
    badge.textContent = "";
  });

  it("shows badge with model type text when ok", () => {
    window.showLoadedModelBadge(true, "lstm");
    const badge = document.getElementById("loadedModelBadge");
    expect(badge.style.display).toBe("inline-block");
    expect(badge.textContent).toBe("Loaded: lstm");
  });

  it("hides badge when not ok", () => {
    window.showLoadedModelBadge(false);
    expect(document.getElementById("loadedModelBadge").style.display).toBe("none");
  });
});

/* ============ showTaskConfigSaved ============ */

describe("showTaskConfigSaved", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    document.getElementById("taskConfigBadge").style.display = "none";
  });

  it("shows badge then hides after 3 seconds", () => {
    window.showTaskConfigSaved();
    const badge = document.getElementById("taskConfigBadge");
    expect(badge.style.display).toBe("inline-block");
    expect(badge.textContent).toBe("Saved");
    vi.advanceTimersByTime(3000);
    expect(badge.style.display).toBe("none");
  });
});

/* ============ onTaskTypeChange ============ */

describe("onTaskTypeChange", () => {
  beforeEach(() => {
    document.getElementById("tsConfigFields").style.display = "none";
    document.getElementById("taskConfigBadge").style.display = "inline-block";
  });

  it("shows ts config fields for time_series", () => {
    document.getElementById("taskTypeSelect").value = "time_series";
    window.onTaskTypeChange();
    expect(document.getElementById("tsConfigFields").style.display).toBe("flex");
  });

  it("hides ts config and badge for general", () => {
    document.getElementById("taskTypeSelect").value = "general";
    window.onTaskTypeChange();
    expect(document.getElementById("tsConfigFields").style.display).toBe("none");
    expect(document.getElementById("taskConfigBadge").style.display).toBe("none");
  });
});

/* ============ goToStep ============ */

describe("goToStep", () => {
  beforeEach(() => {
    document.querySelectorAll(".step-item").forEach((s) => s.classList.remove("active"));
    document.querySelectorAll(".section").forEach((s) => s.classList.remove("active"));
  });

  it("activates the target step and section", () => {
    window.goToStep(3);
    expect(document.querySelector('.step-item[data-step="3"]').classList.contains("active")).toBe(true);
  });
});

/* ============ showTrainError ============ */

describe("showTrainError", () => {
  it("shows error message", () => {
    window.showTrainError("Something went wrong");
    const div = document.getElementById("trainError");
    expect(div.textContent).toBe("Something went wrong");
    expect(div.style.display).toBe("block");
  });
});
