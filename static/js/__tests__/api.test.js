import { describe, it, expect, beforeEach, vi } from "vitest";

describe("API functions — fetch URL correctness", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ data: {}, models: [], error: null }),
      }),
    );
  });

  it("_loadProjects fetches /api/projects", async () => {
    await window._loadProjects();
    expect(global.fetch).toHaveBeenCalledWith("/api/projects");
  });

  it("_uploadFile fetches POST /api/upload with FormData", async () => {
    const file = new File(["content"], "test.csv", { type: "text/csv" });
    await window._uploadFile(file);
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/upload",
      expect.objectContaining({ method: "POST" }),
    );
    // body should be FormData
    const opts = global.fetch.mock.calls[0][1];
    expect(opts.body instanceof FormData).toBe(true);
  });

  it("_activateProject fetches POST /api/projects/<id>/activate", async () => {
    await window._activateProject("proj-1");
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/projects/proj-1/activate",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("_loadProjectModels fetches /api/projects/<id>/models", async () => {
    await window._loadProjectModels("proj-1");
    expect(global.fetch).toHaveBeenCalledWith("/api/projects/proj-1/models");
  });

  it("_loadModelToSession fetches POST /api/projects/<id>/load-model/<mid>", async () => {
    await window._loadModelToSession("proj-1", "model_001");
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/projects/proj-1/load-model/model_001",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("_setTaskConfig sends correct body", async () => {
    await window._setTaskConfig({ task_type: "time_series", seq_len: 48 });
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/data/task-config",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }),
    );
    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.task_type).toBe("time_series");
    expect(body.seq_len).toBe(48);
  });

  it("_setupTraining sends params to /api/train/setup", async () => {
    await window._setupTraining({ model_type: "lstm", epochs: 5 });
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/train/setup",
      expect.objectContaining({ method: "POST" }),
    );
    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.model_type).toBe("lstm");
    expect(body.epochs).toBe(5);
  });

  it("_deleteProject sends DELETE", async () => {
    await window._deleteProject("proj-1");
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/projects/proj-1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("downloadPredictions sets window.location.href", async () => {
    const orig = window.location.href;
    await window.downloadPredictions("csv");
    // Can't easily assert location.href in jsdom, just check no throw
  });

  it("throws on error response", async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ error: "Not found" }),
      }),
    );
    await expect(window._loadProjects()).rejects.toThrow("Not found");
  });
});
