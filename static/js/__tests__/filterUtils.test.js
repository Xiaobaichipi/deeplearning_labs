import { describe, it, expect } from "vitest";

// filterModelsByTask — extracted from filterUtils.js for unit testing.
// The source file uses global-scope function declaration (no export),
// so we mirror the implementation here.  If more functions are extracted
// later, consider a proper export/import pattern.

function filterModelsByTask(models, taskType) {
  const isTs = taskType === "time_series";
  return models.filter((m) => {
    const mTs = m.is_time_series === true;
    return isTs ? mTs : !mTs;
  });
}

describe("filterModelsByTask", () => {
  const models = [
    { id: 1, is_time_series: true },
    { id: 2, is_time_series: false },
    { id: 3 }, // is_time_series undefined
  ];

  it("time_series mode returns only TS models", () => {
    const result = filterModelsByTask(models, "time_series");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe(1);
  });

  it("general mode returns non-TS + undefined models", () => {
    const result = filterModelsByTask(models, "regression");
    expect(result).toHaveLength(2);
  });

  it("empty list returns empty array", () => {
    expect(filterModelsByTask([], "time_series")).toHaveLength(0);
  });

  it("all-TS list returns empty in general mode", () => {
    const tsOnly = [{ id: 1, is_time_series: true }];
    expect(filterModelsByTask(tsOnly, "regression")).toHaveLength(0);
  });

  it("all-general list returns empty in TS mode", () => {
    const genOnly = [{ id: 2, is_time_series: false }];
    expect(filterModelsByTask(genOnly, "time_series")).toHaveLength(0);
  });
});
