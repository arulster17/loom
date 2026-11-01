import { describe, expect, it } from "vitest";

import { CSV_HEADER, groupUsage, parseDayRange, rangeBounds, totalsOf, usageCsv, type UsageRow } from "./usage";

function row(over: Partial<UsageRow>): UsageRow {
  return {
    day: "2026-10-30",
    modelId: "qwen3-8b",
    apiKeyId: "k1",
    keyName: "prod",
    keyPrefix: "loom_sk_AAAAAAAAAAAA",
    playground: false,
    requests: 1,
    promptTokens: 10,
    completionTokens: 5,
    cachedTokens: 0,
    costMicros: 100n,
    ...over,
  };
}

describe("parseDayRange", () => {
  const now = new Date("2026-11-01T12:00:00Z");

  it("defaults to the last 30 days including today (UTC)", () => {
    expect(parseDayRange({}, now)).toEqual({ from: "2026-10-03", to: "2026-11-01" });
  });

  it("accepts valid days and clamps the future and inverted ranges", () => {
    expect(parseDayRange({ from: "2026-10-01", to: "2026-10-15" }, now)).toEqual({ from: "2026-10-01", to: "2026-10-15" });
    expect(parseDayRange({ from: "2026-10-01", to: "2027-01-01" }, now).to).toBe("2026-11-01");
    expect(parseDayRange({ from: "2026-10-20", to: "2026-10-10" }, now)).toEqual({ from: "2026-10-10", to: "2026-10-10" });
  });

  it("ignores junk and impossible dates", () => {
    expect(parseDayRange({ from: "2026-02-30", to: "x" }, now)).toEqual({ from: "2026-10-03", to: "2026-11-01" });
    expect(parseDayRange({ from: ["2026-10-01", "2026-09-01"] }, now).from).toBe("2026-10-01");
  });

  it("limits the range to a year", () => {
    expect(parseDayRange({ from: "2020-01-01" }, now).from).toBe("2025-11-01");
  });

  it("turns the inclusive day range into a half-open time range", () => {
    const [s, e] = rangeBounds({ from: "2026-10-01", to: "2026-10-01" });
    expect(s.toISOString()).toBe("2026-10-01T00:00:00.000Z");
    expect(e.toISOString()).toBe("2026-10-02T00:00:00.000Z");
  });
});

describe("grouping", () => {
  const rows = [
    row({ day: "2026-10-30", modelId: "qwen3-8b", costMicros: 100n }),
    row({ day: "2026-10-31", modelId: "llama-3.3-70b-instruct-fp8", costMicros: 900n, apiKeyId: "k2", keyName: "" }),
    row({ day: "2026-10-31", modelId: "qwen3-8b", costMicros: 50n, apiKeyId: "pg1", playground: true }),
    row({ day: "2026-10-31", modelId: "qwen3-8b", costMicros: 50n, apiKeyId: "pg2", playground: true }),
  ];

  it("totals everything exactly", () => {
    expect(totalsOf(rows)).toEqual({ requests: 4, promptTokens: 40, completionTokens: 20, cachedTokens: 0, costMicros: 1100n });
  });

  it("groups by day (newest first), model and key (costliest first)", () => {
    expect(groupUsage(rows, "day").map((g) => [g.key, g.costMicros])).toEqual([
      ["2026-10-31", 1000n],
      ["2026-10-30", 100n],
    ]);
    expect(groupUsage(rows, "model").map((g) => g.key)).toEqual(["llama-3.3-70b-instruct-fp8", "qwen3-8b"]);
    const byKey = groupUsage(rows, "key");
    expect(byKey.map((g) => g.label)).toEqual(["loom_sk_AAAAAAAAAAAA…", "prod (loom_sk_AAAAAAAAAAAA…)", "Playground"]);
    expect(byKey.find((g) => g.key === "playground")?.costMicros).toBe(100n); // rotated keys merge
  });
});

describe("CSV export", () => {
  it("has a header, exact decimals and CRLF lines", () => {
    const csv = usageCsv([row({ costMicros: 261n })]);
    const lines = csv.split("\r\n");
    expect(lines[0]).toBe(CSV_HEADER.join(","));
    expect(lines[1]).toBe("2026-10-30,qwen3-8b,prod,loom_sk_AAAAAAAAAAAA,1,10,5,0,0.000261");
    expect(csv.endsWith("\r\n")).toBe(true);
  });

  it("quotes delimiters and defuses spreadsheet formulas in key names", () => {
    const csv = usageCsv([row({ keyName: 'a,"b"' }), row({ keyName: "=HYPERLINK(1)" })]);
    expect(csv).toContain('"a,""b"""');
    expect(csv).toContain("'=HYPERLINK(1)");
  });
});
