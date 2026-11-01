import { describe, expect, it } from "vitest";

import { formatPricePerMillion, formatUsd, formatUsdExact, microsToDecimalString, parseUsdToMicros, toMicros } from "./money";

describe("formatUsd", () => {
  it("formats micro-dollars as dollars with 2 decimals", () => {
    expect(formatUsd(5_000_000)).toBe("$5.00");
    expect(formatUsd("5000000")).toBe("$5.00");
    expect(formatUsd(0)).toBe("$0.00");
    expect(formatUsd(1_234_567_890_000n)).toBe("$1,234,567.89");
  });

  it("rounds half away from zero", () => {
    expect(formatUsd(4_995_000)).toBe("$5.00");
    expect(formatUsd(4_994_999)).toBe("$4.99");
    expect(formatUsd(-1_235_000)).toBe("-$1.24");
    expect(formatUsd(-1_000)).toBe("$0.00"); // no "-$0.00"
  });

  it("supports more decimals for small amounts", () => {
    expect(formatUsd(2005, 4)).toBe("$0.0020");
    expect(formatUsd(261, 6)).toBe("$0.000261");
    expect(formatUsd(12_345_678, 0)).toBe("$12");
  });

  it("never loses precision on large amounts", () => {
    expect(formatUsd("9007199254740993000")).toBe("$9,007,199,254,740.99");
  });
});

describe("formatUsdExact and prices", () => {
  it("shows every significant micro-dollar, at least 2 decimals", () => {
    expect(formatUsdExact(261)).toBe("$0.000261");
    expect(formatUsdExact(5_000_000)).toBe("$5.00");
    expect(formatUsdExact(5_100_000)).toBe("$5.10");
    expect(formatUsdExact(-22_414)).toBe("-$0.022414");
  });

  it("formats registry prices per 1M tokens", () => {
    expect(formatPricePerMillion(77_000)).toBe("$0.077");
    expect(formatPricePerMillion(368_000)).toBe("$0.368");
    expect(formatPricePerMillion(571_000)).toBe("$0.571");
  });
});

describe("parseUsdToMicros", () => {
  it("parses what people type", () => {
    expect(parseUsdToMicros("10")).toBe(10_000_000n);
    expect(parseUsdToMicros("12.5")).toBe(12_500_000n);
    expect(parseUsdToMicros(" $1,000.25 ")).toBe(1_000_250_000n);
    expect(parseUsdToMicros("-2.50")).toBe(-2_500_000n);
    expect(parseUsdToMicros("0.000001")).toBe(1n);
    expect(parseUsdToMicros("3.")).toBe(3_000_000n);
  });

  it("refuses anything else", () => {
    for (const bad of ["", "abc", "1.0000001", "1e3", "--1", "1.2.3", "$-1"]) {
      expect(() => parseUsdToMicros(bad), bad).toThrow(RangeError);
    }
  });
});

describe("toMicros and CSV decimals", () => {
  it("accepts pg bigint strings and refuses non-integers", () => {
    expect(toMicros("-42")).toBe(-42n);
    expect(() => toMicros(1.5)).toThrow(RangeError);
    expect(() => toMicros("1.5")).toThrow(RangeError);
  });

  it("writes plain decimals for CSV", () => {
    expect(microsToDecimalString(261n)).toBe("0.000261");
    expect(microsToDecimalString(5_000_000n)).toBe("5.000000");
    expect(microsToDecimalString(-1n)).toBe("-0.000001");
  });
});
