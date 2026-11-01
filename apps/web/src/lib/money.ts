// Money is integer micro-dollars everywhere (1 USD = 1,000,000 micros), as in the ledger.
// Postgres bigint arrives from `pg` as a string, so every helper accepts string | number |
// bigint and does its arithmetic in BigInt: no floating point touches an amount.

export const MICROS_PER_USD = 1_000_000n;

export type MicrosLike = bigint | number | string;

export function toMicros(v: MicrosLike): bigint {
  if (typeof v === "bigint") return v;
  if (typeof v === "number") {
    if (!Number.isSafeInteger(v)) throw new RangeError(`not an integer amount: ${v}`);
    return BigInt(v);
  }
  if (!/^-?\d+$/.test(v)) throw new RangeError(`not an integer amount: ${JSON.stringify(v)}`);
  return BigInt(v);
}

function groupThousands(digits: string): string {
  return digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Dollars with a fixed number of decimals (default 2), rounded half away from zero.
 * formatUsd(5_000_000) -> "$5.00"; formatUsd(-1_234_567) -> "-$1.23".
 */
export function formatUsd(micros: MicrosLike, decimals = 2): string {
  if (!Number.isInteger(decimals) || decimals < 0 || decimals > 6) {
    throw new RangeError("decimals must be 0..6");
  }
  const m = toMicros(micros);
  const neg = m < 0n;
  const abs = neg ? -m : m;
  const unit = 10n ** BigInt(6 - decimals); // micros per displayed unit
  const units = (abs + unit / 2n) / unit; // half away from zero (on the absolute value)
  const scale = 10n ** BigInt(decimals);
  const whole = units / scale;
  const frac = units % scale;
  const body =
    decimals === 0
      ? groupThousands(whole.toString())
      : `${groupThousands(whole.toString())}.${frac.toString().padStart(decimals, "0")}`;
  return `${neg && units !== 0n ? "-" : ""}$${body}`;
}

/**
 * Exact dollars: all six decimals when needed, at least two.
 * 261 -> "$0.000261"; 77_000 -> "$0.077"; 5_000_000 -> "$5.00".
 */
export function formatUsdExact(micros: MicrosLike): string {
  const m = toMicros(micros);
  const neg = m < 0n;
  const abs = neg ? -m : m;
  const whole = abs / MICROS_PER_USD;
  let frac = (abs % MICROS_PER_USD).toString().padStart(6, "0").replace(/0+$/, "");
  if (frac.length < 2) frac = frac.padEnd(2, "0");
  return `${neg ? "-" : ""}$${groupThousands(whole.toString())}.${frac}`;
}

/** Price per 1M tokens from the registry's micro-dollars per 1M: 368_000 -> "$0.368". */
export function formatPricePerMillion(microsPerMtok: MicrosLike): string {
  return formatUsdExact(microsPerMtok);
}

/**
 * Parses a dollar amount typed by a person ("12", "12.5", "$1,000.25", "-2.50") into
 * micros. At most 6 decimals. Throws RangeError on anything else.
 */
export function parseUsdToMicros(input: string): bigint {
  const s = input.trim().replace(/,/g, "");
  const m = /^(-)?\$?(\d+)(?:\.(\d{0,6}))?$/.exec(s);
  if (!m) throw new RangeError("Enter an amount in dollars, like 10 or 12.50.");
  const [, sign, whole, frac = ""] = m;
  const micros = BigInt(whole) * MICROS_PER_USD + BigInt(frac.padEnd(6, "0") || "0");
  return sign ? -micros : micros;
}

/** Micros as a plain decimal string for CSV ("0.000261"), no symbol or separators. */
export function microsToDecimalString(micros: MicrosLike): string {
  const m = toMicros(micros);
  const neg = m < 0n;
  const abs = neg ? -m : m;
  const frac = (abs % MICROS_PER_USD).toString().padStart(6, "0");
  return `${neg ? "-" : ""}${abs / MICROS_PER_USD}.${frac}`;
}
