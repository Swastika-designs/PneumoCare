/** Display formatting shared by tables, cards and chart tooltips. */
export function fmt(value: unknown, digits = 3): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "∞";
    if (Number.isInteger(value)) return value.toLocaleString();
    const abs = Math.abs(value);
    if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
    if (abs > 0 && abs < 0.001) return value.toExponential(2);
    return value.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return value.map((v) => fmt(v)).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`;

export const pctRaw = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "—" : `${v.toFixed(digits)}%`;

export const money = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : v.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export const dateTime = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";
