import type { ReactNode } from "react";
import { fmt } from "../../utils/format";

/** Categorical slots in fixed order (validated reference palette). Never cycle past the list. */
export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
/** Sequential blue ramp, light -> dark, for magnitude (heat-maps). */
export const SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
export const AXIS = { stroke: "#94a3b8", fontSize: 11, tickLine: false } as const;
export const GRID = { stroke: "#e2e8f0", strokeDasharray: "3 3", vertical: false } as const;

export function seriesColor(i: number) {
  return SERIES[Math.min(i, SERIES.length - 1)];
}

export function sequentialColor(t: number) {
  const idx = Math.round(Math.max(0, Math.min(1, t)) * (SEQUENTIAL.length - 1));
  return SEQUENTIAL[idx];
}

/** Diverging (correlations): blue for negative, orange for positive, grey at zero. */
export function divergingColor(v: number) {
  const a = Math.min(1, Math.abs(v));
  const mix = (c: number[]) => {
    const base = [241, 245, 249];
    return `rgb(${base.map((b, i) => Math.round(b + (c[i] - b) * a)).join(",")})`;
  };
  return v >= 0 ? mix([235, 104, 52]) : mix([42, 120, 214]);
}

export function ChartBox({ height = 280, children }: { height?: number; children: ReactNode }) {
  return <div style={{ width: "100%", height }}>{children}</div>;
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: i.color }} aria-hidden />
          {i.label}
        </span>
      ))}
    </div>
  );
}

/** Square-cell heat-map for matrices (confusion matrices, correlations, co-missingness). */
export function MatrixHeatmap({ rows, cols, values, diverging = false, rowTitle, colTitle, cellLabel = true }: {
  rows: string[];
  cols: string[];
  values: number[][];
  diverging?: boolean;
  rowTitle?: string;
  colTitle?: string;
  cellLabel?: boolean;
}) {
  const flat = values.flat();
  const max = Math.max(...flat.map((v) => Math.abs(v)), 1e-9);
  return (
    <div className="overflow-x-auto">
      {colTitle && <p className="mb-1 text-center text-xs text-slate-500">{colTitle}</p>}
      <table className="mx-auto border-separate" style={{ borderSpacing: 2 }}>
        <thead>
          <tr>
            <th className="text-xs font-normal text-slate-400">{rowTitle}</th>
            {cols.map((c) => (
              <th key={c} className="max-w-[90px] truncate px-1 text-[10px] font-medium text-slate-500" title={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r}>
              <th className="max-w-[160px] truncate pr-2 text-right text-[11px] font-medium text-slate-600" title={r}>{r}</th>
              {cols.map((c, j) => {
                const v = values[i][j];
                const bg = diverging ? divergingColor(v) : sequentialColor(v / max);
                const dark = diverging ? Math.abs(v) > 0.6 : v / max > 0.55;
                return (
                  <td
                    key={c}
                    title={`${r} × ${c}: ${fmt(v)}`}
                    className="h-10 min-w-[40px] rounded text-center text-xs tabular-nums"
                    style={{ background: bg, color: dark ? "#fff" : "#0f172a" }}
                  >
                    {cellLabel ? fmt(v, 2) : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Tooltip body styled consistently for Recharts `content` props. */
export function TooltipBox({ title, rows }: { title?: ReactNode; rows: { label: string; value: ReactNode; color?: string }[] }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg">
      {title && <p className="mb-1 font-semibold text-slate-800">{title}</p>}
      {rows.map((r) => (
        <p key={r.label} className="flex items-center gap-2 text-slate-600">
          {r.color && <span className="h-2 w-2 rounded-full" style={{ background: r.color }} />}
          <span>{r.label}:</span>
          <span className="font-medium tabular-nums text-slate-800">{r.value}</span>
        </p>
      ))}
    </div>
  );
}
