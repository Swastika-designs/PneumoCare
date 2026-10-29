import type { ReactNode } from "react";
import { fmt } from "../../utils/format";

export interface Column<T> {
  key: string;
  label: string;
  render?: (row: T) => ReactNode;
  align?: "left" | "right";
}

/** Simple scrollable table. Numbers are formatted automatically unless a column renders itself. */
export function DataTable<T extends object>({ rows, columns, maxRows = 200, caption }: {
  rows: T[];
  columns?: Column<T>[];
  maxRows?: number;
  caption?: string;
}) {
  if (!rows.length) return <p className="py-4 text-sm text-slate-500">No rows.</p>;
  const cols: Column<T>[] =
    columns ?? Object.keys(rows[0]).map((k) => ({ key: k, label: k.replace(/_/g, " ") }));
  const shown = rows.slice(0, maxRows);
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className="table-base">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.key} className={c.align === "right" ? "text-right" : ""}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 bg-white">
          {shown.map((row, i) => (
            <tr key={i} className="hover:bg-slate-50">
              {cols.map((c) => {
                const raw = (row as Record<string, unknown>)[c.key];
                const isNum = typeof raw === "number";
                return (
                  <td key={c.key} className={c.align === "right" || isNum ? "text-right tabular-nums" : ""}>
                    {c.render ? c.render(row) : fmt(raw)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > maxRows && (
        <p className="border-t border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500">
          Showing {maxRows} of {rows.length} rows.
        </p>
      )}
    </div>
  );
}
