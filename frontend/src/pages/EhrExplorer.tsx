import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ChartBox, GRID, Legend, MatrixHeatmap, SERIES, TooltipBox } from "../components/charts";
import { FileDrop } from "../components/images";
import { AsyncView, Badge, Card, Field, KeyValue, Notice, PageHeader, RunButton, Select, StatCard, Tabs } from "../components/ui";
import { DataTable } from "../components/ui/DataTable";
import { useAction, useFetch } from "../hooks/useApi";
import { useTab } from "../hooks/useTab";
import { describeError, getJson, http } from "../services/api";
import type { SourceOption } from "../types/api";
import { fmt, pctRaw } from "../utils/format";

const TABS = [
  { id: "explore", label: "Explore", badge: "T1" },
  { id: "clean", label: "Standardise & clean", badge: "T2" },
  { id: "missing", label: "Missing data & redundancy", badge: "T3" },
] as const;
type TabId = (typeof TABS)[number]["id"];

interface SchemaRow { column: string; dtype: string; semantic_type: string; non_null: number; missing: number; missing_pct: number; unique: number; examples: string[] }
interface Overview {
  rows: number; columns: number; memory_mb: number; duplicate_rows: number; total_missing_cells: number;
  schema: SchemaRow[]; numeric_summary: Record<string, unknown>[]; categorical_summary: Record<string, unknown>[];
  preview: Record<string, unknown>[];
}
interface Profile { column: string; semantic_type: string; chart: string; missing: number; data: { bin: string; count: number }[]; outliers_iqr?: number }

function SourceSelect({ value, onChange, sources, onUploaded }: {
  value: string; onChange: (v: string) => void; sources: SourceOption[]; onUploaded: (id: string) => void;
}) {
  const [msg, setMsg] = useState<string | null>(null);
  const upload = async (f: File) => {
    if (!f.name.toLowerCase().endsWith(".csv")) { setMsg("Only .csv files are accepted."); return; }
    const form = new FormData();
    form.append("file", f);
    try {
      const res = await http.post<{ id: string; rows: number }>("/api/ehr/upload", form);
      setMsg(`Uploaded ${f.name} (${res.data.rows} rows).`);
      onUploaded(res.data.id);
    } catch (e) {
      setMsg(describeError(e).message);
    }
  };
  const groups = Array.from(new Set(sources.map((s) => s.group ?? "")));
  return (
    <Card title="Data source" subtitle="Synthea data are fully synthetic; the teaching variant has documented, deliberately injected errors">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Field label="Dataset">
          <select className="input" value={value} onChange={(e) => onChange(e.target.value)} aria-label="Dataset">
            {groups.map((g) => (
              <optgroup key={g} label={g}>
                {sources.filter((s) => (s.group ?? "") === g).map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
              </optgroup>
            ))}
          </select>
        </Field>
        <div>
          <span className="label">Upload your own CSV</span>
          <FileDrop label="Drop a CSV file" accept=".csv,text/csv" hint="CSV with a header row · max 10 MB, 200,000 rows" onFile={upload} />
          {msg && <p className="mt-1 text-xs text-slate-600" role="status">{msg}</p>}
        </div>
      </div>
    </Card>
  );
}

function ExploreTab({ source }: { source: string }) {
  const overview = useFetch(() => getJson<Overview>("/api/ehr/overview", { source }), [source]);
  const [column, setColumn] = useState("");
  useEffect(() => {
    const s = overview.data?.schema;
    if (s?.length) setColumn((c) => (s.some((x) => x.column === c) ? c : (s.find((x) => x.semantic_type !== "identifier") ?? s[0]).column));
  }, [overview.data]);
  const profile = useFetch(() => (column ? getJson<Profile>("/api/ehr/column", { source, column }) : Promise.resolve(null)), [source, column]);
  return (
    <AsyncView state={overview} loadingLabel="Profiling dataset..." onRetry={overview.run}>
      {(o) => (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            <StatCard label="Rows" value={fmt(o.rows)} />
            <StatCard label="Columns" value={o.columns} />
            <StatCard label="Missing cells" value={fmt(o.total_missing_cells)} />
            <StatCard label="Duplicate rows" value={o.duplicate_rows} />
            <StatCard label="Memory" value={`${fmt(o.memory_mb, 2)} MB`} />
          </div>
          <Card title="Schema" subtitle="Storage type (dtype) vs. inferred meaning (semantic type)">
            <DataTable rows={o.schema} columns={[
              { key: "column", label: "Column" }, { key: "dtype", label: "dtype" },
              { key: "semantic_type", label: "Semantic type", render: (r) => <Badge tone="blue">{r.semantic_type}</Badge> },
              { key: "non_null", label: "Non-null" }, { key: "missing_pct", label: "Missing %", render: (r) => pctRaw(r.missing_pct) },
              { key: "unique", label: "Unique" }, { key: "examples", label: "Examples", render: (r) => <span className="text-xs text-slate-500">{r.examples.join(" · ")}</span> },
            ]} />
          </Card>
          <Card title="Column distribution" actions={
            <div className="w-64"><Select ariaLabel="Column" value={column} onChange={setColumn} options={o.schema.map((s) => ({ value: s.column, label: `${s.column} (${s.semantic_type})` }))} /></div>
          }>
            <AsyncView state={profile} empty={null}>
              {(p) => p && (
                <>
                  <p className="mb-2 text-xs text-slate-500">
                    {p.chart === "histogram" ? "Histogram" : p.chart === "timeline" ? "Records per year" : "Top categories"} · {p.missing} missing
                    {p.outliers_iqr !== undefined && ` · ${p.outliers_iqr} IQR outliers`}
                  </p>
                  <ChartBox height={260}>
                    <ResponsiveContainer>
                      <BarChart data={p.data} margin={{ bottom: 40 }}>
                        <CartesianGrid {...GRID} />
                        <XAxis dataKey="bin" {...AXIS} angle={-35} textAnchor="end" interval={0} height={60} tickFormatter={(v: string) => (v.length > 18 ? `${v.slice(0, 17)}…` : v)} />
                        <YAxis {...AXIS} allowDecimals={false} />
                        <Tooltip content={({ payload }) => payload?.[0] ? <TooltipBox title={payload[0].payload.bin} rows={[{ label: "count", value: fmt(payload[0].payload.count) }]} /> : null} />
                        <Bar isAnimationActive={false} dataKey="count" fill={SERIES[0]} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </>
              )}
            </AsyncView>
          </Card>
          {o.numeric_summary.length > 0 && <Card title="Numeric summary"><DataTable rows={o.numeric_summary} /></Card>}
          {o.categorical_summary.length > 0 && <Card title="Categorical summary"><DataTable rows={o.categorical_summary} /></Card>}
          <Card title="Preview (first 10 rows)"><DataTable rows={o.preview} /></Card>
        </div>
      )}
    </AsyncView>
  );
}

interface CleanReport {
  steps: Record<string, unknown>[];
  dtypes: { column: string; before: string; after: string; unique_before: number; unique_after: number }[];
  value_changes: Record<string, { before: Record<string, number>; after: Record<string, number> }>;
  examples: { column: string; before: string; after: string }[];
  validation_before: { rule: string; violations: number }[];
  validation_after: { rule: string; violations: number }[];
  recovery_vs_ground_truth?: { column: string; match_pct: number; mismatches: number }[];
  cleaned_preview: Record<string, unknown>[];
}

function CleanTab({ source }: { source: string }) {
  const run = useAction(() => getJson<CleanReport>("/api/ehr/clean", { source }));
  const log = useFetch(() => getJson<{ corruptions: { column: string; issue: string; rows: number }[]; note: string }>("/api/ehr/teaching-log"));
  const [col, setCol] = useState<string>("");
  return (
    <div className="space-y-6">
      {source === "teaching:patients_messy" && (
        <Card title="What was broken on purpose" subtitle="Corruption log of the teaching dataset">
          <AsyncView state={log}>{(l) => (<><Notice>{l.note}</Notice><div className="mt-3"><DataTable rows={l.corruptions} /></div></>)}</AsyncView>
        </Card>
      )}
      <div className="flex items-center gap-3">
        <RunButton loading={run.loading} onClick={() => run.run()}>Run cleaning pipeline</RunButton>
        <span className="text-xs text-slate-500">Source: {source}</span>
      </div>
      <AsyncView state={run} loadingLabel="Cleaning...">
        {(r) => {
          const vb = r.validation_before.reduce((a, v) => a + v.violations, 0);
          const va = r.validation_after.reduce((a, v) => a + v.violations, 0);
          const changeCols = Object.keys(r.value_changes);
          const active = changeCols.includes(col) ? col : changeCols[0];
          const vc = active ? r.value_changes[active] : null;
          const validation = r.validation_before.map((b, i) => ({ rule: b.rule, before: b.violations, after: r.validation_after[i]?.violations ?? 0 }));
          return (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <StatCard label="Pipeline steps" value={r.steps.length} />
                <StatCard label="Rule violations before" value={vb} />
                <StatCard label="Rule violations after" value={va} />
                <StatCard label="Columns re-typed" value={r.dtypes.filter((d) => d.before !== d.after).length} />
              </div>
              <Card title="Cleaning log"><DataTable rows={r.steps} /></Card>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card title="Validation rules: before vs after">
                  <Legend items={[{ label: "before", color: SERIES[1] }, { label: "after", color: SERIES[0] }]} />
                  <ChartBox height={Math.max(200, validation.length * 34)}>
                    <ResponsiveContainer>
                      <BarChart data={validation} layout="vertical" margin={{ left: 10 }} barGap={2}>
                        <CartesianGrid {...GRID} horizontal={false} vertical />
                        <XAxis type="number" {...AXIS} allowDecimals={false} />
                        <YAxis type="category" interval={0} dataKey="rule" width={210} {...AXIS} tick={{ fontSize: 10 }} />
                        <Tooltip />
                        <Bar isAnimationActive={false} dataKey="before" fill={SERIES[1]} radius={[0, 4, 4, 0]} />
                        <Bar isAnimationActive={false} dataKey="after" fill={SERIES[0]} radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </Card>
                <Card title="Category values before → after" actions={changeCols.length > 0 && (
                  <div className="w-44"><Select ariaLabel="Column" value={active} onChange={setCol} options={changeCols.map((c) => ({ value: c, label: c }))} /></div>)}>
                  {vc ? (
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      {(["before", "after"] as const).map((k) => (
                        <div key={k}>
                          <p className="label">{k}</p>
                          <ul className="space-y-1">{Object.entries(vc[k]).map(([v, n]) => (
                            <li key={v} className="flex justify-between rounded bg-slate-50 px-2 py-1"><code className="text-xs">{JSON.stringify(v)}</code><span className="tabular-nums">{n}</span></li>
                          ))}</ul>
                        </div>
                      ))}
                    </div>
                  ) : <p className="text-sm text-slate-500">No categorical changes.</p>}
                </Card>
              </div>
              <Card title="Type conversions"><DataTable rows={r.dtypes} columns={[
                { key: "column", label: "Column" }, { key: "before", label: "dtype before" },
                { key: "after", label: "dtype after", render: (d) => d.before !== d.after ? <Badge tone="green">{d.after}</Badge> : d.after },
                { key: "unique_before", label: "unique before" }, { key: "unique_after", label: "unique after" }]} /></Card>
              {r.recovery_vs_ground_truth && (
                <Card title="Did cleaning recover the original values?" subtitle="Comparison with the untouched Synthea data. Deleted values and invalid sentinels cannot be recovered and remain as mismatches.">
                  <DataTable rows={r.recovery_vs_ground_truth} columns={[
                    { key: "column", label: "Column" },
                    { key: "match_pct", label: "Match %", render: (x) => <Badge tone={x.match_pct === 100 ? "green" : "amber"}>{pctRaw(x.match_pct, 2)}</Badge> },
                    { key: "mismatches", label: "Mismatches" }]} />
                </Card>
              )}
              <Card title="Example changes"><DataTable rows={r.examples} /></Card>
            </div>
          );
        }}
      </AsyncView>
    </div>
  );
}

interface Missingness {
  rows: number; null_tokens_converted: number; total_missing_cells: number; missing_cell_pct: number; rows_with_missing: number; complete_rows: number;
  per_column: { column: string; missing: number; missing_pct: number }[];
  co_missing_columns: string[]; co_missing: { a: string; b: string; corr: number }[];
  patterns: { pattern: string; rows: number; missing_columns: string[] }[];
  matrix: { columns: string[]; rows: number[][] };
}
interface Duplicates { rows: number; exact_duplicates: number; near_duplicates: number; key: string | null; key_duplicates: number; rows_after_dedup: number; removed: number; examples: { type: string; row: number; values: Record<string, string | null> }[]; constant_columns: string[]; redundant_column_pairs: { a: string; b: string; abs_corr: number }[] }
interface Imputation { target: string; predictors: string[]; masked_values: number; originally_missing: number; results: { method: string; mae: number; rmse: number; normalized_rmse: number }[]; best_method: string; distribution: { bin: string; observed: number; imputed: number }[]; numeric_columns: string[]; method_note: string }
interface Quality { scores: Record<string, number>; overall: number; validation: { rule: string; violations: number }[] }

function MissingTab({ source }: { source: string }) {
  const miss = useFetch(() => getJson<Missingness>("/api/ehr/missingness", { source }), [source]);
  const dup = useFetch(() => getJson<Duplicates>("/api/ehr/duplicates", { source }), [source]);
  const quality = useFetch(() => getJson<Quality>("/api/ehr/quality", { source }), [source]);
  const [target, setTarget] = useState<string | undefined>(undefined);
  const imp = useFetch(() => getJson<Imputation>("/api/ehr/imputation", { source, target }), [source, target]);
  return (
    <div className="space-y-6">
      <AsyncView state={quality} empty={null}>
        {(q) => (
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            <StatCard label="Quality score" value={fmt(q.overall, 1)} hint="mean of 4 dimensions (0-100)" />
            {Object.entries(q.scores).map(([k, v]) => <StatCard key={k} label={k} value={fmt(v, 1)} />)}
          </div>
        )}
      </AsyncView>
      <AsyncView state={miss} loadingLabel="Analysing missingness...">
        {(m) => (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Missing values per column" subtitle={`${pctRaw(m.missing_cell_pct)} of cells missing · ${m.complete_rows} of ${m.rows} rows complete · ${m.null_tokens_converted} text null tokens converted`}>
              <ChartBox height={Math.max(220, m.per_column.filter((c) => c.missing).length * 22)}>
                <ResponsiveContainer>
                  <BarChart data={m.per_column.filter((c) => c.missing)} layout="vertical" margin={{ left: 10 }}>
                    <CartesianGrid {...GRID} horizontal={false} vertical />
                    <XAxis type="number" domain={[0, 100]} unit="%" {...AXIS} />
                    <YAxis type="category" interval={0} dataKey="column" width={150} {...AXIS} tick={{ fontSize: 10 }} />
                    <Tooltip content={({ payload }) => payload?.[0] ? <TooltipBox title={payload[0].payload.column} rows={[{ label: "missing", value: `${payload[0].payload.missing} (${pctRaw(payload[0].payload.missing_pct)})` }]} /> : null} />
                    <Bar isAnimationActive={false} dataKey="missing_pct" fill={SERIES[1]} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartBox>
            </Card>
            <Card title="Missingness matrix" subtitle={`First ${m.matrix.rows.length} rows; dark = missing`}>
              {m.matrix.columns.length ? (
                <div className="overflow-x-auto">
                  <div className="inline-grid gap-px" style={{ gridTemplateColumns: `repeat(${m.matrix.columns.length}, 14px)` }}>
                    {m.matrix.rows.flatMap((row, i) => row.map((v, j) => (
                      <div key={`${i}-${j}`} title={`row ${i}, ${m.matrix.columns[j]}: ${v ? "missing" : "present"}`} className="h-1.5 w-3.5" style={{ background: v ? "#1c5cab" : "#e2e8f0" }} />
                    )))}
                  </div>
                  <div className="mt-1 flex gap-px">{m.matrix.columns.map((c) => <span key={c} className="w-3.5 origin-top-left translate-x-2 rotate-45 whitespace-nowrap text-[9px] text-slate-500">{c}</span>)}</div>
                  <div className="h-16" />
                </div>
              ) : <p className="text-sm text-slate-500">No missing values.</p>}
            </Card>
            {m.co_missing_columns.length >= 2 && (
              <Card title="Do values go missing together?" subtitle="Correlation of missingness indicators (1 = always missing together)" className="lg:col-span-2">
                <MatrixHeatmap diverging rows={m.co_missing_columns} cols={m.co_missing_columns}
                  values={m.co_missing_columns.map((a) => m.co_missing_columns.map((b) => m.co_missing.find((x) => x.a === a && x.b === b)?.corr ?? 0))} />
              </Card>
            )}
            {m.patterns.length > 0 && <Card title="Most common missingness patterns" className="lg:col-span-2"><DataTable rows={m.patterns.map((p) => ({ rows: p.rows, missing_columns: p.missing_columns.join(", ") || "(none)" }))} /></Card>}
          </div>
        )}
      </AsyncView>
      <AsyncView state={imp} loadingLabel="Comparing imputation methods...">
        {(r) => (
          <Card title={`Imputation comparison - ${r.target}`} subtitle={r.method_note} actions={
            <div className="w-56"><Select ariaLabel="Target column" value={r.target} onChange={setTarget} options={r.numeric_columns.map((c) => ({ value: c, label: c }))} /></div>}>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <div>
                <ChartBox height={220}>
                  <ResponsiveContainer>
                    <BarChart data={r.results}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="method" {...AXIS} />
                      <YAxis {...AXIS} />
                      <Tooltip content={({ payload }) => payload?.[0] ? <TooltipBox title={payload[0].payload.method} rows={[
                        { label: "RMSE", value: fmt(payload[0].payload.rmse) }, { label: "MAE", value: fmt(payload[0].payload.mae) },
                        { label: "RMSE / SD", value: fmt(payload[0].payload.normalized_rmse) }]} /> : null} />
                      <Bar isAnimationActive={false} dataKey="rmse" name="RMSE" radius={[4, 4, 0, 0]}>
                        {r.results.map((x) => <Cell key={x.method} fill={x.method === r.best_method ? SERIES[2] : SERIES[0]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </ChartBox>
                <p className="text-sm text-slate-600">Lowest error: <strong>{r.best_method}</strong> (green). {r.originally_missing} values were originally missing.</p>
              </div>
              <div>
                <p className="mb-1 text-sm font-medium text-slate-700">Observed vs values imputed by {r.best_method}</p>
                <Legend items={[{ label: "observed", color: SERIES[0] }, { label: "imputed", color: SERIES[1] }]} />
                <ChartBox height={200}>
                  <ResponsiveContainer>
                    <BarChart data={r.distribution} barGap={2}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="bin" {...AXIS} />
                      <YAxis {...AXIS} allowDecimals={false} />
                      <Tooltip />
                      <Bar isAnimationActive={false} dataKey="observed" fill={SERIES[0]} radius={[4, 4, 0, 0]} />
                      <Bar isAnimationActive={false} dataKey="imputed" fill={SERIES[1]} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartBox>
              </div>
            </div>
          </Card>
        )}
      </AsyncView>
      <AsyncView state={dup} loadingLabel="Detecting duplicates...">
        {(d) => (
          <Card title="Redundancy: duplicate records and columns">
            <KeyValue items={[
              { label: "Exact duplicate rows", value: d.exact_duplicates },
              { label: "Near duplicates (case/space normalised)", value: d.near_duplicates },
              { label: `Duplicate key (${d.key ?? "none"})`, value: d.key_duplicates },
              { label: "Rows after de-duplication", value: `${d.rows_after_dedup} (removed ${d.removed})` },
              { label: "Constant columns", value: d.constant_columns.join(", ") || "none" },
              { label: "Near-identical numeric column pairs (|r| ≥ 0.98)", value: d.redundant_column_pairs.map((p) => `${p.a} ~ ${p.b}`).join(", ") || "none" },
            ]} />
            {d.examples.length > 0 && <div className="mt-4"><DataTable rows={d.examples.map((e) => ({ type: e.type, row: e.row, ...e.values }))} /></div>}
          </Card>
        )}
      </AsyncView>
    </div>
  );
}

export default function EhrExplorer() {
  const [tab, setTab] = useTab(TABS.map((t) => t.id), "explore" as TabId);
  const sources = useFetch(() => getJson<SourceOption[]>("/api/ehr/sources"));
  const [source, setSource] = useState<string | null>(null);
  const defaults: Record<TabId, string> = { explore: "synthea:patient_profile", clean: "teaching:patients_messy", missing: "synthea:patient_profile" };
  const active = source ?? defaults[tab];
  return (
    <>
      <PageHeader title="EHR Data Explorer" tutorials={["T1", "T2", "T3"]}
        description="Explore, standardise and assess the quality of electronic health record tables (Synthea synthetic EHR or your own CSV)." />
      <Tabs tabs={TABS.map((t) => ({ ...t }))} value={tab} onChange={(t) => { setTab(t); setSource(null); }} />
      <AsyncView state={sources} onRetry={sources.run}>
        {(list) => list.length === 0 ? (
          <Notice tone="warning">No EHR data found. Run <code>python scripts/download_datasets.py --only synthea</code> or upload a CSV.</Notice>
        ) : (
          <div className="space-y-6">
            <SourceSelect value={active} onChange={setSource} sources={list} onUploaded={(id) => { void sources.run(); setSource(id); }} />
            <div key={`${tab}-${active}`}>
              {tab === "explore" && <ExploreTab source={active} />}
              {tab === "clean" && <CleanTab source={active} />}
              {tab === "missing" && <MissingTab source={active} />}
            </div>
          </div>
        )}
      </AsyncView>
    </>
  );
}
