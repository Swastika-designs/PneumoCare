import { Plus, Trash2 } from "lucide-react";
import { Fragment, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { AXIS, ChartBox, GRID, Legend, SERIES, TooltipBox, sequentialColor } from "../components/charts";
import { AsyncView, Badge, Card, Field, KeyValue, Notice, PageHeader, RunButton, Select, StatCard, Tabs } from "../components/ui";
import { DataTable } from "../components/ui/DataTable";
import { useAction, useFetch } from "../hooks/useApi";
import { useTab } from "../hooks/useTab";
import { getJson, postJson } from "../services/api";
import { fmt, money, pctRaw } from "../utils/format";

const TABS = [
  { id: "forecast", label: "Volume & staffing", badge: "T7" },
  { id: "kpis", label: "KPIs & comparisons", badge: "T8" },
  { id: "planning", label: "Utilisation & scenarios", badge: "T9" },
] as const;
type TabId = (typeof TABS)[number]["id"];

interface Forecast {
  freq: string; periods: number; train_periods: number; test_periods: number; features: string[];
  results: { model: string; mae: number; rmse: number; mape: number }[]; best_model: string;
  history: { period: string; actual: number }[];
  test_predictions: Record<string, number | string>[];
  forecast: { period: string; predicted_encounters: number; workload_hours: number; required_fte: number }[];
  staffing_assumptions: Record<string, string | number>; limitations: string[]; encounter_classes: string[];
}

function ForecastTab() {
  const [freq, setFreq] = useState("month");
  const [cls, setCls] = useState("all");
  const [horizon, setHorizon] = useState(6);
  const [hours, setHours] = useState(160);
  const params = { freq, encounter_class: cls, horizon, hours_per_fte: freq === "week" ? hours / 4 : hours };
  const run = useFetch(() => getJson<Forecast>("/api/operations/forecast", params), [freq, cls, horizon, hours]);
  const models = ["Naive (last period)", "Moving average", "Ridge regression", "Random forest"];
  return (
    <div className="space-y-6">
      <Card>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <Field label="Period"><Select value={freq} onChange={setFreq} options={[{ value: "month", label: "Monthly" }, { value: "week", label: "Weekly" }]} /></Field>
          <Field label="Encounter class">
            <Select value={cls} onChange={setCls} options={[{ value: "all", label: "All classes" }, ...(run.data?.encounter_classes ?? []).map((c) => ({ value: c, label: c }))]} />
          </Field>
          <Field label={`Forecast horizon: ${horizon}`}><input type="range" min={1} max={12} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} className="w-full accent-brand-600" /></Field>
          <Field label={`Hours per FTE / month: ${hours}`}><input type="range" min={80} max={200} step={10} value={hours} onChange={(e) => setHours(Number(e.target.value))} className="w-full accent-brand-600" /></Field>
        </div>
      </Card>
      <AsyncView state={run} loadingLabel="Training forecasting models..." onRetry={run.run}>
        {(f) => (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <StatCard label="Periods of history" value={f.periods} hint={`${f.train_periods} train / ${f.test_periods} test (time-ordered)`} />
              <StatCard label="Best model (MAE)" value={f.best_model} />
              <StatCard label="Next-period forecast" value={fmt(f.forecast[0]?.predicted_encounters, 1)} hint="encounters" />
              <StatCard label="Required FTE (next period)" value={fmt(f.forecast[0]?.required_fte, 2)} hint={`at ${(Number(f.staffing_assumptions.utilization_target) * 100).toFixed(0)}% utilisation`} />
            </div>
            <Card title="Model comparison on the held-out (most recent) periods" subtitle="Baselines must be beaten for a model to be useful">
              <DataTable rows={f.results.map((r) => ({ ...r, best: r.model === f.best_model ? "✓" : "" }))} />
            </Card>
            <Card title="Actual vs predicted (test periods)">
              <Legend items={[{ label: "actual", color: "#0f172a" }, ...models.map((m, i) => ({ label: m, color: SERIES[i] }))]} />
              <ChartBox height={280}>
                <ResponsiveContainer>
                  <LineChart data={f.test_predictions}>
                    <CartesianGrid {...GRID} />
                    <XAxis dataKey="period" {...AXIS} minTickGap={24} />
                    <YAxis {...AXIS} />
                    <Tooltip />
                    <Line isAnimationActive={false} dataKey="actual" stroke="#0f172a" strokeWidth={2.5} dot={{ r: 3 }} />
                    {models.map((m, i) => <Line isAnimationActive={false} key={m} dataKey={m} stroke={SERIES[i]} strokeWidth={2} dot={false} strokeDasharray={i < 2 ? "4 3" : undefined} />)}
                  </LineChart>
                </ResponsiveContainer>
              </ChartBox>
            </Card>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card title="Full history and forecast" subtitle={`${f.best_model}, recursive multi-step`}>
                <ChartBox height={260}>
                  <ResponsiveContainer>
                    <ComposedChart data={[...f.history.map((h) => ({ period: h.period, actual: h.actual })), ...f.forecast.map((p) => ({ period: p.period, forecast: p.predicted_encounters }))]}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="period" {...AXIS} minTickGap={30} />
                      <YAxis {...AXIS} />
                      <Tooltip />
                      <Line isAnimationActive={false} dataKey="actual" stroke={SERIES[0]} strokeWidth={2} dot={false} />
                      <Line isAnimationActive={false} dataKey="forecast" stroke={SERIES[1]} strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3 }} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </ChartBox>
                <Legend items={[{ label: "actual encounters", color: SERIES[0] }, { label: "forecast", color: SERIES[1] }]} />
              </Card>
              <Card title="Staffing plan from forecast workload">
                <DataTable rows={f.forecast} />
                <div className="mt-3"><KeyValue items={Object.entries(f.staffing_assumptions).map(([k, v]) => ({ label: k.replace(/_/g, " "), value: fmt(v) }))} /></div>
              </Card>
            </div>
            <Notice tone="warning"><ul className="list-disc pl-4">{f.limitations.map((l) => <li key={l}>{l}</li>)}</ul></Notice>
          </div>
        )}
      </AsyncView>
    </div>
  );
}

interface Kpis {
  window: { start_year: number; end: string };
  headline: Record<string, number | null>;
  group_by: string;
  groups: { group: string; encounters: number; patients: number; avg_cost: number; median_cost: number; total_cost: number; coverage_pct: number; avg_duration_h: number }[];
  comparison_test: { test: string; statistic: number; p_value: number; metric: string; interpretation: string; groups_tested: number } | null;
  monthly_trend: { period: string; encounters: number; avg_cost: number | null; patients: number }[];
  yearly: { YEAR: number; encounters: number; avg_cost: number; patients: number }[];
  group_options: string[];
}

function KpiTab() {
  const [groupBy, setGroupBy] = useState("encounter_class");
  const [metric, setMetric] = useState("TOTAL_CLAIM_COST");
  const [start, setStart] = useState(2016);
  const run = useFetch(() => getJson<Kpis>("/api/operations/kpis", { group_by: groupBy, metric, start_year: start }), [groupBy, metric, start]);
  return (
    <div className="space-y-6">
      <Card>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Field label="Compare groups by">
            <Select value={groupBy} onChange={setGroupBy} options={["encounter_class", "year", "gender", "age_group", "payer", "organization"].map((g) => ({ value: g, label: g.replace("_", " ") }))} />
          </Field>
          <Field label="Metric tested"><Select value={metric} onChange={setMetric} options={[
            { value: "TOTAL_CLAIM_COST", label: "Total claim cost" }, { value: "BASE_ENCOUNTER_COST", label: "Base encounter cost" }, { value: "DURATION_H", label: "Duration (hours)" }]} /></Field>
          <Field label="From year"><Select value={String(start)} onChange={(v) => setStart(Number(v))} options={[2000, 2010, 2016, 2020].map((y) => ({ value: String(y), label: String(y) }))} /></Field>
        </div>
      </Card>
      <AsyncView state={run} loadingLabel="Computing KPIs..." onRetry={run.run}>
        {(k) => {
          const h = k.headline;
          return (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <StatCard label="Encounters" value={fmt(h.encounters)} hint={`${fmt(h.unique_patients)} patients · ${k.window.start_year}–${k.window.end.slice(0, 4)}`} />
                <StatCard label="Avg cost / encounter" value={money(h.avg_cost_per_encounter)} hint={`total ${money(h.total_claim_cost)}`} />
                <StatCard label="Payer coverage" value={pctRaw(h.payer_coverage_pct)} hint={`${pctRaw(h.uninsured_encounter_pct)} encounters uninsured`} />
                <StatCard label="Avg length of stay" value={h.avg_length_of_stay_days !== null ? `${fmt(h.avg_length_of_stay_days, 1)} d` : "—"} hint={`${fmt(h.inpatient_admissions)} admissions · 30-day readmission ${pctRaw(h.readmission_30d_pct)}`} />
              </div>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card title={`Encounters by ${k.group_by.replace("_", " ")}`}>
                  <ChartBox height={280}>
                    <ResponsiveContainer>
                      <BarChart data={k.groups} layout="vertical" margin={{ left: 10 }}>
                        <CartesianGrid {...GRID} horizontal={false} vertical />
                        <XAxis type="number" {...AXIS} />
                        <YAxis type="category" interval={0} dataKey="group" width={140} {...AXIS} tick={{ fontSize: 10 }} tickFormatter={(v: string) => String(v).slice(0, 22)} />
                        <Tooltip content={({ payload }) => payload?.[0] ? <TooltipBox title={String(payload[0].payload.group)} rows={[
                          { label: "encounters", value: fmt(payload[0].payload.encounters) }, { label: "avg cost", value: money(payload[0].payload.avg_cost) },
                          { label: "coverage", value: pctRaw(payload[0].payload.coverage_pct) }]} /> : null} />
                        <Bar isAnimationActive={false} dataKey="encounters" fill={SERIES[0]} radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </Card>
                <Card title={`Average cost by ${k.group_by.replace("_", " ")}`}>
                  <ChartBox height={280}>
                    <ResponsiveContainer>
                      <BarChart data={k.groups} layout="vertical" margin={{ left: 10 }}>
                        <CartesianGrid {...GRID} horizontal={false} vertical />
                        <XAxis type="number" {...AXIS} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                        <YAxis type="category" interval={0} dataKey="group" width={140} {...AXIS} tick={{ fontSize: 10 }} tickFormatter={(v: string) => String(v).slice(0, 22)} />
                        <Tooltip formatter={(v: number) => money(v)} />
                        <Bar isAnimationActive={false} dataKey="avg_cost" name="avg cost" fill={SERIES[1]} radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </Card>
              </div>
              {k.comparison_test && (
                <Notice>
                  <strong>{k.comparison_test.test}</strong> across {k.comparison_test.groups_tested} groups: statistic {fmt(k.comparison_test.statistic, 2)},
                  p = {k.comparison_test.p_value.toExponential(2)}. {k.comparison_test.interpretation} (Rank-based test: costs are heavily skewed, so means are not compared with ANOVA.)
                </Notice>
              )}
              <Card title="Group table"><DataTable rows={k.groups} /></Card>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card title="Monthly encounters (trend)">
                  <ChartBox height={240}>
                    <ResponsiveContainer>
                      <LineChart data={k.monthly_trend}>
                        <CartesianGrid {...GRID} />
                        <XAxis dataKey="period" {...AXIS} minTickGap={30} />
                        <YAxis {...AXIS} />
                        <Tooltip />
                        <Line isAnimationActive={false} dataKey="encounters" stroke={SERIES[0]} strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </Card>
                <Card title="Unique patients seen per year">
                  <ChartBox height={240}>
                    <ResponsiveContainer>
                      <BarChart data={k.yearly}>
                        <CartesianGrid {...GRID} />
                        <XAxis dataKey="YEAR" {...AXIS} />
                        <YAxis {...AXIS} />
                        <Tooltip />
                        <Bar isAnimationActive={false} dataKey="patients" fill={SERIES[2]} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </Card>
              </div>
            </div>
          );
        }}
      </AsyncView>
    </div>
  );
}

interface Util {
  heatmap: { day: string; hour: number; count: number }[]; heatmap_note: string;
  demand_by_year: Record<string, number>[]; demand_growth: { encounter_class: string; previous: number; latest: number; growth_pct: number | null }[];
  provider_workload: { providers: number; mean_encounters_per_provider_year: number; max_encounters_per_provider_year: number; top_providers_share_pct: number; distribution: { bucket: string; providers: number }[] };
  bed_utilization: { period: string; bed_days: number; peak_occupied: number; utilization_pct: number }[];
  bed_capacity_assumed: number; classes: string[];
}
interface ScenarioRow { name: string; demand_change_pct: number; staff_fte: number; minutes_per_encounter?: number | null }
interface ScenarioResult {
  baseline: Record<string, number | string>;
  scenarios: (ScenarioRow & { monthly_encounters: number; workload_hours: number; capacity_hours: number; utilization_pct: number; unmet_hours: number; required_fte_at_85pct: number; projected_monthly_cost: number; status: string })[];
  assumptions: string;
}

function PlanningTab() {
  const [beds, setBeds] = useState(5);
  const util = useFetch(() => getJson<Util>("/api/operations/utilization", { beds }), [beds]);
  const [rows, setRows] = useState<ScenarioRow[]>([
    { name: "Current", demand_change_pct: 0, staff_fte: 0.2 },
    { name: "Demand +30%", demand_change_pct: 30, staff_fte: 0.2 },
    { name: "Demand +30%, extra staff", demand_change_pct: 30, staff_fte: 0.3 },
  ]);
  const scen = useAction(() => postJson<ScenarioResult>("/api/operations/scenarios", { scenarios: rows }));
  const update = (i: number, patch: Partial<ScenarioRow>) => setRows((r) => r.map((x, j) => (j === i ? { ...x, ...patch } : x)));
  return (
    <div className="space-y-6">
      <AsyncView state={util} loadingLabel="Analysing utilisation..." onRetry={util.run}>
        {(u) => {
          const max = Math.max(...u.heatmap.map((c) => c.count), 1);
          const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
          // Largest 7 classes keep their own colour; the rest fold into "other" so every encounter is counted.
          const totals = u.classes.map((c) => ({ c, n: u.demand_by_year.reduce((a, d) => a + (d[c] ?? 0), 0) }))
            .filter((x) => x.n > 0).sort((a, b) => b.n - a.n);
          const top = totals.slice(0, 7).map((x) => x.c);
          const rest = totals.slice(7).map((x) => x.c);
          const classes = rest.length ? [...top, "other"] : top;
          const demand = u.demand_by_year.map((d) => ({ ...d, other: rest.reduce((a, c) => a + (d[c] ?? 0), 0) }));
          return (
            <div className="space-y-6">
              <Card title="Demand heat-map (day × hour)" subtitle={u.heatmap_note}>
                <div className="overflow-x-auto">
                  <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `40px repeat(24, 22px)` }}>
                    <span />
                    {Array.from({ length: 24 }, (_, h) => <span key={h} className="text-center text-[9px] text-slate-400">{h}</span>)}
                    {days.map((d) => (
                      <Fragment key={d}>
                        <span className="pr-1 text-right text-[11px] text-slate-500">{d}</span>
                        {u.heatmap.filter((c) => c.day === d).map((c) => (
                          <div key={`${d}-${c.hour}`} title={`${d} ${c.hour}:00 – ${c.count} encounters`} className="h-5 rounded-sm" style={{ background: sequentialColor(c.count / max) }} />
                        ))}
                      </Fragment>
                    ))}
                  </div>
                </div>
              </Card>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card title="Demand by year and encounter class" subtitle="Stacked: segment = class">
                  <Legend items={classes.map((c, i) => ({ label: c, color: SERIES[i] }))} />
                  <ChartBox height={260}>
                    <ResponsiveContainer>
                      <BarChart data={demand}>
                        <CartesianGrid {...GRID} />
                        <XAxis dataKey="year" {...AXIS} />
                        <YAxis {...AXIS} />
                        <Tooltip />
                        {classes.map((c, i) => <Bar isAnimationActive={false} key={c} dataKey={c} stackId="a" fill={SERIES[i]} stroke="#fff" strokeWidth={1} />)}
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </Card>
                <Card title="Year-over-year growth (last two complete years)">
                  <DataTable rows={u.demand_growth.filter((g) => g.previous + g.latest > 0)} columns={[
                    { key: "encounter_class", label: "Class" }, { key: "previous", label: "Previous" }, { key: "latest", label: "Latest" },
                    { key: "growth_pct", label: "Growth", render: (g) => g.growth_pct === null ? "—" : <Badge tone={g.growth_pct >= 0 ? "green" : "red"}>{pctRaw(g.growth_pct)}</Badge> }]} />
                </Card>
              </div>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card title="Provider workload" subtitle="Average encounters per provider per active year">
                  <KeyValue items={[
                    { label: "Providers", value: u.provider_workload.providers },
                    { label: "Mean encounters / provider-year", value: fmt(u.provider_workload.mean_encounters_per_provider_year, 2) },
                    { label: "Busiest provider", value: fmt(u.provider_workload.max_encounters_per_provider_year, 1) },
                    { label: "Share handled by top 10", value: pctRaw(u.provider_workload.top_providers_share_pct) },
                  ]} />
                  <ChartBox height={180}>
                    <ResponsiveContainer>
                      <BarChart data={u.provider_workload.distribution}>
                        <CartesianGrid {...GRID} />
                        <XAxis dataKey="bucket" {...AXIS} />
                        <YAxis {...AXIS} />
                        <Tooltip />
                        <Bar isAnimationActive={false} dataKey="providers" fill={SERIES[0]} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </Card>
                <Card title="Bed utilisation (inpatient, SNF, hospice)" actions={
                  <div className="w-36"><Field label={`Beds: ${beds}`}><input type="range" min={1} max={20} value={beds} onChange={(e) => setBeds(Number(e.target.value))} className="w-full accent-brand-600" /></Field></div>}>
                  <ChartBox height={220}>
                    <ResponsiveContainer>
                      <BarChart data={u.bed_utilization}>
                        <CartesianGrid {...GRID} />
                        <XAxis dataKey="period" {...AXIS} minTickGap={24} />
                        <YAxis {...AXIS} unit="%" />
                        <Tooltip />
                        <ReferenceLine y={85} stroke="#e34948" strokeDasharray="4 4" label={{ value: "85% target", fontSize: 10, fill: "#64748b", position: "insideTopRight" }} />
                        <Bar isAnimationActive={false} dataKey="utilization_pct" name="utilisation %" fill={SERIES[0]} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </Card>
              </div>
            </div>
          );
        }}
      </AsyncView>
      <Card title="Scenario comparison" subtitle="What-if analysis on the last 12 months of clinic demand">
        <div className="space-y-2">
          {rows.map((r, i) => (
            <div key={i} className="grid grid-cols-2 items-end gap-2 md:grid-cols-5">
              <Field label="Name"><input className="input" value={r.name} onChange={(e) => update(i, { name: e.target.value })} /></Field>
              <Field label="Demand change %"><input className="input" type="number" value={r.demand_change_pct} onChange={(e) => update(i, { demand_change_pct: Number(e.target.value) })} /></Field>
              <Field label="Staff FTE"><input className="input" type="number" step="0.05" min="0.01" value={r.staff_fte} onChange={(e) => update(i, { staff_fte: Number(e.target.value) })} /></Field>
              <Field label="Minutes / encounter"><input className="input" type="number" placeholder="observed" value={r.minutes_per_encounter ?? ""} onChange={(e) => update(i, { minutes_per_encounter: e.target.value ? Number(e.target.value) : null })} /></Field>
              <button className="btn-secondary" onClick={() => setRows((x) => x.filter((_, j) => j !== i))} disabled={rows.length === 1} aria-label={`Remove ${r.name}`}><Trash2 className="h-4 w-4" /></button>
            </div>
          ))}
        </div>
        <div className="mt-3 flex gap-2">
          <button className="btn-secondary" onClick={() => setRows((x) => [...x, { name: `Scenario ${x.length + 1}`, demand_change_pct: 0, staff_fte: 0.2 }])} disabled={rows.length >= 6}><Plus className="h-4 w-4" /> Add scenario</button>
          <RunButton loading={scen.loading} onClick={() => scen.run()}>Compare scenarios</RunButton>
        </div>
        <div className="mt-4">
          <AsyncView state={scen} empty={null}>
            {(s) => (
              <div className="space-y-4">
                <KeyValue items={Object.entries(s.baseline).map(([k, v]) => ({ label: k.replace(/_/g, " "), value: fmt(v) }))} />
                <ChartBox height={240}>
                  <ResponsiveContainer>
                    <BarChart data={s.scenarios}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="name" {...AXIS} />
                      <YAxis {...AXIS} unit="%" />
                      <Tooltip />
                      <ReferenceLine y={100} stroke="#e34948" strokeDasharray="4 4" />
                      <ReferenceLine y={85} stroke="#eda100" strokeDasharray="4 4" />
                      <Bar isAnimationActive={false} dataKey="utilization_pct" name="staff utilisation %" radius={[4, 4, 0, 0]}>
                        {s.scenarios.map((x) => <Cell key={x.name} fill={SERIES[0]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </ChartBox>
                <DataTable rows={s.scenarios} columns={[
                  { key: "name", label: "Scenario" }, { key: "monthly_encounters", label: "Encounters / month" },
                  { key: "workload_hours", label: "Workload h" }, { key: "capacity_hours", label: "Capacity h" },
                  { key: "utilization_pct", label: "Utilisation", render: (x) => pctRaw(x.utilization_pct) },
                  { key: "required_fte_at_85pct", label: "FTE needed @85%" },
                  { key: "projected_monthly_cost", label: "Cost / month", render: (x) => money(x.projected_monthly_cost) },
                  { key: "status", label: "Status", render: (x) => <Badge tone={x.status === "ok" ? "green" : x.status === "high" ? "amber" : "red"}>{x.status}</Badge> },
                ]} />
                <p className="text-xs text-slate-500">{s.assumptions}</p>
              </div>
            )}
          </AsyncView>
        </div>
      </Card>
    </div>
  );
}

export default function Operations() {
  const [tab, setTab] = useTab(TABS.map((t) => t.id), "forecast" as TabId);
  return (
    <>
      <PageHeader title="Healthcare Operations Analytics" tutorials={["T7", "T8", "T9"]}
        description="Patient-volume forecasting, KPI dashboards and capacity planning on Synthea encounters (synthetic data, 111 patients)." />
      <Tabs tabs={TABS.map((t) => ({ ...t }))} value={tab} onChange={setTab} />
      {tab === "forecast" && <ForecastTab />}
      {tab === "kpis" && <KpiTab />}
      {tab === "planning" && <PlanningTab />}
    </>
  );
}
