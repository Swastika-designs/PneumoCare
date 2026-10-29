import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { AXIS, ChartBox, GRID, Legend, MatrixHeatmap, SERIES, TooltipBox } from "../components/charts";
import { Scatter3D } from "../components/charts/Scatter3D";
import { AsyncView, Badge, Card, EmptyState, Field, KeyValue, Notice, PageHeader, RunButton, Select, StatCard, Tabs } from "../components/ui";
import { DataTable } from "../components/ui/DataTable";
import { useAction, useFetch } from "../hooks/useApi";
import { useTab } from "../hooks/useTab";
import { getJson, postJson } from "../services/api";
import type { SourceOption } from "../types/api";
import { fmt, pct, pctRaw } from "../utils/format";

const TABS = [
  { id: "pca", label: "PCA & EDA", badge: "T13" },
  { id: "weight-eda", label: "Adult weight: exploration", badge: "T14" },
  { id: "weight-model", label: "Adult weight: models", badge: "T15" },
] as const;
type TabId = (typeof TABS)[number]["id"];

interface Pca {
  rows_used: number; features: string[]; scaling: string; explained_variance: { component: string; ratio: number; cumulative: number }[];
  components_for_80pct: number; loadings: Record<string, number | string>[]; top_features: Record<string, { feature: string; loading: number }[]>;
  points: { pc1: number; pc2: number; pc3: number; group: string }[]; color_by: string | null; color_options: string[];
  correlation: { columns: string[]; values: number[][] }; interpretation: string;
}

function PcaTab() {
  const sources = useFetch(() => getJson<SourceOption[]>("/api/analytics/pca/sources"));
  const [source, setSource] = useState("nhanes_adults");
  const [scaling, setScaling] = useState("standard");
  const [missing, setMissing] = useState("impute");
  const [colorBy, setColorBy] = useState("");
  const pca = useFetch(() => getJson<Pca>("/api/analytics/pca", { source, scaling, missing, color_by: colorBy || undefined }), [source, scaling, missing, colorBy]);
  const groups = Array.from(new Set(pca.data?.points.map((p) => p.group) ?? [])).slice(0, 3);
  return (
    <div className="space-y-6">
      <Card>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <Field label="Dataset"><Select value={source} onChange={(v) => { setSource(v); setColorBy(""); }} options={(sources.data ?? [{ id: "nhanes_adults", label: "NHANES adults" }]).map((s) => ({ value: s.id, label: s.label }))} /></Field>
          <Field label="Scaling" hint="Without scaling, large-unit variables dominate"><Select value={scaling} onChange={setScaling} options={[{ value: "standard", label: "Standardise (z-score)" }, { value: "minmax", label: "Min-max [0,1]" }, { value: "none", label: "None (centre only)" }]} /></Field>
          <Field label="Missing values"><Select value={missing} onChange={setMissing} options={[{ value: "impute", label: "Median impute" }, { value: "drop", label: "Drop incomplete rows" }]} /></Field>
          <Field label="Colour points by"><Select value={colorBy} onChange={setColorBy} options={[{ value: "", label: "(none)" }, ...(pca.data?.color_options ?? []).map((c) => ({ value: c, label: c }))]} /></Field>
        </div>
      </Card>
      <AsyncView state={pca} loadingLabel="Running PCA..." onRetry={pca.run}>
        {(p) => (
          <div className="space-y-6">
            <Notice>{p.interpretation} {p.rows_used} rows, {p.features.length} numeric features.</Notice>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card title="Scree plot" subtitle="Bars: variance explained per component · line: cumulative">
                <Legend items={[{ label: "per component", color: SERIES[0] }, { label: "cumulative", color: SERIES[1] }]} />
                <ChartBox height={260}>
                  <ResponsiveContainer>
                    <ComposedChart data={p.explained_variance}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="component" {...AXIS} interval={0} tick={{ fontSize: 9 }} />
                      <YAxis {...AXIS} domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip formatter={(v: number) => pct(v)} />
                      <ReferenceLine y={0.8} stroke="#94a3b8" strokeDasharray="4 4" />
                      <Bar isAnimationActive={false} dataKey="ratio" name="per component" fill={SERIES[0]} radius={[4, 4, 0, 0]} />
                      <Line isAnimationActive={false} dataKey="cumulative" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 3 }} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </ChartBox>
              </Card>
              <Card title="PC1 vs PC2 scores" subtitle={p.color_by ? `coloured by ${p.color_by}${groups.length < new Set(p.points.map((x) => x.group)).size ? " (first 3 groups shown)" : ""}` : "choose a colour variable to compare groups"}>
                <Legend items={groups.map((g, i) => ({ label: g, color: SERIES[i] }))} />
                <ChartBox height={260}>
                  <ResponsiveContainer>
                    <ScatterChart>
                      <CartesianGrid {...GRID} vertical />
                      <XAxis dataKey="pc1" type="number" name="PC1" {...AXIS} />
                      <YAxis dataKey="pc2" type="number" name="PC2" {...AXIS} />
                      <ZAxis range={[16, 16]} />
                      <Tooltip />
                      {groups.map((g, i) => <Scatter isAnimationActive={false} key={g} name={g} data={p.points.filter((x) => x.group === g)} fill={SERIES[i]} fillOpacity={0.55} />)}
                    </ScatterChart>
                  </ResponsiveContainer>
                </ChartBox>
              </Card>
            </div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card title="3-D score plot (PC1-PC3)" subtitle="Drag the sliders to rotate"><Scatter3D points={p.points.filter((x) => groups.includes(x.group))} /></Card>
              <Card title="What do the components mean?" subtitle="Largest absolute loadings">
                <div className="space-y-3">
                  {Object.entries(p.top_features).map(([pc, feats]) => (
                    <div key={pc}>
                      <p className="text-sm font-semibold text-slate-700">{pc}</p>
                      <div className="flex flex-wrap gap-1">{feats.map((f) => <Badge key={f.feature} tone={f.loading >= 0 ? "brand" : "blue"}>{f.feature} {f.loading >= 0 ? "+" : ""}{f.loading.toFixed(2)}</Badge>)}</div>
                    </div>
                  ))}
                </div>
                <div className="mt-4"><DataTable rows={p.loadings} /></div>
              </Card>
            </div>
            <Card title="Correlation matrix of the input features" subtitle="Strongly correlated groups of variables collapse into shared components">
              <MatrixHeatmap diverging rows={p.correlation.columns} cols={p.correlation.columns} values={p.correlation.values} cellLabel={p.correlation.columns.length <= 14} />
            </Card>
          </div>
        )}
      </AsyncView>
    </div>
  );
}

interface Explore {
  n_adults: number; variable: string; describe: Record<string, number | string>[]; by_sex: Record<string, number | string>[];
  histogram: { bin: string; count: number }[]; histogram_note: string; bmi_categories: { category: string; count: number }[];
  obesity_prevalence: { unweighted_pct: number; survey_weighted_pct: number; note: string };
  correlations: { a: string; b: string; rho: number }[]; correlation_variables: string[];
  bmi_correlations: { variable: string; label: string; rho: number; p_value: number }[];
  group_bmi: Record<string, { rows: { group: string; n: number; mean_bmi: number; obese_pct: number }[]; kruskal_p: number | null }>;
  variables: { id: string; label: string }[];
}

function WeightEdaTab() {
  const [variable, setVariable] = useState("bmi");
  const [group, setGroup] = useState("activity_level");
  const run = useFetch(() => getJson<Explore>("/api/analytics/weight/explore", { variable }), [variable]);
  return (
    <AsyncView state={run} loadingLabel="Loading NHANES..." onRetry={run.run}>
      {(e) => {
        const g = e.group_bmi[group];
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <StatCard label="Adults (≥20 y, not pregnant)" value={fmt(e.n_adults)} hint="NHANES 2017-2018" />
              <StatCard label="Obesity (BMI ≥ 30), sample" value={pctRaw(e.obesity_prevalence.unweighted_pct)} />
              <StatCard label="Obesity, survey-weighted" value={pctRaw(e.obesity_prevalence.survey_weighted_pct)} hint="population estimate" />
              <StatCard label="Strongest BMI correlate" value={e.bmi_correlations[0].label} hint={`Spearman ρ = ${e.bmi_correlations[0].rho.toFixed(3)}`} />
            </div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card title="Distribution" actions={<div className="w-60"><Select ariaLabel="Variable" value={variable} onChange={setVariable} options={e.variables.map((v) => ({ value: v.id, label: v.label }))} /></div>} subtitle={e.histogram_note}>
                <ChartBox height={240}>
                  <ResponsiveContainer>
                    <BarChart data={e.histogram} barCategoryGap={1}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="bin" {...AXIS} minTickGap={16} />
                      <YAxis {...AXIS} />
                      <Tooltip />
                      <Bar isAnimationActive={false} dataKey="count" fill={SERIES[0]} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartBox>
              </Card>
              <Card title="BMI categories (WHO)">
                <ChartBox height={240}>
                  <ResponsiveContainer>
                    <BarChart data={e.bmi_categories}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="category" {...AXIS} />
                      <YAxis {...AXIS} />
                      <Tooltip />
                      <Bar isAnimationActive={false} dataKey="count" fill={SERIES[0]} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartBox>
                <p className="text-xs text-slate-500">{e.obesity_prevalence.note}</p>
              </Card>
            </div>
            <Card title="Mean BMI by group" subtitle={g?.kruskal_p !== null && g ? `Kruskal-Wallis p = ${g.kruskal_p.toExponential(2)}` : undefined} actions={
              <div className="w-56"><Select ariaLabel="Group" value={group} onChange={setGroup} options={Object.keys(e.group_bmi).map((k) => ({ value: k, label: k.replace(/_/g, " ") }))} /></div>}>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <ChartBox height={240}>
                  <ResponsiveContainer>
                    <BarChart data={g.rows}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="group" {...AXIS} interval={0} tick={{ fontSize: 10 }} />
                      <YAxis {...AXIS} domain={[20, "auto"]} />
                      <Tooltip content={({ payload }) => payload?.[0] ? <TooltipBox title={payload[0].payload.group} rows={[
                        { label: "n", value: payload[0].payload.n }, { label: "mean BMI", value: fmt(payload[0].payload.mean_bmi, 2) },
                        { label: "obese", value: pctRaw(payload[0].payload.obese_pct) }]} /> : null} />
                      <Bar isAnimationActive={false} dataKey="mean_bmi" name="mean BMI" fill={SERIES[0]} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartBox>
                <DataTable rows={g.rows} />
              </div>
              <p className="mt-2 text-xs text-slate-500">The y-axis starts at BMI 20 to make group differences readable; read differences from the table.</p>
            </Card>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card title="Spearman correlation with BMI">
                <DataTable rows={e.bmi_correlations.map((c) => ({ variable: c.label, rho: c.rho, p_value: c.p_value }))} />
              </Card>
              <Card title="Correlation matrix (Spearman)">
                <MatrixHeatmap diverging cellLabel={false} rows={e.correlation_variables} cols={e.correlation_variables}
                  values={e.correlation_variables.map((a) => e.correlation_variables.map((b) => e.correlations.find((c) => c.a === a && c.b === b)?.rho ?? 0))} />
              </Card>
            </div>
            <Card title="Descriptive statistics"><DataTable rows={e.describe} /></Card>
            <Card title="By sex"><DataTable rows={e.by_sex} /></Card>
          </div>
        );
      }}
    </AsyncView>
  );
}

interface WeightModels {
  n_train: number; n_test: number; features: { numeric: string[]; categorical: string[] }; excluded_features: string[];
  regression: Record<string, Record<string, number>>; classification: Record<string, Record<string, number | number[][]>>;
  linear_coefficients: { feature: string; coefficient: number }[]; logistic_odds_ratios: { feature: string; odds_ratio: number }[];
  permutation_importance: { feature: string; importance: number; std: number }[]; roc: Record<string, { fpr: number; tpr: number }[]>;
  scatter: { actual: number; predicted: number }[]; notes: string[]; trained?: false;
}

function WeightModelTab() {
  const saved = useFetch(() => getJson<WeightModels>("/api/analytics/weight/results"));
  const train = useAction(() => postJson<WeightModels>("/api/analytics/weight/train"));
  const m = train.data ?? (saved.data && saved.data.trained !== false ? saved.data : null);
  return (
    <div className="space-y-6">
      <Card title="BMI regression and obesity classification" subtitle="80/20 stratified split + 5-fold cross-validation on the training part" actions={
        <RunButton loading={train.loading} onClick={() => train.run()}>{m ? "Retrain" : "Train models (~1 min)"}</RunButton>}>
        {train.error && <Notice tone="warning">{train.error.message}</Notice>}
        {!m && !train.loading && <EmptyState title="Models not trained yet" message="Train to fit linear/logistic regression and random forests on NHANES." />}
        {train.loading && <p className="text-sm text-slate-500">Fitting models and computing permutation importance...</p>}
        {m && (
          <div className="space-y-2 text-sm">
            <KeyValue items={[
              { label: "Training / test adults", value: `${fmt(m.n_train)} / ${fmt(m.n_test)}` },
              { label: "Predictors", value: [...m.features.numeric, ...m.features.categorical].length },
              { label: "Excluded (would leak the target)", value: m.excluded_features.join(", ") },
            ]} />
          </div>
        )}
      </Card>
      {m && (
        <>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Regression: predict BMI">
              <DataTable rows={Object.entries(m.regression).map(([k, v]) => ({ model: k, ...v }))} />
              <ChartBox height={240}>
                <ResponsiveContainer>
                  <ScatterChart margin={{ bottom: 16 }}>
                    <CartesianGrid {...GRID} vertical />
                    <XAxis dataKey="actual" type="number" name="actual BMI" domain={[10, "auto"]} {...AXIS} label={{ value: "actual BMI", position: "insideBottom", offset: -8, fontSize: 11 }} />
                    <YAxis dataKey="predicted" type="number" name="predicted BMI" domain={[10, "auto"]} {...AXIS} />
                    <ZAxis range={[16, 16]} />
                    <ReferenceLine segment={[{ x: 10, y: 10 }, { x: 90, y: 90 }]} ifOverflow="hidden" stroke="#94a3b8" strokeDasharray="4 4" />
                    <Tooltip />
                    <Scatter isAnimationActive={false} data={m.scatter} fill={SERIES[0]} fillOpacity={0.5} />
                  </ScatterChart>
                </ResponsiveContainer>
              </ChartBox>
              <p className="text-xs text-slate-500">Random forest predictions on test adults. Points far from the dashed line are large errors - predictions shrink towards the mean because lifestyle factors explain little of BMI.</p>
            </Card>
            <Card title="Classification: obese (BMI ≥ 30) vs not">
              <DataTable rows={Object.entries(m.classification).map(([k, v]) => ({ model: k, ...Object.fromEntries(Object.entries(v).filter(([kk]) => kk !== "confusion_matrix")) }))} />
              <Legend items={Object.keys(m.roc).map((k, i) => ({ label: k, color: SERIES[i] }))} />
              <ChartBox height={220}>
                <ResponsiveContainer>
                  <LineChart margin={{ bottom: 16 }}>
                    <CartesianGrid {...GRID} />
                    <XAxis dataKey="fpr" type="number" domain={[0, 1]} {...AXIS} label={{ value: "false positive rate", position: "insideBottom", offset: -8, fontSize: 11 }} />
                    <YAxis type="number" domain={[0, 1]} {...AXIS} />
                    <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" />
                    <Tooltip />
                    {Object.entries(m.roc).map(([k, pts], i) => <Line isAnimationActive={false} key={k} data={pts} dataKey="tpr" name={k} stroke={SERIES[i]} strokeWidth={2} dot={false} type="linear" />)}
                  </LineChart>
                </ResponsiveContainer>
              </ChartBox>
            </Card>
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Permutation importance (random forest, BMI)" subtitle="Drop in test R² when a feature is shuffled">
              <ChartBox height={320}>
                <ResponsiveContainer>
                  <BarChart data={m.permutation_importance} layout="vertical" margin={{ left: 10 }}>
                    <CartesianGrid {...GRID} horizontal={false} vertical />
                    <XAxis type="number" {...AXIS} />
                    <YAxis type="category" interval={0} dataKey="feature" width={140} {...AXIS} tick={{ fontSize: 10 }} />
                    <Tooltip formatter={(v: number) => fmt(v, 4)} />
                    <Bar isAnimationActive={false} dataKey="importance" fill={SERIES[0]} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartBox>
            </Card>
            <Card title="Interpretable coefficients">
              <p className="mb-2 text-xs text-slate-500">Linear regression: BMI change per 1 SD (numeric) or vs reference category.</p>
              <DataTable rows={m.linear_coefficients.slice(0, 10)} />
              <p className="mb-2 mt-4 text-xs text-slate-500">Logistic regression: odds ratio of obesity.</p>
              <DataTable rows={m.logistic_odds_ratios.slice(0, 10)} />
            </Card>
          </div>
          <Notice tone="warning"><p className="font-semibold">Limitations</p><ul className="list-disc pl-4">{m.notes.map((n) => <li key={n}>{n}</li>)}</ul></Notice>
        </>
      )}
    </div>
  );
}

export default function AdvancedAnalytics() {
  const [tab, setTab] = useTab(TABS.map((t) => t.id), "pca" as TabId);
  return (
    <>
      <PageHeader title="Advanced Analytics" tutorials={["T13", "T14", "T15"]}
        description="Dimensionality reduction for EDA, and behavioural factors associated with adult weight using NHANES 2017-2018 (real public survey data)." />
      <Tabs tabs={TABS.map((t) => ({ ...t }))} value={tab} onChange={setTab} />
      {tab === "pca" && <PcaTab />}
      {tab === "weight-eda" && <WeightEdaTab />}
      {tab === "weight-model" && <WeightModelTab />}
    </>
  );
}
