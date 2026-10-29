import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { AXIS, ChartBox, GRID, Legend, MatrixHeatmap, SERIES, TooltipBox } from "../components/charts";
import { AsyncView, Badge, Card, EmptyState, Field, KeyValue, Notice, PageHeader, RunButton, Select, StatCard, Tabs } from "../components/ui";
import { DataTable } from "../components/ui/DataTable";
import { useAction, useFetch } from "../hooks/useApi";
import { useTab } from "../hooks/useTab";
import { getJson, postJson } from "../services/api";
import { fmt, pct } from "../utils/format";

const TABS = [
  { id: "classify", label: "Classification", badge: "T10" },
  { id: "cluster", label: "Clustering", badge: "T10" },
  { id: "ner", label: "Entity extraction", badge: "T11" },
  { id: "ranking", label: "Ranking & disambiguation", badge: "T12" },
] as const;
type TabId = (typeof TABS)[number]["id"];

interface ModelMetrics { name: string; accuracy: number; macro_f1: number; weighted_f1: number; per_class: { class: string; precision: number; recall: number; f1: number; support: number }[]; confusion_matrix: number[][] }
interface Metrics { classes: string[]; train_documents: number; test_documents: number; vectorizer: string; models: Record<string, ModelMetrics>; top_terms: Record<string, string[]>; best_model: string; trained?: false }
interface Classified { predictions: Record<string, { name: string; prediction: string; probabilities?: Record<string, number> }>; explanation: { term: string; contribution: number }[] }
interface Pre { steps: { name: string; tokens: string[]; count: number }[]; tfidf_top_terms?: { term: string; weight: number }[] }

const EXAMPLE = "A 58-year-old man with progressive dyspnoea and angina. Coronary angiography showed severe stenosis of the left anterior descending artery and he underwent bypass surgery.";

function ClassifyTab() {
  const metrics = useFetch(() => getJson<Metrics>("/api/text/metrics"));
  const train = useAction(() => postJson<Metrics>("/api/text/train"));
  const corpus = useFetch(() => getJson<{ train_documents: number; test_documents: number; class_counts: { train: Record<string, number> }; words_per_document: Record<string, number> }>("/api/text/corpus"));
  const [text, setText] = useState(EXAMPLE);
  const classify = useAction(() => postJson<Classified>("/api/text/classify", { text }));
  const pre = useAction(() => postJson<Pre>("/api/text/preprocess", { text }));
  const m = train.data ?? (metrics.data && metrics.data.trained !== false ? metrics.data : null);
  const [modelKey, setModelKey] = useState("logistic_regression");
  return (
    <div className="space-y-6">
      <AsyncView state={corpus} empty={null}>
        {(c) => (
          <Card title="Corpus: Medical Abstracts (5 disease classes)" subtitle={`${fmt(c.train_documents)} training / ${fmt(c.test_documents)} test abstracts · median ${fmt(c.words_per_document.median)} words`}>
            <ChartBox height={180}>
              <ResponsiveContainer>
                <BarChart data={Object.entries(c.class_counts.train).map(([k, v]) => ({ label: k, count: v }))} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid {...GRID} horizontal={false} vertical />
                  <XAxis type="number" {...AXIS} />
                  <YAxis type="category" interval={0} dataKey="label" width={190} {...AXIS} />
                  <Tooltip />
                  <Bar isAnimationActive={false} dataKey="count" name="training documents" fill={SERIES[0]} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartBox>
          </Card>
        )}
      </AsyncView>
      <Card title="Supervised classifiers" subtitle="TF-IDF features + three linear models, evaluated on the official test split" actions={
        <RunButton loading={train.loading} onClick={() => train.run()}>{m ? "Retrain" : "Train classifiers (~1 min)"}</RunButton>}>
        {train.error && <Notice tone="warning">{train.error.message}</Notice>}
        {train.loading && <p className="text-sm text-slate-500">Training on {fmt(corpus.data?.train_documents)} abstracts...</p>}
        {!m && !train.loading && <EmptyState title="Classifiers not trained yet" message="Press the button to fit and evaluate them on the real corpus." />}
        {m && (
          <div className="space-y-4">
            <DataTable rows={Object.entries(m.models).map(([k, v]) => ({ model: v.name, accuracy: pct(v.accuracy), macro_f1: fmt(v.macro_f1), weighted_f1: fmt(v.weighted_f1), best: k === m.best_model ? "✓" : "" }))} />
            <p className="text-xs text-slate-500">{m.vectorizer}. Chance level for this class balance is about 23% accuracy.</p>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <div>
                <div className="mb-2 w-60"><Select ariaLabel="Model" value={modelKey} onChange={setModelKey} options={Object.entries(m.models).map(([k, v]) => ({ value: k, label: v.name }))} /></div>
                <MatrixHeatmap rows={m.classes} cols={m.classes.map((c) => c.split(" ")[0])} values={m.models[modelKey].confusion_matrix} rowTitle="actual" colTitle="predicted" />
              </div>
              <DataTable rows={m.models[modelKey].per_class} />
            </div>
            <div>
              <p className="mb-2 text-sm font-medium text-slate-700">Most indicative terms per class (logistic regression coefficients)</p>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                {Object.entries(m.top_terms).map(([cls, terms]) => (
                  <div key={cls} className="rounded-lg border border-slate-200 p-3">
                    <p className="text-xs font-semibold text-slate-700">{cls}</p>
                    <div className="mt-1 flex flex-wrap gap-1">{terms.slice(0, 10).map((t) => <Badge key={t}>{t}</Badge>)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </Card>
      <Card title="Try it: preprocess and classify your own text">
        <textarea className="input min-h-[110px]" value={text} onChange={(e) => setText(e.target.value)} aria-label="Clinical text" maxLength={20000} />
        <div className="mt-3 flex flex-wrap gap-2">
          <RunButton loading={pre.loading} onClick={() => pre.run()} disabled={!text.trim()}>Show preprocessing</RunButton>
          <RunButton loading={classify.loading} onClick={() => classify.run()} disabled={!text.trim()}>Classify</RunButton>
        </div>
        <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <AsyncView state={pre} empty={null}>
            {(p) => (
              <div className="space-y-3">
                {p.steps.map((s) => (
                  <div key={s.name}>
                    <p className="text-xs font-semibold text-slate-600">{s.name} · {s.count} tokens</p>
                    <div className="mt-1 flex flex-wrap gap-1">{s.tokens.slice(0, 40).map((t, i) => <code key={i} className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px]">{t}</code>)}</div>
                  </div>
                ))}
                {p.tfidf_top_terms && <DataTable rows={p.tfidf_top_terms} />}
              </div>
            )}
          </AsyncView>
          <AsyncView state={classify} empty={null}>
            {(c) => (
              <div className="space-y-3">
                <KeyValue items={Object.values(c.predictions).map((p) => ({ label: p.name, value: <Badge tone="brand">{p.prediction}</Badge> }))} />
                {c.predictions.logistic_regression.probabilities && (
                  <ChartBox height={170}>
                    <ResponsiveContainer>
                      <BarChart data={Object.entries(c.predictions.logistic_regression.probabilities).map(([k, v]) => ({ label: k, p: v }))} layout="vertical" margin={{ left: 10 }}>
                        <XAxis type="number" domain={[0, 1]} {...AXIS} />
                        <YAxis type="category" interval={0} dataKey="label" width={180} {...AXIS} tick={{ fontSize: 10 }} />
                        <Tooltip formatter={(v: number) => pct(v)} />
                        <Bar isAnimationActive={false} dataKey="p" name="probability" fill={SERIES[0]} radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                )}
                <p className="text-xs text-slate-500">Terms pushing the logistic-regression decision:</p>
                <div className="flex flex-wrap gap-1">{c.explanation.map((e) => <Badge key={e.term} tone="blue">{e.term} +{e.contribution.toFixed(2)}</Badge>)}</div>
              </div>
            )}
          </AsyncView>
        </div>
      </Card>
    </div>
  );
}

interface Cluster {
  k: number; documents: number; method: string; silhouette: number; adjusted_rand_index: number; normalized_mutual_info: number; purity: number;
  clusters: { cluster: number; size: number; terms: string[]; dominant_label: string; dominant_share: number }[];
  contingency: { clusters: number[]; labels: string[]; counts: number[][] };
  points: { x: number; y: number; cluster: number; label: string; title: string }[];
  elbow: { k: number; inertia: number }[];
  projection_note: string;
}

function ClusterTab() {
  const [k, setK] = useState(5);
  const run = useAction(() => getJson<Cluster>("/api/text/cluster", { k, sample_size: 3000 }));
  return (
    <div className="space-y-6">
      <Card title="k-means on TF-IDF vectors" subtitle="Unsupervised: labels are used only afterwards to evaluate the clusters">
        <div className="flex flex-wrap items-end gap-4">
          <div className="w-48"><Field label={`Clusters k = ${k}`}><input type="range" min={2} max={10} value={k} onChange={(e) => setK(Number(e.target.value))} className="w-full accent-brand-600" /></Field></div>
          <RunButton loading={run.loading} onClick={() => run.run()}>Cluster 3,000 abstracts</RunButton>
        </div>
      </Card>
      <AsyncView state={run} loadingLabel="Vectorising and clustering...">
        {(c) => (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <StatCard label="Silhouette" value={fmt(c.silhouette, 3)} hint="-1..1, cohesion vs separation" />
              <StatCard label="Adjusted Rand index" value={fmt(c.adjusted_rand_index, 3)} hint="agreement with labels (0 = chance)" />
              <StatCard label="Normalised mutual info" value={fmt(c.normalized_mutual_info, 3)} />
              <StatCard label="Purity" value={pct(c.purity)} />
            </div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card title="Documents in the first two SVD dimensions" subtitle={c.projection_note}>
                <Legend items={c.clusters.slice(0, 8).map((cl) => ({ label: `cluster ${cl.cluster}`, color: SERIES[cl.cluster] ?? SERIES[7] }))} />
                <ChartBox height={300}>
                  <ResponsiveContainer>
                    <ScatterChart>
                      <CartesianGrid {...GRID} vertical />
                      <XAxis dataKey="x" type="number" {...AXIS} name="SVD 1" />
                      <YAxis dataKey="y" type="number" {...AXIS} name="SVD 2" />
                      <ZAxis range={[30, 30]} />
                      <Tooltip content={({ payload }) => payload?.[0] ? <TooltipBox title={payload[0].payload.title} rows={[{ label: "cluster", value: payload[0].payload.cluster }, { label: "true label", value: payload[0].payload.label }]} /> : null} />
                      {c.clusters.map((cl) => (
                        <Scatter isAnimationActive={false} key={cl.cluster} data={c.points.filter((p) => p.cluster === cl.cluster)} fill={SERIES[Math.min(cl.cluster, 7)]} fillOpacity={0.7} stroke="#fff" strokeWidth={1} />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </ChartBox>
              </Card>
              <Card title="Elbow plot" subtitle="Within-cluster sum of squares vs k">
                <ChartBox height={300}>
                  <ResponsiveContainer>
                    <LineChart data={c.elbow}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="k" {...AXIS} />
                      <YAxis {...AXIS} domain={["auto", "auto"]} />
                      <Tooltip />
                      <Line isAnimationActive={false} dataKey="inertia" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartBox>
              </Card>
            </div>
            <Card title="Cluster descriptions">
              <DataTable rows={c.clusters.map((cl) => ({ cluster: cl.cluster, size: cl.size, top_terms: cl.terms.join(", "), dominant_label: cl.dominant_label, share: pct(cl.dominant_share) }))} />
            </Card>
            <Card title="Clusters vs true labels (contingency)">
              <MatrixHeatmap rows={c.contingency.clusters.map((x) => `cluster ${x}`)} cols={c.contingency.labels.map((l) => l.split(" ")[0])} values={c.contingency.counts} />
            </Card>
          </div>
        )}
      </AsyncView>
    </div>
  );
}

interface Entity { text: string; label: string; start: number; end: number; negated: boolean; historical: boolean; family: boolean; source: string }
interface NerResult {
  model: string; pipeline: string[]; text: string; entities: Entity[]; by_label: Record<string, string[]>; colors: Record<string, string>; counts: Record<string, number>;
  structured: { problems_present: string[]; problems_negated: string[]; family_history: string[]; medications: { medication: string; dosage: string | null; negated: boolean }[]; vitals: string[]; labs: string[]; procedures: string[] };
}

function Highlighted({ r }: { r: NerResult }) {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  r.entities.forEach((e, i) => {
    if (e.start > cursor) parts.push(<span key={`t${i}`}>{r.text.slice(cursor, e.start)}</span>);
    const color = r.colors[e.label] ?? "#64748b";
    parts.push(
      <mark key={`e${i}`} title={`${e.label}${e.negated ? " (negated)" : ""}${e.historical ? " (history)" : ""}${e.family ? " (family)" : ""}`}
        className="rounded px-1 py-0.5" style={{ background: `${color}22`, borderBottom: `2px solid ${color}`, textDecoration: e.negated ? "line-through" : undefined }}>
        {r.text.slice(e.start, e.end)}<sup className="ml-0.5 text-[9px] font-semibold" style={{ color }}>{e.label}</sup>
      </mark>,
    );
    cursor = e.end;
  });
  parts.push(<span key="end">{r.text.slice(cursor)}</span>);
  return <p className="leading-8 text-slate-800">{parts}</p>;
}

function NerTab() {
  const notes = useFetch(() => getJson<{ id: string; title: string; text: string }[]>("/api/text/sample-notes"));
  const [text, setText] = useState("");
  const run = useAction(() => postJson<NerResult>("/api/text/entities", { text }));
  const stats = useAction(() => getJson<{ model: string; documents: number; top_entities: Record<string, Record<string, { text: string; count: number }[]>> }>("/api/text/entity-stats", { per_class: 40 }));
  return (
    <div className="space-y-6">
      <Card title="Clinical note" subtitle="Sample notes are synthetic and written for this project">
        <div className="mb-2 flex flex-wrap gap-2">
          {notes.data?.map((n) => <button key={n.id} className="btn-secondary py-1 text-xs" onClick={() => setText(n.text)}>{n.title}</button>)}
        </div>
        <textarea className="input min-h-[130px]" value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste or pick a clinical note..." aria-label="Clinical note" />
        <div className="mt-3"><RunButton loading={run.loading} onClick={() => run.run()} disabled={!text.trim()}>Extract entities</RunButton></div>
      </Card>
      <AsyncView state={run} loadingLabel="Running spaCy pipeline...">
        {(r) => (
          <div className="space-y-6">
            <Card title="Highlighted entities" subtitle={`spaCy model ${r.model} + clinical EntityRuler · struck-through = negated`}>
              <Legend items={Object.keys(r.counts).map((l) => ({ label: `${l} (${r.counts[l]})`, color: r.colors[l] ?? "#64748b" }))} />
              <Highlighted r={r} />
            </Card>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card title="Structured output">
                <KeyValue items={[
                  { label: "Problems present", value: r.structured.problems_present.join(", ") || "—" },
                  { label: "Problems negated", value: r.structured.problems_negated.join(", ") || "—" },
                  { label: "Family history", value: r.structured.family_history.join(", ") || "—" },
                  { label: "Vitals", value: r.structured.vitals.join(", ") || "—" },
                  { label: "Labs", value: r.structured.labs.join(", ") || "—" },
                  { label: "Procedures", value: r.structured.procedures.join(", ") || "—" },
                ]} />
                <div className="mt-3"><DataTable rows={r.structured.medications} /></div>
                <details className="mt-3 text-xs"><summary className="cursor-pointer text-slate-500">JSON</summary><pre className="mt-2 max-h-64 overflow-auto rounded bg-slate-900 p-3 text-slate-100">{JSON.stringify(r.structured, null, 2)}</pre></details>
              </Card>
              <Card title="Entity table"><DataTable rows={r.entities.map(({ text, label, negated, historical, family, source }) => ({ text, label, negated, historical, family, source }))} /></Card>
            </div>
          </div>
        )}
      </AsyncView>
      <Card title="Information extraction at corpus scale" subtitle="Run the same pipeline on 40 abstracts per class and count the entities" actions={<RunButton loading={stats.loading} onClick={() => stats.run()}>Run on corpus</RunButton>}>
        <AsyncView state={stats} empty={null} loadingLabel="Processing 200 abstracts...">
          {(s) => (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
              {Object.entries(s.top_entities).map(([cls, kinds]) => (
                <div key={cls} className="rounded-lg border border-slate-200 p-3 text-xs">
                  <p className="mb-1 font-semibold text-slate-700">{cls}</p>
                  {Object.entries(kinds).map(([kind, items]) => (
                    <p key={kind} className="text-slate-600"><span className="font-medium">{kind}:</span> {items.slice(0, 5).map((i) => `${i.text} (${i.count})`).join(", ")}</p>
                  ))}
                </div>
              ))}
            </div>
          )}
        </AsyncView>
      </Card>
    </div>
  );
}

interface Ranked { query: string; method: string; documents_indexed: number; label_distribution: Record<string, number>; results: { rank: number; doc_id: number; score: number; label: string; title: string; snippet: string; matched_terms: string[] }[] }
interface RankEval { queries: number; k: number; results: Record<string, { precision_at_k: number; map_at_k: number }>; random_baseline_precision: number; protocol: string }
interface Wsd { abbreviation: string; predicted_sense: string; confident: boolean; context_words: string[]; scores: { sense: string; score: number; overlap: string[] }[] }
interface WsdEval { examples: number; accuracy: Record<string, number>; predictions: Record<string, string | boolean>[]; inventory: Record<string, string[]>; data_note: string }

function RankingTab() {
  const [query, setQuery] = useState("antibiotic treatment of pneumonia in children");
  const [method, setMethod] = useState("bm25");
  const rank = useAction(() => postJson<Ranked>("/api/text/rank", { query, method, top_k: 10 }));
  const evaluation = useAction(() => getJson<RankEval>("/api/text/rank-evaluation", { n_queries: 200, k: 10 }));
  const [sentence, setSentence] = useState("SpO2 dropped to 89% on RA, so supplemental oxygen was started.");
  const [variant, setVariant] = useState("corpus_lesk");
  const wsd = useAction(() => postJson<Wsd>("/api/text/wsd", { sentence, variant }));
  const wsdEval = useAction(() => getJson<WsdEval>("/api/text/wsd-evaluation"));
  return (
    <div className="space-y-6">
      <Card title="Document ranking" subtitle="Vector-space retrieval over 11,550 abstracts">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          <div className="md:col-span-3"><Field label="Query"><input className="input" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && rank.run()} /></Field></div>
          <Field label="Scoring"><Select value={method} onChange={setMethod} options={[{ value: "tfidf", label: "TF-IDF cosine" }, { value: "bm25", label: "BM25" }]} /></Field>
        </div>
        <div className="mt-3 flex gap-2">
          <RunButton loading={rank.loading} onClick={() => rank.run()} disabled={!query.trim()}>Search</RunButton>
          <RunButton loading={evaluation.loading} onClick={() => evaluation.run()}>Evaluate both methods</RunButton>
        </div>
        <div className="mt-4 space-y-4">
          <AsyncView state={evaluation} empty={null} loadingLabel="Running 200 queries per method...">
            {(e) => (
              <div className="space-y-2">
                <DataTable rows={Object.entries(e.results).map(([m, r]) => ({ method: m, [`precision@${e.k}`]: fmt(r.precision_at_k), [`MAP@${e.k}`]: fmt(r.map_at_k) }))} />
                <p className="text-xs text-slate-500">{e.protocol} Random baseline precision ≈ {fmt(e.random_baseline_precision)}.</p>
              </div>
            )}
          </AsyncView>
          <AsyncView state={rank} empty={null}>
            {(r) => (
              <ol className="space-y-3">
                {r.results.map((d) => (
                  <li key={d.doc_id} className="rounded-lg border border-slate-200 p-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs font-semibold text-slate-400">#{d.rank}</span>
                      <span className="text-sm font-medium text-slate-800">{d.title}</span>
                      <Badge tone="brand">{d.label}</Badge>
                      <span className="ml-auto text-xs tabular-nums text-slate-500">score {fmt(d.score, 3)}</span>
                    </div>
                    <p className="mt-1 line-clamp-2 text-xs text-slate-500">{d.snippet}</p>
                    <div className="mt-1 flex flex-wrap gap-1">{d.matched_terms.map((t) => <Badge key={t} tone="blue">{t}</Badge>)}</div>
                  </li>
                ))}
              </ol>
            )}
          </AsyncView>
        </div>
      </Card>
      <Card title="Clinical abbreviation disambiguation (simplified Lesk)" subtitle="Supported: MS, RA, PE, CVA, PT, CP">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          <div className="md:col-span-3"><Field label="Sentence"><input className="input" value={sentence} onChange={(e) => setSentence(e.target.value)} /></Field></div>
          <Field label="Signature"><Select value={variant} onChange={setVariant} options={[
            { value: "lesk", label: "Hand-written glosses" }, { value: "corpus_lesk", label: "Glosses + corpus" }, { value: "corpus_only", label: "Corpus only" }]} /></Field>
        </div>
        <div className="mt-3 flex gap-2">
          <RunButton loading={wsd.loading} onClick={() => wsd.run()} disabled={!sentence.trim()}>Disambiguate</RunButton>
          <RunButton loading={wsdEval.loading} onClick={() => wsdEval.run()}>Evaluate on labelled examples</RunButton>
        </div>
        <div className="mt-4 space-y-4">
          <AsyncView state={wsd} empty={null}>
            {(w) => (
              <div className="space-y-2">
                <p className="text-sm"><Badge tone="slate">{w.abbreviation}</Badge> → <strong>{w.predicted_sense}</strong> {!w.confident && <Badge tone="amber">tie / no overlap</Badge>}</p>
                <DataTable rows={w.scores.map((s) => ({ sense: s.sense, overlap_score: s.score, overlapping_words: s.overlap.join(", ") }))} />
                <p className="text-xs text-slate-500">Context words: {w.context_words.join(", ")}</p>
              </div>
            )}
          </AsyncView>
          <AsyncView state={wsdEval} empty={null}>
            {(e) => (
              <div className="space-y-2">
                <Notice tone="warning">{e.data_note}</Notice>
                <DataTable rows={Object.entries(e.accuracy).map(([k, v]) => ({ method: k, accuracy: pct(v) }))} />
                <DataTable rows={e.predictions} maxRows={50} />
              </div>
            )}
          </AsyncView>
        </div>
      </Card>
    </div>
  );
}

export default function ClinicalText() {
  const [tab, setTab] = useTab(TABS.map((t) => t.id), "classify" as TabId);
  return (
    <>
      <PageHeader title="Clinical Text Analytics" tutorials={["T10", "T11", "T12"]}
        description="Classification, clustering, entity extraction, ranking and word-sense disambiguation on the Medical Abstracts corpus and synthetic clinical notes." />
      <Tabs tabs={TABS.map((t) => ({ ...t }))} value={tab} onChange={setTab} />
      {tab === "classify" && <ClassifyTab />}
      {tab === "cluster" && <ClusterTab />}
      {tab === "ner" && <NerTab />}
      {tab === "ranking" && <RankingTab />}
    </>
  );
}
