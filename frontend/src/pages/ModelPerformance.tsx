import { useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { AXIS, ChartBox, GRID, Legend, MatrixHeatmap, SERIES, TooltipBox } from "../components/charts";
import { AsyncView, Card, EmptyState, KeyValue, Notice, PageHeader, StatCard } from "../components/ui";
import { DataTable } from "../components/ui/DataTable";
import { useFetch } from "../hooks/useApi";
import { getJson } from "../services/api";
import type { BinaryMetrics, Evaluation, TrainingHistory } from "../types/api";
import { dateTime, fmt, pct } from "../utils/format";

interface Inspection {
  source: string;
  splits: Record<string, { total: number; class_counts: Record<string, number>; pneumonia_fraction: number }>;
  notes: string[];
  leakage?: {
    hash_collisions_between_splits: Record<string, number>;
    confirmed_near_duplicates: Record<string, number>;
    note: string;
  };
}

function MetricCards({ m }: { m: BinaryMetrics }) {
  const items = [
    ["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall (sensitivity)", m.recall],
    ["Specificity", m.specificity], ["F1-score", m.f1], ["ROC-AUC", m.roc_auc],
  ] as const;
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      {items.map(([label, v]) => (
        <StatCard key={label} label={label} value={label === "ROC-AUC" ? fmt(v, 3) : pct(v as number)} />
      ))}
    </div>
  );
}

export default function ModelPerformance() {
  const evaluation = useFetch(() => getJson<Evaluation>("/api/models/pneumonia/evaluation"));
  const training = useFetch(() => getJson<TrainingHistory>("/api/models/pneumonia/training"));
  const dataset = useFetch(() => getJson<Inspection>("/api/models/pneumonia/dataset"));
  const [atHalf, setAtHalf] = useState(false);

  return (
    <>
      <PageHeader
        title="Model Performance"
        tutorials={["Main project"]}
        description="Metrics, confusion matrix, ROC curve and training curves of the pneumonia classifier - all read from the files written by the training and evaluation scripts."
      />
      <div className="space-y-6">
        <AsyncView state={evaluation} loadingLabel="Loading evaluation..."
          empty={<EmptyState title="No evaluation yet" message="Run python scripts/evaluate_pneumonia.py" />}>
          {(ev) => {
            const m = atHalf ? ev.metrics_at_0_5 : ev.metrics;
            const cm = m.confusion_matrix;
            return (
              <div className="space-y-6">
                <Card
                  title={`Held-out ${ev.split} set · ${ev.samples} images (${ev.data_source})`}
                  subtitle={`Evaluated ${dateTime(ev.evaluated_at)} · checkpoint ${ev.checkpoint}`}
                  actions={
                    <div role="group" aria-label="Decision threshold" className="flex rounded-lg border border-slate-200 p-0.5 text-xs">
                      <button className={`rounded-md px-2 py-1 ${!atHalf ? "bg-brand-600 text-white" : ""}`} onClick={() => setAtHalf(false)}>
                        Validation-chosen threshold ({ev.metrics.threshold.toFixed(3)})
                      </button>
                      <button className={`rounded-md px-2 py-1 ${atHalf ? "bg-brand-600 text-white" : ""}`} onClick={() => setAtHalf(true)}>
                        Threshold 0.5
                      </button>
                    </div>
                  }
                >
                  <MetricCards m={m} />
                  <p className="mt-3 text-xs text-slate-500">
                    The operating threshold was chosen on the validation split (Youden J) before looking at the test set, so test metrics are not tuned to the test data.
                  </p>
                </Card>
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <Card title="Confusion matrix" subtitle="Rows = actual class, columns = predicted class">
                    <MatrixHeatmap rows={["NORMAL", "PNEUMONIA"]} cols={["NORMAL", "PNEUMONIA"]}
                      values={[[cm.tn, cm.fp], [cm.fn, cm.tp]]} rowTitle="actual ↓" colTitle="predicted →" />
                    <KeyValue items={[
                      { label: "True negatives", value: cm.tn }, { label: "False positives", value: cm.fp },
                      { label: "False negatives (missed pneumonia)", value: cm.fn }, { label: "True positives", value: cm.tp },
                    ]} />
                  </Card>
                  <Card title="ROC curve" subtitle={`Area under curve = ${fmt(ev.metrics.roc_auc, 4)}`}>
                    <ChartBox height={300}>
                      <ResponsiveContainer>
                        <AreaChart data={ev.curves.roc} margin={{ top: 8, right: 16, bottom: 20, left: 0 }}>
                          <CartesianGrid {...GRID} />
                          <XAxis dataKey="fpr" type="number" domain={[0, 1]} {...AXIS} label={{ value: "False positive rate", position: "insideBottom", offset: -10, fontSize: 11 }} />
                          <YAxis type="number" domain={[0, 1]} {...AXIS} label={{ value: "True positive rate", angle: -90, position: "insideLeft", fontSize: 11 }} />
                          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" />
                          <Tooltip content={({ payload }) => payload?.[0] ? (
                            <TooltipBox title={`threshold ${fmt(payload[0].payload.threshold, 3)}`} rows={[
                              { label: "TPR", value: fmt(payload[0].payload.tpr, 3) }, { label: "FPR", value: fmt(payload[0].payload.fpr, 3) }]} />
                          ) : null} />
                          <Area isAnimationActive={false} type="stepAfter" dataKey="tpr" stroke={SERIES[0]} strokeWidth={2} fill={SERIES[0]} fillOpacity={0.12} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </ChartBox>
                  </Card>
                </div>
                <Card title="Predicted probability by true class" subtitle="Test images per probability bin; a good model pushes the two classes apart">
                  <Legend items={[{ label: "Actual NORMAL", color: SERIES[2] }, { label: "Actual PNEUMONIA", color: SERIES[1] }]} />
                  <ChartBox height={240}>
                    <ResponsiveContainer>
                      <BarChart data={ev.probability_histogram.map((b) => ({ ...b, bin: b.bin_start.toFixed(2) }))} barGap={2}>
                        <CartesianGrid {...GRID} />
                        <XAxis dataKey="bin" {...AXIS} />
                        <YAxis {...AXIS} allowDecimals={false} />
                        <Tooltip cursor={{ fill: "#f1f5f9" }} />
                        <Bar isAnimationActive={false} dataKey="normal" name="Actual NORMAL" fill={SERIES[2]} radius={[4, 4, 0, 0]} />
                        <Bar isAnimationActive={false} dataKey="pneumonia" name="Actual PNEUMONIA" fill={SERIES[1]} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </Card>
              </div>
            );
          }}
        </AsyncView>

        <AsyncView state={training} loadingLabel="Loading training history..." empty={null}>
          {(t) => (
            <Card title="Training curves" subtitle={`Best epoch ${fmt(t.meta.best_epoch)} of ${fmt(t.meta.epochs_run)} · ${fmt(t.meta.train_samples)} training images · ${fmt(t.meta.training_seconds)} s on ${fmt(t.meta.device)}`}>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div>
                  <p className="mb-1 text-sm font-medium text-slate-700">Loss</p>
                  <Legend items={[{ label: "train", color: SERIES[0] }, { label: "validation", color: SERIES[1] }]} />
                  <ChartBox height={220}>
                    <ResponsiveContainer>
                      <LineChart data={t.history}>
                        <CartesianGrid {...GRID} />
                        <XAxis dataKey="epoch" {...AXIS} />
                        <YAxis {...AXIS} />
                        <Tooltip />
                        <Line isAnimationActive={false} dataKey="train_loss" name="train" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 4 }} />
                        <Line isAnimationActive={false} dataKey="val_loss" name="validation" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </div>
                <div>
                  <p className="mb-1 text-sm font-medium text-slate-700">Validation accuracy and ROC-AUC</p>
                  <Legend items={[{ label: "val accuracy", color: SERIES[0] }, { label: "val ROC-AUC", color: SERIES[2] }]} />
                  <ChartBox height={220}>
                    <ResponsiveContainer>
                      <LineChart data={t.history}>
                        <CartesianGrid {...GRID} />
                        <XAxis dataKey="epoch" {...AXIS} />
                        <YAxis {...AXIS} domain={[0.5, 1]} />
                        <Tooltip />
                        <Line isAnimationActive={false} dataKey="val_accuracy" name="val accuracy" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 4 }} />
                        <Line isAnimationActive={false} dataKey="val_roc_auc" name="val ROC-AUC" stroke={SERIES[2]} strokeWidth={2} dot={{ r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartBox>
                </div>
              </div>
              <div className="mt-4"><DataTable rows={t.history} /></div>
              {t.meta.data_notes && <ul className="mt-3 list-disc pl-5 text-xs text-slate-500">{t.meta.data_notes.map((n) => <li key={n}>{n}</li>)}</ul>}
            </Card>
          )}
        </AsyncView>

        <AsyncView state={dataset} empty={null}>
          {(d) => (
            <Card title="Dataset and leakage checks" subtitle={`Source: ${d.source}`}>
              <DataTable rows={Object.entries(d.splits).map(([split, s]) => ({
                split, total: s.total, normal: s.class_counts.NORMAL, pneumonia: s.class_counts.PNEUMONIA,
                pneumonia_share: pct(s.pneumonia_fraction),
              }))} />
              {d.leakage && (
                <div className="mt-4 space-y-2 text-sm">
                  <KeyValue items={Object.keys(d.leakage.hash_collisions_between_splits).map((k) => ({
                    label: `${k}: hash collisions / confirmed duplicates`,
                    value: `${d.leakage!.hash_collisions_between_splits[k]} / ${d.leakage!.confirmed_near_duplicates[k]}`,
                  }))} />
                  <Notice>{d.leakage.note} Confirmed train-val duplicates are removed from validation during training.</Notice>
                </div>
              )}
            </Card>
          )}
        </AsyncView>
        <AsyncView state={training} empty={null}>
          {() => (
            <Card title="Saved plots" subtitle="PNG files written by the scripts (models/pneumonia/)">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                {["training_curves.png", "confusion_matrix.png", "roc_curve.png"].map((p) => (
                  <img key={p} src={`/api/models/pneumonia/plots/${p}`} alt={p} className="w-full rounded-lg border border-slate-200 bg-white"
                    onError={(e) => ((e.target as HTMLImageElement).style.display = "none")} />
                ))}
              </div>
            </Card>
          )}
        </AsyncView>
      </div>
    </>
  );
}
