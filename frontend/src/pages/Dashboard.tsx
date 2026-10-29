import { BookOpen, CheckCircle2, CircleDashed, Database, History, Stethoscope } from "lucide-react";
import { Link } from "react-router-dom";
import { AsyncView, Badge, Card, EmptyState, PageHeader, StatCard } from "../components/ui";
import { useFetch } from "../hooks/useApi";
import { getJson } from "../services/api";
import type { SystemStatus } from "../types/api";
import { dateTime, pct } from "../utils/format";

const MODULE_LINKS = [
  { to: "/pneumonia", title: "Pneumonia Detection", text: "Upload a chest X-ray, get a model prediction and Grad-CAM." },
  { to: "/image-lab", title: "Biomedical Image Lab", text: "T4-T6: enhancement, segmentation, SIFT/RANSAC, CNN features." },
  { to: "/ehr", title: "EHR Data Explorer", text: "T1-T3: schema, cleaning, missing data, duplicates, imputation." },
  { to: "/operations", title: "Operations Analytics", text: "T7-T9: volume forecasting, KPIs, utilisation, scenarios." },
  { to: "/clinical-text", title: "Clinical Text Analytics", text: "T10-T12: classification, NER, ranking, disambiguation." },
  { to: "/advanced", title: "Advanced Analytics", text: "T13-T15: PCA, adult weight exploration and models." },
];

export default function Dashboard() {
  const status = useFetch(() => getJson<SystemStatus>("/api/system/status"));
  return (
    <>
      <PageHeader
        title="Overview"
        description="System status, data availability and tutorial progress. Every figure on this page is read from the backend."
      />
      <AsyncView state={status} onRetry={status.run} loadingLabel="Loading system status...">
        {(s) => {
          const pneumonia = s.models.find((m) => m.id === "pneumonia_resnet18");
          const auc = pneumonia?.metrics?.roc_auc as number | undefined;
          return (
            <div className="space-y-6">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard icon={<Stethoscope className="h-5 w-5" />} label="Pneumonia model"
                  value={pneumonia?.status === "trained" ? "Trained" : pneumonia?.status === "training" ? "Training" : "Not trained"}
                  hint={auc !== undefined ? `Test ROC-AUC ${auc.toFixed(3)}` : "Run scripts/train_pneumonia.py"} />
                <StatCard icon={<Database className="h-5 w-5" />} label="Required datasets"
                  value={`${s.datasets.required_available} / ${s.datasets.required}`}
                  hint={`checked on disk · ${s.datasets.total - s.datasets.required} optional`} />
                <StatCard icon={<BookOpen className="h-5 w-5" />} label="Tutorials verified"
                  value={`${s.tutorials.verified} / ${s.tutorials.total}`}
                  hint={`${s.tutorials.data_ready} data-ready · ${s.tutorials.explored} explored`} />
                <StatCard icon={<History className="h-5 w-5" />} label="Analyses run"
                  value={s.history.total_runs} hint={`${s.history.predictions} X-ray predictions`} />
              </div>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
                <Card title="Tutorial progress" subtitle="Implemented modules and how often each has been run" className="xl:col-span-2">
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {s.tutorials.items.map((t) => (
                      <Link key={t.id} to={t.route}
                        className="flex items-start gap-2 rounded-lg border border-slate-200 p-3 transition hover:border-brand-300 hover:bg-brand-50/40">
                        {t.runs > 0 ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" aria-label="explored" />
                          : <CircleDashed className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" aria-label="not yet run" />}
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-brand-700">{t.id}</p>
                          <p className="truncate text-sm text-slate-700" title={t.title}>{t.title}</p>
                          <p className="text-[11px] text-slate-400">{t.status} · {t.runs} run{t.runs === 1 ? "" : "s"}{!t.data_ready && " · data missing"}</p>
                        </div>
                      </Link>
                    ))}
                  </div>
                </Card>
                <Card title="Datasets" subtitle="Separate domains, shared infrastructure">
                  <ul className="space-y-2">
                    {s.datasets.items.map((d) => (
                      <li key={d.id} className="flex items-center justify-between gap-2 text-sm">
                        <span className="truncate" title={d.name}>{d.name}</span>
                        <span className="flex shrink-0 gap-1">
                          {d.synthetic && <Badge tone="amber">synthetic</Badge>}
                          <Badge tone={d.available ? "green" : d.required ? "red" : "slate"}>
                            {d.available ? "available" : d.required ? "missing" : "optional"}
                          </Badge>
                        </span>
                      </li>
                    ))}
                  </ul>
                  <Link to="/registry" className="mt-4 inline-block text-sm font-medium text-brand-700 hover:underline">Open registry →</Link>
                </Card>
              </div>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
                <Card title="Recent analyses" className="xl:col-span-2">
                  {s.history.recent.length === 0 ? (
                    <EmptyState title="No analyses yet" message="Run any module and it will appear here." />
                  ) : (
                    <ul className="divide-y divide-slate-100">
                      {s.history.recent.map((h) => (
                        <li key={h.id} className="flex items-center justify-between gap-3 py-2 text-sm">
                          <span className="flex min-w-0 items-center gap-2">
                            {h.tutorial && <Badge tone="brand">{h.tutorial}</Badge>}
                            <span className="truncate">{h.title}</span>
                          </span>
                          <span className="shrink-0 text-xs text-slate-400">{dateTime(h.created_at)}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </Card>
                <Card title="Models">
                  <ul className="space-y-3">
                    {s.models.map((m) => (
                      <li key={m.id} className="text-sm">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-medium text-slate-800">{m.name}</span>
                          <Badge tone={m.status === "trained" || m.status === "pretrained" ? "green" : m.status === "training" ? "amber" : "slate"}>
                            {m.status.replace("_", " ")}
                          </Badge>
                        </div>
                        {m.id === "pneumonia_resnet18" && m.metrics && (
                          <p className="text-xs text-slate-500">
                            accuracy {pct(m.metrics.accuracy as number)} · recall {pct(m.metrics.recall as number)} · specificity {pct(m.metrics.specificity as number)}
                          </p>
                        )}
                      </li>
                    ))}
                  </ul>
                </Card>
              </div>

              <Card title="Modules">
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {MODULE_LINKS.map((m) => (
                    <Link key={m.to} to={m.to} className="rounded-lg border border-slate-200 p-4 transition hover:border-brand-300 hover:shadow-sm">
                      <p className="font-medium text-slate-900">{m.title}</p>
                      <p className="mt-1 text-sm text-slate-500">{m.text}</p>
                    </Link>
                  ))}
                </div>
              </Card>
            </div>
          );
        }}
      </AsyncView>
    </>
  );
}
