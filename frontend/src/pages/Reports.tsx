import { Download, FileJson, FileSpreadsheet, FileText } from "lucide-react";
import { useState } from "react";
import { AsyncView, Card, ErrorState, Notice, PageHeader } from "../components/ui";
import { useFetch } from "../hooks/useApi";
import { describeError, getJson, http, type ApiErrorInfo } from "../services/api";

interface ReportInfo { id: string; title: string; formats: ("json" | "csv" | "pdf")[] }

const ICONS = { json: FileJson, csv: FileSpreadsheet, pdf: FileText };
const DESCRIPTIONS: Record<string, string> = {
  pneumonia_evaluation: "Test-set metrics at both thresholds, confusion matrix and the saved training/ROC plots.",
  predictions: "Every X-ray prediction stored in the database.",
  ehr_quality: "Quality scores before/after cleaning the teaching dataset, cleaning steps and recovery check.",
  operations_kpis: "Headline KPIs plus breakdowns by encounter class and year.",
  text_classification: "Accuracy and F1 of the three text classifiers with per-class results.",
  weight_models: "BMI regression and obesity classification metrics with feature importance.",
  analysis_history: "Log of every analysis run in this installation.",
  registry: "Dataset and model registry snapshot.",
};

export default function Reports() {
  const reports = useFetch(() => getJson<ReportInfo[]>("/api/reports"));
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<{ id: string; info: ApiErrorInfo } | null>(null);

  const download = async (id: string, format: string) => {
    setBusy(`${id}:${format}`);
    setError(null);
    try {
      const res = await http.get(`/api/reports/${id}`, { params: { format }, responseType: "blob" });
      const disposition = String(res.headers["content-disposition"] ?? "");
      const name = /filename="(.+)"/.exec(disposition)?.[1] ?? `${id}.${format}`;
      const url = URL.createObjectURL(res.data as Blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      // Error bodies arrive as Blobs when responseType is blob; parse them for the message.
      let info = describeError(e);
      const data = (e as { response?: { data?: Blob } }).response?.data;
      if (data instanceof Blob) {
        try {
          const parsed = JSON.parse(await data.text());
          if (parsed?.error?.message) info = { ...info, code: parsed.error.code, message: parsed.error.message };
        } catch { /* keep the generic message */ }
      }
      setError({ id, info });
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      <PageHeader title="Reports" description="Export real results as JSON (complete), CSV (tables) or PDF (formatted summary). Reports are generated on demand from stored results." />
      <div className="mb-5"><Notice>If a report depends on a model or analysis that has not been run yet, the export explains what to run instead of producing an empty file.</Notice></div>
      <AsyncView state={reports} onRetry={reports.run}>
        {(list) => (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {list.map((r) => (
              <Card key={r.id} title={r.title} subtitle={DESCRIPTIONS[r.id]}>
                <div className="flex flex-wrap gap-2">
                  {r.formats.map((f) => {
                    const Icon = ICONS[f];
                    return (
                      <button key={f} className="btn-secondary" disabled={busy !== null} onClick={() => download(r.id, f)} aria-label={`Download ${r.title} as ${f.toUpperCase()}`}>
                        {busy === `${r.id}:${f}` ? <Download className="h-4 w-4 animate-bounce" /> : <Icon className="h-4 w-4" />}
                        {f.toUpperCase()}
                      </button>
                    );
                  })}
                </div>
                {error?.id === r.id && <div className="mt-3"><ErrorState error={error.info} /></div>}
              </Card>
            ))}
          </div>
        )}
      </AsyncView>
    </>
  );
}
