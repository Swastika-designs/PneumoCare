import { ShieldAlert } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { FileDrop, SelectedFile, useObjectUrl, validateImageFile } from "../components/images";
import { AsyncView, Badge, Card, EmptyState, ErrorState, LoadingState, Notice, PageHeader, RunButton } from "../components/ui";
import { DataTable } from "../components/ui/DataTable";
import { useAction, useFetch } from "../hooks/useApi";
import { getJson, http } from "../services/api";
import type { Prediction, ScreeningStatus } from "../types/api";
import { dateTime, pct } from "../utils/format";

interface PredictionRow {
  id: number;
  filename: string;
  predicted_label: string;
  probability_pneumonia: number;
  created_at: string;
}

export default function PneumoniaDetection() {
  const status = useFetch(() => getJson<ScreeningStatus>("/api/screening/status"));
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [showCam, setShowCam] = useState(true);
  const preview = useObjectUrl(file);
  const history = useFetch(() => getJson<PredictionRow[]>("/api/screening/predictions", { limit: 10 }));
  const predict = useAction(async (f: File) => {
    const form = new FormData();
    form.append("file", f);
    const res = await http.post<Prediction>("/api/screening/predict", form, { params: { gradcam: true } });
    void history.run();
    return res.data;
  });

  const choose = (f: File) => {
    const err = validateImageFile(f);
    setFileError(err);
    setFile(err ? null : f);
    predict.reset();
  };

  const result = predict.data;
  const modelReady = status.data?.model_available;

  return (
    <>
      <PageHeader
        title="Pneumonia Detection"
        tutorials={["Main project", "T6"]}
        description="Upload a chest X-ray to get a prediction from the trained ResNet-18 model, with a Grad-CAM map of the regions that drove it."
      />
      <div className="mb-5">
        <Notice tone="warning">
          <strong>Educational use only.</strong> The model was trained on a public paediatric chest X-ray dataset and has not
          been clinically validated. Do not use it for diagnosis.
        </Notice>
      </div>
      {status.loading && !status.data && <LoadingState label="Checking model status..." />}
      {status.error && <ErrorState error={status.error} onRetry={status.run} />}
      {status.data && !modelReady && (
        <EmptyState
          title="No trained model available"
          message={`${status.data.message ?? ""} Train with: python scripts/train_pneumonia.py, then python scripts/evaluate_pneumonia.py`}
        />
      )}
      {modelReady && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
          <div className="space-y-4 lg:col-span-2">
            <Card title="1. Choose an X-ray" subtitle={`Model: ${status.data?.model_name} · decision threshold ${status.data?.threshold?.toFixed(3)}`}>
              <FileDrop onFile={choose} />
              {fileError && <p role="alert" className="mt-2 text-sm text-red-600">{fileError}</p>}
              {file && (
                <div className="mt-3 space-y-3">
                  <SelectedFile file={file} onClear={() => { setFile(null); predict.reset(); }} />
                  {preview && <img src={preview} alt="Selected X-ray preview" className="mx-auto max-h-72 rounded-lg bg-slate-900 object-contain" />}
                  <RunButton loading={predict.loading} onClick={() => file && predict.run(file)}>
                    {predict.loading ? "Analysing..." : "2. Run prediction"}
                  </RunButton>
                </div>
              )}
              <p className="mt-3 text-xs text-slate-500">
                No X-ray at hand? Download a sample from the <Link className="text-brand-700 underline" to="/image-lab">Image Laboratory</Link> sample list
                (real images from the held-out test split).
              </p>
            </Card>
          </div>
          <div className="lg:col-span-3">
            <Card title="Result">
              {predict.loading && <LoadingState label="Running the model and computing Grad-CAM..." />}
              {predict.error && <ErrorState error={predict.error} />}
              {!predict.loading && !predict.error && !result && (
                <EmptyState title="No prediction yet" message="Choose an image and run the prediction." />
              )}
              {result && !predict.loading && (
                <div className="space-y-4" data-testid="prediction-result">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className={`rounded-lg px-3 py-1.5 text-lg font-semibold ${result.predicted_label === "PNEUMONIA" ? "bg-red-50 text-red-700" : "bg-emerald-50 text-emerald-700"}`}>
                      {result.predicted_label}
                    </span>
                    <span className="text-sm text-slate-600">
                      P(pneumonia) = <strong>{pct(result.probability_pneumonia)}</strong> · threshold {pct(result.threshold)}
                    </span>
                  </div>
                  <div aria-label="Probability bar" className="relative h-3 overflow-hidden rounded-full bg-slate-100">
                    <div className="h-full rounded-full bg-[#2a78d6]" style={{ width: `${result.probability_pneumonia * 100}%` }} />
                    <div className="absolute inset-y-0 w-0.5 bg-slate-900" style={{ left: `${result.threshold * 100}%` }} title="Decision threshold" />
                  </div>
                  {result.warnings.map((w) => <Notice key={w} tone="warning">{w}</Notice>)}
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <figure>
                      <img src={result.image} alt="Uploaded X-ray" className="aspect-square w-full rounded-lg bg-slate-900 object-contain" />
                      <figcaption className="mt-1 text-xs text-slate-500">Input (grayscale)</figcaption>
                    </figure>
                    {result.gradcam && (
                      <figure>
                        <img src={showCam ? result.gradcam.overlay : result.gradcam.heatmap} alt="Grad-CAM heat-map"
                          className="aspect-square w-full rounded-lg bg-slate-900 object-contain" />
                        <figcaption className="mt-1 flex items-center justify-between text-xs text-slate-500">
                          <span>Grad-CAM for class {result.gradcam.target_class}</span>
                          <button className="text-brand-700 underline" onClick={() => setShowCam((v) => !v)}>
                            {showCam ? "show raw map" : "show overlay"}
                          </button>
                        </figcaption>
                      </figure>
                    )}
                  </div>
                  <p className="text-xs text-slate-500">
                    Grad-CAM highlights regions whose features increased the PNEUMONIA score (red = strongest). It explains the
                    model, not the disease, and can highlight non-anatomical artefacts.
                  </p>
                  <p className="flex items-start gap-2 text-xs text-slate-500"><ShieldAlert className="h-4 w-4 shrink-0" />{result.disclaimer}</p>
                </div>
              )}
            </Card>
          </div>
        </div>
      )}
      <div className="mt-6">
        <Card title="Recent predictions" subtitle="Stored in the local SQLite database">
          <AsyncView state={history} empty={<p className="text-sm text-slate-500">No predictions yet.</p>}>
            {(rows) => rows.length ? (
              <DataTable rows={rows} columns={[
                { key: "filename", label: "File" },
                { key: "predicted_label", label: "Prediction", render: (r) => <Badge tone={r.predicted_label === "PNEUMONIA" ? "red" : "green"}>{r.predicted_label}</Badge> },
                { key: "probability_pneumonia", label: "P(pneumonia)", render: (r) => pct(r.probability_pneumonia), align: "right" },
                { key: "created_at", label: "When", render: (r) => dateTime(r.created_at) },
              ]} />
            ) : <p className="text-sm text-slate-500">No predictions yet.</p>}
          </AsyncView>
        </Card>
      </div>
    </>
  );
}
