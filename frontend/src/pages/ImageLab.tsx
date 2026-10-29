import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { AXIS, ChartBox, GRID, Legend, SERIES, TooltipBox } from "../components/charts";
import { CompareSlider, FileDrop, ImageGrid, SelectedFile, validateImageFile } from "../components/images";
import { AsyncView, Badge, Card, EmptyState, ErrorState, Field, KeyValue, LoadingState, Notice, PageHeader, RunButton, Select, Tabs } from "../components/ui";
import { DataTable } from "../components/ui/DataTable";
import { useAction, useFetch } from "../hooks/useApi";
import { useTab } from "../hooks/useTab";
import { getJson, http, postImage } from "../services/api";
import type { ImagePanel, Prediction } from "../types/api";
import { fmt } from "../utils/format";

const TABS = [
  { id: "enhance", label: "Enhance", badge: "T4" },
  { id: "restore", label: "Restore", badge: "T4" },
  { id: "filters", label: "Filter", badge: "T4" },
  { id: "threshold", label: "Threshold", badge: "T4" },
  { id: "segment", label: "Segment", badge: "T4" },
  { id: "compress", label: "Compress", badge: "T4" },
  { id: "features", label: "SIFT · RANSAC · CNN", badge: "T5" },
  { id: "visualize", label: "Visualise", badge: "T6" },
] as const;
type TabId = (typeof TABS)[number]["id"];

interface Sample { name: string; label: string; url: string }
interface ImageSource { file: File | null; sample: string | null }
type Result = { panels: ImagePanel[] } & Record<string, unknown>;

function HistogramChart({ data, title }: { data: { intensity: number; count: number }[]; title: string }) {
  return (
    <div>
      <p className="mb-1 text-sm font-medium text-slate-700">{title}</p>
      <ChartBox height={180}>
        <ResponsiveContainer>
          <BarChart data={data} barCategoryGap={1}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="intensity" {...AXIS} />
            <YAxis {...AXIS} />
            <Tooltip content={({ payload }) => payload?.[0] ? <TooltipBox title={`intensity ${payload[0].payload.intensity}`} rows={[{ label: "pixels", value: fmt(payload[0].payload.count) }]} /> : null} />
            <Bar isAnimationActive={false} dataKey="count" fill={SERIES[0]} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartBox>
    </div>
  );
}

function SourcePicker({ value, onChange }: { value: ImageSource; onChange: (v: ImageSource) => void }) {
  const samples = useFetch(() => getJson<Sample[]>("/api/imaging/samples"));
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!value.file && !value.sample && samples.data?.length) onChange({ file: null, sample: samples.data[0].name });
  }, [samples.data, value.file, value.sample, onChange]);
  return (
    <Card title="Input image" subtitle="Real test-split X-rays from PneumoniaMNIST, or your own upload">
      <AsyncView state={samples} empty={null}>
        {(list) => list.length ? (
          <div className="grid grid-cols-4 gap-2">
            {list.map((s) => (
              <button key={s.name} onClick={() => onChange({ file: null, sample: s.name })}
                className={`overflow-hidden rounded-lg border-2 ${value.sample === s.name && !value.file ? "border-brand-600" : "border-transparent"}`}
                title={`${s.name} (${s.label})`} aria-label={`Use sample ${s.name}`}>
                <img src={s.url} alt={s.name} className="aspect-square w-full object-cover" />
                <span className={`block text-[10px] ${s.label === "PNEUMONIA" ? "text-red-600" : "text-emerald-700"}`}>{s.label}</span>
              </button>
            ))}
          </div>
        ) : <p className="text-sm text-slate-500">No sample images exported. Run scripts/export_sample_images.py or upload one.</p>}
      </AsyncView>
      <div className="mt-3">
        <FileDrop label="Or upload an image" onFile={(f) => {
          const e = validateImageFile(f);
          setErr(e);
          if (!e) onChange({ file: f, sample: null });
        }} />
        {err && <p role="alert" className="mt-2 text-sm text-red-600">{err}</p>}
        {value.file && <div className="mt-2"><SelectedFile file={value.file} onClear={() => onChange({ file: null, sample: samples.data?.[0]?.name ?? null })} /></div>}
      </div>
    </Card>
  );
}

function useImageOp(endpoint: string, source: ImageSource) {
  return useAction((params: Record<string, unknown> = {}, extra?: Record<string, File | string | null>) =>
    postImage<Result>(`/api/imaging/${endpoint}`, source, params, extra));
}

function OpView({ op, children, columns = 3 }: {
  op: ReturnType<typeof useImageOp>;
  children?: (r: Result) => React.ReactNode;
  columns?: 2 | 3 | 4;
}) {
  if (op.loading && !op.data) return <LoadingState label="Processing image..." />;
  if (op.error) return <ErrorState error={op.error} />;
  if (!op.data) return <EmptyState title="Not run yet" message="Choose parameters and press Run." />;
  return (
    <div className={`space-y-4 ${op.loading ? "opacity-60" : ""}`}>
      {children?.(op.data)}
      <ImageGrid panels={op.data.panels} columns={columns} />
    </div>
  );
}

function Controls({ children, onRun, loading }: { children?: React.ReactNode; onRun: () => void; loading: boolean }) {
  return (
    <div className="mb-4 flex flex-wrap items-end gap-4 rounded-xl border border-slate-200 bg-white p-4">
      {children}
      <RunButton loading={loading} onClick={onRun}>Run</RunButton>
    </div>
  );
}

function Num({ label, value, onChange, min, max, step = 1 }: { label: string; value: number; onChange: (n: number) => void; min: number; max: number; step?: number }) {
  return (
    <div className="w-40">
      <Field label={`${label}: ${value}`}>
        <input type="range" className="w-full accent-brand-600" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      </Field>
    </div>
  );
}

function EnhanceTab({ source }: { source: ImageSource }) {
  const op = useImageOp("enhance", source);
  const [clip, setClip] = useState(2);
  const [gamma, setGamma] = useState(0.8);
  return (
    <>
      <Controls loading={op.loading} onRun={() => op.run({ clip_limit: clip, gamma })}>
        <Num label="CLAHE clip limit" value={clip} onChange={setClip} min={0.5} max={8} step={0.5} />
        <Num label="Gamma" value={gamma} onChange={setGamma} min={0.3} max={2.5} step={0.1} />
      </Controls>
      <OpView op={op}>
        {(r) => (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <HistogramChart title="Histogram - original" data={r.panels[0].histogram ?? []} />
            <HistogramChart title="Histogram - CLAHE" data={r.panels[3].histogram ?? []} />
          </div>
        )}
      </OpView>
    </>
  );
}

function RestoreTab({ source }: { source: ImageSource }) {
  const op = useImageOp("restore", source);
  const [noise, setNoise] = useState("gaussian");
  const [level, setLevel] = useState(20);
  return (
    <>
      <Controls loading={op.loading} onRun={() => op.run({ noise, level })}>
        <div className="w-44"><Field label="Noise model"><Select value={noise} onChange={setNoise} options={[
          { value: "gaussian", label: "Gaussian" }, { value: "salt_pepper", label: "Salt & pepper" }, { value: "speckle", label: "Speckle" }]} /></Field></div>
        <Num label="Noise level" value={level} onChange={setLevel} min={2} max={50} />
      </Controls>
      <OpView op={op}>
        {(r) => (
          <Card title="Restoration quality" subtitle={String(r.note)}>
            <DataTable rows={r.metrics as Record<string, unknown>[]} />
            <p className="mt-2 text-sm text-slate-600">Best by PSNR: <strong>{String(r.best_by_psnr)}</strong></p>
          </Card>
        )}
      </OpView>
    </>
  );
}

function FiltersTab({ source }: { source: ImageSource }) {
  const op = useImageOp("filters", source);
  const [low, setLow] = useState(50);
  const [high, setHigh] = useState(150);
  return (
    <>
      <Controls loading={op.loading} onRun={() => op.run({ low, high })}>
        <Num label="Canny low" value={low} onChange={setLow} min={0} max={255} />
        <Num label="Canny high" value={high} onChange={setHigh} min={0} max={255} />
      </Controls>
      <OpView op={op} columns={4}>{(r) => <p className="text-sm text-slate-600">Canny edge pixels: {fmt(r.edge_pixels_pct, 2)}% of the image.</p>}</OpView>
    </>
  );
}

function ThresholdTab({ source }: { source: ImageSource }) {
  const op = useImageOp("threshold", source);
  const [value, setValue] = useState(128);
  const [block, setBlock] = useState(31);
  return (
    <>
      <Controls loading={op.loading} onRun={() => op.run({ value, block_size: block })}>
        <Num label="Global threshold" value={value} onChange={setValue} min={0} max={255} />
        <Num label="Adaptive block (odd)" value={block} onChange={(n) => setBlock(n % 2 ? n : n + 1)} min={3} max={101} step={2} />
      </Controls>
      <OpView op={op}>
        {(r) => <HistogramChart title={`Intensity histogram (Otsu picks T = ${fmt(r.otsu_threshold, 0)})`} data={r.histogram as { intensity: number; count: number }[]} />}
      </OpView>
    </>
  );
}

interface SegmentationQuality { plausible: boolean; checks: { check: string; passed: boolean; detail: string }[] }

function SegmentTab({ source }: { source: ImageSource }) {
  const op = useImageOp("segment", source);
  return (
    <>
      <Controls loading={op.loading} onRun={() => op.run({})} />
      <OpView op={op}>
        {(r) => {
          const q = r.quality as SegmentationQuality;
          const fields = r.field_area_pct as Record<string, number>;
          return (
            <div className="space-y-3">
              <Notice tone="warning">{String(r.warning)}</Notice>
              <KeyValue items={[
                { label: "Method", value: String(r.method) },
                { label: "Lung-field area", value: `${fmt(r.lung_area_pct, 1)}% of image` },
                { label: "Fields found", value: Object.entries(fields).map(([s, v]) => `image-${s} ${fmt(v, 1)}%`).join(", ") || "none" },
                { label: "Plausibility", value: <Badge tone={q.plausible ? "green" : "amber"}>{q.plausible ? "all checks passed" : "check the mask visually"}</Badge> },
              ]} />
              <ul className="space-y-1 text-sm">
                {q.checks.map((c) => (
                  <li key={c.check} className={c.passed ? "text-slate-600" : "text-amber-800"}>
                    {c.passed ? "✓" : "✗"} {c.check} <span className="text-slate-400">({c.detail})</span>
                  </li>
                ))}
              </ul>
              <ol className="list-decimal pl-5 text-sm text-slate-600">{(r.steps as string[]).map((s) => <li key={s}>{s}</li>)}</ol>
            </div>
          );
        }}
      </OpView>
    </>
  );
}

function CompressTab({ source }: { source: ImageSource }) {
  const op = useImageOp("compress", source);
  return (
    <>
      <Controls loading={op.loading} onRun={() => op.run({})} />
      <OpView op={op}>
        {(r) => {
          const rows = r.table as { format: string; bytes: number; ratio: number; psnr: number | null; ssim: number }[];
          return (
            <Card title="Size vs quality" subtitle={String(r.note)}>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <DataTable rows={rows} />
                <ChartBox height={220}>
                  <ResponsiveContainer>
                    <LineChart data={rows.filter((x) => x.psnr !== null)} margin={{ bottom: 18 }}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="ratio" type="number" {...AXIS} tickFormatter={(v) => `${v.toFixed(0)}x`} label={{ value: "compression ratio", position: "insideBottom", offset: -8, fontSize: 11 }} />
                      <YAxis dataKey="psnr" {...AXIS} unit=" dB" />
                      <Tooltip content={({ payload }) => payload?.[0] ? <TooltipBox title={payload[0].payload.format} rows={[
                        { label: "ratio", value: `${fmt(payload[0].payload.ratio, 1)}x` }, { label: "PSNR", value: `${fmt(payload[0].payload.psnr, 1)} dB` }]} /> : null} />
                      <Line isAnimationActive={false} dataKey="psnr" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartBox>
              </div>
            </Card>
          );
        }}
      </OpView>
    </>
  );
}

function FeaturesTab({ source }: { source: ImageSource }) {
  const sift = useImageOp("sift", source);
  const match = useImageOp("match", source);
  const cnn = useImageOp("cnn-features", source);
  const embedding = useAction(() => getJson<{ points: { x: number; y: number; label: string }[]; linear_probe: Record<string, number>; method: string; images: number }>("/api/imaging/feature-embedding"));
  const [angle, setAngle] = useState(15);
  const [scale, setScale] = useState(0.9);
  const [ratio, setRatio] = useState(0.75);
  const [layer, setLayer] = useState("layer2");
  return (
    <div className="space-y-6">
      <Card title="SIFT keypoints" subtitle="Scale-invariant feature transform: blob-like keypoints with 128-D gradient descriptors">
        <Controls loading={sift.loading} onRun={() => sift.run({})} />
        <OpView op={sift} columns={2}>
          {(r) => (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <KeyValue items={[{ label: "Keypoints", value: fmt(r.keypoints) }, { label: "Descriptor matrix", value: (r.descriptor_shape as number[]).join(" × ") }]} />
              <p className="text-xs text-slate-500 lg:col-span-2">{String(r.explanation)}</p>
            </div>
          )}
        </OpView>
      </Card>
      <Card title="Feature matching + RANSAC homography" subtitle="Image 2 is a synthetic warp of image 1, so the true transform is known and the estimate can be scored">
        <Controls loading={match.loading} onRun={() => match.run({ angle, scale, ratio })}>
          <Num label="Rotation (°)" value={angle} onChange={setAngle} min={-45} max={45} />
          <Num label="Scale" value={scale} onChange={setScale} min={0.6} max={1.4} step={0.05} />
          <Num label="Lowe ratio" value={ratio} onChange={setRatio} min={0.5} max={0.9} step={0.05} />
        </Controls>
        <OpView op={match} columns={2}>
          {(r) => (
            <KeyValue items={[
              { label: "Keypoints (image 1 / image 2)", value: `${fmt(r.keypoints_image_1)} / ${fmt(r.keypoints_image_2)}` },
              { label: "Matches after ratio test", value: fmt(r.good_matches_after_ratio_test) },
              { label: "RANSAC inliers", value: `${fmt(r.ransac_inliers)} (${fmt((r.inlier_ratio as number) * 100, 1)}%)` },
              { label: "Corner error - RANSAC", value: `${fmt(r.corner_error_ransac_px, 2)} px` },
              { label: "Corner error - least squares (no RANSAC)", value: `${fmt(r.corner_error_least_squares_px, 2)} px` },
            ]} />
          )}
        </OpView>
      </Card>
      <Card title="CNN feature maps" subtitle="ImageNet-pretrained ResNet-18; the 12 most active channels of the chosen layer">
        <Controls loading={cnn.loading} onRun={() => cnn.run({ layer })}>
          <div className="w-40"><Field label="Layer"><Select value={layer} onChange={setLayer} options={["layer1", "layer2", "layer3", "layer4"].map((l) => ({ value: l, label: l }))} /></Field></div>
        </Controls>
        {cnn.loading && !cnn.data && <LoadingState />}
        {cnn.error && <ErrorState error={cnn.error} />}
        {cnn.data && (() => {
          const d = cnn.data as unknown as { channels: { channel: number; image: string; mean_activation: number }[]; feature_map_shapes: Record<string, number[]>; embedding_dim: number; embedding_stats: Record<string, number>; explanation: string };
          return (
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
                {d.channels.map((c) => (
                  <figure key={c.channel}><img src={c.image} alt={`channel ${c.channel}`} className="w-full rounded" /><figcaption className="text-center text-[10px] text-slate-500">ch {c.channel} · {fmt(c.mean_activation, 2)}</figcaption></figure>
                ))}
              </div>
              <KeyValue items={[
                ...Object.entries(d.feature_map_shapes).map(([k, v]) => ({ label: `${k} output`, value: v.join(" × ") })),
                { label: "Pooled embedding", value: `${d.embedding_dim}-D, ${fmt(d.embedding_stats.sparsity_pct, 1)}% ~zero` },
              ]} />
              <p className="text-xs text-slate-500">{d.explanation}</p>
            </div>
          );
        })()}
      </Card>
      <Card title="Do generic CNN features separate the classes?" subtitle="Embeddings of balanced test images, projected with PCA; cached after the first run (~20 s)">
        <Controls loading={embedding.loading} onRun={() => embedding.run()} />
        <AsyncView state={embedding} empty={null} loadingLabel="Extracting features for 300 images...">
          {(e) => (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <Legend items={[{ label: "NORMAL", color: SERIES[2] }, { label: "PNEUMONIA", color: SERIES[1] }]} />
                <ChartBox height={300}>
                  <ResponsiveContainer>
                    <ScatterChart>
                      <CartesianGrid {...GRID} vertical />
                      <XAxis dataKey="x" type="number" {...AXIS} name="PC1" />
                      <YAxis dataKey="y" type="number" {...AXIS} name="PC2" />
                      <ZAxis range={[36, 36]} />
                      <Tooltip />
                      {["NORMAL", "PNEUMONIA"].map((lbl) => (
                        <Scatter isAnimationActive={false} key={lbl} name={lbl} data={e.points.filter((p) => p.label === lbl)} fill={lbl === "NORMAL" ? SERIES[2] : SERIES[1]} fillOpacity={0.7} stroke="#fff" strokeWidth={1} />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </ChartBox>
              </div>
              <div className="space-y-2 text-sm">
                <KeyValue items={[
                  { label: "Images", value: e.images },
                  { label: "Linear probe accuracy (5-fold)", value: `${fmt(e.linear_probe.cv_accuracy_mean * 100, 1)}% ± ${fmt(e.linear_probe.cv_accuracy_std * 100, 1)}` },
                  { label: "Linear probe ROC-AUC", value: fmt(e.linear_probe.cv_roc_auc_mean, 3) },
                ]} />
                <p className="text-xs text-slate-500">{e.method}</p>
              </div>
            </div>
          )}
        </AsyncView>
      </Card>
    </div>
  );
}

function VisualizeTab({ source }: { source: ImageSource }) {
  const op = useImageOp("visualize", source);
  const [cmap, setCmap] = useState("viridis");
  const [row, setRow] = useState(0.5);
  const cam = useAction(async () => {
    const form = new FormData();
    if (source.file) form.append("file", source.file);
    else if (source.sample) {
      const blob = await (await fetch(`/api/imaging/samples/${source.sample}`)).blob();
      form.append("file", new File([blob], source.sample, { type: "image/png" }));
    }
    return (await http.post<Prediction>("/api/screening/predict", form, { params: { context: "T6" } })).data;
  });
  return (
    <div className="space-y-6">
      <Controls loading={op.loading} onRun={() => op.run({ colormap: cmap, row })}>
        <div className="w-40"><Field label="Colour map"><Select value={cmap} onChange={setCmap} options={["bone", "viridis", "inferno", "hot", "jet"].map((c) => ({ value: c, label: c }))} /></Field></div>
        <Num label="Profile row (fraction)" value={row} onChange={setRow} min={0.05} max={0.95} step={0.05} />
      </Controls>
      <OpView op={op}>
        {(r) => {
          const cmp = r.comparison as { original: string; enhanced: string; label: string };
          return (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card title="Side-by-side comparison" subtitle="Drag the slider"><CompareSlider before={cmp.original} after={cmp.enhanced} afterLabel={cmp.label} /></Card>
              <Card title="Intensity profile" subtitle={`Grey level along row ${fmt(r.profile_row)} (red line)`}>
                <ChartBox height={200}>
                  <ResponsiveContainer>
                    <LineChart data={r.profile as { x: number; intensity: number }[]}>
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="x" {...AXIS} />
                      <YAxis domain={[0, 255]} {...AXIS} />
                      <Tooltip />
                      <Line isAnimationActive={false} dataKey="intensity" stroke={SERIES[0]} strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartBox>
                <HistogramChart title="Histogram" data={r.histogram as { intensity: number; count: number }[]} />
              </Card>
            </div>
          );
        }}
      </OpView>
      <Card title="Model attention (Grad-CAM)" subtitle="Uses the trained pneumonia classifier; the prediction is also saved to history">
        <RunButton loading={cam.loading} onClick={() => cam.run()}>Compute Grad-CAM</RunButton>
        <div className="mt-4">
          <AsyncView state={cam} empty={null}>
            {(p) => (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <img src={p.image} alt="input" className="rounded-lg bg-slate-900" />
                {p.gradcam && <img src={p.gradcam.heatmap} alt="Grad-CAM heat-map" className="rounded-lg" />}
                {p.gradcam && <img src={p.gradcam.overlay} alt="Grad-CAM overlay" className="rounded-lg" />}
                <p className="text-sm sm:col-span-3">Prediction <strong>{p.predicted_label}</strong>, P(pneumonia) = {fmt(p.probability_pneumonia * 100, 1)}%.</p>
              </div>
            )}
          </AsyncView>
        </div>
      </Card>
    </div>
  );
}

export default function ImageLab() {
  const [tab, setTab] = useTab(TABS.map((t) => t.id), "enhance" as TabId);
  const [source, setSource] = useState<ImageSource>({ file: null, sample: null });
  const key = source.file ? `file:${source.file.name}:${source.file.size}` : `sample:${source.sample}`;
  const body: Record<TabId, React.ReactNode> = {
    enhance: <EnhanceTab source={source} />, restore: <RestoreTab source={source} />, filters: <FiltersTab source={source} />,
    threshold: <ThresholdTab source={source} />, segment: <SegmentTab source={source} />, compress: <CompressTab source={source} />,
    features: <FeaturesTab source={source} />, visualize: <VisualizeTab source={source} />,
  };
  return (
    <>
      <PageHeader title="Biomedical Image Laboratory" tutorials={["T4", "T5", "T6"]}
        description="Classical image processing, feature extraction and visualisation on chest X-rays. All outputs are computed by OpenCV / PyTorch on the backend." />
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-4">
        <div className="xl:col-span-1"><SourcePicker value={source} onChange={setSource} /></div>
        <div className="xl:col-span-3">
          <Tabs tabs={TABS.map((t) => ({ ...t }))} value={tab} onChange={setTab} />
          {/* Re-mount the tab when the input image changes so stale results are cleared. */}
          <div key={`${tab}-${key}`}>{source.file || source.sample ? body[tab] : <EmptyState title="Choose an input image" />}</div>
        </div>
      </div>
    </>
  );
}
