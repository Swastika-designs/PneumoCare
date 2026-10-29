import { ImageUp, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ImagePanel } from "../types/api";
import { fmt } from "../utils/format";

const ACCEPT = "image/png,image/jpeg,image/bmp,image/tiff";
const MAX_MB = 10;

/** Client-side checks mirror the backend's validation so users get instant feedback. */
export function validateImageFile(file: File): string | null {
  if (!file.type.startsWith("image/") && !/\.(png|jpe?g|bmp|tiff?)$/i.test(file.name)) {
    return `"${file.name}" is not an image. Use PNG, JPEG, BMP or TIFF.`;
  }
  if (file.size > MAX_MB * 1024 * 1024) return `"${file.name}" is larger than ${MAX_MB} MB.`;
  if (file.size === 0) return `"${file.name}" is empty.`;
  return null;
}

export function FileDrop({ onFile, label = "Drop an X-ray here or click to browse", accept = ACCEPT,
  hint = `PNG, JPEG, BMP or TIFF · max ${MAX_MB} MB` }: {
  onFile: (file: File) => void;
  label?: string;
  accept?: string;
  hint?: string;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onFile(f);
      }}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-8 text-center transition ${
        drag ? "border-brand-500 bg-brand-50" : "border-slate-300 bg-slate-50 hover:border-brand-400"
      }`}
    >
      <ImageUp className="h-8 w-8 text-brand-600" aria-hidden />
      <p className="mt-2 text-sm font-medium text-slate-700">{label}</p>
      <p className="text-xs text-slate-500">{hint}</p>
      <input
        ref={input}
        data-testid="file-input"
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}

export function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return url;
}

export function ImageGrid({ panels, columns = 3 }: { panels: ImagePanel[]; columns?: 2 | 3 | 4 }) {
  const cols = { 2: "sm:grid-cols-2", 3: "sm:grid-cols-2 lg:grid-cols-3", 4: "sm:grid-cols-2 lg:grid-cols-4" }[columns];
  return (
    <div className={`grid grid-cols-1 gap-4 ${cols}`}>
      {panels.map((p) => (
        <figure key={p.title} className="overflow-hidden rounded-xl border border-slate-200 bg-white">
          <img src={p.image} alt={p.title} className="aspect-square w-full bg-slate-900 object-contain" />
          <figcaption className="space-y-1 p-3">
            <p className="text-sm font-medium text-slate-800">{p.title}</p>
            {p.description && <p className="text-xs text-slate-500">{p.description}</p>}
            <div className="flex flex-wrap gap-x-3 text-[11px] text-slate-500">
              {p.psnr !== undefined && <span>PSNR {fmt(p.psnr, 2)} dB</span>}
              {p.ssim !== undefined && <span>SSIM {fmt(p.ssim, 3)}</span>}
              {p.stats && <span>mean {fmt(p.stats.mean, 1)} · sd {fmt(p.stats.std, 1)} · H {fmt(p.stats.entropy_bits, 2)} bits</span>}
              {p.foreground_pct !== undefined && <span>foreground {fmt(p.foreground_pct, 1)}%</span>}
              {p.bytes !== undefined && <span>{fmt(p.bytes)} bytes</span>}
            </div>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

/** Drag-to-compare slider for two images of the same size. */
export function CompareSlider({ before, after, beforeLabel = "Original", afterLabel = "Processed" }: {
  before: string;
  after: string;
  beforeLabel?: string;
  afterLabel?: string;
}) {
  const [pos, setPos] = useState(50);
  return (
    <div>
      <div className="relative mx-auto aspect-square max-w-md overflow-hidden rounded-xl bg-slate-900 select-none">
        <img src={after} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" />
        <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
          <img src={before} alt={beforeLabel} className="absolute inset-0 h-full w-full object-contain" />
        </div>
        <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white" style={{ left: `${pos}%` }} />
        <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs text-white">{beforeLabel}</span>
        <span className="absolute right-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs text-white">{afterLabel}</span>
      </div>
      <input
        aria-label="Comparison position"
        type="range"
        min={0}
        max={100}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        className="mx-auto mt-2 block w-full max-w-md accent-brand-600"
      />
    </div>
  );
}

export function SelectedFile({ file, onClear }: { file: File; onClear: () => void }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm">
      <span className="truncate">{file.name} <span className="text-slate-400">({(file.size / 1024).toFixed(0)} KB)</span></span>
      <button onClick={onClear} aria-label="Remove file" className="text-slate-400 hover:text-red-600"><X className="h-4 w-4" /></button>
    </div>
  );
}
