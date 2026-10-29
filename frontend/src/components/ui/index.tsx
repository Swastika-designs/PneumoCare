import { AlertTriangle, Database, Inbox, Info, Loader2, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";
import type { ApiErrorInfo } from "../../services/api";

export function PageHeader({ title, description, tutorials, actions }: {
  title: string;
  description?: string;
  tutorials?: string[];
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-2xl font-semibold text-slate-900">{title}</h1>
          {tutorials?.map((t) => <Badge key={t} tone="brand">{t}</Badge>)}
        </div>
        {description && <p className="mt-1 max-w-3xl text-sm text-slate-500">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function Card({ title, subtitle, actions, children, className = "" }: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card p-5 ${className}`}>
      {(title || actions) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-2">
          <div>
            {title && <h2 className="text-base font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

const tones = {
  brand: "bg-brand-50 text-brand-700 ring-brand-200",
  slate: "bg-slate-100 text-slate-700 ring-slate-200",
  green: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  amber: "bg-amber-50 text-amber-800 ring-amber-200",
  red: "bg-red-50 text-red-700 ring-red-200",
  blue: "bg-blue-50 text-blue-700 ring-blue-200",
};

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: keyof typeof tones }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function StatCard({ label, value, hint, icon }: { label: string; value: ReactNode; hint?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="card flex items-start gap-3 p-4">
      {icon && <div className="rounded-lg bg-brand-50 p-2 text-brand-700">{icon}</div>}
      <div className="min-w-0">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
        <p className="mt-1 truncate text-2xl font-semibold text-slate-900">{value}</p>
        {hint && <p className="mt-0.5 text-xs text-slate-500">{hint}</p>}
      </div>
    </div>
  );
}

export function LoadingState({ label = "Loading..." }: { label?: string }) {
  return (
    <div role="status" className="flex items-center justify-center gap-2 py-12 text-sm text-slate-500">
      <Loader2 className="h-5 w-5 animate-spin text-brand-600" aria-hidden />
      {label}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: ApiErrorInfo; onRetry?: () => void }) {
  const isData = error.code === "dataset_unavailable";
  const isModel = error.code === "model_unavailable";
  return (
    <div role="alert" className="flex flex-col gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 sm:flex-row sm:items-start">
      {isData ? <Database className="h-5 w-5 shrink-0" aria-hidden /> : <AlertTriangle className="h-5 w-5 shrink-0" aria-hidden />}
      <div className="flex-1">
        <p className="font-semibold">
          {isData ? "Dataset not available" : isModel ? "Model not available" : error.status === null ? "Connection problem" : "Something went wrong"}
        </p>
        <p className="mt-1 whitespace-pre-line">{error.message}</p>
      </div>
      {onRetry && (
        <button className="btn-secondary" onClick={onRetry}>
          <RefreshCw className="h-4 w-4" aria-hidden /> Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, message, action }: { title: string; message?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white px-6 py-12 text-center">
      <Inbox className="h-8 w-8 text-slate-400" aria-hidden />
      <p className="mt-3 font-medium text-slate-700">{title}</p>
      {message && <p className="mt-1 max-w-md text-sm text-slate-500">{message}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warning" }) {
  const cls = tone === "warning" ? "border-amber-200 bg-amber-50 text-amber-900" : "border-blue-200 bg-blue-50 text-blue-900";
  return (
    <div className={`flex gap-2 rounded-lg border p-3 text-sm ${cls}`}>
      {tone === "warning" ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden /> : <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />}
      <div>{children}</div>
    </div>
  );
}

/** Render the right state for an async result: loading, error, empty or content. */
export function AsyncView<T>({ state, children, empty, loadingLabel, onRetry }: {
  state: { data: T | null; error: ApiErrorInfo | null; loading: boolean };
  children: (data: T) => ReactNode;
  empty?: ReactNode;
  loadingLabel?: string;
  onRetry?: () => void;
}) {
  if (state.loading && !state.data) return <LoadingState label={loadingLabel} />;
  if (state.error) return <ErrorState error={state.error} onRetry={onRetry} />;
  if (!state.data) return <>{empty ?? null}</>;
  return (
    <div className={state.loading ? "pointer-events-none opacity-60 transition" : "transition"}>
      {children(state.data)}
    </div>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: {
  tabs: { id: T; label: string; badge?: string }[];
  value: T;
  onChange: (id: T) => void;
}) {
  return (
    <div role="tablist" className="mb-5 flex flex-wrap gap-1 rounded-xl border border-slate-200 bg-white p-1">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={value === t.id}
          onClick={() => onChange(t.id)}
          className={`flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition ${
            value === t.id ? "bg-brand-600 text-white shadow-sm" : "text-slate-600 hover:bg-slate-100"
          }`}
        >
          {t.label}
          {t.badge && (
            <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${value === t.id ? "bg-white/20" : "bg-slate-100 text-slate-500"}`}>
              {t.badge}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-slate-400">{hint}</span>}
    </label>
  );
}

export function Select<T extends string>({ value, onChange, options, ariaLabel }: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
  ariaLabel?: string;
}) {
  return (
    <select className="input" value={value} aria-label={ariaLabel} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

export function RunButton({ loading, onClick, children = "Run analysis", disabled }: {
  loading: boolean;
  onClick: () => void;
  children?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <button className="btn-primary" onClick={onClick} disabled={loading || disabled}>
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

export function KeyValue({ items }: { items: { label: string; value: ReactNode }[] }) {
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
      {items.map((i) => (
        <div key={i.label} className="flex justify-between gap-4 border-b border-slate-100 py-1">
          <dt className="text-slate-500">{i.label}</dt>
          <dd className="text-right font-medium text-slate-800">{i.value}</dd>
        </div>
      ))}
    </dl>
  );
}
