import { ExternalLink } from "lucide-react";
import { AsyncView, Badge, Card, PageHeader } from "../components/ui";
import { useFetch } from "../hooks/useApi";
import { getJson } from "../services/api";
import type { DatasetInfo, ModelInfo } from "../types/api";
import { dateTime, fmt } from "../utils/format";

export default function Registry() {
  const datasets = useFetch(() => getJson<DatasetInfo[]>("/api/datasets"));
  const models = useFetch(() => getJson<ModelInfo[]>("/api/models"));
  return (
    <>
      <PageHeader title="Datasets & Models" description="Sources, access requirements and availability (checked on disk), plus the status of every model." />
      <h2 className="mb-3 text-lg font-semibold text-slate-900">Datasets</h2>
      <AsyncView state={datasets} onRetry={datasets.run}>
        {(list) => (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {list.map((d) => (
              <Card key={d.id} title={d.name} subtitle={d.domain.replace("_", " ")} actions={
                <div className="flex gap-1">
                  {d.synthetic && <Badge tone="amber">synthetic</Badge>}
                  <Badge tone={d.available ? "green" : d.required ? "red" : "slate"}>
                    {d.available ? "available" : d.required ? "not downloaded" : "optional · not installed"}
                  </Badge>
                </div>}>
                <p className="text-sm text-slate-600">{d.description}</p>
                {d.note && <p className="mt-2 text-xs text-slate-500">{d.note}</p>}
                <dl className="mt-3 grid grid-cols-[110px_1fr] gap-y-1 text-xs">
                  <dt className="text-slate-500">Used by</dt><dd>{d.used_by.join(", ")}</dd>
                  <dt className="text-slate-500">Access</dt><dd>{d.access}</dd>
                  <dt className="text-slate-500">Licence</dt><dd>{d.license}</dd>
                  {d.size_mb !== null && <><dt className="text-slate-500">Size</dt><dd>{fmt(d.size_mb)} MB</dd></>}
                  {d.downloaded_at && <><dt className="text-slate-500">Downloaded</dt><dd>{dateTime(d.downloaded_at)}</dd></>}
                  {d.path && <><dt className="text-slate-500">Location</dt><dd><code>data/{d.path}</code></dd></>}
                  {!d.available && d.download && <><dt className="text-slate-500">How to get it</dt><dd><code className="break-all">{d.download}</code></dd></>}
                </dl>
                {d.source_url.startsWith("http") && (
                  <a href={d.source_url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1 text-xs text-brand-700 hover:underline">
                    Source <ExternalLink className="h-3 w-3" />
                  </a>
                )}
              </Card>
            ))}
          </div>
        )}
      </AsyncView>
      <h2 className="mb-3 mt-8 text-lg font-semibold text-slate-900">Models</h2>
      <AsyncView state={models} onRetry={models.run}>
        {(list) => (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {list.map((m) => (
              <Card key={m.id} title={m.name} subtitle={m.task} actions={
                <Badge tone={m.status === "trained" || m.status === "pretrained" ? "green" : m.status === "training" ? "amber" : "slate"}>{m.status.replace("_", " ")}</Badge>}>
                <dl className="grid grid-cols-[110px_1fr] gap-y-1 text-xs">
                  <dt className="text-slate-500">Tutorials</dt><dd>{m.tutorials.join(", ")}</dd>
                  <dt className="text-slate-500">Artefact</dt><dd><code>{m.path ?? "—"}</code></dd>
                  <dt className="text-slate-500">Updated</dt><dd>{dateTime(m.updated_at)}</dd>
                  <dt className="text-slate-500">How to train</dt><dd><code className="break-all">{m.how_to_train}</code></dd>
                </dl>
                {m.metrics && (
                  <pre className="mt-3 max-h-40 overflow-auto rounded bg-slate-50 p-2 text-[11px] text-slate-700">{JSON.stringify(m.metrics, (_k, v) => (typeof v === "number" ? Number(v.toFixed(4)) : v), 2)}</pre>
                )}
              </Card>
            ))}
          </div>
        )}
      </AsyncView>
    </>
  );
}
