import { ArrowRight, Search } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { AsyncView, Badge, Card, EmptyState, KeyValue, PageHeader } from "../components/ui";
import { useFetch } from "../hooks/useApi";
import { getJson } from "../services/api";
import type { Tutorial } from "../types/api";

function StatusBadge({ t }: { t: Tutorial }) {
  const tone = t.status === "verified" ? "green" : t.status === "verification failed" ? "red" : "slate";
  const checks = t.verification ? ` ${t.verification.checks_passed}/${t.verification.checks_total}` : "";
  return <Badge tone={tone}>{t.status}{checks}</Badge>;
}

export function TutorialLibrary() {
  const tutorials = useFetch(() => getJson<Tutorial[]>("/api/tutorials"));
  const [q, setQ] = useState("");
  return (
    <>
      <PageHeader title="Tutorial Library" description="All 15 Healthcare Analytics tutorials, the module that implements each one, and whether its data is ready." />
      <div className="relative mb-5 max-w-md">
        <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" aria-hidden />
        <input className="input pl-9" placeholder="Search tutorials, concepts..." value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search tutorials" />
      </div>
      <AsyncView state={tutorials} onRetry={tutorials.run}>
        {(list) => {
          const needle = q.toLowerCase();
          const shown = list.filter((t) => !needle || [t.id, t.title, t.module, ...t.concepts].join(" ").toLowerCase().includes(needle));
          if (!shown.length) return <EmptyState title="No tutorials match" message={`Nothing matches "${q}".`} />;
          return (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
              {shown.map((t) => (
                <article key={t.id} className="card flex flex-col p-5">
                  <div className="flex items-center justify-between">
                    <Badge tone="brand">{t.id}</Badge>
                    <div className="flex gap-1">
                      <StatusBadge t={t} />
                      <Badge tone={t.data_ready ? "blue" : "red"}>{t.data_ready ? "data ready" : "data missing"}</Badge>
                    </div>
                  </div>
                  <h2 className="mt-3 font-semibold text-slate-900">{t.title}</h2>
                  <p className="mt-1 text-xs text-slate-500">{t.module}</p>
                  <p className="mt-2 flex-1 text-sm text-slate-600">{t.objective}</p>
                  <div className="mt-3 flex flex-wrap gap-1">{t.concepts.slice(0, 4).map((c) => <Badge key={c}>{c}</Badge>)}</div>
                  <div className="mt-4 flex items-center justify-between">
                    <Link to={`/tutorials/${t.id}`} className="text-sm text-slate-600 hover:underline">Details</Link>
                    <Link to={t.route} className="btn-primary py-1.5">Open workspace <ArrowRight className="h-4 w-4" /></Link>
                  </div>
                </article>
              ))}
            </div>
          );
        }}
      </AsyncView>
    </>
  );
}

export function TutorialDetail() {
  const { id = "" } = useParams();
  const tutorial = useFetch(() => getJson<Tutorial>(`/api/tutorials/${id}`), [id]);
  return (
    <AsyncView state={tutorial} onRetry={tutorial.run}>
      {(t) => (
        <>
          <PageHeader title={`${t.id} - ${t.title}`} description={t.objective}
            actions={<Link to={t.route} className="btn-primary">Open workspace <ArrowRight className="h-4 w-4" /></Link>} />
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Implementation">
              <KeyValue items={[
                { label: "Module", value: t.module },
                { label: "Status", value: <StatusBadge t={t} /> },
                { label: "Last verified", value: t.verification ? new Date(t.verification.verified_at).toLocaleString() : "never" },
                { label: "Data ready", value: t.data_ready ? "yes" : "no - see Datasets & Models" },
                { label: "Times run", value: t.runs },
                { label: "Datasets", value: t.datasets.join(", ") },
                { label: "Documentation", value: <code className="text-xs">{t.doc}</code> },
              ]} />
            </Card>
            <Card title="Concepts covered">
              <div className="flex flex-wrap gap-1.5">{t.concepts.map((c) => <Badge key={c} tone="brand">{c}</Badge>)}</div>
            </Card>
            <Card title="Inputs"><p className="text-sm text-slate-600">{t.inputs}</p></Card>
            <Card title="Outputs"><p className="text-sm text-slate-600">{t.outputs}</p></Card>
            <Card title="Verification checks" subtitle="From scripts/verify_tutorials.py, run against the live API" className="lg:col-span-2">
              {t.verification ? (
                <ul className="space-y-1 text-sm">
                  {t.verification.checks.map((c) => (
                    <li key={c.check} className="flex gap-2">
                      <span className={c.passed ? "text-emerald-600" : "text-red-600"}>{c.passed ? "✓" : "✗"}</span>
                      <span>{c.check}{c.detail && <span className="text-slate-500"> · {c.detail}</span>}</span>
                    </li>
                  ))}
                  {t.verification.error && <li className="text-red-600">{t.verification.error}</li>}
                </ul>
              ) : <p className="text-sm text-slate-500">Not verified yet. Run python scripts/verify_tutorials.py with the backend running.</p>}
            </Card>
            <Card title="API endpoints" className="lg:col-span-2">
              <ul className="space-y-1">{t.endpoints.map((e) => <li key={e}><code className="rounded bg-slate-100 px-2 py-0.5 text-xs">{e}</code></li>)}</ul>
            </Card>
          </div>
          <Link to="/tutorials" className="mt-6 inline-block text-sm text-brand-700 hover:underline">← All tutorials</Link>
        </>
      )}
    </AsyncView>
  );
}
