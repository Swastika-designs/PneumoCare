import {
  Activity, BarChart3, BookOpen, Brain, Database, FileDown, FileText, FlaskConical, LayoutDashboard, Menu, ScanLine,
  Stethoscope, TableProperties, X,
} from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useFetch } from "../hooks/useApi";
import { getJson } from "../services/api";

export const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/pneumonia", label: "Pneumonia Detection", icon: Stethoscope },
  { to: "/model-performance", label: "Model Performance", icon: BarChart3 },
  { to: "/image-lab", label: "Image Laboratory", icon: ScanLine, tutorials: "T4-T6" },
  { to: "/ehr", label: "EHR Data Explorer", icon: TableProperties, tutorials: "T1-T3" },
  { to: "/operations", label: "Operations Analytics", icon: Activity, tutorials: "T7-T9" },
  { to: "/clinical-text", label: "Clinical Text", icon: FileText, tutorials: "T10-T12" },
  { to: "/advanced", label: "Advanced Analytics", icon: Brain, tutorials: "T13-T15" },
  { to: "/tutorials", label: "Tutorial Library", icon: BookOpen },
  { to: "/registry", label: "Datasets & Models", icon: Database },
  { to: "/reports", label: "Reports", icon: FileDown },
];

function ApiStatus() {
  const health = useFetch(() => getJson<{ status: string; version: string }>("/api/health"));
  const ok = health.data?.status === "ok";
  return (
    <div className="flex items-center gap-2 text-xs text-slate-400" aria-live="polite">
      <span className={`h-2 w-2 rounded-full ${health.loading ? "bg-amber-400" : ok ? "bg-emerald-400" : "bg-red-500"}`} />
      {health.loading ? "Connecting to API..." : ok ? `API online · v${health.data?.version}` : "API offline"}
    </div>
  );
}

export function AppLayout() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);

  const sidebar = (
    <nav aria-label="Main navigation" className="flex h-full flex-col bg-slate-900 text-slate-300">
      <div className="flex items-center gap-2 px-5 py-5">
        <div className="rounded-lg bg-brand-500 p-1.5 text-white"><FlaskConical className="h-5 w-5" aria-hidden /></div>
        <div>
          <p className="text-base font-semibold text-white">PneumoCare</p>
          <p className="text-[11px] text-slate-400">Healthcare Analytics</p>
        </div>
      </div>
      <ul className="flex-1 space-y-0.5 overflow-y-auto px-3">
        {NAV.map((item) => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                  isActive ? "bg-brand-600 text-white" : "hover:bg-slate-800 hover:text-white"
                }`
              }
            >
              <item.icon className="h-4 w-4 shrink-0" aria-hidden />
              <span className="flex-1">{item.label}</span>
              {item.tutorials && <span className="text-[10px] text-slate-400">{item.tutorials}</span>}
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="border-t border-slate-800 px-5 py-4">
        <ApiStatus />
        <p className="mt-2 text-[11px] leading-snug text-slate-500">
          Educational project. Not a medical device.
        </p>
      </div>
    </nav>
  );

  return (
    <div className="min-h-screen lg:pl-64">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 lg:block">{sidebar}</aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-slate-900/50" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-64">{sidebar}</aside>
        </div>
      )}
      <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-slate-200 bg-white/90 px-4 py-3 backdrop-blur lg:hidden">
        <button className="btn-secondary px-2 py-1.5" onClick={() => setOpen((v) => !v)} aria-label="Toggle navigation">
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
        <span className="font-semibold text-slate-900">PneumoCare</span>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <Outlet />
      </main>
    </div>
  );
}
