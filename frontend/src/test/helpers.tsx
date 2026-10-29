import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import App from "../App";
import { http } from "../services/api";

type Handler = (url: string, config?: { params?: Record<string, unknown> }) => unknown;

/** Route mocked GET/POST calls by URL prefix. Unmatched URLs reject like a 404. */
export function mockApi(get: Record<string, unknown | Handler>, post: Record<string, unknown | Handler> = {}) {
  const resolve = (table: Record<string, unknown | Handler>, url: string, config?: { params?: Record<string, unknown> }) => {
    const key = Object.keys(table).sort((a, b) => b.length - a.length).find((k) => url.startsWith(k));
    if (key === undefined) return Promise.reject(Object.assign(new Error(`unmocked ${url}`), { isAxiosError: true, response: { status: 404, data: {} } }));
    const value = table[key];
    try {
      const data = typeof value === "function" ? (value as Handler)(url, config) : value;
      return data instanceof Promise ? data.then((d) => ({ data: d, headers: {} })) : Promise.resolve({ data, headers: {} });
    } catch (err) {
      return Promise.reject(err);
    }
  };
  vi.spyOn(http, "get").mockImplementation(((url: string, config?: { params?: Record<string, unknown> }) => resolve(get, url, config)) as never);
  vi.spyOn(http, "post").mockImplementation(((url: string, _body?: unknown, config?: { params?: Record<string, unknown> }) => resolve(post, url, config)) as never);
}

export function apiError(status: number, code: string, message: string) {
  return Object.assign(new Error(message), { isAxiosError: true, response: { status, data: { error: { code, message } } } });
}

export function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

export const STATUS = {
  database: true,
  datasets: { total: 3, available: 1, required: 2, required_available: 1, items: [
    { id: "synthea", name: "Synthea", domain: "ehr", available: true, synthetic: true, required: true },
    { id: "nhanes", name: "NHANES", domain: "lifestyle", available: false, synthetic: false, required: true },
    { id: "kaggle_chest_xray", name: "Kaggle X-rays", domain: "imaging", available: false, synthetic: false, required: false },
  ] },
  models: [{ id: "pneumonia_resnet18", name: "Pneumonia classifier", status: "trained", metrics: { accuracy: 0.93, recall: 0.99, specificity: 0.82, roc_auc: 0.994 }, updated_at: null }],
  tutorials: { total: 15, verified: 14, data_ready: 14, explored: 1, items: [
    { id: "T1", title: "Introduction to EHR", module: "EHR Data Explorer", route: "/ehr?tab=explore", data_ready: true, runs: 2, status: "verified" },
  ] },
  history: { total_runs: 2, predictions: 0, recent: [
    { id: 1, module: "ehr", tutorial: "T1", title: "Explored synthea:patients", dataset: "synthea:patients", summary: {}, created_at: "2026-10-26T10:00:00Z" },
  ] },
};
