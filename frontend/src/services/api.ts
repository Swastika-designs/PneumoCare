import axios, { AxiosError } from "axios";

/** Single axios instance. In development Vite proxies /api to FastAPI (see vite.config.ts). */
export const http = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "",
  timeout: 180_000,
});

export interface ApiErrorInfo {
  status: number | null;
  code: string;
  message: string;
}

/** Turn any thrown value into a readable message, using the backend's {"error": {...}} envelope. */
export function describeError(error: unknown): ApiErrorInfo {
  if (axios.isAxiosError(error)) {
    const err = error as AxiosError<{ error?: { code?: string; message?: string }; detail?: unknown }>;
    if (!err.response) {
      return {
        status: null,
        code: "network",
        message: "Cannot reach the PneumoCare API. Is the backend running on port 8000?",
      };
    }
    const body = err.response.data;
    if (body?.error?.message) {
      return { status: err.response.status, code: body.error.code ?? "error", message: body.error.message };
    }
    if ([500, 502, 503, 504].includes(err.response.status)) {
      // The Vite dev proxy answers 500/502 without our error envelope when the backend is down.
      return {
        status: err.response.status,
        code: "http",
        message: `The API did not respond correctly (HTTP ${err.response.status}). Is the backend running on port 8000?`,
      };
    }
    return { status: err.response.status, code: "http", message: `Request failed with status ${err.response.status}` };
  }
  return { status: null, code: "unknown", message: error instanceof Error ? error.message : String(error) };
}

export async function getJson<T>(url: string, params?: Record<string, unknown>): Promise<T> {
  const res = await http.get<T>(url, { params, paramsSerializer: { indexes: null } });
  return res.data;
}

export async function postJson<T>(url: string, body?: unknown, params?: Record<string, unknown>): Promise<T> {
  const res = await http.post<T>(url, body, { params });
  return res.data;
}

/** POST multipart form data: an uploaded file or the name of a bundled sample image. */
export async function postImage<T>(
  url: string,
  image: { file?: File | null; sample?: string | null },
  params?: Record<string, unknown>,
  extra?: Record<string, File | string | null | undefined>,
): Promise<T> {
  const form = new FormData();
  if (image.file) form.append("file", image.file);
  else if (image.sample) form.append("sample", image.sample);
  Object.entries(extra ?? {}).forEach(([key, value]) => {
    if (value) form.append(key, value);
  });
  const res = await http.post<T>(url, form, { params });
  return res.data;
}

export function reportUrl(id: string, format: "json" | "csv" | "pdf", params?: Record<string, string>): string {
  const query = new URLSearchParams({ format, ...(params ?? {}) });
  return `${http.defaults.baseURL ?? ""}/api/reports/${id}?${query.toString()}`;
}
