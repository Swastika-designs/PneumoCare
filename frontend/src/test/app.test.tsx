import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { describeError } from "../services/api";
import { fmt, pct } from "../utils/format";
import { STATUS, apiError, mockApi, renderAt } from "./helpers";

afterEach(() => vi.restoreAllMocks());

const HEALTH = { "/api/health": { status: "ok", version: "1.0.0" } };

describe("navigation and dashboard", () => {
  it("shows loading, then real status values from the API", async () => {
    mockApi({ ...HEALTH, "/api/system/status": STATUS });
    renderAt("/");
    expect(screen.getByRole("status")).toHaveTextContent(/loading system status/i);
    expect(await screen.findByText("14 / 15")).toBeInTheDocument();
    expect(screen.getByText("optional")).toBeInTheDocument();
    expect(screen.getByText("Explored synthea:patients")).toBeInTheDocument();
    expect(screen.getByText(/Test ROC-AUC 0.994/)).toBeInTheDocument();
  });

  it("shows a connection error with retry when the backend is down", async () => {
    mockApi({ "/api/health": () => { throw Object.assign(new Error("Network Error"), { isAxiosError: true }); },
      "/api/system/status": () => { throw Object.assign(new Error("Network Error"), { isAxiosError: true }); } });
    renderAt("/");
    expect(await screen.findByText(/Cannot reach the PneumoCare API/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("navigates to the tutorial library via the sidebar", async () => {
    mockApi({ ...HEALTH, "/api/system/status": STATUS, "/api/tutorials": [
      { id: "T12", title: "Clinical Text Ranking and Word Disambiguation", module: "Clinical Text Analytics", route: "/clinical-text?tab=ranking",
        datasets: [], objective: "Rank documents", concepts: ["BM25", "Lesk algorithm"], inputs: "", outputs: "", endpoints: [], status: "verified",
        verification: { passed: true, verified_at: "2026-10-26T10:00:00Z", error: null, checks_passed: 4, checks_total: 4, checks: [] },
        data_ready: true, runs: 0, doc: "docs/tutorials/T12.md" },
      { id: "T13", title: "Dimensionality Reduction for EDA", module: "Advanced Analytics", route: "/advanced?tab=pca",
        datasets: [], objective: "PCA", concepts: ["explained variance"], inputs: "", outputs: "", endpoints: [], status: "not verified", verification: null,
        data_ready: false, runs: 0, doc: "docs/tutorials/T13.md" },
    ] });
    renderAt("/");
    const nav = screen.getAllByRole("navigation")[0];
    await userEvent.click(within(nav).getByRole("link", { name: /tutorial library/i }));
    expect(await screen.findByText("Clinical Text Ranking and Word Disambiguation")).toBeInTheDocument();
    expect(screen.getByText("data missing")).toBeInTheDocument();
    expect(screen.getByText("verified 4/4")).toBeInTheDocument();
    expect(screen.getByText("not verified")).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText(/search tutorials/i), "lesk");
    expect(screen.queryByText("Dimensionality Reduction for EDA")).not.toBeInTheDocument();
    expect(screen.getByText("Clinical Text Ranking and Word Disambiguation")).toBeInTheDocument();
  });

  it("renders a not-found page for unknown routes", async () => {
    mockApi(HEALTH);
    renderAt("/does-not-exist");
    expect(screen.getByText("Page not found")).toBeInTheDocument();
  });
});

describe("pneumonia detection", () => {
  it("explains how to train when no model exists", async () => {
    mockApi({ ...HEALTH, "/api/screening/status": { model_available: false, message: "No trained pneumonia model was found." }, "/api/screening/predictions": [] });
    renderAt("/pneumonia");
    expect(await screen.findByText("No trained model available")).toBeInTheDocument();
    expect(screen.getByText(/scripts\/train_pneumonia.py/)).toBeInTheDocument();
  });

  it("rejects non-image uploads before calling the API", async () => {
    mockApi({ ...HEALTH, "/api/screening/status": { model_available: true, model_name: "resnet18", threshold: 0.5 }, "/api/screening/predictions": [] });
    renderAt("/pneumonia");
    await screen.findByText(/1. Choose an X-ray/);
    const input = screen.getByTestId("file-input") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(["hello"], "notes.txt", { type: "text/plain" })] } });
    expect(await screen.findByRole("alert")).toHaveTextContent(/not an image/);
  });

  it("uploads an image, shows loading and then the prediction with Grad-CAM", async () => {
    let resolvePredict: (v: unknown) => void = () => {};
    mockApi(
      { ...HEALTH, "/api/screening/status": { model_available: true, model_name: "resnet18", threshold: 0.4 }, "/api/screening/predictions": [] },
      { "/api/screening/predict": () => new Promise((r) => { resolvePredict = r; }) },
    );
    globalThis.URL.createObjectURL = vi.fn(() => "blob:preview");
    globalThis.URL.revokeObjectURL = vi.fn();
    renderAt("/pneumonia");
    await screen.findByText(/1. Choose an X-ray/);
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [new File([new Uint8Array([137, 80])], "chest.png", { type: "image/png" })] } });
    await userEvent.click(await screen.findByRole("button", { name: /run prediction/i }));
    expect(await screen.findByText(/Running the model/)).toBeInTheDocument();
    resolvePredict({
      id: 1, filename: "chest.png", predicted_label: "PNEUMONIA", probability_pneumonia: 0.91, probabilities: { NORMAL: 0.09, PNEUMONIA: 0.91 },
      threshold: 0.4, model_name: "resnet18", image: "data:image/png;base64,AA", warnings: [], disclaimer: "Not a medical device.",
      gradcam: { overlay: "data:image/png;base64,AA", heatmap: "data:image/png;base64,AA", target_class: "PNEUMONIA", method: "Grad-CAM" },
    });
    const result = await screen.findByTestId("prediction-result");
    expect(within(result).getByText("PNEUMONIA", { selector: "span" })).toBeInTheDocument();
    expect(within(result).getByText("91.0%")).toBeInTheDocument();
    expect(within(result).getByAltText("Grad-CAM heat-map")).toBeInTheDocument();
  });

  it("shows the backend's error message when prediction fails", async () => {
    mockApi(
      { ...HEALTH, "/api/screening/status": { model_available: true, model_name: "resnet18", threshold: 0.4 }, "/api/screening/predictions": [] },
      { "/api/screening/predict": () => { throw apiError(422, "validation_failed", "Image is too small (10x10)"); } },
    );
    globalThis.URL.createObjectURL = vi.fn(() => "blob:preview");
    globalThis.URL.revokeObjectURL = vi.fn();
    renderAt("/pneumonia");
    await screen.findByText(/1. Choose an X-ray/);
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [new File([new Uint8Array([1])], "tiny.png", { type: "image/png" })] } });
    await userEvent.click(await screen.findByRole("button", { name: /run prediction/i }));
    expect(await screen.findByText("Image is too small (10x10)")).toBeInTheDocument();
  });
});

describe("other pages", () => {
  it("model performance shows metrics from the evaluation file", async () => {
    const m = { threshold: 0.03, accuracy: 0.9295, precision: 0.9, recall: 0.997, sensitivity: 0.997, specificity: 0.816, f1: 0.946, roc_auc: 0.9939,
      confusion_matrix: { tn: 191, fp: 43, fn: 1, tp: 389 }, support: { normal: 234, pneumonia: 390 } };
    mockApi({ ...HEALTH, "/api/models/pneumonia/evaluation": { checkpoint: "x.pt", data_source: "pneumoniamnist", split: "test", samples: 624, test_loss: 0.2,
      metrics: m, metrics_at_0_5: { ...m, threshold: 0.5, accuracy: 0.968 }, curves: { roc: [{ fpr: 0, tpr: 0, threshold: 1 }, { fpr: 1, tpr: 1, threshold: 0 }], pr: [] },
      probability_histogram: [], evaluated_at: "2026-10-26T10:00:00Z", model_meta: {} } });
    renderAt("/model-performance");
    expect(await screen.findByText("93.0%")).toBeInTheDocument();
    expect(screen.getByText("Area under curve = 0.9939")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /threshold 0.5/i }));
    expect(screen.getByText("96.8%")).toBeInTheDocument();
  });

  it("EHR explorer explains when Synthea is missing", async () => {
    mockApi({ ...HEALTH, "/api/ehr/sources": [] });
    renderAt("/ehr");
    expect(await screen.findByText(/No EHR data found/)).toBeInTheDocument();
  });

  it("operations page surfaces dataset-unavailable errors", async () => {
    mockApi({ ...HEALTH, "/api/operations/forecast": () => { throw apiError(409, "dataset_unavailable", "The Synthea dataset has not been downloaded."); } });
    renderAt("/operations?tab=forecast");
    expect(await screen.findByText("Dataset not available")).toBeInTheDocument();
    expect(screen.getByText(/has not been downloaded/)).toBeInTheDocument();
  });

  it("reports page lists exports in three formats", async () => {
    mockApi({ ...HEALTH, "/api/reports": [{ id: "registry", title: "Dataset & model registry", formats: ["json", "csv", "pdf"] }] });
    renderAt("/reports");
    expect(await screen.findByRole("button", { name: /as PDF/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /as CSV/i })).toBeInTheDocument();
  });

  it("clinical text tab is selected from the URL", async () => {
    mockApi({ ...HEALTH, "/api/text/sample-notes": [{ id: "n1", title: "ED note (synthetic)", text: "Denies fever." }] });
    renderAt("/clinical-text?tab=ner");
    expect(await screen.findByRole("tab", { name: /entity extraction/i })).toHaveAttribute("aria-selected", "true");
    await userEvent.click(await screen.findByRole("button", { name: "ED note (synthetic)" }));
    await waitFor(() => expect(screen.getByLabelText("Clinical note")).toHaveValue("Denies fever."));
  });
});

describe("utilities", () => {
  it("formats numbers and percentages", () => {
    expect(fmt(null)).toBe("—");
    expect(fmt(1234)).toBe("1,234");
    expect(fmt(0.123456)).toBe("0.123");
    expect(pct(0.9295)).toBe("93.0%");
  });

  it("reads the backend error envelope", () => {
    expect(describeError(apiError(409, "model_unavailable", "Train first"))).toEqual({ status: 409, code: "model_unavailable", message: "Train first" });
  });
});
