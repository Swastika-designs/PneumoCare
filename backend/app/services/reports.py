"""Export results as CSV, JSON or PDF.

Each builder returns a ``Report`` made of sections (text, tables, images) using only stored
or freshly computed real results. If the underlying result does not exist yet (e.g. the
model is not trained) the builder raises a clear error instead of producing an empty file.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ModelUnavailable, NotFound, ValidationFailed
from app.core.utils import to_jsonable
from app.models.entities import AnalysisRun, Prediction


@dataclass
class Section:
    heading: str
    text: str | None = None
    table: list[dict] | None = None
    images: list[Path] = field(default_factory=list)


@dataclass
class Report:
    id: str
    title: str
    sections: list[Section]
    payload: dict


def _metric_rows(metrics: dict, keys: tuple[str, ...]) -> list[dict]:
    return [{"metric": k, "value": round(metrics[k], 4) if isinstance(metrics.get(k), float) else metrics.get(k)}
            for k in keys if k in metrics]


def build_pneumonia(_db: Session, **_) -> Report:
    from app.ml.pneumonia.model import checkpoint_dir

    folder = checkpoint_dir()
    path = folder / "evaluation.json"
    if not path.exists():
        raise ModelUnavailable("No evaluation results yet. Run scripts/train_pneumonia.py and "
                               "scripts/evaluate_pneumonia.py first.")
    ev = json.loads(path.read_text())
    keys = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "threshold")
    cm = ev["metrics"]["confusion_matrix"]
    sections = [
        Section("Summary", text=f"Checkpoint {ev['checkpoint']} evaluated on the held-out {ev['split']} split of "
                                f"{ev['data_source']} ({ev['samples']} images) on {ev['evaluated_at'][:19]} UTC. "
                                "Educational model - not a clinically validated diagnostic tool."),
        Section("Test metrics at the validation-selected threshold", table=_metric_rows(ev["metrics"], keys)),
        Section("Test metrics at threshold 0.5", table=_metric_rows(ev["metrics_at_0_5"], keys)),
        Section("Confusion matrix", table=[{"actual": "NORMAL", "pred_NORMAL": cm["tn"], "pred_PNEUMONIA": cm["fp"]},
                                           {"actual": "PNEUMONIA", "pred_NORMAL": cm["fn"],
                                            "pred_PNEUMONIA": cm["tp"]}]),
        Section("Plots", images=[p for p in (folder / "training_curves.png", folder / "confusion_matrix.png",
                                             folder / "roc_curve.png") if p.exists()]),
    ]
    return Report("pneumonia_evaluation", "Pneumonia model evaluation", sections, ev)


def build_predictions(db: Session, **_) -> Report:
    rows = db.execute(select(Prediction).order_by(Prediction.created_at.desc())).scalars().all()
    table = [{"id": r.id, "file": r.filename, "prediction": r.predicted_label,
              "p_pneumonia": round(r.probability_pneumonia, 4), "threshold": round(r.threshold, 4),
              "model": r.model_name, "created_at": r.created_at.isoformat()} for r in rows]
    if not table:
        raise NotFound("No predictions have been made yet.")
    return Report("predictions", "Pneumonia screening predictions",
                  [Section("Predictions", text="Model outputs for uploaded images (educational use only).",
                           table=table)], {"predictions": table})


def build_history(db: Session, **_) -> Report:
    rows = db.execute(select(AnalysisRun).order_by(AnalysisRun.created_at.desc())).scalars().all()
    table = [{"id": r.id, "tutorial": r.tutorial or "", "module": r.module, "title": r.title,
              "dataset": r.dataset or "", "created_at": r.created_at.isoformat()} for r in rows]
    if not table:
        raise NotFound("No analyses have been run yet.")
    return Report("analysis_history", "Analysis history", [Section("Runs", table=table)],
                  {"runs": [dict(t, summary=r.summary) for t, r in zip(table, rows)]})


def build_ehr_quality(db: Session, source: str = "teaching:patients_messy", **_) -> Report:
    from app.services.ehr_analysis import cleaning_report, quality_report, standardize
    from app.services.ehr_data import load_source, make_messy_patients

    df = load_source(source, db)
    before = quality_report(df)
    cleaned = standardize(df)[0]
    if "Id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset="Id")
    after = quality_report(cleaned.drop_duplicates())
    truth = make_messy_patients()[1] if source == "teaching:patients_messy" else None
    cleaning = cleaning_report(df, truth)
    scores = [{"dimension": k, "before": before["scores"][k], "after": after["scores"][k]} for k in before["scores"]]
    scores.append({"dimension": "overall", "before": before["overall"], "after": after["overall"]})
    sections = [
        Section("Source", text=f"Data source: {source}. Rows: {len(df)}; columns: {df.shape[1]}."),
        Section("Quality scores (0-100) before and after cleaning + de-duplication", table=scores),
        Section("Cleaning steps", table=[{k: v for k, v in s.items()} for s in cleaning["steps"]]),
        Section("Validation rules before cleaning", table=before["validation"]),
    ]
    if cleaning.get("recovery_vs_ground_truth"):
        sections.append(Section("Recovery vs original Synthea values", table=cleaning["recovery_vs_ground_truth"]))
    return Report("ehr_quality", "EHR data-quality report", sections,
                  {"source": source, "before": before, "after": after, "cleaning": cleaning})


def build_operations(_db: Session, start_year: int = 2016, **_) -> Report:
    from app.services.operations import kpis

    k = kpis(start_year=start_year)
    head = [{"kpi": key, "value": round(v, 3) if isinstance(v, float) else v} for key, v in k["headline"].items()]
    sections = [
        Section("Window", text=f"Synthea encounters from {start_year} to {k['window']['end']} (synthetic data)."),
        Section("Headline KPIs", table=head),
        Section("By encounter class", table=k["groups"]),
        Section("By year", table=k["yearly"]),
    ]
    return Report("operations_kpis", "Healthcare operations KPIs", sections, k)


def build_text(_db: Session, **_) -> Report:
    from app.services.text_analytics import classifier_metrics

    m = classifier_metrics()
    if not m:
        raise ModelUnavailable("Text classifiers are not trained yet.")
    summary = [{"model": v["name"], "accuracy": round(v["accuracy"], 4), "macro_f1": round(v["macro_f1"], 4),
                "weighted_f1": round(v["weighted_f1"], 4)} for v in m["models"].values()]
    best = m["models"][m["best_model"]]
    sections = [Section("Setup", text=f"{m['train_documents']} training / {m['test_documents']} test abstracts. "
                                      f"{m['vectorizer']}."),
                Section("Model comparison (test set)", table=summary),
                Section(f"Per-class results - {best['name']}", table=best["per_class"])]
    return Report("text_classification", "Clinical text classification", sections, m)


def build_weight(_db: Session, **_) -> Report:
    from app.services.lifestyle import model_results

    m = model_results()
    if not m:
        raise ModelUnavailable("Adult weight models are not trained yet.")
    reg = [{"model": k, **{kk: round(vv, 4) for kk, vv in v.items()}} for k, v in m["regression"].items()]
    clf = [{"model": k, **{kk: round(vv, 4) for kk, vv in v.items() if kk != "confusion_matrix"}}
           for k, v in m["classification"].items()]
    sections = [Section("Setup", text=f"NHANES 2017-2018 adults; {m['n_train']} train / {m['n_test']} test. "
                                      "Excluded predictors: " + ", ".join(m["excluded_features"]) + "."),
                Section("BMI regression (test set)", table=reg),
                Section("Obesity classification (test set)", table=clf),
                Section("Permutation importance (random forest, BMI)", table=m["permutation_importance"]),
                Section("Notes", text=" ".join(m["notes"]))]
    return Report("weight_models", "Adult weight models", sections, m)


def build_registry(_db: Session, **_) -> Report:
    from app.services.registry import dataset_status, model_status

    ds = [{"id": d["id"], "name": d["name"], "domain": d["domain"], "available": d["available"],
           "synthetic": d["synthetic"], "access": d["access"], "source": d["source_url"]} for d in dataset_status()]
    ms = [{"id": m["id"], "name": m["name"], "status": m["status"], "updated_at": m["updated_at"] or ""}
          for m in model_status()]
    return Report("registry", "Dataset and model registry",
                  [Section("Datasets", table=ds), Section("Models", table=ms)], {"datasets": ds, "models": ms})


BUILDERS = {
    "pneumonia_evaluation": ("Pneumonia model evaluation", build_pneumonia),
    "predictions": ("Screening predictions", build_predictions),
    "ehr_quality": ("EHR data-quality report (T2-T3)", build_ehr_quality),
    "operations_kpis": ("Operations KPIs (T8)", build_operations),
    "text_classification": ("Clinical text classification (T10)", build_text),
    "weight_models": ("Adult weight models (T15)", build_weight),
    "analysis_history": ("Analysis history", build_history),
    "registry": ("Dataset & model registry", build_registry),
}


def available_reports() -> list[dict]:
    return [{"id": k, "title": v[0], "formats": ["json", "csv", "pdf"]} for k, v in BUILDERS.items()]


def build(report_id: str, db: Session, **params) -> Report:
    if report_id not in BUILDERS:
        raise NotFound(f"Unknown report '{report_id}'.")
    return BUILDERS[report_id][1](db, **params)


def render(report: Report, fmt: str) -> tuple[bytes, str, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"pneumocare_{report.id}_{stamp}.{fmt}"
    if fmt == "json":
        body = json.dumps(to_jsonable({"report": report.id, "title": report.title,
                                       "generated_at": datetime.now(timezone.utc).isoformat(),
                                       "data": report.payload}), indent=2).encode()
        return body, "application/json", name
    if fmt == "csv":
        return _csv(report), "text/csv", name
    if fmt == "pdf":
        return _pdf(report), "application/pdf", name
    raise ValidationFailed("format must be json, csv or pdf")


def _csv(report: Report) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([report.title])
    for section in report.sections:
        if not section.table:
            continue
        writer.writerow([])
        writer.writerow([f"# {section.heading}"])
        cols = list(dict.fromkeys(k for row in section.table for k in row))
        writer.writerow(cols)
        for row in section.table:
            writer.writerow([_cell(row.get(c)) for c in cols])
    return buf.getvalue().encode("utf-8-sig")


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def _pdf(report: Report) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title=report.title)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"PneumoCare - {report.title}", styles["Title"]),
             Paragraph(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", styles["Normal"]),
             Spacer(1, 10)]
    width = A4[0] - 3.2 * cm
    for section in report.sections:
        story.append(Paragraph(section.heading, styles["Heading2"]))
        if section.text:
            story.append(Paragraph(section.text, styles["BodyText"]))
        if section.table:
            cols = list(dict.fromkeys(k for row in section.table for k in row))[:8]
            data = [cols] + [[_cell(row.get(c))[:40] for c in cols] for row in section.table[:60]]
            table = Table(data, repeatRows=1, colWidths=[width / len(cols)] * len(cols))
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(table)
            if len(section.table) > 60:
                story.append(Paragraph(f"... {len(section.table) - 60} more rows in the CSV/JSON export.",
                                       styles["Italic"]))
        for image in section.images:
            img = RLImage(str(image))
            scale = min(width / img.drawWidth, 7 * cm / img.drawHeight)
            img.drawWidth *= scale
            img.drawHeight *= scale
            story += [Spacer(1, 6), img]
        story.append(Spacer(1, 10))
    doc.build(story)
    return buf.getvalue()
