"""Run every tutorial's intended workflow against the running API and verify the outputs.

Each tutorial (T1-T15) is exercised over HTTP exactly as the web app does it, on its intended
dataset, and its key outputs are checked for correctness (not just for HTTP 200). Every call is
also recorded by the backend as a genuine analysis run.

Usage (backend must be running):
    python scripts/verify_tutorials.py [--base http://127.0.0.1:8000] [--only T10 T15]
Writes reports/tutorial_verification.json, which the Tutorial Library uses to show "verified".
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "tutorial_verification.json"
SAMPLE_DIR = ROOT / "data" / "sample" / "xray"


class Verifier:
    def __init__(self, base: str):
        self.http = httpx.Client(base_url=base, timeout=600)
        self.sample = sorted(SAMPLE_DIR.glob("pneumonia_*.png"))[0].name if SAMPLE_DIR.exists() else None
        self.checks: list[dict] = []
        self.evidence: dict = {}

    def get(self, url, **params):
        r = self.http.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def post(self, url, json_body=None, params=None, data=None, files=None):
        r = self.http.post(url, json=json_body, params=params, data=data, files=files)
        r.raise_for_status()
        return r.json()

    def image(self, endpoint, **params):
        return self.post(f"/api/imaging/{endpoint}", data={"sample": self.sample}, params=params)

    def check(self, name, ok, detail=""):
        self.checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})


def t1(v):
    sources = [s["id"] for s in v.get("/api/ehr/sources")]
    v.check("Synthea sources listed", "synthea:patients" in sources and "synthea:patient_profile" in sources)
    ov = v.get("/api/ehr/overview", source="synthea:patients")
    v.check(
        "patients table profiled",
        ov["rows"] > 0 and any(s["column"] == "GENDER" for s in ov["schema"]),
        f"{ov['rows']} rows x {ov['columns']} columns",
    )
    prof = v.get("/api/ehr/overview", source="synthea:patient_profile")
    v.check(
        "numeric summary computed",
        len(prof["numeric_summary"]) >= 10,
        f"{len(prof['numeric_summary'])} numeric columns",
    )
    col = v.get("/api/ehr/column", source="synthea:patient_profile", column="AGE")
    v.check("histogram sums to non-missing rows", sum(b["count"] for b in col["data"]) == prof["rows"] - col["missing"])
    enc = v.get("/api/ehr/column", source="synthea:encounters", column="START")
    v.check("timeline for dates", enc["chart"] == "timeline" and len(enc["data"]) > 10)
    v.evidence["T1"] = {"patients": ov["rows"], "profile_columns": prof["columns"]}


def t2(v):
    log = v.get("/api/ehr/teaching-log")
    v.check(
        "corruption log available", len(log["corruptions"]) >= 10, f"{len(log['corruptions'])} documented corruptions"
    )
    r = v.get("/api/ehr/clean", source="teaching:patients_messy")
    before = sum(x["violations"] for x in r["validation_before"])
    after = sum(x["violations"] for x in r["validation_after"])
    v.check("validation violations eliminated", before > 0 and after == 0, f"{before} -> {after}")
    rec = {x["column"]: x["match_pct"] for x in r["recovery_vs_ground_truth"]}
    targeted = ["GENDER", "RACE", "STATE", "BIRTHDATE", "HEALTHCARE_EXPENSES", "ZIP"]
    v.check(
        "ground truth recovered for corrupted columns",
        all(rec[c] == 100.0 for c in targeted),
        {c: rec[c] for c in targeted},
    )
    v.check("types converted", any(d["before"] == "object" and d["after"].startswith("float") for d in r["dtypes"]))
    v.evidence["T2"] = {"violations_before": before, "violations_after": after}


def t3(v):
    m = v.get("/api/ehr/missingness", source="synthea:patient_profile")
    v.check(
        "missingness quantified",
        m["total_missing_cells"] > 0 and len(m["co_missing_columns"]) >= 2,
        f"{m['missing_cell_pct']}% cells missing, {m['complete_rows']} complete rows",
    )
    d = v.get("/api/ehr/duplicates", source="teaching:patients_messy")
    v.check(
        "injected duplicates found",
        d["exact_duplicates"] == 5 and d["near_duplicates"] == 3 and d["removed"] == 8,
        f"exact={d['exact_duplicates']} near={d['near_duplicates']} removed={d['removed']}",
    )
    imp = v.get("/api/ehr/imputation", source="synthea:patient_profile", target="LDL")
    methods = {x["method"]: x["rmse"] for x in imp["results"]}
    v.check("4 imputation methods compared", set(methods) == {"mean", "median", "knn", "iterative"}, methods)
    v.check("best method has lowest RMSE", methods[imp["best_method"]] == min(methods.values()), imp["best_method"])
    q_messy = v.get("/api/ehr/quality", source="teaching:patients_messy")
    q_clean = v.get("/api/ehr/quality", source="synthea:patients")
    v.check(
        "quality score lower for corrupted data",
        q_messy["overall"] < q_clean["overall"],
        f"messy {q_messy['overall']} vs original {q_clean['overall']}",
    )
    v.evidence["T3"] = {
        "ldl_rmse": methods,
        "quality_messy": q_messy["overall"],
        "quality_original": q_clean["overall"],
    }


def t4(v):
    e = v.image("enhance")
    v.check("6 enhancement outputs", len(e["panels"]) == 6)
    r = v.image("restore", noise="gaussian", level=20)
    noisy = r["metrics"][0]["psnr"]
    best = max(x["psnr"] for x in r["metrics"][1:])
    v.check("denoising raises PSNR", best > noisy + 3, f"{noisy:.1f} dB -> {best:.1f} dB ({r['best_by_psnr']})")
    f = v.image("filters")
    v.check("edge filters", len(f["panels"]) == 7 and f["edge_pixels_pct"] > 0)
    t = v.image("threshold")
    v.check("Otsu threshold computed", 0 < t["otsu_threshold"] < 255, t["otsu_threshold"])
    s = v.image("segment")
    v.check(
        "two lung fields segmented", s["components"] == 2 and 10 < s["lung_area_pct"] < 60, f"{s['lung_area_pct']:.1f}%"
    )
    # Pneumonia samples may legitimately fail (consolidated lung is not dark), so plausibility is checked on normals.
    normals = [p.name for p in sorted(SAMPLE_DIR.glob("normal_*.png"))]
    plausible = [v.post("/api/imaging/segment", data={"sample": n})["quality"]["plausible"] for n in normals]
    v.check("plausible masks on all normal samples", normals and all(plausible), f"{sum(plausible)}/{len(normals)}")
    c = v.image("compress")
    jpeg = [x for x in c["table"] if x["format"].startswith("JPEG")]
    v.check(
        "JPEG size falls with quality", [x["bytes"] for x in jpeg] == sorted([x["bytes"] for x in jpeg], reverse=True)
    )
    v.evidence["T4"] = {"sample": v.sample, "psnr_noisy": noisy, "psnr_best": best, "lung_area_pct": s["lung_area_pct"],
                        "segmentation_method": s["method"]}


def t5(v):
    s = v.image("sift")
    v.check(
        "SIFT keypoints + 128-D descriptors", s["keypoints"] > 0 and s["descriptor_shape"][1] == 128, s["keypoints"]
    )
    m = v.image("match", angle=15, scale=0.9)
    v.check(
        "RANSAC recovers known homography", m["corner_error_ransac_px"] < 2.0, f"{m['corner_error_ransac_px']:.3f} px"
    )
    v.check(
        "RANSAC beats least squares",
        m["corner_error_ransac_px"] < m["corner_error_least_squares_px"],
        f"{m['corner_error_ransac_px']:.3f} vs {m['corner_error_least_squares_px']:.3f} px",
    )
    cnn = v.image("cnn-features", layer="layer4")
    v.check("CNN feature maps", cnn["feature_map_shapes"]["layer4"] == [512, 7, 7] and len(cnn["channels"]) == 12)
    emb = v.get("/api/imaging/feature-embedding")
    v.check(
        "linear probe above chance",
        emb["linear_probe"]["cv_accuracy_mean"] > 0.6,
        f"{emb['linear_probe']['cv_accuracy_mean']:.3f}",
    )
    v.evidence["T5"] = {
        "inliers": m["ransac_inliers"],
        "corner_error_px": m["corner_error_ransac_px"],
        "probe_accuracy": emb["linear_probe"]["cv_accuracy_mean"],
    }


def t6(v):
    viz = v.image("visualize", colormap="viridis", row=0.5)
    v.check("visualisation panels", len(viz["panels"]) == 6 and len(viz["profile"]) > 50)
    v.check("comparison images", viz["comparison"]["original"].startswith("data:image/png"))
    with open(SAMPLE_DIR / v.sample, "rb") as fh:
        p = v.post("/api/screening/predict", params={"context": "T6"}, files={"file": (v.sample, fh, "image/png")})
    v.check(
        "Grad-CAM produced by trained model",
        p["gradcam"] is not None and p["gradcam"]["overlay"].startswith("data:"),
        f"{p['predicted_label']} p={p['probability_pneumonia']:.3f}",
    )
    v.evidence["T6"] = {"sample": v.sample, "prediction": p["predicted_label"]}


def t7(v):
    f = v.get("/api/operations/forecast", freq="month", encounter_class="all", horizon=6)
    maes = {x["model"]: x["mae"] for x in f["results"]}
    v.check("4 models incl. baselines", len(maes) == 4 and "Naive (last period)" in maes)
    v.check("best model = lowest test MAE", maes[f["best_model"]] == min(maes.values()), f"{f['best_model']} {maes}")
    v.check(
        "time-ordered split",
        f["train_periods"] > f["test_periods"] > 0
        and f["test_predictions"][0]["period"] > f["history"][f["train_periods"]]["period"],
    )
    a = f["staffing_assumptions"]
    fc = f["forecast"][0]
    expected = (
        fc["predicted_encounters"]
        * a["hours_per_encounter"]
        / (a["hours_per_fte_per_period"] * a["utilization_target"])
    )
    v.check(
        "FTE formula consistent", abs(fc["required_fte"] - expected) < 0.01, f"{fc['required_fte']} vs {expected:.3f}"
    )
    w = v.get("/api/operations/forecast", freq="week", encounter_class="ambulatory", horizon=4)
    v.check("weekly per-class forecast", len(w["forecast"]) == 4)
    v.evidence["T7"] = {"best_model": f["best_model"], "mae": maes}


def t8(v):
    k = v.get("/api/operations/kpis", group_by="encounter_class")
    h = k["headline"]
    v.check(
        "headline KPIs", h["encounters"] > 0 and 0 <= h["payer_coverage_pct"] <= 100, f"{h['encounters']} encounters"
    )
    v.check("groups add up", sum(g["encounters"] for g in k["groups"]) == h["encounters"])
    v.check("significance test", k["comparison_test"]["p_value"] < 0.05, k["comparison_test"]["test"])
    p = v.get("/api/operations/kpis", group_by="payer")
    v.check("payer comparison", len(p["groups"]) > 3 and len(p["monthly_trend"]) > 12)
    v.evidence["T8"] = {"encounters": h["encounters"], "avg_cost": h["avg_cost_per_encounter"]}


def t9(v):
    u = v.get("/api/operations/utilization", beds=5)
    v.check("demand heat-map 7x24", len(u["heatmap"]) == 168 and sum(c["count"] for c in u["heatmap"]) > 0)
    v.check("bed utilisation series", len(u["bed_utilization"]) > 0)
    s = v.post(
        "/api/operations/scenarios",
        json_body={
            "scenarios": [
                {"name": "base", "staff_fte": 0.2},
                {"name": "double", "demand_change_pct": 100, "staff_fte": 0.2},
            ]
        },
    )
    b, d = s["scenarios"]
    v.check(
        "scenario arithmetic",
        abs(d["utilization_pct"] - 2 * b["utilization_pct"]) < 0.2,
        f"{b['utilization_pct']}% -> {d['utilization_pct']}%",
    )
    v.evidence["T9"] = {
        "providers": u["provider_workload"]["providers"],
        "scenario_util": [b["utilization_pct"], d["utilization_pct"]],
    }


def t10(v):
    pre = v.post("/api/text/preprocess", json_body={"text": "The patients were treated with antibiotics."})
    v.check("lemmatisation", "treat" in pre["steps"][3]["tokens"])
    m = v.post("/api/text/train")
    best = m["models"][m["best_model"]]
    v.check(
        "classifiers above chance (0.23)",
        best["accuracy"] > 0.45,
        f"{m['best_model']} acc={best['accuracy']:.3f} " f"macroF1={best['macro_f1']:.3f}",
    )
    c = v.post(
        "/api/text/classify",
        json_body={"text": "Carcinoma of the breast with lymph node metastasis treated by chemotherapy."},
    )
    v.check("neoplasm text classified", c["predictions"]["logistic_regression"]["prediction"] == "neoplasms")
    cl = v.get("/api/text/cluster", k=5, sample_size=3000)
    v.check(
        "clustering metrics",
        -1 <= cl["silhouette"] <= 1 and cl["normalized_mutual_info"] > 0,
        f"silhouette={cl['silhouette']:.3f} NMI={cl['normalized_mutual_info']:.3f}",
    )
    v.evidence["T10"] = {k: {"accuracy": x["accuracy"], "macro_f1": x["macro_f1"]} for k, x in m["models"].items()}


def t11(v):
    notes = v.get("/api/text/sample-notes")
    r = v.post("/api/text/entities", json_body={"text": notes[0]["text"]})
    s = r["structured"]
    v.check("negation detected", {"chest pain", "hemoptysis", "pleural effusion"} <= set(s["problems_negated"]))
    v.check("medication-dose linking", any(x["medication"] == "ceftriaxone" and x["dosage"] for x in s["medications"]))
    v.check("vitals extracted", len(s["vitals"]) >= 4, s["vitals"])
    st = v.get("/api/text/entity-stats", per_class=20)
    v.check("corpus-scale extraction", len(st["top_entities"]) == 5, f"{st['documents']} abstracts")
    v.evidence["T11"] = {"entities": len(r["entities"]), "model": r["model"]}


def t12(v):
    r = v.post(
        "/api/text/rank",
        json_body={"query": "coronary artery disease myocardial infarction", "method": "bm25", "top_k": 10},
    )
    v.check(
        "ranked results with matched terms", len(r["results"]) == 10 and all(x["matched_terms"] for x in r["results"])
    )
    e = v.get("/api/text/rank-evaluation", n_queries=200, k=10)
    v.check(
        "retrieval beats random",
        e["results"]["bm25"]["precision_at_k"] > e["random_baseline_precision"],
        {m: round(x["precision_at_k"], 3) for m, x in e["results"].items()},
    )
    w = v.post(
        "/api/text/wsd", json_body={"sentence": "SpO2 92% on RA without supplemental oxygen", "variant": "corpus_lesk"}
    )
    v.check("abbreviation disambiguated", w["predicted_sense"] == "room air")
    we = v.get("/api/text/wsd-evaluation")
    v.check(
        "corpus-only WSD beats baseline",
        we["accuracy"]["corpus_only"] > we["accuracy"]["first_sense_baseline"],
        we["accuracy"],
    )
    v.evidence["T12"] = {"p_at_10": {m: x["precision_at_k"] for m, x in e["results"].items()}, "wsd": we["accuracy"]}


def t13(v):
    p = v.get("/api/analytics/pca", source="nhanes_adults", scaling="standard", color_by="bmi_category")
    v.check("variance sums to 1", abs(p["explained_variance"][-1]["cumulative"] - 1) < 1e-6)
    v.check(
        "ratios decrease",
        all(a["ratio"] >= b["ratio"] for a, b in zip(p["explained_variance"], p["explained_variance"][1:])),
    )
    raw = v.get("/api/analytics/pca", source="nhanes_adults", scaling="none")
    v.check(
        "scaling matters",
        raw["explained_variance"][0]["ratio"] > p["explained_variance"][0]["ratio"],
        f"PC1 {raw['explained_variance'][0]['ratio']:.2f} unscaled vs {p['explained_variance'][0]['ratio']:.2f} scaled",
    )
    s = v.get("/api/analytics/pca", source="synthea:patient_profile", color_by="GENDER")
    v.check("second dataset", s["components_for_80pct"] <= len(s["features"]))
    v.evidence["T13"] = {
        "nhanes_components_80": p["components_for_80pct"],
        "synthea_components_80": s["components_for_80pct"],
    }


def t14(v):
    e = v.get("/api/analytics/weight/explore", variable="bmi")
    pr = e["obesity_prevalence"]
    v.check("adult sample size", e["n_adults"] > 4000, e["n_adults"])
    v.check(
        "weighted prevalence plausible (CDC ~42%)",
        35 < pr["survey_weighted_pct"] < 50,
        f"{pr['survey_weighted_pct']:.1f}%",
    )
    v.check("correlations computed", len(e["bmi_correlations"]) == 11)
    v.evidence["T14"] = {"n": e["n_adults"], "obesity_weighted": pr["survey_weighted_pct"]}


def t15(v):
    m = v.post("/api/analytics/weight/train")
    reg, clf = m["regression"], m["classification"]
    v.check(
        "no leaky predictors",
        not {"weight_kg", "height_cm", "waist_cm", "bmi"}
        & set(m["features"]["numeric"] + m["features"]["categorical"]),
    )
    v.check(
        "regression beats mean baseline",
        reg["random_forest_regressor"]["r2"] > reg["baseline_mean"]["r2"],
        f"R2 {reg['random_forest_regressor']['r2']:.3f}",
    )
    v.check(
        "classifier beats chance",
        clf["random_forest_classifier"]["roc_auc"] > 0.6,
        f"AUC {clf['random_forest_classifier']['roc_auc']:.3f}",
    )
    v.check("interpretation produced", len(m["permutation_importance"]) == 15 and len(m["logistic_odds_ratios"]) > 0)
    v.evidence["T15"] = m["summary"]


RUNNERS = {f"T{i}": f for i, f in enumerate([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15], 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    previous = json.loads(OUT.read_text()) if OUT.exists() else {"tutorials": {}}
    results = previous.get("tutorials", {})
    for tid, fn in RUNNERS.items():
        if args.only and tid not in args.only:
            continue
        v = Verifier(args.base)
        started = time.time()
        error = None
        try:
            fn(v)
        except Exception as exc:  # record the failure instead of stopping the whole run
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, httpx.HTTPStatusError):
                error += f" -> {exc.response.text[:300]}"
        passed = error is None and v.checks and all(c["passed"] for c in v.checks)
        results[tid] = {
            "passed": bool(passed),
            "checks": v.checks,
            "error": error,
            "evidence": v.evidence.get(tid),
            "seconds": round(time.time() - started, 1),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        print(
            f"{tid:4s} {'PASS' if passed else 'FAIL'}  {sum(c['passed'] for c in v.checks)}/{len(v.checks)} checks"
            f"  {results[tid]['seconds']}s" + (f"  ERROR {error}" if error else "")
        )
        for c in v.checks:
            if not c["passed"]:
                print(f"      failed: {c['check']} ({c['detail']})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "base_url": args.base,
                "tutorials": dict(sorted(results.items(), key=lambda kv: int(kv[0][1:]))),
            },
            indent=2,
        )
    )
    ok = all(r["passed"] for r in results.values())
    print(
        f"\n{sum(r['passed'] for r in results.values())}/{len(results)} tutorials verified -> {OUT.relative_to(ROOT)}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
