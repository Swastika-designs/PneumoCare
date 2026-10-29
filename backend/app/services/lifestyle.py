"""T14-T15: behavioural factors and adult weight (NHANES 2017-2018).

Variables are recoded from the NHANES codebooks; "refused"/"don't know" codes become missing.
Body weight, height and waist circumference are *excluded* from the predictors because they
define BMI (target leakage).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import settings
from app.core.errors import DatasetUnavailable, ValidationFailed
from app.core.utils import to_jsonable

NHANES_FILES = ["DEMO_J", "BMX_J", "PAQ_J", "SLQ_J", "SMQ_J", "ALQ_J", "DR1TOT_J"]
NUMERIC_FEATURES = ["age", "income_poverty_ratio", "activity_min_week", "sedentary_min_day", "sleep_hours",
                    "drinks_per_week", "energy_kcal", "sugar_g", "fiber_g", "fat_g", "protein_g"]
CATEGORICAL_FEATURES = ["sex", "race_ethnicity", "education", "smoking_status"]
LABELS = {
    "age": "Age (years)", "income_poverty_ratio": "Income-to-poverty ratio",
    "activity_min_week": "Moderate-equivalent activity (min/week)", "sedentary_min_day": "Sedentary time (min/day)",
    "sleep_hours": "Sleep (h, weekdays)", "drinks_per_week": "Alcoholic drinks per week",
    "energy_kcal": "Energy intake (kcal, day 1)", "sugar_g": "Total sugars (g)", "fiber_g": "Dietary fibre (g)",
    "fat_g": "Total fat (g)", "protein_g": "Protein (g)", "bmi": "BMI (kg/m^2)", "waist_cm": "Waist (cm)",
}
RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian", 7: "Other / multiracial"}
EDUCATION = {1: "< 9th grade", 2: "9-11th grade", 3: "High school / GED", 4: "Some college / AA",
             5: "College graduate +"}
ALCOHOL_DAYS_PER_YEAR = {0: 0, 1: 365, 2: 300, 3: 182, 4: 104, 5: 52, 6: 30, 7: 12, 8: 9, 9: 4.5, 10: 1.5}


def nhanes_dir() -> Path:
    return settings.raw_dir / "nhanes"


def models_dir() -> Path:
    return settings.models_dir / "lifestyle"


def require_nhanes() -> None:
    missing = [f for f in NHANES_FILES if not (nhanes_dir() / f"{f}.xpt").exists()]
    if missing:
        raise DatasetUnavailable(
            f"NHANES files missing: {', '.join(missing)}. Run `python scripts/download_datasets.py --only nhanes`.",
            details={"dataset": "nhanes"})


def _clean_codes(s: pd.Series, invalid: tuple) -> pd.Series:
    return s.where(~s.isin(invalid))


def _minutes(df: pd.DataFrame, flag: str, days: str, minutes: str) -> pd.Series:
    d = _clean_codes(df[days], (77, 99))
    m = _clean_codes(df[minutes], (7777, 9999))
    value = d * m
    return value.where(df[flag] == 1, np.where(df[flag] == 2, 0.0, np.nan))


@lru_cache(maxsize=1)
def _adults_cached(stamp: float) -> pd.DataFrame:
    frames = {f: pd.read_sas(nhanes_dir() / f"{f}.xpt", format="xport") for f in NHANES_FILES}
    df = frames["DEMO_J"]
    for f in NHANES_FILES[1:]:
        df = df.merge(frames[f], on="SEQN", how="left")
    df = df[(df["RIDAGEYR"] >= 20) & (df.get("RIDEXPRG", pd.Series(np.nan, index=df.index)) != 1)]
    df = df[df["BMXBMI"].notna()].copy()

    vigorous = (_minutes(df, "PAQ605", "PAQ610", "PAD615").fillna(0)
                + _minutes(df, "PAQ650", "PAQ655", "PAD660").fillna(0))
    moderate = (_minutes(df, "PAQ620", "PAQ625", "PAD630").fillna(0)
                + _minutes(df, "PAQ635", "PAQ640", "PAD645").fillna(0)
                + _minutes(df, "PAQ665", "PAQ670", "PAD675").fillna(0))
    answered_pa = df[["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]].isin([1, 2]).any(axis=1)

    smoking = np.select(
        [df["SMQ020"] == 2, (df["SMQ020"] == 1) & df["SMQ040"].isin([1, 2]), (df["SMQ020"] == 1) & (df["SMQ040"] == 3)],
        ["never", "current", "former"], default=None)
    drink_days = df["ALQ121"].map(ALCOHOL_DAYS_PER_YEAR)
    drinks = _clean_codes(df["ALQ130"], (777, 999))
    drinks = drinks.where(df["ALQ121"] != 0, 0)
    reliable_diet = df["DR1DRSTZ"] == 1 if "DR1DRSTZ" in df else pd.Series(True, index=df.index)

    out = pd.DataFrame({
        "seqn": df["SEQN"].astype(int),
        "age": df["RIDAGEYR"],
        "sex": df["RIAGENDR"].map({1: "Male", 2: "Female"}),
        "race_ethnicity": df["RIDRETH3"].map(RACE),
        "education": df["DMDEDUC2"].map(EDUCATION),
        "income_poverty_ratio": df.get("INDFMPIR"),
        "activity_min_week": (2 * vigorous + moderate).where(answered_pa),
        "sedentary_min_day": _clean_codes(df["PAD680"], (7777, 9999)),
        "sleep_hours": df["SLD012"],
        "smoking_status": smoking,
        "drinks_per_week": drink_days / 52 * drinks,
        "energy_kcal": df["DR1TKCAL"].where(reliable_diet),
        "sugar_g": df["DR1TSUGR"].where(reliable_diet),
        "fiber_g": df["DR1TFIBE"].where(reliable_diet),
        "fat_g": df["DR1TTFAT"].where(reliable_diet),
        "protein_g": df["DR1TPROT"].where(reliable_diet),
        "bmi": df["BMXBMI"],
        "waist_cm": df["BMXWAIST"],
        "weight_kg": df["BMXWT"],
        "height_cm": df["BMXHT"],
        "exam_weight": df["WTMEC2YR"],
    })
    out["bmi_category"] = pd.cut(out["bmi"], [0, 18.5, 25, 30, 200],
                                 labels=["Underweight", "Normal", "Overweight", "Obese"], right=False).astype(str)
    out["obese"] = (out["bmi"] >= 30).astype(int)
    return out.reset_index(drop=True)


def adults() -> pd.DataFrame:
    require_nhanes()
    stamp = max((nhanes_dir() / f"{f}.xpt").stat().st_mtime for f in NHANES_FILES)
    return _adults_cached(stamp).copy()


# ------------------------------------------------------------------ T14: exploration


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    mask = values.notna() & weights.notna() & (weights > 0)
    return float(np.average(values[mask], weights=weights[mask])) if mask.any() else None


def explore(variable: str = "bmi") -> dict:
    df = adults()
    if variable not in NUMERIC_FEATURES + ["bmi", "waist_cm"]:
        raise ValidationFailed(f"Unknown variable '{variable}'.")
    describe = []
    for col in ["bmi", "waist_cm"] + NUMERIC_FEATURES:
        s = df[col]
        describe.append({"variable": col, "label": LABELS.get(col, col), "n": int(s.notna().sum()),
                         "missing_pct": float(s.isna().mean() * 100), "mean": s.mean(), "sd": s.std(),
                         "median": s.median(), "q1": s.quantile(0.25), "q3": s.quantile(0.75)})
    by_sex = df.groupby("sex")[["bmi", "waist_cm", "activity_min_week", "sleep_hours", "energy_kcal"]].mean()
    values = df[variable].dropna()
    upper = values.quantile(0.99)
    counts, edges = np.histogram(values.clip(upper=upper), bins=30)
    histogram = [{"bin": f"{edges[i]:.0f}", "start": float(edges[i]), "count": int(c)} for i, c in enumerate(counts)]
    corr_cols = ["bmi", "waist_cm"] + NUMERIC_FEATURES
    corr = df[corr_cols].corr(method="spearman")
    correlations = [{"a": a, "b": b, "rho": float(corr.loc[a, b])} for a in corr_cols for b in corr_cols]
    bmi_corr = sorted([{"variable": c, "label": LABELS[c], "rho": float(corr.loc["bmi", c]),
                        "p_value": float(stats.spearmanr(df["bmi"], df[c], nan_policy="omit").pvalue)}
                       for c in NUMERIC_FEATURES], key=lambda r: -abs(r["rho"]))
    group_bmi = {}
    df["activity_level"] = pd.cut(df["activity_min_week"], [-1, 0, 149, 1e9],
                                  labels=["Inactive (0)", "Some (1-149)", "Meets guideline (150+)"]).astype(str)
    df["sleep_band"] = pd.cut(df["sleep_hours"], [0, 6, 9, 24], labels=["< 6 h", "6-9 h", "> 9 h"],
                              right=False).astype(str)
    for group in ["smoking_status", "education", "activity_level", "sleep_band", "sex", "race_ethnicity"]:
        sub = df[df[group].notna() & (df[group] != "nan") & (df[group] != "None")]
        g = sub.groupby(group).agg(n=("bmi", "size"), mean_bmi=("bmi", "mean"), obese_pct=("obese", "mean"))
        g["obese_pct"] *= 100
        samples = [x["bmi"].values for _, x in sub.groupby(group) if len(x) > 10]
        p = float(stats.kruskal(*samples).pvalue) if len(samples) > 1 else None
        group_bmi[group] = {"rows": g.reset_index().rename(columns={group: "group"}).to_dict(orient="records"),
                            "kruskal_p": p}
    category_counts = df["bmi_category"].value_counts().reindex(["Underweight", "Normal", "Overweight", "Obese"])
    return to_jsonable({
        "n_adults": len(df),
        "variable": variable,
        "describe": describe,
        "by_sex": by_sex.reset_index().to_dict(orient="records"),
        "histogram": histogram,
        "histogram_note": f"Values above the 99th percentile ({upper:.1f}) are grouped into the last bin.",
        "bmi_categories": [{"category": k, "count": int(v)} for k, v in category_counts.items()],
        "obesity_prevalence": {
            "unweighted_pct": float(df["obese"].mean() * 100),
            "survey_weighted_pct": _weighted_mean(df["obese"].astype(float), df["exam_weight"]) * 100,
            "note": "Weighted estimate uses the MEC exam weight WTMEC2YR (point estimate only; no design-based "
                    "variance).",
        },
        "correlations": correlations,
        "correlation_variables": corr_cols,
        "bmi_correlations": bmi_corr,
        "group_bmi": group_bmi,
        "variables": [{"id": c, "label": LABELS.get(c, c)} for c in ["bmi", "waist_cm"] + NUMERIC_FEATURES],
    })


# ------------------------------------------------------------------ T15: models


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
         NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CATEGORICAL_FEATURES),
    ])


def _feature_names(pre: ColumnTransformer) -> list[str]:
    names = []
    for name in pre.get_feature_names_out():
        kind, _, rest = name.partition("__")
        names.append(rest)
    return names


def train_models(seed: int = 42) -> dict:
    df = adults()
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].replace({None: np.nan})
    y_reg, y_clf = df["bmi"], df["obese"]
    x_tr, x_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(x, y_reg, y_clf, test_size=0.2, random_state=seed,
                                                              stratify=y_clf)
    regs = {
        "linear_regression": Pipeline([("pre", _preprocessor()), ("model", LinearRegression())]),
        "random_forest_regressor": Pipeline([("pre", _preprocessor()), ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=-1))]),
    }
    clfs = {
        "logistic_regression": Pipeline([("pre", _preprocessor()), ("model", LogisticRegression(max_iter=2000))]),
        "random_forest_classifier": Pipeline([("pre", _preprocessor()), ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=-1))]),
    }
    regression, classification = {}, {}
    for name, pipe in regs.items():
        pipe.fit(x_tr, yr_tr)
        pred = pipe.predict(x_te)
        cv = cross_val_score(pipe, x_tr, yr_tr, cv=5, scoring="r2")
        regression[name] = {"r2": r2_score(yr_te, pred), "mae": mean_absolute_error(yr_te, pred),
                            "rmse": float(np.sqrt(mean_squared_error(yr_te, pred))),
                            "cv_r2_mean": float(cv.mean()), "cv_r2_std": float(cv.std())}
    baseline_pred = np.full(len(yr_te), yr_tr.mean())
    regression["baseline_mean"] = {"r2": r2_score(yr_te, baseline_pred),
                                   "mae": mean_absolute_error(yr_te, baseline_pred),
                                   "rmse": float(np.sqrt(mean_squared_error(yr_te, baseline_pred)))}
    roc_points = {}
    for name, pipe in clfs.items():
        pipe.fit(x_tr, yc_tr)
        prob = pipe.predict_proba(x_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        cv = cross_val_score(pipe, x_tr, yc_tr, cv=5, scoring="roc_auc")
        classification[name] = {
            "accuracy": accuracy_score(yc_te, pred), "precision": precision_score(yc_te, pred, zero_division=0),
            "recall": recall_score(yc_te, pred), "f1": f1_score(yc_te, pred), "roc_auc": roc_auc_score(yc_te, prob),
            "cv_auc_mean": float(cv.mean()), "cv_auc_std": float(cv.std()),
            "confusion_matrix": confusion_matrix(yc_te, pred).tolist(),
        }
        fpr, tpr, _ = roc_curve(yc_te, prob)
        idx = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 100)).astype(int))
        roc_points[name] = [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]
    classification["majority_baseline"] = {"accuracy": float(max(yc_te.mean(), 1 - yc_te.mean())), "roc_auc": 0.5}

    lin = regs["linear_regression"]
    names = _feature_names(lin.named_steps["pre"])
    coefficients = sorted([{"feature": n, "coefficient": float(c)} for n, c in
                           zip(names, lin.named_steps["model"].coef_)], key=lambda r: -abs(r["coefficient"]))
    logit = clfs["logistic_regression"]
    odds = sorted([{"feature": n, "odds_ratio": float(np.exp(c))} for n, c in
                   zip(names, logit.named_steps["model"].coef_[0])], key=lambda r: -abs(np.log(r["odds_ratio"])))
    perm = permutation_importance(regs["random_forest_regressor"], x_te, yr_te, n_repeats=5, random_state=seed,
                                  n_jobs=1)
    importance = sorted([{"feature": f, "importance": float(m), "std": float(s)} for f, m, s in
                         zip(x.columns, perm.importances_mean, perm.importances_std)], key=lambda r: -r["importance"])
    rf_pred = regs["random_forest_regressor"].predict(x_te)
    sample = np.random.default_rng(seed).choice(len(yr_te), size=min(400, len(yr_te)), replace=False)
    scatter = [{"actual": float(yr_te.iloc[i]), "predicted": float(rf_pred[i])} for i in sample]

    best_reg = max((k for k in regression if k != "baseline_mean"), key=lambda k: regression[k]["r2"])
    best_clf = max((k for k in classification if k != "majority_baseline"), key=lambda k: classification[k]["roc_auc"])
    result = to_jsonable({
        "n_train": len(x_tr), "n_test": len(x_te),
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "excluded_features": ["weight_kg", "height_cm", "waist_cm"],
        "regression": regression, "classification": classification,
        "linear_coefficients": coefficients, "logistic_odds_ratios": odds,
        "permutation_importance": importance, "roc": roc_points, "scatter": scatter,
        "summary": {"best_regressor": best_reg, "r2": regression[best_reg]["r2"],
                    "best_classifier": best_clf, "roc_auc": classification[best_clf]["roc_auc"]},
        "notes": [
            "Numeric predictors are standardised, so linear coefficients are BMI change per 1 SD.",
            "Odds ratios are per 1 SD (numeric) or vs the reference category (categorical, first level dropped).",
            "Survey weights are not used in model fitting; results describe the sample, not the US population.",
            "Cross-sectional data: associations, not causal effects. Diet is a single 24-hour recall.",
        ],
    })
    out = models_dir()
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"regression": regs, "classification": clfs}, out / "weight_models.joblib")
    (out / "weight_models.json").write_text(json.dumps(result, indent=2))
    return result


def model_results() -> dict | None:
    path = models_dir() / "weight_models.json"
    return json.loads(path.read_text()) if path.exists() else None
