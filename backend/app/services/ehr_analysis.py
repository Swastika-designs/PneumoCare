"""T1-T3: EHR exploration, standardisation/cleaning, missing data and redundancy."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from app.core.errors import ValidationFailed
from app.core.utils import to_jsonable

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "unknown", "?", "-", "missing"}
DATE_HINT = re.compile(r"DATE|START|STOP|TIME|^DOB$", re.IGNORECASE)
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}
CATEGORY_MAPS = {
    "GENDER": {"f": "F", "female": "F", "m": "M", "male": "M"},
    "SEX": {"f": "F", "female": "F", "m": "M", "male": "M"},
    "MARITAL": {"m": "M", "married": "M", "s": "S", "single": "S", "d": "D", "divorced": "D", "w": "W",
                "widowed": "W"},
}


# ------------------------------------------------------------------ T1: exploration


def semantic_type(series: pd.Series) -> str:
    name = str(series.name)
    non_null = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if non_null.nunique() <= 2:
            return "binary"
        return "numeric"
    if name.upper() == "ID" or name.upper().endswith("ID") or name.upper() in {"PATIENT", "ENCOUNTER"}:
        return "identifier"
    if DATE_HINT.search(name):
        return "datetime-like text"
    unique = non_null.nunique()
    if unique <= 50 or unique / max(len(non_null), 1) < 0.05:
        return "categorical"
    return "text"


def overview(df: pd.DataFrame) -> dict:
    schema = []
    for col in df.columns:
        s = df[col]
        schema.append({
            "column": col,
            "dtype": str(s.dtype),
            "semantic_type": semantic_type(s),
            "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
            "examples": [str(v)[:40] for v in s.dropna().unique()[:3]],
        })
    numeric = df.select_dtypes(include="number")
    numeric_summary = []
    for col in numeric.columns:
        d = numeric[col].describe()
        numeric_summary.append({
            "column": col, "count": d["count"], "mean": d["mean"], "std": d["std"], "min": d["min"],
            "q1": d["25%"], "median": d["50%"], "q3": d["75%"], "max": d["max"],
            "skew": float(numeric[col].skew()) if d["count"] > 2 else None,
        })
    categorical_summary = []
    for item in schema:
        if item["semantic_type"] in {"categorical", "binary"} and item["column"] in df:
            counts = df[item["column"]].value_counts(dropna=True)
            categorical_summary.append({
                "column": item["column"], "unique": item["unique"],
                "top": str(counts.index[0]) if len(counts) else None,
                "top_freq": int(counts.iloc[0]) if len(counts) else 0,
            })
    return to_jsonable({
        "rows": len(df),
        "columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "schema": schema,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "preview": df.head(10).astype(object).where(df.head(10).notna(), None).to_dict(orient="records"),
    })


def column_profile(df: pd.DataFrame, column: str, bins: int = 20) -> dict:
    if column not in df.columns:
        raise ValidationFailed(f"Column '{column}' not found.")
    s = df[column]
    kind = semantic_type(s)
    result = {"column": column, "semantic_type": kind, "missing": int(s.isna().sum())}
    if kind in {"numeric", "binary"} and pd.api.types.is_numeric_dtype(s):
        values = s.dropna()
        counts, edges = np.histogram(values, bins=bins)
        result["chart"] = "histogram"
        result["data"] = [{"bin": f"{edges[i]:.3g}-{edges[i + 1]:.3g}", "start": edges[i], "count": int(c)}
                          for i, c in enumerate(counts)]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        result["outliers_iqr"] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
    elif kind in {"datetime", "datetime-like text"}:
        dates = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        by_year = dates.dt.year.value_counts().sort_index()
        result["chart"] = "timeline"
        result["data"] = [{"bin": str(int(y)), "count": int(c)} for y, c in by_year.items()]
    else:
        counts = s.astype(str).where(s.notna(), "(missing)").value_counts().head(15)
        result["chart"] = "bar"
        result["data"] = [{"bin": k, "count": int(v)} for k, v in counts.items()]
    return to_jsonable(result)


# ------------------------------------------------------------------ T2: standardisation & cleaning


def _null_token_mask(s: pd.Series) -> pd.Series:
    return s.map(lambda v: isinstance(v, str) and v.strip().lower() in NULL_TOKENS)


def _numeric_candidate(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(s):
        return None
    non_null = s.dropna().astype(str).str.strip()
    if non_null.empty:
        return None
    cleaned = non_null.str.replace(r"[$,\s]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if parsed.notna().mean() >= 0.9:
        return pd.to_numeric(s.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    return None


def _date_candidate(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return None
    non_null = s.dropna().astype(str)
    if non_null.empty or not (DATE_HINT.search(str(s.name)) or non_null.str.match(r"^\d{1,4}[-/ ]").mean() > 0.9):
        return None
    parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    if parsed[s.notna()].notna().mean() >= 0.9:
        return parsed.dt.tz_localize(None)
    return None


def validation_issues(df: pd.DataFrame) -> list[dict]:
    """Rule-based validity checks. Returns one entry per rule with the number of violating rows."""
    issues = []

    def numeric(col):
        return pd.to_numeric(df[col].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce") \
            if df[col].dtype == object else df[col]

    for col in df.columns:
        upper = col.upper()
        if any(k in upper for k in ("INCOME", "EXPENSE", "COST", "COVERAGE", "AGE")):
            v = numeric(col)
            issues.append({"rule": f"{col} must be >= 0", "column": col, "violations": int((v < 0).sum())})
        if upper == "LAT":
            v = numeric(col)
            issues.append({"rule": "LAT must be within [-90, 90]", "column": col,
                           "violations": int(((v < -90) | (v > 90)).sum())})
        if upper == "LON":
            v = numeric(col)
            issues.append({"rule": "LON must be within [-180, 180]", "column": col,
                           "violations": int(((v < -180) | (v > 180)).sum())})
        if upper in {"GENDER", "SEX"}:
            bad = ~df[col].isna() & ~df[col].isin(["F", "M"])
            issues.append({"rule": f"{col} must be coded F or M", "column": col, "violations": int(bad.sum())})
        if upper == "ZIP":
            bad = df[col].notna() & ~df[col].astype(str).str.fullmatch(r"\d{5}")
            issues.append({"rule": "ZIP must be 5 digits", "column": col, "violations": int(bad.sum())})
    if {"BIRTHDATE", "DEATHDATE"} <= set(df.columns):
        b = pd.to_datetime(df["BIRTHDATE"], errors="coerce", format="mixed")
        d = pd.to_datetime(df["DEATHDATE"], errors="coerce", format="mixed")
        issues.append({"rule": "DEATHDATE must not precede BIRTHDATE", "column": "DEATHDATE",
                       "violations": int((d < b).sum())})
    if "BIRTHDATE" in df.columns:
        not_iso = df["BIRTHDATE"].notna() & ~df["BIRTHDATE"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}")
        issues.append({"rule": "BIRTHDATE must be an ISO date (YYYY-MM-DD)", "column": "BIRTHDATE",
                       "violations": int(not_iso.sum())})
    return issues


def standardize(df: pd.DataFrame, invalid_to_null: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the cleaning pipeline and return (clean_df, step_log)."""
    out = df.copy()
    steps: list[dict] = []

    obj_cols = [c for c in out.columns if out[c].dtype == object]
    changed = 0
    for col in obj_cols:
        before = out[col]
        stripped = before.map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
        changed += int((stripped != before)[before.notna()].sum())
        out[col] = stripped
    steps.append({"step": "Trim and collapse whitespace", "cells_changed": changed})

    changed = 0
    for col in obj_cols:
        mask = _null_token_mask(out[col])
        changed += int(mask.sum())
        out.loc[mask, col] = np.nan
    steps.append({"step": "Convert null tokens (N/A, null, unknown, '') to missing", "cells_changed": changed})

    for col in obj_cols:
        upper = col.upper()
        before = out[col].copy()
        if upper in CATEGORY_MAPS:
            mapping = CATEGORY_MAPS[upper]
            out[col] = out[col].map(lambda v, m=mapping: m.get(v.lower(), v) if isinstance(v, str) else v)
        elif upper == "STATE":
            out[col] = out[col].map(lambda v: US_STATES.get(v.lower(), v.upper() if len(v) == 2 else v)
                                    if isinstance(v, str) else v)
        elif out[col].nunique() <= 50 and out[col].dropna().astype(str).str.lower().nunique() < out[col].nunique():
            # Only columns that actually contain case variants ("WHITE" vs "white") are lower-cased.
            out[col] = out[col].map(lambda v: v.lower() if isinstance(v, str) else v)
        else:
            continue
        n = int((out[col] != before)[before.notna()].sum())
        if n:
            steps.append({"step": f"Normalise categories in {col}", "cells_changed": n,
                          "categories_before": int(before.nunique()), "categories_after": int(out[col].nunique())})

    for col in list(out.columns):
        if col.upper() == "ZIP":
            before = out[col].copy()
            out[col] = out[col].map(lambda v: str(v).split(".")[0].zfill(5) if pd.notna(v) else v)
            steps.append({"step": "Zero-pad ZIP codes to 5 digits",
                          "cells_changed": int((out[col] != before)[before.notna()].sum())})
            continue
        converted = _numeric_candidate(out[col])
        if converted is not None:
            steps.append({"step": f"Convert {col} to numeric", "from_dtype": str(out[col].dtype),
                          "to_dtype": str(converted.dtype),
                          "cells_changed": int(out[col].notna().sum())})
            out[col] = converted
            continue
        converted = _date_candidate(out[col])
        if converted is not None:
            steps.append({"step": f"Parse {col} as dates (mixed formats)", "from_dtype": str(out[col].dtype),
                          "to_dtype": "datetime64", "unparseable": int((out[col].notna() & converted.isna()).sum())})
            out[col] = converted

    if invalid_to_null:
        nulled = 0
        for col in out.columns:
            upper = col.upper()
            if not pd.api.types.is_numeric_dtype(out[col]):
                continue
            mask = pd.Series(False, index=out.index)
            if any(k in upper for k in ("INCOME", "EXPENSE", "COST", "COVERAGE", "AGE")):
                mask = out[col] < 0
            elif upper == "LAT":
                mask = (out[col] < -90) | (out[col] > 90)
            elif upper == "LON":
                mask = (out[col] < -180) | (out[col] > 180)
            nulled += int(mask.sum())
            out.loc[mask, col] = np.nan
        steps.append({"step": "Set rule-violating values (negative money, impossible coordinates) to missing",
                      "cells_changed": nulled})
    return out, steps


def cleaning_report(raw: pd.DataFrame, ground_truth: pd.DataFrame | None = None) -> dict:
    cleaned, steps = standardize(raw)
    dtypes = [{"column": c, "before": str(raw[c].dtype), "after": str(cleaned[c].dtype),
               "unique_before": int(raw[c].nunique()), "unique_after": int(cleaned[c].nunique())}
              for c in raw.columns]
    value_changes = {}
    for col in raw.columns:
        if raw[col].dtype == object and cleaned[col].nunique() <= 15 and raw[col].nunique() <= 40:
            value_changes[col] = {
                "before": {str(k): int(v) for k, v in raw[col].value_counts(dropna=False).head(15).items()},
                "after": {str(k): int(v) for k, v in cleaned[col].value_counts(dropna=False).items()},
            }
    examples = []
    for col in raw.columns:
        diff = raw[col].astype(str) != cleaned[col].astype(str)
        for idx in raw.index[diff][:2]:
            examples.append({"column": col, "before": str(raw.at[idx, col]), "after": str(cleaned.at[idx, col])})
    report = {
        "steps": steps,
        "dtypes": dtypes,
        "value_changes": value_changes,
        "examples": examples[:30],
        "validation_before": validation_issues(raw),
        "validation_after": validation_issues(cleaned.assign(**{
            c: cleaned[c].dt.strftime("%Y-%m-%d") for c in cleaned.columns
            if pd.api.types.is_datetime64_any_dtype(cleaned[c])})),
        "cleaned_preview": cleaned.head(10).astype(object).where(cleaned.head(10).notna(), None)
        .to_dict(orient="records"),
    }
    if ground_truth is not None:
        report["recovery_vs_ground_truth"] = _recovery(cleaned, ground_truth)
    return to_jsonable(report)


def _recovery(cleaned: pd.DataFrame, truth: pd.DataFrame) -> list[dict]:
    """Compare the cleaned teaching data with the original Synthea values (first row per Id).

    The original values are passed through the same pipeline first, so both sides use the
    same canonical format (e.g. 'Massachusetts' -> 'MA'). Values that were deleted or replaced
    by invalid sentinels cannot be recovered by cleaning and show up as mismatches.
    """
    dedup = cleaned.drop_duplicates(subset="Id", keep="first").set_index("Id")
    truth = standardize(truth, invalid_to_null=False)[0].set_index("Id").loc[dedup.index]
    rows = []
    for col in truth.columns:
        a, b = dedup[col], truth[col]
        if pd.api.types.is_datetime64_any_dtype(a):
            a = a.dt.strftime("%Y-%m-%d")
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            match = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            match = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
        rows.append({"column": col, "match_pct": round(float(np.mean(match) * 100), 2),
                     "mismatches": int((~np.asarray(match)).sum())})
    return rows


# ------------------------------------------------------------------ T3: missing data & redundancy


def missingness(df: pd.DataFrame, normalize_tokens: bool = True, sample_rows: int = 80) -> dict:
    data = df.copy()
    tokens_found = 0
    if normalize_tokens:
        for col in data.columns:
            if data[col].dtype == object:
                mask = _null_token_mask(data[col])
                tokens_found += int(mask.sum())
                data.loc[mask, col] = np.nan
    miss = data.isna()
    per_column = sorted(
        [{"column": c, "missing": int(miss[c].sum()), "missing_pct": round(float(miss[c].mean() * 100), 2)}
         for c in data.columns], key=lambda r: -r["missing"])
    partial = [c for c in data.columns if 0 < miss[c].mean() < 1]
    partial = sorted(partial, key=lambda c: -miss[c].mean())[:12]
    co_missing = []
    if len(partial) >= 2:
        corr = miss[partial].astype(float).corr().fillna(0)
        co_missing = [{"a": a, "b": b, "corr": float(corr.loc[a, b])} for a in partial for b in partial]
    patterns = (miss[partial].astype(int).astype(str).agg("".join, axis=1).value_counts().head(8)
                if partial else pd.Series(dtype=int))
    rows_with_missing = int(miss.any(axis=1).sum())
    matrix_cols = [c for c in data.columns if miss[c].any()][:20]
    return to_jsonable({
        "rows": len(data),
        "columns": data.shape[1],
        "null_tokens_converted": tokens_found,
        "total_missing_cells": int(miss.values.sum()),
        "missing_cell_pct": round(float(miss.values.mean() * 100), 2),
        "rows_with_missing": rows_with_missing,
        "complete_rows": len(data) - rows_with_missing,
        "per_column": per_column,
        "co_missing_columns": partial,
        "co_missing": co_missing,
        "patterns": [{"pattern": p, "rows": int(n),
                      "missing_columns": [partial[i] for i, ch in enumerate(p) if ch == "1"]}
                     for p, n in patterns.items()],
        "matrix": {"columns": matrix_cols,
                   "rows": miss[matrix_cols].head(sample_rows).astype(int).values.tolist() if matrix_cols else []},
    })


def duplicates(df: pd.DataFrame, key: str | None = None, near_subset: list[str] | None = None) -> dict:
    exact = df.duplicated(keep="first")
    key = key if key in df.columns else ("Id" if "Id" in df.columns else None)
    key_dupes = df.duplicated(subset=[key], keep="first") if key else pd.Series(False, index=df.index)
    normalized = df.copy()
    for col in normalized.columns:
        if normalized[col].dtype == object:
            normalized[col] = normalized[col].map(lambda v: re.sub(r"\s+", " ", v).strip().lower()
                                                  if isinstance(v, str) else v)
    subset = [c for c in (near_subset or []) if c in df.columns] or None
    near = normalized.duplicated(subset=subset, keep="first") & ~exact
    examples = []
    for mask, kind in ((exact, "exact"), (near, "near (after case/space normalisation)"),
                       (key_dupes & ~exact & ~near, f"same {key}")):
        for idx in df.index[mask][:3]:
            row = df.loc[idx]
            examples.append({"type": kind, "row": int(idx),
                             "values": {str(k): (None if pd.isna(v) else str(v)[:40]) for k, v in
                                        row.head(8).items()}})
    deduped = df[~(exact | near | key_dupes)]
    return to_jsonable({
        "rows": len(df),
        "exact_duplicates": int(exact.sum()),
        "near_duplicates": int(near.sum()),
        "key": key,
        "key_duplicates": int(key_dupes.sum()),
        "rows_after_dedup": len(deduped),
        "removed": len(df) - len(deduped),
        "examples": examples,
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "redundant_column_pairs": _redundant_columns(df),
    })


def _redundant_columns(df: pd.DataFrame, threshold: float = 0.98) -> list[dict]:
    """Numeric columns that are near-perfectly correlated carry redundant information."""
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr().abs()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if corr.loc[a, b] >= threshold:
                pairs.append({"a": a, "b": b, "abs_corr": float(corr.loc[a, b])})
    return pairs


IMPUTERS = {
    "mean": lambda: SimpleImputer(strategy="mean"),
    "median": lambda: SimpleImputer(strategy="median"),
    "knn": lambda: KNNImputer(n_neighbors=5),
    "iterative": lambda: IterativeImputer(max_iter=10, random_state=0),
}


def imputation_comparison(df: pd.DataFrame, target: str | None = None, mask_fraction: float = 0.15,
                          seed: int = 0) -> dict:
    """Mask known numeric values, impute them back with each method, and measure the error."""
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, (numeric.notna().mean() >= 0.5) & (numeric.nunique() > 2)]
    if numeric.shape[1] < 2:
        raise ValidationFailed("Need at least two numeric columns with >= 50% observed values for imputation.")
    target = target if target in numeric.columns else numeric.isna().mean().sort_values().index[-1]
    rng = np.random.default_rng(seed)
    observed = np.flatnonzero(numeric[target].notna().values)
    n_mask = max(5, int(len(observed) * mask_fraction))
    if len(observed) <= n_mask:
        raise ValidationFailed(f"Column '{target}' has too few observed values to evaluate imputation.")
    masked_idx = rng.choice(observed, size=n_mask, replace=False)
    truth = numeric[target].values[masked_idx].astype(float)
    work = numeric.copy()
    work.iloc[masked_idx, work.columns.get_loc(target)] = np.nan
    scale = float(numeric[target].std()) or 1.0
    results = []
    imputed_values = {}
    for name, factory in IMPUTERS.items():
        filled = factory().fit_transform(work)
        col_idx = list(work.columns).index(target)
        pred = filled[masked_idx, col_idx]
        err = pred - truth
        results.append({"method": name, "mae": float(np.mean(np.abs(err))),
                        "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "normalized_rmse": float(np.sqrt(np.mean(err ** 2)) / scale)})
        imputed_values[name] = pred
    best = min(results, key=lambda r: r["rmse"])["method"]
    edges = np.histogram_bin_edges(numeric[target].dropna(), bins=15)
    obs_hist, _ = np.histogram(numeric[target].dropna(), bins=edges)
    best_full = IMPUTERS[best]().fit_transform(numeric)[:, list(numeric.columns).index(target)]
    imputed_only = best_full[numeric[target].isna().values]
    imp_hist, _ = np.histogram(imputed_only, bins=edges) if len(imputed_only) else (np.zeros(len(edges) - 1), None)
    return to_jsonable({
        "target": target,
        "predictors": [c for c in numeric.columns if c != target],
        "masked_values": n_mask,
        "originally_missing": int(numeric[target].isna().sum()),
        "results": sorted(results, key=lambda r: r["rmse"]),
        "best_method": best,
        "distribution": [{"bin": f"{edges[i]:.3g}", "observed": int(obs_hist[i]), "imputed": int(imp_hist[i])}
                         for i in range(len(obs_hist))],
        "numeric_columns": list(numeric.columns),
        "method_note": f"{n_mask} observed values of {target} were hidden at random, re-imputed with each "
                       "method using the other numeric columns, and compared with the true values.",
    })


def quality_report(df: pd.DataFrame) -> dict:
    """Simple data-quality scorecard combining completeness, uniqueness, validity and consistency."""
    miss = missingness(df)
    dup = duplicates(df)
    issues = validation_issues(df)
    checked_cells = max(sum(1 for _ in issues) * len(df), 1)
    violations = sum(i["violations"] for i in issues)
    inconsistent = 0
    cat_cols = 0
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() <= 50:
            cat_cols += 1
            raw = df[col].dropna().astype(str)
            if raw.nunique() > raw.str.strip().str.lower().nunique():
                inconsistent += 1
    scores = {
        "completeness": 100 - miss["missing_cell_pct"],
        "uniqueness": 100 * (1 - dup["removed"] / max(len(df), 1)),
        "validity": 100 * (1 - violations / checked_cells),
        "consistency": 100 * (1 - inconsistent / max(cat_cols, 1)),
    }
    return to_jsonable({
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "overall": round(float(np.mean(list(scores.values()))), 2),
        "validation": issues,
        "inconsistent_categorical_columns": inconsistent,
        "categorical_columns": cat_cols,
        "missing": {k: miss[k] for k in ("total_missing_cells", "missing_cell_pct", "rows_with_missing")},
        "duplicates": {k: dup[k] for k in ("exact_duplicates", "near_duplicates", "key_duplicates", "removed")},
    })
