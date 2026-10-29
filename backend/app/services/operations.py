"""T7-T9: patient-volume prediction, healthcare KPIs, operational analysis and scenarios.

All numbers come from the Synthea ``encounters`` table (synthetic data). Synthea has no
staff rosters, so staffing is derived from *workload* (encounters x observed encounter
duration) under explicit, user-editable assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.core.errors import ValidationFailed
from app.core.utils import to_jsonable
from app.services.ehr_data import read_synthea, reference_date

LONG_STAY_CLASSES = {"inpatient", "snf", "hospice"}
FREQ = {"week": "W-MON", "month": "MS"}


def encounters(start_year: int = 2016) -> pd.DataFrame:
    enc = read_synthea("encounters")
    enc["START"] = pd.to_datetime(enc["START"], utc=True).dt.tz_localize(None)
    enc["STOP"] = pd.to_datetime(enc["STOP"], utc=True).dt.tz_localize(None)
    enc = enc[enc["START"].dt.year >= start_year].copy()
    enc["DURATION_H"] = (enc["STOP"] - enc["START"]).dt.total_seconds() / 3600
    patients = read_synthea("patients")[["Id", "GENDER", "BIRTHDATE"]]
    enc = enc.merge(patients, left_on="PATIENT", right_on="Id", how="left", suffixes=("", "_P"))
    age = (enc["START"] - pd.to_datetime(enc["BIRTHDATE"])).dt.days / 365.25
    enc["AGE_GROUP"] = pd.cut(age, [-1, 17, 39, 64, 200], labels=["0-17", "18-39", "40-64", "65+"]).astype(str)
    payers = read_synthea("payers")[["Id", "NAME"]].rename(columns={"Id": "PAYER", "NAME": "PAYER_NAME"})
    enc = enc.merge(payers, on="PAYER", how="left")
    orgs = read_synthea("organizations")[["Id", "NAME"]].rename(columns={"Id": "ORGANIZATION",
                                                                          "NAME": "ORGANIZATION_NAME"})
    enc = enc.merge(orgs, on="ORGANIZATION", how="left")
    enc["YEAR"] = enc["START"].dt.year
    return enc


# ------------------------------------------------------------------ T7: volume prediction & staffing


def volume_series(enc: pd.DataFrame, freq: str = "month", encounter_class: str | None = None) -> pd.Series:
    if freq not in FREQ:
        raise ValidationFailed("freq must be 'week' or 'month'")
    data = enc if not encounter_class or encounter_class == "all" else enc[enc["ENCOUNTERCLASS"] == encounter_class]
    if data.empty:
        raise ValidationFailed(f"No encounters for class '{encounter_class}'.")
    series = data.set_index("START").resample(FREQ[freq]).size()
    # Drop the last (incomplete) period ending after the reference date.
    return series.iloc[:-1] if len(series) > 1 else series


def _features(series: pd.Series, lags: int, season: int) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.values}, index=series.index)
    for k in range(1, lags + 1):
        df[f"lag_{k}"] = df["y"].shift(k)
    df["rolling_mean"] = df["y"].shift(1).rolling(lags).mean()
    position = (series.index.month - 1) / 12 if season == 12 else (series.index.isocalendar().week.values - 1) / 52
    df["season_sin"] = np.sin(2 * np.pi * np.asarray(position, dtype=float))
    df["season_cos"] = np.cos(2 * np.pi * np.asarray(position, dtype=float))
    df["trend"] = np.arange(len(df))
    return df.dropna()


def forecast_volume(freq: str = "month", encounter_class: str = "all", horizon: int = 6, test_fraction: float = 0.2,
                    start_year: int = 2016, hours_per_fte: float | None = None, utilization_target: float = 0.85
                    ) -> dict:
    enc = encounters(start_year)
    series = volume_series(enc, freq, encounter_class)
    lags, season = (3, 12) if freq == "month" else (4, 52)
    data = _features(series, lags, season)
    if len(data) < 20:
        raise ValidationFailed("Not enough history for forecasting; choose an earlier start year or monthly periods.")
    split = int(len(data) * (1 - test_fraction))
    train, test = data.iloc[:split], data.iloc[split:]
    feature_cols = [c for c in data.columns if c != "y"]
    models = {
        "Naive (last period)": None,
        "Moving average": None,
        "Ridge regression": Ridge(alpha=1.0),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0),
    }
    results, predictions = [], {}
    for name, model in models.items():
        if name.startswith("Naive"):
            pred = test["lag_1"].values
        elif name == "Moving average":
            pred = test["rolling_mean"].values
        else:
            model.fit(train[feature_cols], train["y"])
            pred = model.predict(test[feature_cols])
        predictions[name] = pred
        y = test["y"].values
        results.append({
            "model": name,
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mape": float(np.mean(np.abs(y - pred) / np.maximum(y, 1)) * 100),
        })
    best = min(results, key=lambda r: r["mae"])["model"]

    # Recursive multi-step forecast with the best learned model (or naive baseline).
    future = []
    history = series.copy()
    final_model = models[best]
    if final_model is not None:
        final_model.fit(data[feature_cols], data["y"])
    step = pd.offsets.MonthBegin(1) if freq == "month" else pd.offsets.Week(1, weekday=0)
    for _ in range(horizon):
        next_idx = history.index[-1] + step
        extended = pd.concat([history, pd.Series([np.nan], index=[next_idx])])
        row = _features(extended.fillna(0), lags, season).iloc[[-1]]
        if final_model is not None:
            value = float(final_model.predict(row[feature_cols])[0])
        elif best == "Moving average":
            value = float(row["rolling_mean"].iloc[0])
        else:
            value = float(history.iloc[-1])
        value = max(value, 0.0)
        future.append({"period": next_idx.strftime("%Y-%m-%d"), "predicted_encounters": round(value, 1)})
        history = pd.concat([history, pd.Series([value], index=[next_idx])])

    # Staffing from workload. Durations of multi-day stays are excluded: they need beds, not clinic hours.
    clinic = enc[~enc["ENCOUNTERCLASS"].isin(LONG_STAY_CLASSES)]
    if encounter_class and encounter_class != "all":
        clinic = clinic[clinic["ENCOUNTERCLASS"] == encounter_class]
    hours_per_encounter = float(clinic["DURATION_H"].median()) if len(clinic) else 0.5
    hours_per_fte = hours_per_fte or (40.0 if freq == "week" else 160.0)
    for row in future:
        workload = row["predicted_encounters"] * hours_per_encounter
        row["workload_hours"] = round(workload, 1)
        row["required_fte"] = round(workload / (hours_per_fte * utilization_target), 2)

    history_rows = [{"period": idx.strftime("%Y-%m-%d"), "actual": int(v)} for idx, v in series.items()]
    test_rows = [{"period": idx.strftime("%Y-%m-%d"), "actual": float(y),
                  **{name: float(predictions[name][i]) for name in predictions}}
                 for i, (idx, y) in enumerate(test["y"].items())]
    return to_jsonable({
        "freq": freq,
        "encounter_class": encounter_class,
        "periods": len(series),
        "train_periods": len(train),
        "test_periods": len(test),
        "features": feature_cols,
        "results": results,
        "best_model": best,
        "history": history_rows,
        "test_predictions": test_rows,
        "forecast": future,
        "staffing_assumptions": {
            "hours_per_encounter": round(hours_per_encounter, 3),
            "hours_per_encounter_source": "median observed encounter duration (Synthea STOP - START), "
                                          "excluding inpatient/SNF/hospice stays",
            "hours_per_fte_per_period": hours_per_fte,
            "utilization_target": utilization_target,
        },
        "limitations": [
            "Synthea encounters are simulated; volumes are small (111 patients), so error metrics are noisy.",
            "Staffing is a workload-based estimate. Synthea contains no rosters or staffing labels.",
            "The last incomplete period is excluded from training.",
        ],
        "encounter_classes": sorted(enc["ENCOUNTERCLASS"].unique().tolist()),
    })


# ------------------------------------------------------------------ T8: KPIs and group comparisons

GROUP_COLUMNS = {
    "encounter_class": "ENCOUNTERCLASS",
    "year": "YEAR",
    "gender": "GENDER",
    "age_group": "AGE_GROUP",
    "payer": "PAYER_NAME",
    "organization": "ORGANIZATION_NAME",
}


def _readmission_rate(enc: pd.DataFrame, days: int = 30) -> tuple[float | None, int]:
    inpatient = enc[enc["ENCOUNTERCLASS"] == "inpatient"].sort_values(["PATIENT", "START"])
    if inpatient.empty:
        return None, 0
    next_start = inpatient.groupby("PATIENT")["START"].shift(-1)
    readmit = (next_start - inpatient["STOP"]).dt.days.between(0, days)
    return float(readmit.mean()), len(inpatient)


def kpis(start_year: int = 2016, group_by: str = "encounter_class", metric: str = "TOTAL_CLAIM_COST") -> dict:
    enc = encounters(start_year)
    if group_by not in GROUP_COLUMNS:
        raise ValidationFailed(f"group_by must be one of {sorted(GROUP_COLUMNS)}")
    if metric not in {"TOTAL_CLAIM_COST", "BASE_ENCOUNTER_COST", "DURATION_H"}:
        raise ValidationFailed("metric must be TOTAL_CLAIM_COST, BASE_ENCOUNTER_COST or DURATION_H")
    readmit, n_inpatient = _readmission_rate(enc)
    inpatient = enc[enc["ENCOUNTERCLASS"] == "inpatient"]
    n_patients = enc["PATIENT"].nunique()
    headline = {
        "encounters": len(enc),
        "unique_patients": n_patients,
        "encounters_per_patient": len(enc) / max(n_patients, 1),
        "total_claim_cost": float(enc["TOTAL_CLAIM_COST"].sum()),
        "avg_cost_per_encounter": float(enc["TOTAL_CLAIM_COST"].mean()),
        "payer_coverage_pct": float(enc["PAYER_COVERAGE"].sum() / max(enc["TOTAL_CLAIM_COST"].sum(), 1) * 100),
        "uninsured_encounter_pct": float((enc["PAYER_NAME"] == "NO_INSURANCE").mean() * 100),
        "inpatient_admissions": n_inpatient,
        "avg_length_of_stay_days": float(inpatient["DURATION_H"].mean() / 24) if len(inpatient) else None,
        "readmission_30d_pct": readmit * 100 if readmit is not None else None,
        "emergency_visits_per_1000_patients": float((enc["ENCOUNTERCLASS"] == "emergency").sum()
                                                    / max(n_patients, 1) * 1000),
    }
    col = GROUP_COLUMNS[group_by]
    grouped = enc.groupby(col).agg(
        encounters=("Id", "size"), patients=("PATIENT", "nunique"), avg_cost=("TOTAL_CLAIM_COST", "mean"),
        median_cost=("TOTAL_CLAIM_COST", "median"), total_cost=("TOTAL_CLAIM_COST", "sum"),
        coverage=("PAYER_COVERAGE", "sum"), avg_duration_h=("DURATION_H", "mean"),
    ).reset_index().sort_values("encounters", ascending=False)
    grouped["coverage_pct"] = grouped["coverage"] / grouped["total_cost"].clip(lower=1) * 100
    grouped = grouped.drop(columns="coverage").head(15)

    groups = [g[metric].dropna().values for _, g in enc.groupby(col) if len(g) >= 5]
    test = None
    if len(groups) == 2:
        stat, p = stats.mannwhitneyu(*groups)
        test = {"test": "Mann-Whitney U", "statistic": float(stat), "p_value": float(p)}
    elif len(groups) > 2:
        stat, p = stats.kruskal(*groups)
        test = {"test": "Kruskal-Wallis H", "statistic": float(stat), "p_value": float(p)}
    if test:
        test["metric"] = metric
        test["groups_tested"] = len(groups)
        test["interpretation"] = (
            f"{metric} differs significantly between {group_by} groups (p < 0.05)." if test["p_value"] < 0.05
            else f"No significant difference in {metric} between {group_by} groups at the 5% level."
        )

    monthly = enc.set_index("START").resample("MS").agg({"Id": "size", "TOTAL_CLAIM_COST": "mean",
                                                         "PATIENT": "nunique"})
    trend = [{"period": i.strftime("%Y-%m"), "encounters": int(r["Id"]), "avg_cost": r["TOTAL_CLAIM_COST"],
              "patients": int(r["PATIENT"])} for i, r in monthly.iterrows()]
    yearly = enc.groupby("YEAR").agg(encounters=("Id", "size"), avg_cost=("TOTAL_CLAIM_COST", "mean"),
                                     patients=("PATIENT", "nunique")).reset_index()
    return to_jsonable({
        "window": {"start_year": start_year, "end": reference_date().strftime("%Y-%m-%d")},
        "headline": headline,
        "group_by": group_by,
        "groups": grouped.rename(columns={col: "group"}).to_dict(orient="records"),
        "comparison_test": test,
        "monthly_trend": trend,
        "yearly": yearly.to_dict(orient="records"),
        "group_options": list(GROUP_COLUMNS),
    })


# ------------------------------------------------------------------ T9: operations & scenarios


def operations(start_year: int = 2016, beds: int = 5) -> dict:
    enc = encounters(start_year)
    dow = enc["START"].dt.dayofweek
    hour = enc["START"].dt.hour
    heat = pd.crosstab(dow, hour).reindex(index=range(7), columns=range(24), fill_value=0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heatmap = [{"day": days[d], "hour": h, "count": int(heat.loc[d, h])} for d in range(7) for h in range(24)]

    per_year_class = enc.pivot_table(index="YEAR", columns="ENCOUNTERCLASS", values="Id", aggfunc="size",
                                     fill_value=0)
    demand = [{"year": int(y), **{c: int(v) for c, v in row.items()}} for y, row in per_year_class.iterrows()]
    growth = []
    if len(per_year_class) >= 3:
        last_full = per_year_class.iloc[-3:-1]  # two most recent complete years
        for cls in per_year_class.columns:
            a, b = last_full[cls].iloc[0], last_full[cls].iloc[1]
            growth.append({"encounter_class": cls, "previous": int(a), "latest": int(b),
                           "growth_pct": float((b - a) / a * 100) if a else None})

    provider_load = enc.groupby(["PROVIDER", "YEAR"]).size().groupby("PROVIDER").mean()
    workload = {
        "providers": int(enc["PROVIDER"].nunique()),
        "mean_encounters_per_provider_year": float(provider_load.mean()),
        "max_encounters_per_provider_year": float(provider_load.max()),
        "top_providers_share_pct": float(provider_load.sort_values(ascending=False).head(10).sum()
                                         / provider_load.sum() * 100),
        "distribution": [{"bucket": str(b), "providers": int(n)} for b, n in
                         pd.cut(provider_load, [0, 1, 2, 5, 10, 20, 1000]).value_counts(sort=False).items()],
    }

    # Bed utilisation: count occupied bed-days of long stays per month vs. an assumed bed capacity.
    stays = enc[enc["ENCOUNTERCLASS"].isin(LONG_STAY_CLASSES)]
    bed_days = pd.Series(0.0, index=pd.date_range(enc["START"].min().normalize(), enc["STOP"].max().normalize()))
    for _, s in stays.iterrows():
        bed_days.loc[s["START"].normalize():s["STOP"].normalize()] += 1
    monthly_beds = bed_days.resample("MS").agg(["sum", "max"])
    month_days = monthly_beds.index.days_in_month
    bed_util = [{"period": i.strftime("%Y-%m"), "bed_days": float(r["sum"]), "peak_occupied": int(r["max"]),
                 "utilization_pct": float(r["sum"] / (beds * d) * 100)}
                for (i, r), d in zip(monthly_beds.iterrows(), month_days)]

    return to_jsonable({
        "heatmap": heatmap,
        "heatmap_note": "Synthea timestamps are simulated and stored in UTC; time-of-day patterns reflect the "
                        "simulator, not real clinic hours.",
        "demand_by_year": demand,
        "demand_growth": growth,
        "provider_workload": workload,
        "bed_utilization": bed_util[-36:],
        "bed_capacity_assumed": beds,
        "classes": sorted(enc["ENCOUNTERCLASS"].unique().tolist()),
    })


def scenarios(items: list[dict], start_year: int = 2016, hours_per_fte_month: float = 160.0) -> dict:
    """Compare what-if scenarios against the last 12 months of observed demand."""
    if not items:
        raise ValidationFailed("Provide at least one scenario.")
    enc = encounters(start_year)
    end = enc["START"].max()
    recent = enc[enc["START"] > end - pd.DateOffset(months=12)]
    clinic = recent[~recent["ENCOUNTERCLASS"].isin(LONG_STAY_CLASSES)]
    base_volume = len(clinic) / 12
    base_hours = float(clinic["DURATION_H"].median())
    base_cost = float(clinic["TOTAL_CLAIM_COST"].mean())
    out = []
    for item in items:
        demand_change = float(item.get("demand_change_pct", 0))
        staff = float(item.get("staff_fte", 1))
        minutes = item.get("minutes_per_encounter")
        hours = float(minutes) / 60 if minutes else base_hours
        if staff <= 0 or hours <= 0:
            raise ValidationFailed("staff_fte and minutes_per_encounter must be positive.")
        volume = base_volume * (1 + demand_change / 100)
        workload = volume * hours
        capacity = staff * hours_per_fte_month
        utilization = workload / capacity * 100
        out.append({
            "name": item.get("name", "Scenario"),
            "demand_change_pct": demand_change,
            "staff_fte": staff,
            "minutes_per_encounter": round(hours * 60, 1),
            "monthly_encounters": round(volume, 1),
            "workload_hours": round(workload, 1),
            "capacity_hours": round(capacity, 1),
            "utilization_pct": round(utilization, 1),
            "unmet_hours": round(max(workload - capacity, 0), 1),
            "required_fte_at_85pct": round(workload / (hours_per_fte_month * 0.85), 2),
            "projected_monthly_cost": round(volume * base_cost, 2),
            "status": "over capacity" if utilization > 100 else ("high" if utilization > 85 else "ok"),
        })
    return to_jsonable({
        "baseline": {"monthly_encounters": round(base_volume, 1), "hours_per_encounter": round(base_hours, 3),
                     "avg_cost_per_encounter": round(base_cost, 2), "window_end": end.strftime("%Y-%m-%d")},
        "scenarios": out,
        "assumptions": f"Baseline = clinic encounters (excluding inpatient/SNF/hospice) in the 12 months before "
                       f"{end:%Y-%m-%d}. One FTE = {hours_per_fte_month:g} working hours per month.",
    })
