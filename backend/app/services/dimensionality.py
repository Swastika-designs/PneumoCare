"""T13: dimensionality reduction (PCA) for exploratory data analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.core.errors import ValidationFailed
from app.core.utils import to_jsonable

PCA_SOURCES = {
    "nhanes_adults": "NHANES 2017-2018 adults - body measures and lifestyle",
    "synthea:patient_profile": "Synthea patient profile - vitals, labs and utilisation",
}


def load_pca_source(source: str, db=None) -> tuple[pd.DataFrame, list[str]]:
    """Return the frame and the categorical columns available for colouring the scatter plot."""
    if source == "nhanes_adults":
        from app.services.lifestyle import adults

        df = adults().drop(columns=["seqn", "exam_weight", "obese"])
        return df, ["bmi_category", "sex", "smoking_status", "education", "race_ethnicity"]
    from app.services.ehr_data import load_source

    df = load_source(source, db)
    cats = [c for c in df.columns if df[c].dtype == object and 1 < df[c].nunique() <= 12]
    return df, cats


def run_pca(df: pd.DataFrame, color_options: list[str], scaling: str = "standard", n_components: int = 5,
            missing: str = "impute", color_by: str | None = None, columns: list[str] | None = None,
            max_points: int = 1500, seed: int = 0) -> dict:
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.nunique() > 2]
    if columns:
        unknown = [c for c in columns if c not in numeric.columns]
        if unknown:
            raise ValidationFailed(f"Not numeric or unknown columns: {', '.join(unknown)}")
        numeric = numeric[columns]
    numeric = numeric.loc[:, numeric.notna().mean() >= 0.3]
    if numeric.shape[1] < 3:
        raise ValidationFailed("PCA needs at least 3 numeric columns with >= 30% observed values.")
    if missing == "drop":
        numeric = numeric.dropna()
    elif missing == "impute":
        numeric = numeric.fillna(numeric.median())
    else:
        raise ValidationFailed("missing must be 'drop' or 'impute'")
    if len(numeric) < 10:
        raise ValidationFailed("Too few complete rows for PCA; try missing='impute'.")
    if scaling == "standard":
        x = StandardScaler().fit_transform(numeric)
    elif scaling == "minmax":
        x = MinMaxScaler().fit_transform(numeric)
    elif scaling == "none":
        x = numeric.values - numeric.values.mean(axis=0)
    else:
        raise ValidationFailed("scaling must be 'standard', 'minmax' or 'none'")
    k = int(min(max(n_components, 2), numeric.shape[1], len(numeric)))
    full = PCA().fit(x)
    pca = PCA(n_components=max(k, 3) if numeric.shape[1] >= 3 else k).fit(x)
    scores = pca.transform(x)
    color = None
    if color_by and color_by in df.columns:
        color = df.loc[numeric.index, color_by].astype(str).values
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(scores), size=min(max_points, len(scores)), replace=False)
    points = [{"pc1": float(scores[i, 0]), "pc2": float(scores[i, 1]), "pc3": float(scores[i, 2]),
               "group": color[i] if color is not None else "all"} for i in idx]
    loadings = [{"feature": f, **{f"pc{j + 1}": float(pca.components_[j, i]) for j in range(min(k, 5))}}
                for i, f in enumerate(numeric.columns)]
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_80 = int(np.searchsorted(cumulative, 0.8) + 1)
    top_features = {}
    for j in range(min(k, 3)):
        order = np.argsort(np.abs(pca.components_[j]))[::-1][:4]
        top_features[f"PC{j + 1}"] = [{"feature": numeric.columns[i], "loading": float(pca.components_[j, i])}
                                      for i in order]
    corr = numeric.corr()
    return to_jsonable({
        "rows_used": len(numeric),
        "features": list(numeric.columns),
        "scaling": scaling,
        "missing_strategy": missing,
        "explained_variance": [{"component": f"PC{i + 1}", "ratio": float(r), "cumulative": float(c)}
                               for i, (r, c) in enumerate(zip(full.explained_variance_ratio_, cumulative))],
        "components_for_80pct": n_80,
        "loadings": loadings,
        "top_features": top_features,
        "points": points,
        "color_by": color_by if color is not None else None,
        "color_options": color_options,
        "correlation": {"columns": list(corr.columns), "values": corr.values.round(3).tolist()},
        "interpretation": f"{n_80} of {numeric.shape[1]} components explain 80% of the variance after "
                          f"{scaling} scaling.",
    })
