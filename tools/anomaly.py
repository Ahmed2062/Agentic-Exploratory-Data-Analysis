"""anomaly.py -- IQR, robust (median/MAD) Z-score, and Isolation Forest,
plus a consensus view of rows flagged by multiple methods."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MAX_INDICES = 50


def _py(i: Any) -> Any:
    return int(i) if isinstance(i, (int, np.integer)) else i


def _result(method: str, df: pd.DataFrame, idx: list, **extra: Any) -> dict[str, Any]:
    return {
        "method": method,
        "outlier_count": len(idx),
        "outlier_pct": round(len(idx) / len(df) * 100, 2) if len(df) else 0.0,
        "outlier_indices": [_py(i) for i in idx[:MAX_INDICES]],
        "all_indices": [_py(i) for i in idx],
        **extra,
    }


def detect_iqr_outliers(df: pd.DataFrame, column: str, k: float = 1.5) -> dict[str, Any]:
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    series = df[column].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    idx = df[(df[column] < lower) | (df[column] > upper)].index.tolist()
    return _result(
        "iqr", df, idx, column=column, k=k,
        lower_bound=round(float(lower), 4), upper_bound=round(float(upper), 4),
        explanation=f"Values of {column} outside [{round(float(lower), 2)}, {round(float(upper), 2)}] (1.5x IQR) are flagged.",
    )


def iqr_summary(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, int]:
    """Outlier count per numeric column -- a quick map of where the oddities live."""
    out = {}
    for col in numeric_columns:
        r = detect_iqr_outliers(df, col)
        if "error" not in r:
            out[col] = r["outlier_count"]
    return out


def detect_zscore_outliers(df: pd.DataFrame, column: str, threshold: float = 3.5) -> dict[str, Any]:
    """Modified Z-score (Iglewicz & Hoaglin): uses median and MAD, which the
    outliers themselves can't inflate -- avoids the 'masking' problem of mean/std."""
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    series = df[column].dropna()
    if series.empty:
        return _result("robust_zscore", df, [], column=column, explanation="No data.")
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return _result("robust_zscore", df, [], column=column, explanation="MAD is zero; the column is near-constant.")
    mz = 0.6745 * (series - median) / mad
    idx = series[mz.abs() > threshold].index.tolist()
    return _result(
        "robust_zscore", df, idx, column=column, threshold=threshold,
        explanation=f"Values of {column} with modified Z-score above {threshold} (median/MAD based) are flagged.",
    )


def detect_isolation_forest_anomalies(
    df: pd.DataFrame, numeric_columns: list[str], contamination: float = 0.02
) -> dict[str, Any]:
    cols = [c for c in numeric_columns if c in df.columns]
    if len(cols) < 2:
        return {"error": "Isolation Forest needs at least 2 numeric columns."}
    sub = df[cols].dropna()
    if len(sub) < 20:
        return {"error": "Not enough complete rows for a reliable fit."}
    X = StandardScaler().fit_transform(sub.values)  # equal weight regardless of units
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    preds = model.fit_predict(X)
    scores = -model.score_samples(X)
    flagged = sub.index[preds == -1]
    order = np.argsort(-scores[preds == -1])
    idx = flagged[order].tolist()  # most anomalous first
    return _result(
        "isolation_forest", df, idx, columns_used=cols,
        contamination=contamination,
        explanation=(
            f"The {round(contamination * 100, 2)}% most isolated rows in the combined space of {', '.join(cols)} "
            "(features scaled). This flags a fixed share by design, so treat it as a ranking and confirm with consensus."
        ),
    )


def consensus_anomalies(results: dict[str, dict[str, Any]], n_rows: int, min_methods: int = 2) -> dict[str, Any]:
    """Rows flagged by >= min_methods independent methods -- the ones worth a human look."""
    usable = {m: r for m, r in results.items() if "outlier_indices" in r}
    if len(usable) < min_methods:
        return {"error": f"Need at least {min_methods} anomaly methods for consensus."}
    counts = Counter(i for r in usable.values() for i in set(r.get("all_indices", r["outlier_indices"])))
    idx = sorted(i for i, c in counts.items() if c >= min_methods)
    return {
        "method": "consensus",
        "methods_compared": list(usable),
        "min_methods": min_methods,
        "outlier_count": len(idx),
        "outlier_pct": round(len(idx) / n_rows * 100, 2) if n_rows else 0.0,
        "outlier_indices": idx[:MAX_INDICES],
        "all_indices": idx,
        "explanation": f"Rows flagged by at least {min_methods} of: {', '.join(usable)}.",
    }
