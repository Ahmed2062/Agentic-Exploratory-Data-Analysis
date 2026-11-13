"""statistics.py -- correlation, group comparison, trend, and regression.
Every result reports association, never causation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MAX_CORR_COLUMNS = 15
ALPHA = 0.05


def _strength_label(r: float) -> str:
    a = abs(r)
    return "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "negligible"


def _sig(p: float) -> float:
    """Keep p-values readable but don't round tiny ones to 0."""
    return float(f"{p:.3g}")


def correlation_analysis(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    """Pairwise-complete Pearson + Spearman with Benjamini-Hochberg FDR correction."""
    cols = [c for c in numeric_columns if c in df.columns][:MAX_CORR_COLUMNS]
    if len(cols) < 2:
        return {"pairs": []}
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            sub = df[[a, b]].dropna()  # pairwise deletion, not listwise
            if len(sub) < 3 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
                continue
            r, p = scipy_stats.pearsonr(sub[a], sub[b])
            rho, _ = scipy_stats.spearmanr(sub[a], sub[b])
            # If Pearson and Spearman disagree, outliers or non-linearity are
            # distorting Pearson; the rank-based Spearman is the safer read.
            disagree = abs(float(rho) - float(r)) > 0.2
            pairs.append(
                {
                    "column_a": a,
                    "column_b": b,
                    "n": int(len(sub)),
                    "r": round(float(r), 4),
                    "spearman_rho": round(float(rho), 4),
                    "p_value": _sig(p),
                    "strength": _strength_label(rho if disagree else r),
                    "strength_basis": "spearman" if disagree else "pearson",
                    "nonlinear_or_outlier_driven": bool(disagree),
                }
            )
    if pairs:
        adjusted = scipy_stats.false_discovery_control([p["p_value"] for p in pairs], method="bh")
        for pair, p_adj in zip(pairs, adjusted):
            pair["p_adjusted"] = _sig(float(p_adj))
            pair["significant"] = bool(p_adj < ALPHA)
    pairs.sort(key=lambda x: max(abs(x["r"]), abs(x["spearman_rho"])), reverse=True)
    return {
        "pairs": pairs,
        "columns_used": cols,
        "note": "Correlation indicates association, not causation. p-values are Benjamini-Hochberg adjusted.",
    }


def groupby_analysis(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    if group_col not in df.columns or value_col not in df.columns:
        return {"error": f"Missing column(s): {group_col}, {value_col}"}
    sub = df[[group_col, value_col]].dropna()
    grouped = sub.groupby(group_col)[value_col].agg(["count", "mean", "median"]).reset_index()
    grouped = grouped.sort_values("mean", ascending=False)
    samples = [g[value_col].values for _, g in sub.groupby(group_col) if len(g) >= 2]
    test = None
    if len(samples) >= 2:
        h, p = scipy_stats.kruskal(*samples)  # rank-based: robust to skew/outliers
        test = {"test": "kruskal_wallis", "h_statistic": round(float(h), 4), "p_value": _sig(p), "significant": bool(p < ALPHA)}
    return {
        "group_col": group_col,
        "value_col": value_col,
        "groups": grouped.head(20).round(4).to_dict(orient="records"),
        "n_groups": int(len(grouped)),
        "difference_test": test,
    }


def time_series_analysis(df: pd.DataFrame, date_col: str, value_col: str) -> dict[str, Any]:
    if date_col not in df.columns or value_col not in df.columns:
        return {"error": f"Missing column(s): {date_col}, {value_col}"}
    sub = df[[date_col, value_col]].dropna().copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce", format="mixed")
    sub = sub.dropna(subset=[date_col]).sort_values(date_col)
    if sub.empty:
        return {"error": "No parseable dates."}

    span_days = (sub[date_col].max() - sub[date_col].min()).days
    freq, freq_label = ("ME", "monthly") if span_days > 180 else ("W", "weekly") if span_days > 21 else ("D", "daily")
    # median per period: a few extreme rows can't fake a trend
    series = sub.set_index(date_col)[value_col].resample(freq).median().dropna()
    if len(series) < 3:
        return {"trend": "insufficient_data", "date_col": date_col, "value_col": value_col, "periods": int(len(series))}

    lr = scipy_stats.linregress(np.arange(len(series)), series.values)
    if lr.pvalue >= ALPHA:
        trend = "no significant trend"
    else:
        trend = "increasing" if lr.slope > 0 else "decreasing"
    # change along the fitted line: consistent with the slope, unlike noisy first/last points
    fitted_start, fitted_end = lr.intercept, lr.intercept + lr.slope * (len(series) - 1)
    pct_change = round(float((fitted_end - fitted_start) / abs(fitted_start) * 100), 2) if fitted_start != 0 else None
    return {
        "date_col": date_col,
        "value_col": value_col,
        "frequency": freq_label,
        "periods": int(len(series)),
        "trend": trend,
        "slope_per_period": round(float(lr.slope), 4),
        "trend_p_value": _sig(lr.pvalue),
        "aggregation": "median per period",
        "pct_change_fitted": pct_change,
    }


def regression_analysis(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> dict[str, Any]:
    """Standardized linear regression with a holdout R^2 when there's enough data."""
    features = [c for c in feature_cols if c in df.columns and c != target_col]
    if target_col not in df.columns or not features:
        return {"error": "Target or feature columns not found."}
    sub = df[[target_col] + features].dropna()
    if len(sub) < max(20, 5 * len(features)):
        return {"error": f"Not enough complete rows ({len(sub)}) for a reliable regression on {len(features)} features."}

    X = StandardScaler().fit_transform(sub[features].values)
    y = sub[target_col].values
    result: dict[str, Any] = {"target_col": target_col, "feature_cols": features, "n": int(len(sub))}

    if len(sub) >= 50:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        model = LinearRegression().fit(X_tr, y_tr)
        result["r_squared_train"] = round(float(r2_score(y_tr, model.predict(X_tr))), 4)
        result["r_squared_test"] = round(float(r2_score(y_te, model.predict(X_te))), 4)
        result["validation"] = "25% holdout"
    else:
        model = LinearRegression().fit(X, y)
        result["r_squared_train"] = round(float(r2_score(y, model.predict(X))), 4)
        result["validation"] = "in-sample only (too few rows for holdout)"
    result["r_squared"] = result.get("r_squared_test", result["r_squared_train"])

    coefs = {f: round(float(c), 4) for f, c in zip(features, model.coef_)}
    result["standardized_coefficients"] = dict(sorted(coefs.items(), key=lambda kv: -abs(kv[1])))
    result["top_features"] = list(result["standardized_coefficients"])[:3]
    result["note"] = (
        "Coefficients are per 1 standard deviation of each feature, so magnitudes are comparable. "
        "They describe linear association, not causal effect."
    )
    return result
