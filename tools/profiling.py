"""profiling.py -- deterministic profiling/data-quality tools. Every
number here is computed, never generated."""
from __future__ import annotations

from typing import Any

import pandas as pd

MISSING_WARN_PCT = 5.0


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def missing_value_report(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    missing = df.isna().sum()
    report = {
        col: {
            "missing_count": int(missing[col]),
            "missing_pct": round(float(missing[col]) / total * 100, 2) if total else 0.0,
        }
        for col in df.columns
        if missing[col] > 0
    }
    return {"columns_with_missing": report, "any_missing": bool(missing.sum() > 0)}


def duplicate_report(df: pd.DataFrame) -> dict[str, Any]:
    dup_mask = df.duplicated(keep="first")
    return {
        "duplicate_row_count": int(dup_mask.sum()),
        "duplicate_pct": round(float(dup_mask.mean()) * 100, 2) if len(df) else 0.0,
    }


def descriptive_statistics(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for col in numeric_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4) if series.count() > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "median": round(float(series.median()), 4),
            "max": round(float(series.max()), 4),
            "skew": round(float(series.skew()), 4) if series.count() > 2 else 0.0,
        }
    return stats


def data_quality_issues(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    missing = missing_value_report(df)
    for col, info in sorted(missing["columns_with_missing"].items(), key=lambda kv: -kv[1]["missing_pct"]):
        if info["missing_pct"] >= MISSING_WARN_PCT or len(issues) == 0:
            issues.append(f"Column '{col}' has {info['missing_pct']}% missing values.")
    dup = duplicate_report(df)
    if dup["duplicate_row_count"] > 0:
        issues.append(f"{dup['duplicate_row_count']} duplicate rows detected ({dup['duplicate_pct']}%).")
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            issues.append(f"Column '{col}' is constant and carries no information.")
    for col in schema.get("numeric_columns", []):
        s = df[col].dropna()
        if len(s) > 2 and abs(float(s.skew())) > 2:
            issues.append(f"Column '{col}' is highly skewed (skew {round(float(s.skew()), 4)}); prefer medians and rank-based tests.")
    if schema.get("id_columns"):
        issues.append(f"Identifier columns excluded from analysis: {', '.join(map(str, schema['id_columns']))}.")
    if df.shape[0] < 30:
        issues.append("Dataset has fewer than 30 rows -- statistical tests will have low power.")
    return issues
