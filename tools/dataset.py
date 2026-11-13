"""dataset.py -- schema inspection. Operates purely in-memory on a
DataFrame Streamlit already loaded; no disk I/O."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

TARGET_KEYWORDS = ("revenue", "sales", "target", "label", "price", "amount", "profit", "income", "cost")
ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|\bid\b|uuid|identifier|index)", re.IGNORECASE)


def _is_id_like(name: str, series: pd.Series) -> bool:
    """Row identifiers carry no signal and pollute correlations/anomalies."""
    if ID_NAME_RE.search(name):
        return True
    if pd.api.types.is_integer_dtype(series) and len(series) > 20:
        s = series.dropna()
        return s.is_unique and (s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return False


def _normalize(name: str) -> str:
    return re.sub(r"[_\-]+", " ", name.lower()).strip()


def pick_target(numeric_cols: list[str], question: str = "") -> str | None:
    """Prefer a numeric column the user explicitly named; else a keyword match."""
    q = question.lower()
    named = [c for c in numeric_cols if _normalize(c) and _normalize(c) in q]
    if named:
        # the longest name is the most specific match ("unit price" over "price")
        return max(named, key=len)
    for kw in TARGET_KEYWORDS:
        for col in numeric_cols:
            if kw in col.lower():
                return col
    return None


def inspect_schema(df: pd.DataFrame, question: str = "") -> dict[str, Any]:
    numeric_cols, categorical_cols, datetime_cols, boolean_cols, text_cols, id_cols = [], [], [], [], [], []

    for col in df.columns:
        series = df[col]
        if _is_id_like(str(col), series):
            id_cols.append(col)
        elif pd.api.types.is_bool_dtype(series):
            boolean_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        else:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            parse_rate = parsed.notna().mean() if len(series) else 0
            looks_like_date = any(k in str(col).lower() for k in ("date", "time", "timestamp", "day", "month"))
            if parse_rate > 0.8 and (looks_like_date or parse_rate > 0.95):
                datetime_cols.append(col)
            else:
                nunique = series.nunique(dropna=True)
                if nunique <= max(50, int(0.05 * len(series))):
                    categorical_cols.append(col)
                else:
                    text_cols.append(col)

    return {
        "columns": list(df.columns),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "boolean_columns": boolean_cols,
        "text_columns": text_cols,
        "id_columns": id_cols,
        "likely_target_column": pick_target(numeric_cols, question),
    }


def best_group_column(df: pd.DataFrame, categorical_cols: list[str], max_levels: int = 20) -> str | None:
    """Most useful categorical for comparisons: 2..max_levels levels, fewest missing."""
    candidates = []
    for c in categorical_cols:
        n = df[c].nunique(dropna=True)
        if 2 <= n <= max_levels:
            candidates.append((df[c].isna().mean(), n, c))
    return sorted(candidates)[0][2] if candidates else None
