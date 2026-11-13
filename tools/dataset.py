"""dataset.py -- schema inspection. Operates purely in-memory on a
DataFrame Streamlit already loaded; no disk I/O."""
from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import pandas as pd

DATE_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y",
    "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    "%Y%m%d", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y", "%Y-%m",
]
CURRENCY_RE = re.compile(r"[\s$€£¥₹,%]|^\((?=.*\)$)|\)$")


def load_table(data: bytes, filename: str) -> pd.DataFrame:
    """Read CSV/TSV/TXT/XLSX robustly: sniff the delimiter (handles ';' CSVs)
    and fall back through common encodings instead of failing on the first."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(data))
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        try:
            return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001 -- sniffer can fail on one-column files
            last_error = exc
            try:
                return pd.read_csv(io.StringIO(text))
            except Exception as exc2:  # noqa: BLE001
                last_error = exc2
    raise ValueError(f"Could not read {filename}: {last_error}")


def _best_date_format(series: pd.Series) -> tuple[str | None, float]:
    """Pick ONE format for the whole column. Parsing element-by-element
    ('mixed') silently reads 05/01/2023 as May 1 and 15/01/2023 as Jan 15."""
    sample = series.dropna().astype(str).head(500)
    if sample.empty:
        return None, 0.0
    best, best_rate = None, 0.0
    for fmt in DATE_FORMATS:
        rate = pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean()
        if rate > best_rate:
            best, best_rate = fmt, rate
    return best, float(best_rate)


def _decimal_comma(values: pd.Series) -> bool:
    """True if numbers use ',' as the decimal mark (1.234,50 / 12,5)."""
    sample = values.head(500)
    both = sample[sample.str.contains(r"\.", regex=True) & sample.str.contains(",", regex=False)]
    if len(both):
        return (both.str.rfind(",") > both.str.rfind(".")).mean() > 0.5
    comma_only = sample.str.fullmatch(r"\D*-?\d+,\d{1,2}\D*")
    return bool(comma_only.mean() > 0.8)


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fix the things real-world CSVs get wrong, and say what was changed."""
    df = df.copy()
    notes: list[str] = []
    df.columns = [str(c).strip() for c in df.columns]

    numeric = df.select_dtypes(include="number")
    n_inf = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if not numeric.empty else 0
    if n_inf:
        df[numeric.columns] = numeric.replace([np.inf, -np.inf], np.nan)
        notes.append(f"{n_inf} infinite values were treated as missing.")

    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        non_null = s.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        if non_null.empty:
            continue
        # numbers stored as text: "1,234.50", "$12.50", "45%", "(300)" for negatives,
        # and European decimal commas: "1.234,50" or "12,5"
        if _decimal_comma(non_null):
            non_null = non_null.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        cleaned = non_null.str.replace(CURRENCY_RE, "", regex=True)
        negative = non_null.str.match(r"^\(.*\)$")
        as_num = pd.to_numeric(cleaned, errors="coerce")
        if as_num.notna().mean() >= 0.95 and not non_null.str.fullmatch(r"\d{4}-\d{2}(-\d{2})?.*").any():
            as_num[negative] = -as_num[negative]
            converted = pd.Series(np.nan, index=s.index)
            converted[as_num.index] = as_num
            df[col] = converted
            notes.append(f"Column '{col}' contained numbers stored as text and was converted to numeric.")
            continue
        fmt, rate = _best_date_format(non_null)
        if fmt and rate >= 0.9:
            df[col] = pd.to_datetime(s.astype(str).str.strip(), format=fmt, errors="coerce")
            if fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y") and rate < 1.0:
                notes.append(f"Column '{col}' parsed as dates using {fmt}; {round((1 - rate) * 100, 2)}% of values did not match.")
    return df, notes

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


def _mentioned(col: str, question: str) -> bool:
    """Whole-word match ('a' must not match inside 'are'), plus a stem match so
    'survival' finds 'Survived' and 'prices' finds 'price'."""
    name = _normalize(col)
    if not name:
        return False
    if re.search(rf"\b{re.escape(name)}\b", question):
        return True
    words = re.findall(r"[a-z]{5,}", question)
    return len(name) >= 5 and " " not in name and any(w[:5] == name[:5] and abs(len(w) - len(name)) <= 3 for w in words)


def pick_target(numeric_cols: list[str], question: str = "") -> str | None:
    """Prefer a numeric column the user explicitly named; else a keyword match."""
    q = question.lower()
    named = [c for c in numeric_cols if _mentioned(c, q)]
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

    # integer codes with few levels (Pclass, rating 1-5, 0/1 flags) are useful as groups
    discrete = [
        c for c in numeric_cols
        if 2 <= df[c].nunique(dropna=True) <= 10 and (df[c].dropna() % 1 == 0).all()
    ]
    return {
        "columns": list(df.columns),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "boolean_columns": boolean_cols,
        "text_columns": text_cols,
        "id_columns": id_cols,
        "discrete_numeric_columns": discrete,
        "discrete_levels": {c: int(df[c].nunique(dropna=True)) for c in discrete},
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
