"""visualization.py -- chart tools returning matplotlib Figure objects.

Uses the object-oriented `matplotlib.figure.Figure` API rather than
pyplot: pyplot keeps global state, which leaks memory across reruns and
is not safe when several Streamlit sessions render at once.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from matplotlib.figure import Figure

SCATTER_MAX_POINTS = 5000
MAX_BOX_GROUPS = 12


def _sample(df: pd.DataFrame, n: int = SCATTER_MAX_POINTS) -> pd.DataFrame:
    return df.sample(n, random_state=42) if len(df) > n else df


def create_histogram(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.hist(df[column].dropna(), bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return {"chart_type": "histogram", "figure": fig, "description": f"Histogram of {column}."}


def create_boxplot(df: pd.DataFrame, value_col: str, group_col: str | None = None) -> dict[str, Any]:
    if value_col not in df.columns:
        return {"error": f"Column not found: {value_col}"}
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    if group_col and group_col in df.columns:
        sub = df[[group_col, value_col]].dropna()
        top = sub[group_col].value_counts().head(MAX_BOX_GROUPS).index
        groups = [(str(g), sub.loc[sub[group_col] == g, value_col].values) for g in top]
        ax.boxplot([v for _, v in groups], tick_labels=[g for g, _ in groups])
        ax.set_title(f"{value_col} by {group_col}")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.boxplot(df[value_col].dropna().values)
        ax.set_title(f"Boxplot of {value_col}")
    ax.set_ylabel(value_col)
    fig.tight_layout()
    return {
        "chart_type": "boxplot",
        "figure": fig,
        "description": f"Boxplot of {value_col}" + (f" by {group_col}." if group_col else "."),
    }


def create_scatterplot(
    df: pd.DataFrame, x_col: str, y_col: str, highlight_index: list | None = None
) -> dict[str, Any]:
    if x_col not in df.columns or y_col not in df.columns:
        return {"error": f"Missing column(s): {x_col}, {y_col}"}
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    sub = _sample(df[[x_col, y_col]].dropna())
    ax.scatter(sub[x_col], sub[y_col], alpha=0.4, s=12, color="#DD8452", label="rows")
    if highlight_index:
        hl = df.loc[[i for i in highlight_index if i in df.index], [x_col, y_col]].dropna()
        if not hl.empty:
            ax.scatter(hl[x_col], hl[y_col], s=40, facecolors="none", edgecolors="#C44E52", linewidths=1.5, label="consensus anomalies")
            ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    note = " Sampled to 5,000 points." if len(df) > SCATTER_MAX_POINTS else ""
    return {"chart_type": "scatterplot", "figure": fig, "description": f"{y_col} vs {x_col}.{note}"}


def create_heatmap(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    cols = [c for c in numeric_columns if c in df.columns][:15]
    if len(cols) < 2:
        return {"error": "Need at least 2 numeric columns."}
    corr = df[cols].corr()
    size = 1.5 + len(cols) * 0.7
    fig = Figure(figsize=(size, size))
    ax = fig.subplots()
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation heatmap (Pearson)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return {"chart_type": "heatmap", "figure": fig, "description": "Pairwise Pearson correlations."}


def create_time_series_plot(df: pd.DataFrame, date_col: str, value_col: str) -> dict[str, Any]:
    if date_col not in df.columns or value_col not in df.columns:
        return {"error": f"Missing column(s): {date_col}, {value_col}"}
    sub = df[[date_col, value_col]].dropna().copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce", format="mixed")
    sub = sub.dropna(subset=[date_col]).sort_values(date_col)
    if sub.empty:
        return {"error": "No parseable dates."}
    span_days = (sub[date_col].max() - sub[date_col].min()).days
    freq, label = ("W", "weekly") if span_days > 60 else ("D", "daily")
    agg = sub.set_index(date_col)[value_col].resample(freq).mean().dropna()
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    ax.plot(agg.index, agg.values, color="#55A868")
    ax.set_title(f"{value_col} over time ({label} mean)")
    ax.set_ylabel(value_col)
    fig.autofmt_xdate()
    fig.tight_layout()
    return {"chart_type": "time_series", "figure": fig, "description": f"{label.capitalize()} mean of {value_col} over {date_col}."}
