"""anomaly_agent.py -- runs whichever anomaly-detection methods were
planned, then computes a consensus set. Pure computation, no LLM."""
from __future__ import annotations

import pandas as pd

from agents.base import add_trace, run_tasks
from graph.state import EDAState
from tools import anomaly as anomaly_tools


def anomaly_agent(state: EDAState, df: pd.DataFrame) -> dict:
    schema = state.get("schema", {})
    numeric_cols = schema.get("numeric_columns", [])
    target = schema.get("likely_target_column")
    focus_col = target or (numeric_cols[0] if numeric_cols else None)

    handlers = {
        "anomaly_iqr": lambda: anomaly_tools.detect_iqr_outliers(df, focus_col),
        "anomaly_zscore": lambda: anomaly_tools.detect_zscore_outliers(df, focus_col),
        "anomaly_isolation_forest": lambda: anomaly_tools.detect_isolation_forest_anomalies(df, numeric_cols),
    }
    results, completed, errors = run_tasks(state, "anomaly", handlers)

    anomalies = dict(state.get("anomalies", {}))
    for task, res in results.items():
        anomalies[task.removeprefix("anomaly_")] = res
    if results and numeric_cols:
        anomalies["iqr_by_column"] = anomaly_tools.iqr_summary(df, numeric_cols)
    method_results = {k: v for k, v in anomalies.items() if k in ("iqr", "zscore", "isolation_forest")}
    if len(method_results) >= 2:
        anomalies["consensus"] = anomaly_tools.consensus_anomalies(method_results, n_rows=len(df))

    counts = ", ".join(f"{k}={v['outlier_count']}" for k, v in anomalies.items() if isinstance(v, dict) and "outlier_count" in v)
    return {
        "anomalies": anomalies,
        "completed_tasks": completed,
        "task_errors": errors,
        "insights_stale": True,
        "trace": add_trace(state, "anomaly", f"Anomaly counts: {counts or 'none'}.", ran=list(results)),
    }
