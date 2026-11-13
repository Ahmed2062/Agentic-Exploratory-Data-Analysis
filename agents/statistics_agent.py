"""statistics_agent.py -- pure computation, no LLM."""
from __future__ import annotations

import pandas as pd

from agents.base import add_trace, run_tasks
from graph.state import EDAState
from tools import statistics as stats_tools


def statistics_agent(state: EDAState, df: pd.DataFrame) -> dict:
    schema = state.get("schema", {})
    numeric_cols = schema.get("numeric_columns", [])
    categorical_cols = schema.get("categorical_columns", [])
    datetime_cols = schema.get("datetime_columns", [])
    target = schema.get("likely_target_column")
    focus = target or (numeric_cols[0] if numeric_cols else None)

    def _groupby() -> dict:
        """Test every usable grouping; report the one whose groups differ most.
        If the target is itself a class code (iris species, 0/1 churn), compare
        each numeric feature ACROSS the target's classes instead."""
        if focus in schema.get("discrete_numeric_columns", []):
            features = [c for c in numeric_cols if c != focus][:10]
            results = [stats_tools.groupby_analysis(df, focus, f) for f in features]
            results = [r for r in results if "error" not in r]
            if results:
                results.sort(key=lambda r: (r.get("difference_test") or {}).get("p_value", 1.0))
                best = results[0]
                best["other_groupings"] = [
                    {"group_col": f"{focus} (for {r['value_col']})", "p_value": (r.get("difference_test") or {}).get("p_value")}
                    for r in results[1:5]
                ]
                return best
        discrete = [c for c in schema.get("discrete_numeric_columns", []) if c != focus]
        candidates = [c for c in categorical_cols + discrete if 2 <= df[c].nunique(dropna=True) <= 20][:8]
        if not candidates:
            return {"error": "No categorical column with 2-20 levels to compare."}
        results = [stats_tools.groupby_analysis(df, c, focus) for c in candidates]
        results = [r for r in results if "error" not in r]
        if not results:
            return {"error": "Group comparison failed for every categorical column."}
        results.sort(key=lambda r: (r.get("difference_test") or {}).get("p_value", 1.0))
        best = results[0]
        best["other_groupings"] = [
            {"group_col": r["group_col"], "p_value": (r.get("difference_test") or {}).get("p_value")} for r in results[1:]
        ]
        return best

    handlers = {
        "correlation": lambda: stats_tools.correlation_analysis(df, numeric_cols),
        "groupby": _groupby,
        "time_series": lambda: stats_tools.time_series_analysis(df, datetime_cols[0], focus),
        "regression": lambda: stats_tools.regression_analysis(df, target, [c for c in numeric_cols if c != target]),
    }
    results, completed, errors = run_tasks(state, "statistics", handlers)

    statistical_results = dict(state.get("statistical_results", {}))
    statistical_results.update(results)
    failed = [t for t in errors if t in handlers and t not in state.get("task_errors", {})]
    msg = f"Ran {', '.join(results) or 'nothing'}" + (f"; failed: {', '.join(failed)}" if failed else "") + "."
    return {
        "statistical_results": statistical_results,
        "completed_tasks": completed,
        "task_errors": errors,
        "insights_stale": True,
        "trace": add_trace(state, "statistics", msg, ran=list(results), failed=failed),
    }
