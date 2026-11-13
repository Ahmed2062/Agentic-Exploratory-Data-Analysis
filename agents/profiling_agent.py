"""profiling_agent.py -- pure computation, no LLM."""
from __future__ import annotations

import pandas as pd

from agents.base import add_trace, run_tasks
from graph.state import EDAState
from tools import profiling


def profiling_agent(state: EDAState, df: pd.DataFrame) -> dict:
    schema = state.get("schema", {})
    numeric_cols = schema.get("numeric_columns", [])

    def _profile() -> dict:
        return {
            "profile": profiling.profile_dataset(df),
            "missing_values": profiling.missing_value_report(df),
            "duplicates": profiling.duplicate_report(df),
            "descriptive_statistics": profiling.descriptive_statistics(df, numeric_cols),
            "data_quality_issues": profiling.data_quality_issues(df, schema),
        }

    results, completed, errors = run_tasks(state, "profiling", {"profiling": _profile})
    tool_outputs = dict(state.get("tool_outputs", {}))
    tool_outputs.update(results)
    n_issues = len(results.get("profiling", {}).get("data_quality_issues", []))
    return {
        "tool_outputs": tool_outputs,
        "completed_tasks": completed,
        "task_errors": errors,
        "insights_stale": True,
        "trace": add_trace(state, "profiling", f"Profiled dataset; {n_issues} data-quality issue(s).", ran=list(results)),
    }
