"""visualization_agent.py -- picks a small, relevant set of charts
rather than plotting every column (avoids chart spam). Runs after the
statistics agent so it can plot the strongest relationship it found."""
from __future__ import annotations

import pandas as pd

from agents.base import add_trace, run_tasks
from graph.state import EDAState
from tools import visualization as viz_tools
from tools.dataset import best_group_column


def visualization_agent(state: EDAState, df: pd.DataFrame) -> dict:
    schema = state.get("schema", {})
    numeric_cols = schema.get("numeric_columns", [])
    categorical_cols = schema.get("categorical_columns", [])
    datetime_cols = schema.get("datetime_columns", [])
    target = schema.get("likely_target_column")
    focus = target or (numeric_cols[0] if numeric_cols else None)

    def _charts() -> dict:
        charts = []
        if focus:
            charts.append(viz_tools.create_histogram(df, focus))
        # plot exactly the comparison the statistics agent chose
        gb = state.get("statistical_results", {}).get("groupby") or {}
        group_col = gb.get("group_col") or best_group_column(df, categorical_cols)
        value_col = gb.get("value_col") or focus
        if group_col and value_col and group_col != value_col:
            charts.append(viz_tools.create_boxplot(df, value_col, group_col))
        if len(numeric_cols) >= 2:
            charts.append(viz_tools.create_heatmap(df, numeric_cols))

        pairs = state.get("statistical_results", {}).get("correlation", {}).get("pairs", [])
        # pairs involving the target are more relevant to the question than the overall top pair
        target_pairs = [p for p in pairs if focus in (p["column_a"], p["column_b"])]
        top = (target_pairs or pairs or [None])[0]
        if top:
            charts.append(viz_tools.create_scatterplot(df, top["column_a"], top["column_b"]))
        elif len(numeric_cols) >= 2:
            charts.append(viz_tools.create_scatterplot(df, numeric_cols[0], numeric_cols[1]))

        if datetime_cols and focus:
            charts.append(viz_tools.create_time_series_plot(df, datetime_cols[0], focus))
        charts = [c for c in charts if "error" not in c]
        if not charts:
            return {"error": "No chart could be produced for this schema."}
        return {"charts": charts}

    results, completed, errors = run_tasks(state, "visualization", {"visualization": _charts})
    charts = results.get("visualization", {}).get("charts", [])
    # replace, never append: re-running must not duplicate charts
    visualizations = charts if charts else list(state.get("visualizations", []))
    return {
        "visualizations": visualizations,
        "completed_tasks": completed,
        "task_errors": errors,
        "trace": add_trace(state, "visualization", f"Produced {len(charts)} chart(s)."),
    }
