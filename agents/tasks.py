"""tasks.py -- single source of truth for what tasks exist, which agent
owns each one, whether a dataset can support it, and which tasks a
question's intent requires.

The Supervisor uses it to build and validate plans, the dispatcher uses
it to route to the next agent, and the Critic uses it to check that the
evidence actually covers what the user asked.
"""
from __future__ import annotations

import re
from typing import Any

# Task -> agent node that executes it. Order of AGENT_ORDER encodes data
# dependencies (visualization reads correlation results; insight reads all).
TASK_AGENT = {
    "profiling": "profiling",
    "correlation": "statistics",
    "groupby": "statistics",
    "time_series": "statistics",
    "regression": "statistics",
    "visualization": "visualization",
    "anomaly_iqr": "anomaly",
    "anomaly_zscore": "anomaly",
    "anomaly_isolation_forest": "anomaly",
}
ALL_TASKS = list(TASK_AGENT)
AGENT_ORDER = ["profiling", "statistics", "visualization", "anomaly"]


def is_feasible(task: str, schema: dict[str, Any]) -> tuple[bool, str]:
    """Can this dataset support the task at all? Returns (ok, reason)."""
    numeric = schema.get("numeric_columns", [])
    categorical = schema.get("categorical_columns", [])
    datetime_cols = schema.get("datetime_columns", [])
    rows = schema.get("shape", {}).get("rows", 0)
    target = schema.get("likely_target_column")

    if task not in TASK_AGENT:
        return False, f"unknown task '{task}'"
    if task == "correlation" and len(numeric) < 2:
        return False, "needs at least 2 numeric columns"
    discrete = schema.get("discrete_numeric_columns", [])
    if task == "groupby" and not ((categorical or len(discrete) >= 2 or (discrete and len(numeric) >= 2)) and numeric):
        return False, "needs a categorical (or low-cardinality integer) column and a numeric column"
    if task == "time_series" and not (datetime_cols and numeric):
        return False, "needs a datetime column and a numeric column"
    if task == "regression":
        if not target:
            return False, "no target column could be identified"
        if len(numeric) < 2:
            return False, "needs a target plus at least 1 numeric feature"
        levels = schema.get("discrete_levels", {}).get(target)
        if levels is not None and levels > 2:
            return False, f"target '{target}' is a class code with {levels} levels; linear regression would treat categories as quantities"
    if task == "visualization" and not (numeric or datetime_cols):
        return False, "no numeric or datetime columns to plot"
    if task in ("anomaly_iqr", "anomaly_zscore") and not numeric:
        return False, "needs a numeric column"
    if task == "anomaly_isolation_forest":
        if len(numeric) < 2:
            return False, "needs at least 2 numeric columns"
        if rows < 20:
            return False, "needs at least 20 rows"
    return True, ""


# Question intent -> tasks whose evidence answers it. Shared by the
# planner (to plan) and the Critic (to verify the evidence exists).
INTENTS: dict[str, dict[str, Any]] = {
    "anomaly": {
        "pattern": r"anomal|outlier|unusual|abnormal|suspicious|weird|odd\b",
        "tasks": ["anomaly_iqr", "anomaly_zscore", "anomaly_isolation_forest"],
    },
    "trend": {
        "pattern": r"trend|over time|growth|growing|declin|season|monthly|weekly|forecast",
        "tasks": ["time_series"],
    },
    "drivers": {
        "pattern": r"driv|factor|cause|predict|influenc|impact|explain|affect",
        "tasks": ["regression", "correlation"],
    },
    "relationship": {
        "pattern": r"correlat|relat|associat|depend",
        "tasks": ["correlation"],
    },
    "segments": {
        "pattern": r"\bby\b|compare|comparison|segment|which (region|group|category|channel)|across|per\b",
        "tasks": ["groupby"],
    },
}


def detect_intents(question: str) -> list[str]:
    q = question.lower()
    return [name for name, spec in INTENTS.items() if re.search(spec["pattern"], q)]


def required_tasks_for(question: str, schema: dict[str, Any]) -> dict[str, list[str]]:
    """intent -> feasible tasks that would provide evidence for it."""
    out: dict[str, list[str]] = {}
    for intent in detect_intents(question):
        feasible = [t for t in INTENTS[intent]["tasks"] if is_feasible(t, schema)[0]]
        if feasible:
            out[intent] = feasible
    return out
