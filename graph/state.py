"""state.py -- the shared object every agent node reads/writes.

LangGraph merges each node's partial-dict return into this state. No
custom reducers are used, so every agent copies a list/dict, updates
it, and returns the full new value (never mutates state in place).
"""
from __future__ import annotations

from typing import Any, TypedDict


class EDAState(TypedDict, total=False):
    # inputs / run config
    user_query: str
    use_llm: bool
    demo_drop_tasks: list[str]          # fault injection for live demos

    # planning
    schema: dict[str, Any]
    execution_plan: list[str]
    skipped_tasks: dict[str, str]       # task -> why it was not run
    completed_tasks: list[str]
    task_errors: dict[str, dict[str, Any]]  # task -> {"error", "retryable"}
    retry_count: int
    status: str

    # evidence (computed by tools, never by the LLM)
    tool_outputs: dict[str, Any]
    statistical_results: dict[str, Any]
    visualizations: list[dict[str, Any]]
    anomalies: dict[str, Any]

    # interpretation
    insights: list[dict[str, Any]]
    insight_mode: str                   # "llm" | "deterministic"
    insights_stale: bool                # evidence changed since last insight run
    insight_feedback: list[str]         # critic feedback passed to the LLM on retry

    # verification
    critic_feedback: dict[str, Any]
    critic_history: list[dict[str, Any]]

    # output
    final_report: str
    trace: list[dict[str, Any]]         # human-readable event log
