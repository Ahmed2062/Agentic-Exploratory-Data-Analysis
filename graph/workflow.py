"""
workflow.py -- wires every agent into a LangGraph StateGraph.

    START -> dataset_inspector -> supervisor --(route_next)--> profiling / statistics /
             visualization / anomaly --(route_next)--> ... -> insight -> critic
    critic --(route_after_critic)--> report -> END
                                 \\-> replan -> supervisor  (targeted, capped by MAX_RETRIES)

The DataFrame is bound to tool nodes with functools.partial at build time
rather than stored in state: it isn't JSON-serializable (would break a
checkpointer) and copying it through every state update is wasteful.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Any, Iterator

import pandas as pd
from langgraph.graph import END, StateGraph

from agents.anomaly_agent import anomaly_agent
from agents.base import add_trace
from agents.critic_agent import critic_agent
from agents.insight_agent import insight_agent
from agents.profiling_agent import profiling_agent
from agents.report_agent import report_agent
from agents.statistics_agent import statistics_agent
from agents.supervisor import supervisor_agent
from agents.tasks import AGENT_ORDER
from agents.visualization_agent import visualization_agent
from graph.router import route_after_critic, route_next
from graph.state import EDAState
from tools import dataset as dataset_tools

RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "60"))
DISPATCH_TARGETS = {name: name for name in AGENT_ORDER + ["insight", "critic"]}


def _dataset_inspector_node(state: EDAState, df: pd.DataFrame) -> dict:
    schema = dataset_tools.inspect_schema(df, state.get("user_query", ""))
    msg = (
        f"{schema['shape']['rows']} rows; {len(schema['numeric_columns'])} numeric, "
        f"{len(schema['categorical_columns'])} categorical, {len(schema['datetime_columns'])} datetime; "
        f"target = {schema['likely_target_column']}."
    )
    return {"schema": schema, "trace": add_trace(state, "dataset_inspector", msg)}


def _replan_node(state: EDAState) -> dict:
    n = state.get("retry_count", 0) + 1
    return {"retry_count": n, "trace": add_trace({**state, "retry_count": n}, "replan", f"Starting retry round {n}.")}


def build_workflow(df: pd.DataFrame):
    graph = StateGraph(EDAState)

    graph.add_node("dataset_inspector", partial(_dataset_inspector_node, df=df))
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("profiling", partial(profiling_agent, df=df))
    graph.add_node("statistics", partial(statistics_agent, df=df))
    graph.add_node("visualization", partial(visualization_agent, df=df))
    graph.add_node("anomaly", partial(anomaly_agent, df=df))
    graph.add_node("insight", insight_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("replan", _replan_node)
    graph.add_node("report", report_agent)

    graph.set_entry_point("dataset_inspector")
    graph.add_edge("dataset_inspector", "supervisor")
    for node in ["supervisor", *AGENT_ORDER]:
        graph.add_conditional_edges(node, route_next, DISPATCH_TARGETS)
    graph.add_edge("insight", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"report": "report", "replan": "replan"})
    graph.add_edge("replan", "supervisor")
    graph.add_edge("report", END)
    return graph.compile()


def initial_state(user_query: str, use_llm: bool = True, demo_drop_tasks: list[str] | None = None) -> EDAState:
    return {
        "user_query": user_query,
        "use_llm": use_llm,
        "demo_drop_tasks": demo_drop_tasks or [],
        "retry_count": 0,
        "completed_tasks": [],
        "task_errors": {},
        "skipped_tasks": {},
        "tool_outputs": {},
        "statistical_results": {},
        "visualizations": [],
        "anomalies": {},
        "insights": [],
        "critic_history": [],
        "trace": [],
        "status": "planning",
    }


def stream_eda(df: pd.DataFrame, user_query: str, **kwargs: Any) -> Iterator[tuple[str, dict, EDAState]]:
    """Yields (node_name, node_update, merged_state) after every step, for live UIs.
    No reducers are defined, so merging is a plain dict update -- same as LangGraph's."""
    app = build_workflow(df)
    state = initial_state(user_query, **kwargs)
    for chunk in app.stream(state, config={"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
        for node, update in chunk.items():
            state = {**state, **(update or {})}
            yield node, update or {}, state


def run_eda(df: pd.DataFrame, user_query: str, **kwargs: Any) -> EDAState:
    state: EDAState = initial_state(user_query, **kwargs)
    for _, _, state in stream_eda(df, user_query, **kwargs):
        pass
    return state
