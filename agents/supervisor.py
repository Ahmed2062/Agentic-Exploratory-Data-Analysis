"""supervisor.py -- plans the analysis, and re-plans from Critic feedback.

First pass: a rule-based plan from schema + question intent (always
works, even with no LLM), plus an optional LLM pass that may only ADD
whitelisted, feasible tasks.

Re-plan pass: acts on the Critic's structured feedback --
  * add tasks the evidence was missing (coverage gaps),
  * re-queue tasks that failed for retryable reasons,
  * mark insights stale (with feedback) so only the Insight agent re-runs.
Successful work is kept, so a retry re-runs only what's needed.
"""
from __future__ import annotations

from agents.base import add_trace
from agents.llm import safe_json_call
from agents.tasks import ALL_TASKS, detect_intents, is_feasible
from graph.state import EDAState


def rule_based_plan(schema: dict, user_query: str) -> list[str]:
    intents = set(detect_intents(user_query))
    candidates = ["profiling", "correlation", "groupby", "time_series"]
    if "drivers" in intents:
        candidates.append("regression")
    candidates.append("visualization")
    if "anomaly" in intents:
        candidates += ["anomaly_iqr", "anomaly_zscore", "anomaly_isolation_forest"]
    else:
        candidates.append("anomaly_iqr")  # a cheap sanity check on every run
    return [t for t in dict.fromkeys(candidates) if is_feasible(t, schema)[0]]


def _initial_plan(state: EDAState) -> dict:
    schema = state.get("schema", {})
    user_query = state.get("user_query", "")
    plan = rule_based_plan(schema, user_query)
    skipped: dict[str, str] = {}

    system = (
        "You are the Supervisor of a data-analysis agent team. "
        f"Valid tasks are exactly: {ALL_TASKS}. Return JSON: "
        '{"additional_tasks": [...]} using only tasks NOT already planned '
        "that would materially help answer the question. Empty list if the plan is sufficient."
    )
    user = f"Schema: {schema}\nQuestion: {user_query}\nAlready planned: {plan}"
    result = safe_json_call(system, user, fallback={"additional_tasks": []}, enabled=state.get("use_llm", True))
    suggested = result.get("additional_tasks", [])
    added = []
    for t in suggested if isinstance(suggested, list) else []:
        if t not in ALL_TASKS or t in plan:
            continue
        ok, reason = is_feasible(t, schema)
        if ok:
            plan.append(t)
            added.append(t)
        else:
            skipped[t] = f"suggested by LLM planner, but {reason}"

    trace = add_trace(state, "supervisor", f"Initial plan: {', '.join(plan)}.", plan=list(plan), llm_added=added)

    # Fault injection for live demos: simulate a planner that missed tasks,
    # so the Critic's coverage check and the re-plan loop can be shown.
    dropped = [t for t in state.get("demo_drop_tasks", []) if t in plan]
    if dropped:
        plan = [t for t in plan if t not in dropped]
        trace.append({"round": 0, "node": "supervisor", "message": f"[demo fault] dropped from plan: {', '.join(dropped)}."})

    return {
        "execution_plan": plan,
        "skipped_tasks": skipped,
        "completed_tasks": [],
        "task_errors": {},
        "retry_count": 0,
        "insights_stale": True,
        "insight_feedback": [],
        "status": "running",
        "trace": trace,
    }


def _replan(state: EDAState) -> dict:
    feedback = state.get("critic_feedback", {})
    schema = state.get("schema", {})
    plan = list(state.get("execution_plan", []))
    errors = dict(state.get("task_errors", {}))
    actions = []

    to_add = [t for t in feedback.get("missing_analysis", []) if t in ALL_TASKS and t not in plan and is_feasible(t, schema)[0]]
    if to_add:
        plan += to_add
        actions.append(f"added {', '.join(to_add)}")

    retry = [t for t in feedback.get("retry_tasks", []) if t in errors]
    for t in retry:
        errors.pop(t)
    if retry:
        actions.append(f"re-queued {', '.join(retry)}")

    update: dict = {"execution_plan": plan, "task_errors": errors, "status": "running"}
    if feedback.get("regenerate_insights"):
        update["insights_stale"] = True
        update["insight_feedback"] = [i["message"] for i in feedback.get("issues", []) if i.get("kind") in ("ungrounded_number", "causal_language", "no_insights")]
        actions.append("asked Insight agent to regenerate with feedback")

    update["trace"] = add_trace(state, "supervisor", "Re-plan: " + ("; ".join(actions) or "no actionable change") + ".")
    return update


def supervisor_agent(state: EDAState) -> dict:
    return _initial_plan(state) if state.get("retry_count", 0) == 0 and not state.get("execution_plan") else _replan(state)
