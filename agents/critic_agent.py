"""critic_agent.py -- deterministic verification (the safety net must not
depend on the component it's checking, so there is no LLM here).

Each issue has a severity: "blocking" issues fail the run and trigger a
targeted re-plan; "warning" issues can't be fixed by retrying and are
surfaced in the report's Limitations section instead.

Checks:
  1. Plan sanity and task completion (retryable vs. permanent failures)
  2. Coverage: does the evidence answer every intent in the question?
  3. Evidence exists at all; sample size is adequate
  4. Numeric grounding: every number in an insight exists in tool output
  5. No unsupported causal language
"""
from __future__ import annotations

import os

from agents.base import add_trace, pending_tasks
from agents.tasks import AGENT_ORDER, INTENTS, detect_intents, is_feasible
from agents.verification import causal_phrases, collect_evidence_numbers, ungrounded_numbers
from graph.state import EDAState

MIN_SAMPLE_SIZE = 30
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


def evidence_numbers(state: EDAState) -> list[float]:
    return collect_evidence_numbers(
        state.get("tool_outputs", {}),
        state.get("statistical_results", {}),
        state.get("anomalies", {}),
        state.get("schema", {}).get("shape", {}),
    )


def _issue(severity: str, kind: str, message: str) -> dict:
    return {"severity": severity, "kind": kind, "message": message}


def critic_agent(state: EDAState) -> dict:
    issues: list[dict] = []
    missing_analysis: list[str] = []
    retry_tasks: list[str] = []
    schema = state.get("schema", {})
    plan = state.get("execution_plan", [])
    completed = set(state.get("completed_tasks", []))
    errors = state.get("task_errors", {})
    skipped = state.get("skipped_tasks", {})
    retries_left = state.get("retry_count", 0) < MAX_RETRIES

    # 1. plan + task completion
    if not plan:
        issues.append(_issue("blocking", "empty_plan", "Execution plan is empty."))
        missing_analysis.append("profiling")
    for task, err in errors.items():
        if err.get("retryable") and retries_left:
            issues.append(_issue("blocking", "task_failed_retryable", f"'{task}' crashed ({err['error']}); retrying."))
            retry_tasks.append(task)
        else:
            issues.append(_issue("warning", "task_failed", f"'{task}' could not run: {err['error']}"))
    for agent in AGENT_ORDER:
        for task in pending_tasks(state, agent):
            issues.append(_issue("blocking", "not_run", f"Planned task '{task}' never ran."))
            missing_analysis.append(task)
    for task, reason in skipped.items():
        issues.append(_issue("warning", "skipped", f"'{task}' skipped: {reason}"))

    # 2. coverage: every intent in the question needs at least one piece of evidence
    question = state.get("user_query", "")
    for intent in detect_intents(question):
        tasks = INTENTS[intent]["tasks"]
        if any(t in completed for t in tasks):
            continue
        candidates = [t for t in tasks if is_feasible(t, schema)[0] and t not in errors and t not in plan]
        if candidates:
            issues.append(_issue("blocking", "coverage_gap", f"Question asks about {intent}, but no {intent} evidence was produced; adding {', '.join(candidates)}."))
            missing_analysis += candidates
        else:
            issues.append(_issue("warning", "unanswerable_intent", f"Question asks about {intent}, but this dataset can't support it."))

    # 3. evidence + sample size
    if not (state.get("tool_outputs") or state.get("statistical_results") or state.get("anomalies")):
        issues.append(_issue("blocking", "no_evidence", "No tool outputs were produced."))
    rows = schema.get("shape", {}).get("rows")
    if rows is not None and rows < MIN_SAMPLE_SIZE:
        issues.append(_issue("warning", "small_sample", f"Only {rows} rows (< {MIN_SAMPLE_SIZE}); statistical power is low."))

    # 4 + 5. verify the insights against the evidence
    insights = state.get("insights", [])
    evidence = evidence_numbers(state)
    insight_problem = False
    if not insights:
        issues.append(_issue("blocking", "no_insights", "No insights were produced."))
        insight_problem = True
    for i, finding in enumerate(insights, 1):
        text = f"{finding.get('fact', '')} {finding.get('interpretation', '')}"
        bad_numbers = ungrounded_numbers(text, evidence)
        if bad_numbers:
            issues.append(_issue("blocking", "ungrounded_number", f"Finding {i} cites numbers not found in any tool output: {', '.join(bad_numbers)}."))
            insight_problem = True
        causal = causal_phrases(text)
        if causal:
            issues.append(_issue("blocking", "causal_language", f"Finding {i} uses causal language ({', '.join(causal)}) not supported by associational evidence."))
            insight_problem = True

    blocking = [i for i in issues if i["severity"] == "blocking"]
    feedback = {
        "passed": not blocking,
        "issues": issues,
        "missing_analysis": list(dict.fromkeys(missing_analysis)),
        "retry_tasks": retry_tasks,
        "regenerate_insights": insight_problem,
        "insights_rejected": insight_problem,
    }
    history = list(state.get("critic_history", []))
    history.append({"round": state.get("retry_count", 0), "passed": feedback["passed"], "blocking": len(blocking), "warnings": len(issues) - len(blocking), "issues": issues})
    verdict = "passed" if feedback["passed"] else f"failed with {len(blocking)} blocking issue(s)"
    return {
        "critic_feedback": feedback,
        "critic_history": history,
        "status": "critiquing",
        "trace": add_trace(state, "critic", f"Verification {verdict}, {len(issues) - len(blocking)} warning(s)."),
    }
