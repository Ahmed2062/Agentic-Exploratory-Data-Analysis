"""base.py -- shared bookkeeping for the tool-running agents.

Every planned task ends a round in exactly one bucket:
  completed_tasks   -> tool ran and returned a usable result
  task_errors       -> tool failed; `retryable` says whether a retry could help
  skipped_tasks     -> the Supervisor decided not to run it (with a reason)

A tool returning {"error": ...} is a *precondition* failure on this data:
re-running identical deterministic code can't fix it, so it's
non-retryable. The same holds for most exceptions from pandas/SciPy code,
so only genuinely transient errors (memory, I/O, timeouts) are retried --
a pointless retry would also re-run the slow LLM insight step.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from agents.tasks import TASK_AGENT
from graph.state import EDAState

logger = logging.getLogger(__name__)
TRANSIENT_ERRORS = (MemoryError, TimeoutError, OSError)


def pending_tasks(state: EDAState, agent: str) -> list[str]:
    done = set(state.get("completed_tasks", []))
    errored = set(state.get("task_errors", {}))
    skipped = set(state.get("skipped_tasks", {}))
    return [
        t for t in state.get("execution_plan", [])
        if TASK_AGENT.get(t) == agent and t not in done and t not in errored and t not in skipped
    ]


def add_trace(state: EDAState, node: str, message: str, **data: Any) -> list[dict[str, Any]]:
    trace = list(state.get("trace", []))
    trace.append({"round": state.get("retry_count", 0), "node": node, "message": message, **data})
    return trace


def run_tasks(
    state: EDAState,
    agent: str,
    handlers: dict[str, Callable[[], dict[str, Any]]],
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    """Run each pending task's handler. Returns (results, completed, errors)."""
    completed = list(state.get("completed_tasks", []))
    errors = dict(state.get("task_errors", {}))
    results: dict[str, Any] = {}
    for task in pending_tasks(state, agent):
        handler = handlers.get(task)
        if handler is None:
            errors[task] = {"error": f"agent '{agent}' has no handler for '{task}'", "retryable": False}
            continue
        try:
            out = handler()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Task %s crashed", task)
            errors[task] = {"error": f"{type(exc).__name__}: {exc}", "retryable": isinstance(exc, TRANSIENT_ERRORS)}
            continue
        if isinstance(out, dict) and "error" in out:
            errors[task] = {"error": out["error"], "retryable": False}
            continue
        results[task] = out
        if task not in completed:
            completed.append(task)
    return results, completed, errors
