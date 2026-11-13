"""router.py -- conditional-edge logic.

route_next: the dispatcher. After the Supervisor and after every agent,
it picks the next node that actually has work, in dependency order.
Agents with nothing planned are never visited, and a re-plan re-runs
only the agents whose tasks changed.

route_after_critic: pass -> report; fail -> replan, capped by MAX_RETRIES.
"""
from __future__ import annotations

from agents.base import pending_tasks
from agents.critic_agent import MAX_RETRIES
from agents.tasks import AGENT_ORDER
from graph.state import EDAState


def route_next(state: EDAState) -> str:
    for agent in AGENT_ORDER:
        if pending_tasks(state, agent):
            return agent
    if state.get("insights_stale", True):
        return "insight"
    return "critic"


def route_after_critic(state: EDAState) -> str:
    if state.get("critic_feedback", {}).get("passed", False):
        return "report"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "report"
    return "replan"
