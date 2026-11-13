import json

import agents.insight_agent as insight_mod
import agents.report_agent as report_mod
from graph.workflow import run_eda
from tests.conftest import QUESTION


def test_happy_path_passes_first_time(sales_df):
    state = run_eda(sales_df, QUESTION, use_llm=False)
    assert state["critic_feedback"]["passed"]
    assert state["retry_count"] == 0
    assert state["status"] == "done"
    assert "## Anomalies" in state["final_report"]
    assert {"correlation", "groupby", "time_series", "regression", "anomaly_iqr"} <= set(state["completed_tasks"])


def test_dispatcher_skips_agents_with_no_work(sales_df):
    state = run_eda(sales_df[["revenue"]], "Describe revenue", use_llm=False)
    nodes = [t["node"] for t in state["trace"]]
    assert "statistics" not in nodes  # no stats task is feasible on one column


def test_planner_miss_triggers_targeted_replan(sales_df):
    dropped = ["anomaly_iqr", "anomaly_zscore", "anomaly_isolation_forest"]
    state = run_eda(sales_df, QUESTION, use_llm=False, demo_drop_tasks=dropped)
    assert state["critic_history"][0]["passed"] is False
    assert state["critic_feedback"]["passed"]
    assert state["retry_count"] == 1
    round1 = [t["node"] for t in state["trace"] if t["round"] == 1]
    assert "anomaly" in round1
    assert "statistics" not in round1 and "profiling" not in round1  # successful work is kept


def test_precondition_failure_is_a_warning_not_a_retry(sales_df):
    # no datetime column -> time_series infeasible; LLM-style injected task must not loop
    df = sales_df.drop(columns=["order_date"])
    state = run_eda(df, "How is revenue trending over time?", use_llm=False)
    assert state["retry_count"] == 0
    kinds = {i["kind"] for i in state["critic_feedback"]["issues"]}
    assert "unanswerable_intent" in kinds


def _fake_llm(responses):
    calls = iter(responses)
    return lambda system, user, json_mode=False: next(calls)


def test_hallucinated_insight_is_regenerated_with_feedback(sales_df, monkeypatch):
    bad = json.dumps({"findings": [{"fact": "Revenue rose 37% and discounts drive sales.", "interpretation": "x", "source": "time_series"}]})
    good = json.dumps({"findings": [{"fact": "18 rows (1.49%) were flagged by IQR.", "interpretation": "Review them.", "source": "anomaly"}]})
    seen_prompts = []

    def llm(system, user, json_mode=False):
        seen_prompts.append(user)
        return bad if len(seen_prompts) == 1 else good

    monkeypatch.setattr(insight_mod, "llm_enabled", lambda flag=True: True)
    monkeypatch.setattr(insight_mod, "call_llm", llm)
    state = run_eda(sales_df, QUESTION, use_llm=True)

    first = state["critic_history"][0]
    assert not first["passed"]
    assert {"ungrounded_number", "causal_language"} <= {i["kind"] for i in first["issues"]}
    assert "REJECTED" in seen_prompts[1]  # critic feedback reached the LLM
    assert state["critic_feedback"]["passed"] and state["insight_mode"] == "llm"


def test_persistent_hallucination_fails_closed(sales_df, monkeypatch):
    bad = json.dumps({"findings": [{"fact": "Revenue rose 37%.", "interpretation": "x", "source": "time_series"}]})
    monkeypatch.setattr(insight_mod, "llm_enabled", lambda flag=True: True)
    monkeypatch.setattr(insight_mod, "call_llm", lambda *a, **k: bad)
    state = run_eda(sales_df, QUESTION, use_llm=True)
    assert not state["critic_feedback"]["passed"]
    assert "37%" not in state["final_report"]
    assert "LLM findings failed verification" in state["final_report"]


def test_ungrounded_summary_is_replaced(sales_df, monkeypatch):
    monkeypatch.setattr(report_mod, "safe_text_call", lambda *a, **k: "Revenue grew 999.9% because of marketing.")
    state = run_eda(sales_df, QUESTION, use_llm=False)
    assert "999.9" not in state["final_report"]
    assert "template (LLM summary failed verification)" in state["final_report"]
