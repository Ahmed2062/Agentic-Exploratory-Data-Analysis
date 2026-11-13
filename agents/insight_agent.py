"""insight_agent.py -- turns tool-computed results into findings.

The LLM sees ONLY computed results and must separate fact from
interpretation. On a retry it also receives the Critic's specific
complaints. If the LLM is off, fails, or returns malformed output, a
deterministic builder produces findings that are grounded by construction.
"""
from __future__ import annotations

import json

from agents.base import add_trace
from agents.llm import call_llm, llm_enabled
from graph.state import EDAState

SYSTEM_PROMPT = """You are the Insight Agent in a data-analysis pipeline.
You are given ONLY tool-computed statistics. Rules:
- Every number you write must appear in the input JSON. Copy numbers exactly; never compute, estimate, or round new ones.
- Separate "fact" (the literal computed result) from "interpretation" (what it plausibly means).
- Never claim causation. Do not use: causes, due to, because of, leads to, results in, drives. Use "is associated with".
- Return JSON: {"findings": [{"fact": "...", "interpretation": "...", "source": "<task name>"}]} with 3-4 short findings that answer the user's question."""

VALID_SOURCES = {"profiling", "correlation", "groupby", "time_series", "regression", "anomaly", "consensus"}


def _fmt_pct(x: float) -> str:
    return f"{x}%"


def deterministic_findings(state: EDAState) -> list[dict]:
    """Template findings that only quote numbers present in state."""
    findings: list[dict] = []
    stats = state.get("statistical_results", {})
    anomalies = state.get("anomalies", {})
    target = state.get("schema", {}).get("likely_target_column")

    reg = stats.get("regression")
    if reg and "error" not in reg:
        coefs = reg["standardized_coefficients"]
        top = ", ".join(f"{f} ({coefs[f]})" for f in reg["top_features"])
        findings.append({
            "fact": f"A linear model of {reg['target_col']} reaches R² = {reg['r_squared']} ({reg['validation']}); strongest standardized coefficients: {top}.",
            "interpretation": "These features are most associated with the target, holding the others constant. This is association, not proof of effect.",
            "source": "regression",
        })

    pairs = stats.get("correlation", {}).get("pairs", [])
    target_pairs = [p for p in pairs if target in (p["column_a"], p["column_b"])] if target else []
    for p in (target_pairs or pairs)[:3]:
        if p["strength"] in ("negligible", "weak") and findings:
            continue
        sig = "statistically significant after FDR correction" if p.get("significant") else "not significant after FDR correction"
        extra = " Pearson and Spearman disagree, so the relationship is non-linear or distorted by outliers; the strength is judged on Spearman." if p.get("nonlinear_or_outlier_driven") else ""
        findings.append({
            "fact": f"{p['column_a']} and {p['column_b']} have r = {p['r']} (Spearman {p['spearman_rho']}, n = {p['n']}), {sig}.",
            "interpretation": f"A {p['strength']} association.{extra}",
            "source": "correlation",
        })

    gb = stats.get("groupby")
    if gb and "error" not in gb and gb.get("groups"):
        best, worst = gb["groups"][0], gb["groups"][-1]
        test = gb.get("difference_test") or {}
        verdict = "differ significantly (Kruskal-Wallis)" if test.get("significant") else "do not differ significantly (Kruskal-Wallis)"
        findings.append({
            "fact": f"Mean {gb['value_col']} is highest for {gb['group_col']} = {best[gb['group_col']]} ({best['mean']}) and lowest for {worst[gb['group_col']]} ({worst['mean']}).",
            "interpretation": f"Groups {verdict}.",
            "source": "groupby",
        })

    ts = stats.get("time_series")
    if ts and "error" not in ts and ts.get("trend") != "insufficient_data":
        pct = ts.get("pct_change_fitted")
        trend_text = "no significant trend" if ts["trend"] == "no significant trend" else f"a significant {ts['trend']} trend"
        findings.append({
            "fact": f"{ts['value_col']} shows {trend_text} over {ts['periods']} {ts['frequency']} periods" + (f" ({_fmt_pct(pct)} along the fitted line)." if pct is not None and ts["trend"] != "no significant trend" else "."),
            "interpretation": "Tested with a linear fit on period medians; it does not account for seasonality.",
            "source": "time_series",
        })

    cons = anomalies.get("consensus")
    if cons and "error" not in cons:
        findings.append({
            "fact": f"{cons['outlier_count']} rows ({_fmt_pct(cons['outlier_pct'])}) were flagged by at least {cons['min_methods']} independent anomaly methods.",
            "interpretation": "Rows that several methods agree on are the best candidates for manual review: data errors or genuine extremes.",
            "source": "consensus",
        })
    else:
        for method in ("iqr", "zscore", "isolation_forest"):
            r = anomalies.get(method)
            if r and "outlier_count" in r:
                findings.append({
                    "fact": f"{method} flagged {r['outlier_count']} rows ({_fmt_pct(r['outlier_pct'])}).",
                    "interpretation": "Warrants manual review: may be errors or genuine extremes.",
                    "source": "anomaly",
                })

    for issue in state.get("tool_outputs", {}).get("profiling", {}).get("data_quality_issues", [])[:3]:
        findings.append({"fact": issue, "interpretation": "Address before relying on the results.", "source": "profiling"})
    return findings


def _llm_payload(state: EDAState) -> str:
    """Only the fields a finding could cite. Smaller prompt = faster on CPU,
    and fewer distracting numbers for the model to misquote."""
    stats = state.get("statistical_results", {})
    out: dict = {
        "user_question": state.get("user_query", ""),
        "target_column": state.get("schema", {}).get("likely_target_column"),
        "rows": state.get("schema", {}).get("shape", {}).get("rows"),
        "data_quality_issues": state.get("tool_outputs", {}).get("profiling", {}).get("data_quality_issues", [])[:3],
    }
    pairs = stats.get("correlation", {}).get("pairs", [])
    if pairs:
        out["correlations"] = [
            {k: p[k] for k in ("column_a", "column_b", "r", "spearman_rho", "strength", "significant")} for p in pairs[:5]
        ]
    reg = stats.get("regression")
    if reg and "error" not in reg:
        out["regression"] = {k: reg[k] for k in ("target_col", "r_squared", "validation", "standardized_coefficients")}
    gb = stats.get("groupby")
    if gb and "error" not in gb:
        out["groups"] = {
            "group_col": gb["group_col"], "value_col": gb["value_col"],
            "means": {str(g[gb["group_col"]]): g["mean"] for g in gb["groups"][:6]},
            "difference_significant": (gb.get("difference_test") or {}).get("significant"),
        }
    ts = stats.get("time_series")
    if ts and "error" not in ts:
        out["trend"] = {k: ts.get(k) for k in ("value_col", "trend", "periods", "frequency", "pct_change_fitted")}
    anomalies = state.get("anomalies", {})
    out["anomalies"] = {
        m: {"count": r["outlier_count"], "pct": r.get("outlier_pct"), "how": r.get("explanation", "")}
        for m, r in anomalies.items() if isinstance(r, dict) and "outlier_count" in r
    }
    return json.dumps(out, default=str)


def _parse(raw: str) -> list[dict]:
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("findings", [])
    if not isinstance(data, list):
        raise ValueError("findings is not a list")
    out = []
    for f in data:
        if isinstance(f, dict) and str(f.get("fact", "")).strip():
            source = str(f.get("source", "")).removeprefix("anomaly_")
            out.append({
                "fact": str(f["fact"]).strip(),
                "interpretation": str(f.get("interpretation", "")).strip(),
                "source": source if source in VALID_SOURCES else "unspecified",
            })
    if not out:
        raise ValueError("no usable findings")
    return out


def insight_agent(state: EDAState) -> dict:
    feedback = state.get("insight_feedback", [])
    mode = "deterministic"
    note = ""
    if llm_enabled(state.get("use_llm", True)):
        user = _llm_payload(state)
        if feedback:
            user += "\n\nYour previous findings were REJECTED by the verifier for these reasons. Fix them:\n- " + "\n- ".join(feedback)
        try:
            findings = _parse(call_llm(SYSTEM_PROMPT, user, json_mode=True))
            mode = "llm"
        except Exception as exc:  # noqa: BLE001
            findings = deterministic_findings(state)
            note = f" (LLM output unusable: {type(exc).__name__}; used deterministic findings)"
    else:
        findings = deterministic_findings(state)

    completed = list(state.get("completed_tasks", []))
    if "insight" not in completed:
        completed.append("insight")
    return {
        "insights": findings,
        "insight_mode": mode,
        "insights_stale": False,
        "completed_tasks": completed,
        "trace": add_trace(state, "insight", f"Wrote {len(findings)} finding(s) via {mode}{note}" + (" using critic feedback." if feedback and mode == "llm" else ".")),
    }
