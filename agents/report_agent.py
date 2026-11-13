"""report_agent.py -- assembles the final markdown report.

Every number comes straight from state. The LLM writes only the
Executive Summary, and that prose goes through the same grounding and
causal-language checks as the insights; if it fails, a templated summary
is used instead. If the Critic still rejects the LLM's insights after
all retries, the report FAILS CLOSED: it swaps in deterministic findings
rather than publishing unverified claims.
"""
from __future__ import annotations

from datetime import datetime, timezone

from agents.base import add_trace
from agents.critic_agent import evidence_numbers
from agents.insight_agent import deterministic_findings
from agents.llm import safe_text_call
from agents.verification import causal_phrases, ungrounded_numbers
from graph.state import EDAState

SUMMARY_SYSTEM_PROMPT = """You write the Executive Summary for a data analysis report.
You are given the user's question and verified findings as JSON.
Write 3-5 sentences of plain language that answer the question. Only use numbers
that appear in the findings, copied exactly. Never use causal words (causes, due to,
because of, leads to, drives); say "is associated with". Return prose only."""


def _fallback_summary(state: EDAState, findings: list[dict]) -> str:
    rows = state.get("schema", {}).get("shape", {}).get("rows", "unknown")
    lead = " ".join(f["fact"] for f in findings[:3])
    return f"Analysis of {rows} rows for the question \"{state.get('user_query', '')}\". {lead}".strip()


def _table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _section_quality(state: EDAState) -> str:
    issues = state.get("tool_outputs", {}).get("profiling", {}).get("data_quality_issues", [])
    return "\n".join(f"- {i}" for i in issues) if issues else "No significant data-quality issues detected."


def _section_relationships(state: EDAState) -> str:
    parts = []
    stats = state.get("statistical_results", {})
    pairs = stats.get("correlation", {}).get("pairs", [])
    if pairs:
        rows = [[f"{p['column_a']} / {p['column_b']}", p["r"], p["spearman_rho"], p["p_adjusted"], "yes" if p["significant"] else "no", f"{p['strength']} ({p['strength_basis']})"] for p in pairs[:8]]
        parts.append("**Correlations** (Benjamini-Hochberg adjusted p-values; strength judged on Spearman where it disagrees with Pearson)\n\n" + _table(["Pair", "Pearson r", "Spearman ρ", "p (adj.)", "Significant", "Strength"], rows))
    reg = stats.get("regression")
    if reg and "error" not in reg:
        rows = [[f, c] for f, c in reg["standardized_coefficients"].items()]
        r2 = f"R² = {reg['r_squared']} ({reg['validation']})"
        if "r_squared_test" in reg:
            r2 += f"; train R² = {reg['r_squared_train']}"
        parts.append(f"**Linear model of {reg['target_col']}**: {r2}, n = {reg['n']}\n\n" + _table(["Feature", "Std. coefficient"], rows))
    gb = stats.get("groupby")
    if gb and "error" not in gb:
        rows = [[g[gb["group_col"]], g["count"], g["mean"], g["median"]] for g in gb["groups"][:10]]
        test = gb.get("difference_test")
        t = f" Kruskal-Wallis p = {test['p_value']}." if test else ""
        others = gb.get("other_groupings") or []
        if others:
            t += " Also tested: " + ", ".join(f"{o['group_col']} (p = {o['p_value']})" for o in others) + "."
        parts.append(f"**{gb['value_col']} by {gb['group_col']}**.{t}\n\n" + _table([gb["group_col"], "count", "mean", "median"], rows))
    return "\n\n".join(parts) or "No statistical tests were applicable to this question."


def _section_trends(state: EDAState) -> str:
    ts = state.get("statistical_results", {}).get("time_series")
    if not ts or "error" in ts:
        return "No time-series analysis was applicable."
    if ts.get("trend") == "insufficient_data":
        return f"Not enough periods to assess a trend for {ts['value_col']}."
    pct = ts.get("pct_change_fitted")
    return (
        f"- **{ts['value_col']}** ({ts['frequency']} medians, {ts['periods']} periods): **{ts['trend']}** "
        f"(slope {ts['slope_per_period']} per period, p = {ts['trend_p_value']})"
        + (f", {pct}% change along the fitted line." if pct is not None else ".")
    )


def _section_anomalies(state: EDAState) -> str:
    anomalies = state.get("anomalies", {})
    lines = []
    for method in ("iqr", "zscore", "isolation_forest", "consensus"):
        r = anomalies.get(method)
        if r and "outlier_count" in r:
            sample = ", ".join(map(str, r.get("outlier_indices", [])[:10]))
            lines.append(f"- **{method}**: {r['outlier_count']} rows ({r.get('outlier_pct', 0)}%). {r.get('explanation', '')}" + (f" Example row indices: {sample}." if sample else ""))
    by_col = anomalies.get("iqr_by_column")
    if by_col:
        lines.append("- **IQR outliers per column**: " + ", ".join(f"{c}: {n}" for c, n in by_col.items()))
    return "\n".join(lines) if lines else "No anomaly detection was run."


def _section_findings(findings: list[dict]) -> str:
    if not findings:
        return "No verified findings available."
    return "\n".join(f"- **Fact:** {f.get('fact', '')}\n  **Interpretation:** {f.get('interpretation', '')}" for f in findings)


def _section_verification(state: EDAState, replaced: bool, summary_source: str) -> str:
    history = state.get("critic_history", [])
    lines = [f"- Round {h['round']}: {'passed' if h['passed'] else 'failed'} ({h['blocking']} blocking, {h['warnings']} warnings)" for h in history]
    lines.append(f"- Findings written by: {'deterministic templates (LLM findings failed verification)' if replaced else state.get('insight_mode', 'n/a')}")
    lines.append(f"- Executive summary written by: {summary_source}")
    return "\n".join(lines)


def _section_limitations(state: EDAState) -> str:
    lines = [f"- {i['message']}" for i in state.get("critic_feedback", {}).get("issues", []) if i["severity"] == "warning"]
    unresolved = [i for i in state.get("critic_feedback", {}).get("issues", []) if i["severity"] == "blocking"]
    lines += [f"- Unresolved after retries: {i['message']}" for i in unresolved]
    lines += [
        "- Findings describe this dataset only and may not generalize.",
        "- Correlation and regression results describe association, not causation.",
    ]
    return "\n".join(lines)


def report_agent(state: EDAState) -> dict:
    feedback = state.get("critic_feedback", {})
    findings = state.get("insights", [])
    replaced = False
    if not feedback.get("passed") and feedback.get("insights_rejected"):
        findings = deterministic_findings(state)  # fail closed
        replaced = True

    fallback = _fallback_summary(state, findings)
    summary = safe_text_call(
        SUMMARY_SYSTEM_PROMPT,
        f"Question: {state.get('user_query')}\nFindings: {findings}",
        fallback=fallback,
        enabled=state.get("use_llm", True),
    )
    summary_source = "LLM" if summary != fallback else "template"
    if summary_source == "LLM":
        evidence = evidence_numbers(state)
        if ungrounded_numbers(summary, evidence) or causal_phrases(summary):
            summary, summary_source = fallback, "template (LLM summary failed verification)"

    schema = state.get("schema", {})
    shape = schema.get("shape", {})
    report = f"""# EDA report

*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*

**Question:** {state.get('user_query', '')}

## Executive summary
{summary}

## Dataset overview
- Rows: {shape.get('rows', 'n/a')} | Columns: {shape.get('columns', 'n/a')}
- Target column: {schema.get('likely_target_column') or 'none identified'}
- Numeric: {', '.join(schema.get('numeric_columns', [])) or 'none'}
- Categorical: {', '.join(schema.get('categorical_columns', [])) or 'none'}
- Datetime: {', '.join(schema.get('datetime_columns', [])) or 'none'}
- Excluded identifiers: {', '.join(schema.get('id_columns', [])) or 'none'}

## Data quality
{_section_quality(state)}

## Relationships
{_section_relationships(state)}

## Trends
{_section_trends(state)}

## Anomalies
{_section_anomalies(state)}

## Findings
{_section_findings(findings)}

## Verification
{_section_verification(state, replaced, summary_source)}

## Limitations
{_section_limitations(state)}

## Recommended next steps
- Resolve the data-quality issues above before acting on the results.
- Validate the strongest associations with domain knowledge or an experiment before assuming causality.
- Review the consensus anomalies row by row.
"""
    completed = list(state.get("completed_tasks", []))
    if "report" not in completed:
        completed.append("report")
    msg = "Report written" + ("; LLM findings replaced with deterministic ones (fail closed)." if replaced else ".")
    return {
        "final_report": report,
        "insights": findings,
        "completed_tasks": completed,
        "status": "done",
        "trace": add_trace(state, "report", msg),
    }
