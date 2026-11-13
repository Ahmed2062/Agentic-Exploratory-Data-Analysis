"""
app.py -- Streamlit front end. The LangGraph pipeline runs in-process and
streams each agent step into the page as it happens. Nothing is written
to disk; everything lives in session state for the life of the tab.

Run:  streamlit run app.py      (Ollama optional: `ollama pull gemma3:1b`)
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.critic_agent import MAX_RETRIES
from agents.llm import OLLAMA_MODEL, ollama_available
from graph.workflow import stream_eda

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_sales.csv"
SAMPLE_QUESTIONS = [
    "What factors are most associated with revenue, how is it trending, and are there any unusual orders?",
    "Which region performs best, and is the difference significant?",
    "Are there any anomalies in revenue?",
]
NODE_LABELS = {
    "dataset_inspector": "Schema",
    "supervisor": "Supervisor",
    "profiling": "Profiling",
    "statistics": "Statistics",
    "visualization": "Visualization",
    "anomaly": "Anomaly detection",
    "insight": "Insight",
    "critic": "Critic",
    "replan": "Re-plan",
    "report": "Report",
}

st.set_page_config(page_title="Agentic EDA", layout="wide")
for key, default in (("df", None), ("df_name", None), ("result", None), ("timings", []), ("question", "")):
    st.session_state.setdefault(key, default)

# ---------- sidebar: run settings ----------
with st.sidebar:
    st.subheader("Run settings")
    llm_up = ollama_available(ttl_s=5)
    if llm_up:
        st.success(f"Ollama is running ({OLLAMA_MODEL}).")
    else:
        st.info("Ollama isn't reachable, so findings will come from deterministic templates. Start it with `ollama serve`.")
    use_llm = st.toggle("Use the local LLM", value=llm_up, disabled=not llm_up,
                        help="Off: planning add-ons, findings and the summary come from deterministic templates.")
    st.caption(f"The Critic allows up to {MAX_RETRIES} re-plan rounds.")

    st.subheader("Demo controls")
    simulate_miss = st.toggle(
        "Simulate a planner miss",
        help="Removes the anomaly tasks from the first plan. If your question asks about anomalies, "
             "the Critic notices the missing evidence and the Supervisor re-plans.",
    )

# ---------- main ----------
st.title("Agentic EDA")
st.caption("Ask a question about a CSV. A Supervisor plans the analysis, specialist agents compute it with "
           "pandas, SciPy and scikit-learn, and a Critic checks every claim against the computed results before the report is written.")

st.subheader("Dataset")
c1, c2 = st.columns([3, 1])
uploaded = c1.file_uploader("Upload a CSV", type=["csv"], label_visibility="collapsed")
if uploaded is not None and st.session_state.df_name != uploaded.name:
    try:
        st.session_state.df = pd.read_csv(uploaded)
        st.session_state.df_name = uploaded.name
        st.session_state.result = None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't read that file as CSV: {exc}")
if c2.button("Use sample sales data", use_container_width=True, disabled=not SAMPLE_PATH.exists()):
    st.session_state.df = pd.read_csv(SAMPLE_PATH)
    st.session_state.df_name = SAMPLE_PATH.name
    st.session_state.result = None

df = st.session_state.df
if df is not None:
    st.write(f"**{st.session_state.df_name}**: {df.shape[0]:,} rows, {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)
else:
    st.write("Upload a CSV or load the sample data to begin.")

st.subheader("Question")
cols = st.columns(len(SAMPLE_QUESTIONS))
for col, q in zip(cols, SAMPLE_QUESTIONS):
    if col.button(q, use_container_width=True):
        st.session_state.question = q
question = st.text_area("Your question", key="question", label_visibility="collapsed",
                        placeholder="For example: what is most associated with revenue, and are there unusual orders?")

if st.button("Run analysis", type="primary", disabled=df is None or not question.strip()):
    drop = ["anomaly_iqr", "anomaly_zscore", "anomaly_isolation_forest"] if simulate_miss else []
    timings, state = [], None
    with st.status("Running the agent pipeline", expanded=True) as status:
        t_prev = time.perf_counter()
        for node, update, state in stream_eda(df, question, use_llm=use_llm, demo_drop_tasks=drop):
            now = time.perf_counter()
            timings.append({"step": len(timings) + 1, "node": node, "seconds": round(now - t_prev, 2)})
            t_prev = now
            msg = (update.get("trace") or [{}])[-1].get("message", "")
            st.write(f"**{NODE_LABELS.get(node, node)}** · round {state.get('retry_count', 0)} · {msg}")
        passed = state["critic_feedback"].get("passed")
        status.update(label="Analysis complete" if passed else "Analysis complete with unresolved issues",
                      state="complete" if passed else "error", expanded=False)
    st.session_state.result, st.session_state.timings = state, timings

# ---------- results ----------
result = st.session_state.result
if result:
    critic = result.get("critic_feedback", {})
    rounds = len(result.get("critic_history", []))
    if critic.get("passed"):
        st.success(f"The Critic verified the findings after {rounds} round{'s' if rounds != 1 else ''}.")
    else:
        st.warning("The Critic couldn't verify everything within the retry limit. Unverified LLM findings were replaced "
                   "with deterministic ones; see Limitations in the report.")

    tab_report, tab_charts, tab_trace, tab_raw = st.tabs(["Report", "Charts", "Agent trace", "Raw data"])
    with tab_report:
        report = result.get("final_report", "")
        st.download_button("Download report (Markdown)", data=report, file_name="eda_report.md", mime="text/markdown")
        st.markdown(report)

    with tab_charts:
        charts = result.get("visualizations", [])
        if not charts:
            st.write("No charts apply to this dataset and question.")
        grid = st.columns(2)
        for i, chart in enumerate(charts):
            with grid[i % 2]:
                st.pyplot(chart["figure"])
                st.caption(chart.get("description", ""))

    with tab_trace:
        st.markdown("**Execution order**: which agent ran, in which round, and what it did.")
        trace = pd.DataFrame(result.get("trace", []))
        if not trace.empty:
            trace["node"] = trace["node"].map(lambda n: NODE_LABELS.get(n, n))
            st.dataframe(trace[["round", "node", "message"]], use_container_width=True, hide_index=True)

        st.markdown("**Critic rounds**")
        for h in result.get("critic_history", []):
            label = f"Round {h['round']}: {'passed' if h['passed'] else 'failed'} ({h['blocking']} blocking, {h['warnings']} warnings)"
            with st.expander(label, expanded=not h["passed"]):
                if not h["issues"]:
                    st.write("No issues.")
                for issue in h["issues"]:
                    (st.error if issue["severity"] == "blocking" else st.info)(f"{issue['kind']}: {issue['message']}")

        c1, c2, c3 = st.columns(3)
        c1.markdown("**Completed**\n\n" + "\n".join(f"- {t}" for t in result.get("completed_tasks", [])))
        skipped = result.get("skipped_tasks", {})
        errors = result.get("task_errors", {})
        c2.markdown("**Skipped**\n\n" + ("\n".join(f"- {t}: {r}" for t, r in skipped.items()) or "None"))
        c3.markdown("**Failed**\n\n" + ("\n".join(f"- {t}: {e['error']}" for t, e in errors.items()) or "None"))

        if st.session_state.timings:
            st.markdown("**Time per step** (seconds)")
            st.bar_chart(pd.DataFrame(st.session_state.timings).set_index("step")["seconds"])

    with tab_raw:
        st.markdown(f"**Findings** (written by: {result.get('insight_mode', 'n/a')})")
        st.json(result.get("insights", []))
        st.markdown("**Schema**")
        st.json(result.get("schema", {}))
        st.markdown("**Statistical results**")
        st.json(result.get("statistical_results", {}), expanded=False)
