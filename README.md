# Agentic EDA

Upload a CSV or Excel file, ask a question in plain English, and get a verified,
evidence-backed exploratory data analysis report. A LangGraph multi-agent
pipeline plans the analysis, computes it with pandas, SciPy and
scikit-learn, and has a deterministic Critic check every claim before the
report is released. Runs fully locally with Streamlit and Ollama, and still
works with no model at all.

**Core rule: the LLM plans and writes prose; tools compute.** Every number
in the report comes from `tools/`. Numbers the LLM writes are checked
against tool output, and anything that can't be verified is rejected.

## How it works

```
CSV -> dataset_inspector -> supervisor --dispatch--> profiling -> statistics
                                                  -> visualization -> anomaly
                                                  -> insight -> critic
critic --passed--> report
       --failed--> replan -> supervisor (targeted; capped by MAX_RETRIES)
```

- **Dataset inspector** detects numeric, categorical, datetime and ID
  columns, and picks a target column (one named in the question wins).
- **Supervisor** builds a rule-based plan from the schema and the
  question's intents. An optional LLM pass can only add whitelisted,
  feasible tasks. On re-plan it acts on the Critic's structured feedback.
- **Dispatcher** (`graph/router.py`) routes to the next agent that has
  pending work, in dependency order, so agents with nothing to do never run.
  A retry re-runs only what changed.
- **Tool agents** do pure computation. A task ends each round as completed,
  failed (retryable or not), or skipped (with a reason).
- **Insight agent** turns results into fact/interpretation findings (LLM,
  with a deterministic fallback). On retry it receives the Critic's complaints.
- **Critic** runs deterministic checks with blocking/warning severity:
  - task completion;
  - question coverage (did we produce evidence for every intent asked about?);
  - sample size;
  - numeric grounding (every number in a finding must exist in tool output);
  - unsupported causal language.
- **Report** is templated from state. If LLM findings still fail
  verification after all retries, the report fails closed and uses
  deterministic findings. The LLM's summary is verified the same way.

## Statistics and anomaly methods

| Analysis | Method |
|---|---|
| Correlation | Pearson and Spearman, pairwise deletion, Benjamini-Hochberg FDR correction. Strength is judged on Spearman when the two disagree. |
| Group comparison | Every usable categorical is tested with Kruskal-Wallis; the most different grouping is reported. |
| Trend | Linear fit on period medians (frequency adapts to date span), with p-value and change along the fitted line. |
| Regression | Standardized features (comparable coefficients), 25% holdout R² when n ≥ 50. |
| Anomalies | IQR, robust modified Z-score (median/MAD), scaled Isolation Forest, and a consensus of rows flagged by ≥ 2 methods. |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull gemma3:1b          # optional; a 7-8B model writes much better findings
streamlit run app.py
```

Optional `.env` settings: `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `MAX_RETRIES` (default 2),
`LLM_TIMEOUT_S` (default 180), `LLM_MAX_TOKENS` (default 600),
`LLM_KEEP_ALIVE` (default 30m), `LLM_DISABLED=1`.

## Demo

Click **Use sample sales data**, pick the first suggested question, and run.
`data/sample_sales.csv` comes from `scripts/make_sample_data.py` with known,
planted structure:
- North sells more than other regions;
- revenue grows over time;
- 12 orders have corrupted revenue (their indices are in `data/sample_sales_truth.json`);
- some ratings are missing, and there are duplicate rows.

Turn on **Simulate a planner miss** in the sidebar to watch the Critic catch
missing anomaly evidence and the Supervisor re-plan.

## Tests

```bash
pytest -q
```

The 36 tests cover the tools, messy real-world inputs, recovery of the planted signal, the
grounding checks, targeted re-planning, and LLM hallucination handling. A
fake LLM invents numbers, and the tests check that it is caught,
regenerated with feedback, and fails closed. Tests never need Ollama.

## Project layout

```
app.py                Streamlit UI with live streaming of agent steps
graph/                state, dispatcher/router, LangGraph wiring
agents/               supervisor, tool agents, insight, critic, report
  tasks.py            task registry: owners, feasibility, question intents
  verification.py     numeric grounding + causal-language checks
  base.py             shared task bookkeeping and trace log
tools/                pure, deterministic pandas/SciPy/scikit-learn functions
scripts/              sample data generator
tests/                pytest suite
```
