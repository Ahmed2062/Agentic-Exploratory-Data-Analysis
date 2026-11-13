# What changed and why

## Orchestration
- **Real routing.** The fixed linear chain is now a dispatcher
  (`route_next`) that sends control to the next agent with pending work.
  Agents with nothing planned are never visited.
- **Re-planning that does something.** Previously the replan step added
  only tasks already in the plan, so a retry re-ran the identical pipeline.
  The Critic now returns structured feedback:
  - `missing_analysis`: coverage gaps to add;
  - `retry_tasks`: crashed tasks worth retrying;
  - `regenerate_insights`: whether the findings need rewriting.

  The Supervisor acts on each, and successful work is kept, so a retry
  re-runs only what changed.
- **Success-aware completion.** Tasks that returned `{"error": ...}` were
  previously marked complete. A task now ends each round as completed,
  failed (retryable or not), or skipped with a reason.
- **No duplicate work on retry.** Charts were appended and completed
  tasks re-added on every pass; both are fixed.
- **LLM-suggested tasks are validated** for feasibility. An impossible
  task, such as regression with no target, is skipped with a reason
  instead of failing the Critic on every round.

## Verification
- **Numeric grounding.** Every number in a finding or in the summary must
  match a tool-computed value within rounding. Row indices are excluded
  from the evidence, so they can't launder made-up integers.
- **Severity levels.** Issues that retries can't fix (small sample,
  infeasible task) are warnings that go into Limitations. They no longer
  burn retries.
- **Coverage check.** If the question asks about anomalies, trends,
  drivers or segments and no such evidence exists, the Critic requests it.
- **Causal-language check** covers more phrasings and ignores negations.
- **Fail closed.** If the LLM's findings still fail after the retry
  limit, the report uses deterministic findings and says so.
- **Critic feedback reaches the LLM** on retry, so regeneration is
  targeted rather than a blind re-prompt.

## Statistics
- Correlation:
  - pairwise deletion instead of listwise;
  - Spearman alongside Pearson;
  - Benjamini-Hochberg FDR correction;
  - p-values no longer round to 0;
  - strength judged on Spearman when outliers distort Pearson.
- Group comparison tests every usable categorical with Kruskal-Wallis and
  reports the most informative one. It previously used the first column only.
- Trend: adaptive frequency, period medians, slope p-value, and change
  along the fitted line rather than noisy first-to-last points.
- Regression: standardized coefficients and a holdout R².
- Anomalies:
  - the robust median/MAD Z-score replaces mean/std, which suffered from masking;
  - Isolation Forest scales its features;
  - a consensus view of rows flagged by multiple methods;
  - IQR counts per column.
- Schema: ID columns are excluded, and a target named in the question wins.

## Engineering
- Object-oriented matplotlib `Figure`, so there is no global pyplot state,
  no memory leak, and it is safe across Streamlit sessions.
- Ollama health check with caching, a request timeout, and an
  `LLM_DISABLED` switch.
- Pinned `pyarrow` (newer releases need NumPy 2 and broke table rendering).
- Sample dataset with planted ground truth, 21 pytest tests, and `.gitignore`.
  Committed `__pycache__` files and `.DS_Store` were removed.
- The UI streams each agent step live and shows the agent trace, the
  Critic rounds, timings, and skipped or failed tasks.
