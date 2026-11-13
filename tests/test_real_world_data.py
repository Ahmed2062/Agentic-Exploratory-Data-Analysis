"""Datasets an interviewer might hand over: messy formats, odd values,
and columns that look like something they aren't. Each case came from
stress-testing the pipeline on unfamiliar data."""
import io

import numpy as np
import pandas as pd

from graph.workflow import run_eda
from tools.dataset import inspect_schema, load_table, pick_target, prepare_dataframe

Q = "What are the main patterns, what is associated with the target, and are there any outliers?"
rng = np.random.default_rng(3)


def test_numbers_stored_as_text_are_converted():
    df = pd.DataFrame({"Sales": ["$1,234.50", "$99.00", "(12.00)", "$3,000.10"] * 10, "Store": ["A", "B"] * 20})
    out, notes = prepare_dataframe(df)
    assert pd.api.types.is_numeric_dtype(out["Sales"])
    assert out["Sales"].iloc[2] == -12.0
    assert any("Sales" in n for n in notes)


def test_day_first_dates_parse_consistently():
    dates = pd.date_range("2023-01-01", periods=400).strftime("%d/%m/%Y")
    out, _ = prepare_dataframe(pd.DataFrame({"Date": dates, "v": range(400)}))
    assert out["Date"].is_monotonic_increasing  # 'mixed' parsing scrambles this


def test_infinite_values_become_missing_without_wasted_retries():
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    df.loc[:5, "x"] = np.inf
    state = run_eda(df, Q, use_llm=False)
    assert state["retry_count"] == 0
    assert not state["task_errors"]
    assert any("infinite" in i for i in state["tool_outputs"]["profiling"]["data_quality_issues"])


def test_short_column_names_do_not_match_inside_words():
    assert pick_target(["a", "y", "revenue"], "what are the patterns in revenue?") == "revenue"


def test_question_stem_matches_column():
    assert pick_target(["Survived", "Fare"], "what affects survival?") == "Survived"


def test_low_cardinality_integers_can_be_groups():
    df = pd.DataFrame({"Pclass": rng.integers(1, 4, 300), "Fare": rng.gamma(2, 15, 300)})
    df["Fare"] += (4 - df["Pclass"]) * 20
    state = run_eda(df, "How does fare differ by class?", use_llm=False)
    assert state["statistical_results"]["groupby"]["group_col"] == "Pclass"


def test_semicolon_csv_and_latin1_encoding():
    raw = "Stadt;Umsatz;Datum\nKöln;1.5;2024-01-01\nMünchen;2.5;2024-01-02\n".encode("latin-1")
    df = load_table(raw, "data.csv")
    assert list(df.columns) == ["Stadt", "Umsatz", "Datum"]
    assert df["Stadt"].iloc[0] == "Köln"


def test_excel_upload():
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_excel(buf, index=False)
    assert load_table(buf.getvalue(), "file.xlsx").shape == (3, 2)


def test_categorical_only_data_still_produces_a_report():
    df = pd.DataFrame({"color": rng.choice(["red", "blue"], 100), "size": rng.choice(["S", "M", "L"], 100)})
    state = run_eda(df, Q, use_llm=False)
    assert state["status"] == "done" and state["final_report"]


def test_class_label_target_compares_features_across_classes():
    from sklearn.datasets import load_iris
    state = run_eda(load_iris(as_frame=True).frame, "What distinguishes the target classes?", use_llm=False)
    gb = state["statistical_results"]["groupby"]
    assert gb["group_col"] == "target" and gb["difference_test"]["significant"]
    assert "regression" in state["skipped_tasks"] or "regression" not in state["execution_plan"]
    assert not state["task_errors"]


def test_european_decimal_comma():
    df = pd.DataFrame({"Preis": ["1.234,50 €", "12,00 €", "999,99 €"] * 10, "Menge": ["12,5", "3,0", "7,25"] * 10})
    out, _ = prepare_dataframe(df)
    assert out["Preis"].iloc[0] == 1234.5
    assert out["Menge"].iloc[0] == 12.5


def test_us_thousands_still_parse():
    out, _ = prepare_dataframe(pd.DataFrame({"x": ["1,234.50", "99.00", "12,000"] * 10}))
    assert list(out["x"].iloc[:3]) == [1234.5, 99.0, 12000.0]
