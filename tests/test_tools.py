import numpy as np
import pandas as pd

from tools import anomaly, dataset, statistics


def test_schema_excludes_ids_and_finds_target(sales_df):
    schema = dataset.inspect_schema(sales_df)
    assert schema["id_columns"] == ["order_id"]
    assert "order_id" not in schema["numeric_columns"]
    assert schema["datetime_columns"] == ["order_date"]
    assert schema["likely_target_column"] == "revenue"


def test_target_named_in_question_wins(sales_df):
    schema = dataset.inspect_schema(sales_df, "What is associated with customer rating?")
    assert schema["likely_target_column"] == "customer_rating"


def test_correlation_is_fdr_adjusted_and_robust():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(size=300), "noise": rng.normal(size=300)})
    res = statistics.correlation_analysis(df, ["x", "y", "noise"])
    top = res["pairs"][0]
    assert {top["column_a"], top["column_b"]} == {"x", "y"}
    assert top["significant"] and top["strength"] == "strong"
    assert all(p["p_adjusted"] >= p["p_value"] for p in res["pairs"])


def test_groupby_detects_planted_region_effect(sales_df):
    res = statistics.groupby_analysis(sales_df, "region", "revenue")
    assert res["groups"][0]["region"] == "North"
    assert res["difference_test"]["significant"]


def test_trend_detects_planted_growth(sales_df):
    res = statistics.time_series_analysis(sales_df, "order_date", "revenue")
    assert res["trend"] == "increasing"
    assert res["pct_change_fitted"] > 0


def test_regression_reports_holdout(sales_df):
    res = statistics.regression_analysis(sales_df, "revenue", ["units", "unit_price", "discount_pct", "marketing_spend"])
    assert res["validation"] == "25% holdout"
    assert set(res["top_features"][:2]) == {"units", "unit_price"}


def test_robust_zscore_recovers_planted_anomalies(sales_df, planted_anomalies):
    res = anomaly.detect_zscore_outliers(sales_df, "revenue")
    recall = len(set(planted_anomalies) & set(res["all_indices"])) / len(planted_anomalies)
    assert recall >= 0.9


def test_consensus_requires_agreement():
    results = {"a": {"outlier_indices": [1, 2, 3]}, "b": {"outlier_indices": [2, 3, 4]}}
    res = anomaly.consensus_anomalies(results, n_rows=10)
    assert res["outlier_indices"] == [2, 3]


def test_tools_report_errors_instead_of_raising():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert "error" in statistics.regression_analysis(df, "a", ["missing"])
    assert "error" in anomaly.detect_isolation_forest_anomalies(df, ["a"])
