from agents.verification import causal_phrases, collect_evidence_numbers, ungrounded_numbers

EVIDENCE = [0.8123, 18, 1.49, 1375.6661, 42.24]


def test_grounded_numbers_pass_with_rounding():
    assert ungrounded_numbers("r = 0.81, 18 rows (1.49%), mean 1375.67, up 42.24%", EVIDENCE) == []


def test_invented_numbers_are_flagged():
    assert ungrounded_numbers("Revenue grew 37% and the mean was 2000.5", EVIDENCE) == ["37", "2000.5"]


def test_thresholds_small_ints_and_years_are_ignored():
    assert ungrounded_numbers("p < 0.05 for the top 3 regions in 2024", EVIDENCE) == []


def test_row_indices_are_not_evidence():
    nums = collect_evidence_numbers({"outlier_indices": [777], "all_indices": [888], "outlier_count": 2})
    assert 777 not in nums and 888 not in nums and 2 in nums


def test_causal_language_flagged_but_negation_allowed():
    assert causal_phrases("Discounts drive units, due to pricing.") == ["drive", "due to"]
    assert causal_phrases("This does not cause churn; it is not driven by price.") == []
