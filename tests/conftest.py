import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["LLM_DISABLED"] = "1"  # tests never depend on a running model


@pytest.fixture(scope="session")
def sales_df() -> pd.DataFrame:
    path = ROOT / "data" / "sample_sales.csv"
    if not path.exists():
        from scripts.make_sample_data import main
        main()
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def planted_anomalies() -> list[int]:
    return json.loads((ROOT / "data" / "sample_sales_truth.json").read_text())["planted_anomaly_indices"]


QUESTION = "What factors are most associated with revenue, how is it trending, and are there any unusual orders?"
