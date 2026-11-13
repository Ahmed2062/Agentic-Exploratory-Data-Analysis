"""Generate data/sample_sales.csv: a synthetic orders dataset with KNOWN
structure, so a live demo can show the system finding planted signal.

Planted ground truth:
  * revenue = units * unit_price  -> units strongly associated with revenue
  * units rise with marketing_spend and discount_pct
  * revenue trends upward over the two years
  * region "North" sells more than the other regions
  * 12 corrupted rows with revenue inflated 6-10x  (anomalies, indices saved)
  * ~4% missing customer_rating, 6 exact duplicate rows
  * order_id is an identifier and should be excluded
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N = 1200
OUT = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 730, N)), unit="D")
    region = rng.choice(["North", "South", "East", "West"], N, p=[0.3, 0.25, 0.25, 0.2])
    channel = rng.choice(["Online", "Retail", "Partner"], N)
    marketing = rng.gamma(4, 50, N).round(2)
    discount = rng.uniform(0, 30, N).round(1)
    t = (dates - dates.min()).days / 730
    base = 20 + 0.04 * marketing + 0.5 * discount + 15 * t + np.where(region == "North", 10, 0)
    units = np.maximum(1, rng.normal(base, 6)).round().astype(int)
    unit_price = rng.normal(25, 4, N).clip(10).round(2)
    revenue = (units * unit_price).round(2)
    rating = rng.normal(4.1, 0.5, N).clip(1, 5).round(1)

    df = pd.DataFrame({
        "order_id": np.arange(100001, 100001 + N),
        "order_date": dates.strftime("%Y-%m-%d"),
        "region": region,
        "channel": channel,
        "marketing_spend": marketing,
        "discount_pct": discount,
        "units": units,
        "unit_price": unit_price,
        "revenue": revenue,
        "customer_rating": rating,
    })
    planted = rng.choice(N, 12, replace=False)
    df.loc[planted, "revenue"] = (df.loc[planted, "revenue"] * rng.uniform(6, 10, 12)).round(2)
    df.loc[rng.choice(N, int(0.04 * N), replace=False), "customer_rating"] = np.nan
    df = pd.concat([df, df.sample(6, random_state=SEED)], ignore_index=True)

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "sample_sales.csv", index=False)
    (OUT / "sample_sales_truth.json").write_text(json.dumps({"planted_anomaly_indices": sorted(int(i) for i in planted)}, indent=2))
    print(f"Wrote {len(df)} rows to {OUT / 'sample_sales.csv'}")


if __name__ == "__main__":
    main()
