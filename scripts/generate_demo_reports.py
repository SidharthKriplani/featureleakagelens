from pathlib import Path
import numpy as np
import pandas as pd

from featureleakagelens import LeakageAuditConfig, audit_dataframe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(7)
n = 500
application_ts = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 60, size=n), unit="D")
outcome_ts = application_ts + pd.to_timedelta(rng.integers(30, 90, size=n), unit="D")
credit_score = rng.normal(680, 55, size=n).round(0)
income = rng.normal(90000, 18000, size=n).round(0)
base_prob = 1 / (1 + np.exp((credit_score - 650) / 35))
defaulted = rng.binomial(1, np.clip(base_prob, 0.05, 0.80))
# Planted leakage: post-outcome payment flag and timestamp.
payment_received_flag = 1 - defaulted
payment_received_ts = outcome_ts + pd.to_timedelta(rng.integers(1, 10, size=n), unit="D")
# Proxy-like column name and high-cardinality ID.
default_status_final = np.where(defaulted == 1, "DEFAULTED", "CURRENT")
account_id = [f"ACC-{100000+i}" for i in range(n)]
split = np.where(np.arange(n) < 350, "train", "test")
region = np.where(rng.random(n) < np.where(split == "test", 0.65, 0.35), "north", "south")

df = pd.DataFrame({
    "account_id": account_id,
    "application_ts": application_ts,
    "outcome_ts": outcome_ts,
    "credit_score": credit_score,
    "income": income,
    "region": region,
    "payment_received_flag": payment_received_flag,
    "payment_received_ts": payment_received_ts,
    "default_status_final": default_status_final,
    "split": split,
    "defaulted": defaulted,
})

data_path = DATA / "demo_leakage_dataset.csv"
df.to_csv(data_path, index=False)

config = LeakageAuditConfig(
    target_col="defaulted",
    split_col="split",
    outcome_time_col="outcome_ts",
    feature_time_cols={"payment_received_flag": "payment_received_ts"},
    ignore_cols=["application_ts"],
)
report = audit_dataframe(df, config)
report.save(OUT)
print(report.to_markdown())
print(f"\nSaved demo data to {data_path}")
print(f"Saved reports to {OUT}")
