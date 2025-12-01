"""
Utility helpers for loading data, deriving features, and basic statistics.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def load_merged(path: str | Path = "data/merged_sampled.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["reviews_per_day_90d"] = out["review_count"] / 90
    out["log_price"] = np.log1p(out["release_price"].fillna(0))
    out["log_owners"] = np.log1p(out["owners_median"].fillna(0))
    out["engagement_ratio"] = out["review_count"] / out["owners_median"].replace(0, np.nan)
    out["price_bucket"] = pd.cut(
        out["release_price"],
        bins=[0, 5, 10, 20, 60, np.inf],
        labels=["$0-5", "$5-10", "$10-20", "$20-60", "$60+"],
    )
    return out


def cohen_d(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a = sample_a.dropna()
    b = sample_b.dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    pooled_std = np.sqrt(((a.var(ddof=1) * (len(a) - 1)) + (b.var(ddof=1) * (len(b) - 1))) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_std


def simple_regression(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "r": np.nan}
    slope, intercept, r, _, _ = stats.linregress(x[mask], y[mask])
    return {"slope": slope, "intercept": intercept, "r": r}


def mannwhitney_p(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a = sample_a.dropna()
    b = sample_b.dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p
