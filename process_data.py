"""
Answer 10 developer-oriented questions using merged_sampled.csv.

Each question has one function. Results are saved to data/analysis_results.json.
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import add_features, cohen_d, load_merged, mannwhitney_p, simple_regression


def q1_optimal_price_bucket(df: pd.DataFrame) -> Dict[str, object]:
    grouped = df.groupby("price_bucket")["positive_ratio"].agg(["mean", "count"])
    bucket_vs_rest = {}
    for bucket, _ in grouped.iterrows():
        in_b = df[df["price_bucket"] == bucket]["positive_ratio"]
        rest = df[df["price_bucket"] != bucket]["positive_ratio"]
        bucket_vs_rest[str(bucket)] = {
            "mean": round(in_b.mean(), 2),
            "count": int(in_b.count()),
            "p_value": mannwhitney_p(in_b, rest),
            "effect_size_d": cohen_d(in_b, rest),
        }
    return {"bucket_stats": grouped.round(2).to_dict(), "bucket_vs_rest": bucket_vs_rest}


def q2_review_speed_targets(df: pd.DataFrame) -> Dict[str, float]:
    series = df["reviews_per_day_90d"].dropna()
    stats = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return {k: round(float(v), 4) for k, v in stats.to_dict().items()}


def q3_engagement_targets(df: pd.DataFrame) -> Dict[str, float]:
    series = df["engagement_ratio"].dropna()
    stats = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return {k: round(float(v), 6) for k, v in stats.to_dict().items()}


def q4_price_effect_on_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    res = simple_regression(df["log_price"], df["positive_ratio"])
    return {k: (None if np.isnan(v) else round(v, 4)) for k, v in res.items()}


def q5_playtime_effect_on_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    res = simple_regression(df["avg_playtime"], df["positive_ratio"])
    mask = df["avg_playtime"].notna() & df["positive_ratio"].notna()
    corr = df.loc[mask, ["avg_playtime", "positive_ratio"]].corr().iloc[0, 1]
    return {
        "corr": round(float(corr), 4) if pd.notna(corr) else None,
        "slope": None if np.isnan(res["slope"]) else round(res["slope"], 6),
        "intercept": None if np.isnan(res["intercept"]) else round(res["intercept"], 4),
    }


def q6_scale_effect_on_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    mask = df["owners_median"].notna() & df["positive_ratio"].notna()
    corr = df.loc[mask, ["owners_median", "positive_ratio"]].corr().iloc[0, 1]
    return {"corr": round(float(corr), 4) if pd.notna(corr) else None}


def q7_update_cadence_effect(df: pd.DataFrame) -> Dict[str, float]:
    if "update_count" not in df.columns:
        return {"corr_updates_positive": None, "corr_gap_positive": None}
    mask1 = df["update_count"].notna() & df["positive_ratio"].notna()
    corr_updates = df.loc[mask1, ["update_count", "positive_ratio"]].corr().iloc[0, 1] if mask1.any() else np.nan
    mask2 = df["max_update_gap"].notna() & df["positive_ratio"].notna()
    corr_gap = df.loc[mask2, ["max_update_gap", "positive_ratio"]].corr().iloc[0, 1] if mask2.any() else np.nan
    return {
        "corr_updates_positive": round(float(corr_updates), 4) if pd.notna(corr_updates) else None,
        "corr_gap_positive": round(float(corr_gap), 4) if pd.notna(corr_gap) else None,
    }


def q8_community_effect(df: pd.DataFrame) -> Dict[str, float]:
    if "community_posts" not in df.columns:
        return {"corr_comm_positive": None, "mean_high": None, "mean_low": None}
    mask = df["community_posts"].notna() & df["positive_ratio"].notna()
    if not mask.any():
        return {"corr_comm_positive": None, "mean_high": None, "mean_low": None}
    corr = df.loc[mask, ["community_posts", "positive_ratio"]].corr().iloc[0, 1]
    median_posts = df.loc[mask, "community_posts"].median()
    high = df.loc[mask & (df["community_posts"] >= median_posts), "positive_ratio"]
    low = df.loc[mask & (df["community_posts"] < median_posts), "positive_ratio"]
    return {
        "corr_comm_positive": round(float(corr), 4) if pd.notna(corr) else None,
        "mean_high": round(high.mean(), 2) if not high.empty else None,
        "mean_low": round(low.mean(), 2) if not low.empty else None,
    }


def q9_risky_titles(df: pd.DataFrame, min_reviews: int = 5, price_threshold: float = 15.0, rating_threshold: float = 60.0, top_n: int = 10) -> List[Dict[str, object]]:
    subset = df[
        (df["release_price"] >= price_threshold)
        & (df["positive_ratio"] <= rating_threshold)
        & (df["review_count"] >= min_reviews)
    ]
    cols = ["app_id", "release_price", "positive_ratio", "review_count"]
    return subset[cols].sort_values("positive_ratio").head(top_n).round(2).to_dict(orient="records")


def q10_benchmark_titles(df: pd.DataFrame, min_reviews: int = 20, top_n: int = 10) -> List[Dict[str, object]]:
    subset = df[df["review_count"] >= min_reviews].sort_values("positive_ratio", ascending=False)
    cols = ["app_id", "positive_ratio", "review_count", "release_price"]
    return subset[cols].head(top_n).round(2).to_dict(orient="records")


QUESTION_FUNCS = [
    q1_optimal_price_bucket,
    q2_review_speed_targets,
    q3_engagement_targets,
    q4_price_effect_on_sentiment,
    q5_playtime_effect_on_sentiment,
    q6_scale_effect_on_sentiment,
    q7_update_cadence_effect,
    q8_community_effect,
    q9_risky_titles,
    q10_benchmark_titles,
]


def main() -> None:
    df_raw = load_merged()
    df = add_features(df_raw)
    results: Dict[str, object] = {}
    for func in QUESTION_FUNCS:
        results[func.__name__] = func(df)
    out_path = Path("data/analysis_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(results).to_json(out_path, indent=2, force_ascii=False)
    print(f"[DONE] Saved results to {out_path}")


if __name__ == "__main__":
    main()
