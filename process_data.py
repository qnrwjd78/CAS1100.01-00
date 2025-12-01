"""
Answer 6 developer-oriented questions using merged_sampled.csv.

Each question has one function. Results are saved to data/analysis_results.json.
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import add_features, cohen_d, load_merged, mannwhitney_p, simple_regression


# Q1 최적 출시가 구간은?
def q1_optimal_price_bucket(df: pd.DataFrame) -> Dict[str, object]:
    grouped = df.groupby("price_bucket", observed=False)["positive_ratio"].agg(["mean", "count"])
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


# Q2 출시 직후 리뷰 속도 목표는?
def q2_review_speed_targets(df: pd.DataFrame) -> Dict[str, float]:
    series = df["reviews_per_day_90d"].dropna()
    stats = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return {k: round(float(v), 4) for k, v in stats.to_dict().items()}


# Q3 초기 참여도 목표는?
def q3_engagement_targets(df: pd.DataFrame) -> Dict[str, float]:
    series = df["engagement_ratio"].dropna()
    stats = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return {k: round(float(v), 6) for k, v in stats.to_dict().items()}


# Q4 가격이 평점에 주는 영향은?
def q4_price_effect_on_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    res = simple_regression(df["log_price"], df["positive_ratio"])
    return {k: (None if np.isnan(v) else round(v, 4)) for k, v in res.items()}


# Q5 플레이타임이 평점에 주는 영향은?
def q5_playtime_effect_on_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    res = simple_regression(df["avg_playtime"], df["positive_ratio"])
    mask = df["avg_playtime"].notna() & df["positive_ratio"].notna()
    corr = df.loc[mask, ["avg_playtime", "positive_ratio"]].corr().iloc[0, 1]
    return {
        "corr": round(float(corr), 4) if pd.notna(corr) else None,
        "slope": None if np.isnan(res["slope"]) else round(res["slope"], 6),
        "intercept": None if np.isnan(res["intercept"]) else round(res["intercept"], 4),
    }


# Q6 판매 규모가 평점에 주는 영향은?
def q6_scale_effect_on_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    mask = df["owners_median"].notna() & df["positive_ratio"].notna()
    corr = df.loc[mask, ["owners_median", "positive_ratio"]].corr().iloc[0, 1]
    return {"corr": round(float(corr), 4) if pd.notna(corr) else None}


QUESTION_FUNCS = [
    q1_optimal_price_bucket,
    q2_review_speed_targets,
    q3_engagement_targets,
    q4_price_effect_on_sentiment,
    q5_playtime_effect_on_sentiment,
    q6_scale_effect_on_sentiment,
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
    print(f"분석 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
