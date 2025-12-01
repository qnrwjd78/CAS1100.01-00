from pathlib import Path

import matplotlib
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["font.sans-serif"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import add_features, load_merged


def plot_price_vs_sentiment(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="release_price", y="positive_ratio", hue="price_bucket")
    plt.title("Q1: 가격 vs 평점 (산점)")
    plt.xlabel("Release price")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "Q1_price_vs_sentiment.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_review_velocity(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["reviews_per_day_90d"].dropna(), bins=30, kde=True)
    plt.title("Q2: 리뷰 속도 분포 (90일)")
    plt.xlabel("Reviews per day")
    out = out_dir / "Q2_review_velocity.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_engagement(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="price_bucket", y="engagement_ratio")
    plt.yscale("log")
    plt.title("Q3: 가격대별 참여도 (박스플롯)")
    plt.xlabel("Price bucket")
    plt.ylabel("Engagement (reviews / owners)")
    out = out_dir / "Q3_engagement_by_price.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_price_effect(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.regplot(data=df, x="log_price", y="positive_ratio", scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
    plt.title("Q4: 가격(Log) vs 평점")
    plt.xlabel("Log(Release Price + 1)")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "Q4_price_effect.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_playtime_effect(df: pd.DataFrame, out_dir: Path) -> None:
    if "avg_playtime" not in df.columns:
        return
    subset = df[df["avg_playtime"].notna() & df["positive_ratio"].notna()]
    if subset.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.regplot(data=subset, x="avg_playtime", y="positive_ratio", scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
    plt.xscale("log")
    plt.title("Q5: 플레이타임 vs 평점")
    plt.xlabel("Average Playtime (min, log scale)")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "Q5_playtime_effect.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_scale_effect(df: pd.DataFrame, out_dir: Path) -> None:
    if "owners_median" not in df.columns:
        return
    subset = df[df["owners_median"].notna() & df["positive_ratio"].notna()]
    if subset.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.regplot(data=subset, x="owners_median", y="positive_ratio", scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
    plt.xscale("log")
    plt.title("Q6: 판매 규모(Owners) vs 평점")
    plt.xlabel("Owners Median (log scale)")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "Q6_scale_effect.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_price_buckets(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="price_bucket", y="positive_ratio", estimator="mean", errorbar=None)
    plt.title("Q1: 가격 버킷별 평균 평점")
    plt.xlabel("Price bucket")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "Q1_price_bucket_avg.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="price_bucket", y="engagement_ratio", estimator="mean", errorbar=None)
    plt.title("Q1: 가격 버킷별 평균 참여도")
    plt.xlabel("Price bucket")
    plt.ylabel("Engagement (reviews / owners)")
    plt.yscale("log")
    out = out_dir / "Q1_price_bucket_engagement.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_speed_vs_sentiment(df: pd.DataFrame, out_dir: Path) -> None:
    if "reviews_per_day_90d" not in df.columns:
        return
    subset = df[df["reviews_per_day_90d"].notna() & df["positive_ratio"].notna()]
    if subset.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=subset, x="reviews_per_day_90d", y="positive_ratio")
    plt.xscale("log")
    plt.title("Q2: 리뷰 속도 vs 평점")
    plt.xlabel("Reviews per day (log scale)")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "Q2_speed_vs_sentiment.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    out_dir = Path("data/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_features(load_merged())
    plot_price_vs_sentiment(df, out_dir)
    plot_review_velocity(df, out_dir)
    plot_engagement(df, out_dir)
    plot_price_effect(df, out_dir)
    plot_playtime_effect(df, out_dir)
    plot_scale_effect(df, out_dir)
    plot_price_buckets(df, out_dir)
    plot_speed_vs_sentiment(df, out_dir)
    print(f"[DONE] Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
