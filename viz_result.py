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
    out = out_dir / "price_vs_sentiment.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_review_velocity(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["reviews_per_day_90d"].dropna(), bins=30, kde=True)
    plt.title("Q2: 리뷰 속도 분포 (90일)")
    plt.xlabel("Reviews per day")
    out = out_dir / "review_velocity.png"
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
    out = out_dir / "engagement_by_price.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_updates_vs_sentiment(df: pd.DataFrame, out_dir: Path) -> None:
    if "update_count" not in df.columns:
        return
    subset = df[df["update_count"].notna() & df["positive_ratio"].notna()]
    if subset.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=subset, x="update_count", y="positive_ratio")
    plt.title("Q7: 업데이트 횟수 vs 평점")
    plt.xlabel("Update count")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "updates_vs_sentiment.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_community_vs_sentiment(df: pd.DataFrame, out_dir: Path) -> None:
    if "community_posts" not in df.columns:
        return
    subset = df[df["community_posts"].notna() & df["positive_ratio"].notna()]
    if subset.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=subset, x="community_posts", y="positive_ratio")
    plt.title("Q8: 커뮤니티 게시글 vs 평점")
    plt.xlabel("Community posts (first page count)")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "community_vs_sentiment.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_price_buckets(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="price_bucket", y="positive_ratio", estimator="mean", errorbar=None)
    plt.title("Q1: 가격 버킷별 평균 평점")
    plt.xlabel("Price bucket")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "price_bucket_avg.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="price_bucket", y="engagement_ratio", estimator="mean", errorbar=None)
    plt.title("Q1: 가격 버킷별 평균 참여도")
    plt.xlabel("Price bucket")
    plt.ylabel("Engagement (reviews / owners)")
    plt.yscale("log")
    out = out_dir / "price_bucket_engagement.png"
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
    out = out_dir / "speed_vs_sentiment.png"
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
    plot_updates_vs_sentiment(df, out_dir)
    plot_community_vs_sentiment(df, out_dir)
    plot_price_buckets(df, out_dir)
    plot_speed_vs_sentiment(df, out_dir)
    print(f"[DONE] Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
