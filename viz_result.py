"""
Simple visualization of analysis results and merged data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import add_features, load_merged


def plot_price_vs_sentiment(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="release_price", y="positive_ratio", hue="price_bucket")
    plt.title("Price vs Positive Ratio (90d)")
    plt.xlabel("Release price")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "price_vs_sentiment.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_review_velocity(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["reviews_per_day_90d"].dropna(), bins=30, kde=True)
    plt.title("Review velocity (90d)")
    plt.xlabel("Reviews per day")
    out = out_dir / "review_velocity.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_engagement(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="price_bucket", y="engagement_ratio")
    plt.yscale("log")
    plt.title("Engagement ratio by price bucket")
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
    plt.title("Update count vs Positive Ratio")
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
    plt.title("Community posts vs Positive Ratio")
    plt.xlabel("Community posts (first page count)")
    plt.ylabel("Positive ratio (%)")
    out = out_dir / "community_vs_sentiment.png"
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
    print(f"[DONE] Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
