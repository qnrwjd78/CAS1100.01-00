"""
End-to-end data loader:
- Sample app_ids from review partitions
- Fetch store + SteamSpy metadata
- Enrich with SteamDB/Community (best effort; fills blanks on failure)
- Save final merged CSV only

Configuration: edit the constants below.
"""
from pathlib import Path
import random

import pandas as pd

from loaders.load_data_from_dataset import collect_app_ids_from_reviews
from loaders.load_data_from_api import fetch_meta_api
from loaders.load_data_from_web import fetch_meta_web

# -------------------
# Configurable params
# -------------------
REVIEW_DIR = Path("data/reviews")
PATTERN = "reviews-*.csv"
TARGET = 600
SAMPLE_MULTIPLIER = 2.0  # sample extra ids to offset missing release_date
OUT_DIR = Path("data")
SLEEP_STORE = 0.01
SLEEP_STEAMSPY = 0.01
WEB_DELAY = 1.0  # delay between SteamDB requests


def main() -> None:
    candidate_count = int(TARGET * SAMPLE_MULTIPLIER)
    print(f"[DATASET] Collecting up to {candidate_count} app_ids from reviews...")
    ids = collect_app_ids_from_reviews(REVIEW_DIR, PATTERN, max_ids=candidate_count)
    random.shuffle(ids)
    ids = ids[:candidate_count]
    print(f"[DATASET] Collected {len(ids)} app_ids")

    print("[API] Fetching store + SteamSpy metadata...")
    api_df = fetch_meta_api(
        ids,
        sleep_store=SLEEP_STORE,
        sleep_steamspy=SLEEP_STEAMSPY,
        progress_ratio=0.1,
        target=candidate_count,  # stop once we've attempted enough to cover target after filtering
    )

    api_df["release_date"] = pd.to_datetime(api_df.get("release_date"), errors="coerce")
    api_df = api_df.dropna(subset=["release_date"])
    api_df = api_df.head(TARGET)
    final_ids = api_df["app_id"].dropna().astype(int).tolist()
    print(f"[API] Kept {len(final_ids)} app_ids with release_date")

    print("[WEB] Enriching via SteamDB/Community (best effort)...")
    web_df = fetch_meta_web(final_ids, delay=WEB_DELAY)

    merged = api_df.merge(web_df, on="app_id", how="left", suffixes=("", "_web"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / "merged_sampled.csv", index=False)

    print(f"[DONE] merged_sampled.csv ({len(merged)} rows)")


if __name__ == "__main__":
    main()
