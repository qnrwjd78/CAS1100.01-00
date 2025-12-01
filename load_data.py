from pathlib import Path
import random

import pandas as pd

from loaders.load_data_from_dataset import collect_app_ids_from_reviews
from loaders.load_data_from_api import fetch_meta_api
from loaders.load_data_from_web import fetch_meta_web
from loaders.load_data_from_review import aggregate_reviews

REVIEW_DIR = Path("data/reviews")
PATTERN = "reviews-*.csv"
TARGET = 600
SAMPLE_MULTIPLIER = 2.0
OUT_DIR = Path("data")
SLEEP_STORE = 0.001
SLEEP_STEAMSPY = 0.001
WEB_DELAY = 0.1
REVIEW_WINDOW_DAYS = 90


def main() -> None:
    candidate_count = int(TARGET * SAMPLE_MULTIPLIER)
    print(f"리뷰 데이터에서 {candidate_count}개 app_id 수집 시도...")
    ids = collect_app_ids_from_reviews(REVIEW_DIR, PATTERN, max_ids=candidate_count)
    random.shuffle(ids)
    ids = ids[:candidate_count]
    print(f"수집된 app_id: {len(ids)}개")

    print("스토어+SteamSpy 메타 수집 중...")
    api_df = fetch_meta_api(
        ids,
        sleep_store=SLEEP_STORE,
        sleep_steamspy=SLEEP_STEAMSPY,
        progress_ratio=0.1,
        target=TARGET,
    )

    api_df["release_date"] = pd.to_datetime(api_df.get("release_date"), errors="coerce")
    api_df = api_df.dropna(subset=["release_date"]).head(TARGET)
    final_ids = api_df["app_id"].dropna().astype(int).tolist()
    print(f"release_date 있는 app_id: {len(final_ids)}개")

    print("리뷰 집계(출시~90일) 중...")
    review_df = aggregate_reviews(final_ids, api_df[["app_id", "release_date"]], REVIEW_DIR, PATTERN, window_days=REVIEW_WINDOW_DAYS)

    print("SteamDB/커뮤니티 보강 시도 중...")
    web_df = fetch_meta_web(final_ids, delay=WEB_DELAY)

    merged = (
        api_df.merge(review_df, on="app_id", how="left")
        .merge(web_df, on="app_id", how="left", suffixes=("", "_web"))
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / "merged_sampled.csv", index=False)

    print(f"최종 저장: merged_sampled.csv ({len(merged)}행)")


if __name__ == "__main__":
    main()
