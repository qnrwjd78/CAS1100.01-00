from pathlib import Path
from typing import Iterable

import pandas as pd


def aggregate_reviews(app_ids: Iterable[int], release_df: pd.DataFrame, review_dir: Path, pattern: str, window_days: int = 90) -> pd.DataFrame:
    app_set = set(app_ids)
    release_map = {
        int(r.app_id): pd.to_datetime(r.release_date, errors="coerce").timestamp()
        for _, r in release_df.dropna().iterrows()
    }
    counters = {}
    window_sec = window_days * 86400
    for path in sorted(review_dir.glob(pattern)):
        try:
            reader = pd.read_csv(
                path,
                usecols=["appid", "unix_timestamp_created", "voted_up"],
                chunksize=200_000,
            )
        except Exception as e:
            print(f"{path.name} 읽기 실패: {e}")
            continue
        for chunk in reader:
            chunk["appid"] = pd.to_numeric(chunk["appid"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["appid"].isin(app_set)]
            if chunk.empty:
                continue
            chunk["ts"] = pd.to_numeric(chunk["unix_timestamp_created"], errors="coerce")
            chunk = chunk.dropna(subset=["ts"])
            if chunk.empty:
                continue
            chunk["ts"] = chunk["ts"].astype(int)
            chunk["voted_up"] = (
                chunk["voted_up"]
                .astype(str)
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(int)
            )
            for app_id, group in chunk.groupby("appid"):
                rel_ts = release_map.get(int(app_id))
                if rel_ts is None:
                    continue
                mask = (group["ts"] >= rel_ts) & (group["ts"] < rel_ts + window_sec)
                if not mask.any():
                    continue
                sub = group[mask]
                rec = counters.get(int(app_id), {"review_count": 0, "positive_count": 0, "ts_min": None, "ts_max": None})
                rec["review_count"] += len(sub)
                rec["positive_count"] += int(sub["voted_up"].sum())
                ts_min = sub["ts"].min()
                ts_max = sub["ts"].max()
                rec["ts_min"] = ts_min if rec["ts_min"] is None else min(rec["ts_min"], ts_min)
                rec["ts_max"] = ts_max if rec["ts_max"] is None else max(rec["ts_max"], ts_max)
                counters[int(app_id)] = rec
    rows = []
    for app_id, rec in counters.items():
        review_count = rec["review_count"]
        positive_count = rec["positive_count"]
        positive_ratio = (positive_count / review_count * 100) if review_count else None
        rows.append(
            {
                "app_id": app_id,
                "review_count": review_count,
                "positive_count": positive_count,
                "ts_min": rec["ts_min"],
                "ts_max": rec["ts_max"],
                "positive_ratio": positive_ratio,
            }
        )
    return pd.DataFrame(rows)
