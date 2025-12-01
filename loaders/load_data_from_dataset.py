from pathlib import Path
from typing import List, Set

import pandas as pd


def collect_app_ids_from_reviews(review_dir: Path, pattern: str, max_ids: int) -> List[int]:
    ids: Set[int] = set()
    paths = sorted(review_dir.glob(pattern))
    for p in paths:
        for chunk in pd.read_csv(p, usecols=["appid"], chunksize=200_000):
            for appid in chunk["appid"].dropna().astype(int):
                if appid not in ids:
                    ids.add(appid)
                    if len(ids) >= max_ids:
                        return list(ids)
    return list(ids)
