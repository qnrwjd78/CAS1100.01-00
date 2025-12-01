from typing import List, Optional

import numpy as np
import pandas as pd
import requests

STORE_API = "https://store.steampowered.com/api/appdetails"
STEAMSPY_ENDPOINT = "https://steamspy.com/api.php"


class SteamStoreFetcher:
    def __init__(self, sleep_seconds: float = 0.5):
        self.session = requests.Session()
        self.sleep_seconds = sleep_seconds

    def fetch(self, app_id: int):
        import time

        params = {"appids": app_id, "cc": "us", "l": "en"}
        resp = self.session.get(STORE_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get(str(app_id), {}).get("data", {})
        release_date = data.get("release_date", {}).get("date")
        price_info = data.get("price_overview", {}) or {}
        final_price = price_info.get("final")  # cents
        initial_price = price_info.get("initial")  # cents
        discount = price_info.get("discount_percent")
        currency = price_info.get("currency")
        time.sleep(self.sleep_seconds)
        return {
            "app_id": app_id,
            "release_date": pd.to_datetime(release_date, errors="coerce"),
            "release_price": (final_price / 100) if final_price is not None else None,
            "initial_list_price": (initial_price / 100) if initial_price is not None else None,
            "discount_percent": discount,
            "currency": currency,
        }


class SteamSpyFetcher:
    def __init__(self, sleep_seconds: float = 0.5):
        self.session = requests.Session()
        self.sleep_seconds = sleep_seconds

    def fetch(self, app_id: int):
        import time

        resp = self.session.get(STEAMSPY_ENDPOINT, params={"request": "appdetails", "appid": app_id}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        time.sleep(self.sleep_seconds)
        price_raw = data.get("price")
        price = None
        try:
            price = float(price_raw) if price_raw is not None else None
        except (TypeError, ValueError):
            price = None
        owners = data.get("owners")
        owners_min = owners_max = owners_median = None
        if owners:
            parts = str(owners).replace(",", "").split("..")
            if len(parts) == 2:
                try:
                    owners_min = float(parts[0])
                    owners_max = float(parts[1])
                    owners_median = (owners_min + owners_max) / 2
                except ValueError:
                    pass
        return {
            "app_id": app_id,
            "release_price_steamspy": (price / 100) if price is not None else None,
            "avg_playtime": data.get("average_forever"),
            "owners_min": owners_min,
            "owners_max": owners_max,
            "owners_median": owners_median,
        }


def fetch_meta_api(
    app_ids: List[int],
    sleep_store: float = 0.05,
    sleep_steamspy: float = 0.05,
    progress_ratio: float = 0.1,
    target: Optional[int] = None,
) -> pd.DataFrame:
    store_fetcher = SteamStoreFetcher(sleep_seconds=sleep_store)
    steamspy_fetcher = SteamSpyFetcher(sleep_seconds=sleep_steamspy)

    progress_every = max(1, int(len(app_ids) * progress_ratio))
    records = []
    kept = 0
    for idx, app_id in enumerate(app_ids, start=1):
        base = {"app_id": int(app_id)}
        try:
            base.update(store_fetcher.fetch(app_id))
        except requests.RequestException:
            pass
        try:
            base.update(steamspy_fetcher.fetch(app_id))
        except requests.RequestException:
            pass
        has_data = any(
            (v is not None) and not (isinstance(v, float) and np.isnan(v))
            for k, v in base.items()
            if k != "app_id"
        )
        if len(base) > 1 and has_data:
            records.append(base)
            kept += 1
        if idx % progress_every == 0:
            print(f"[API] {idx}/{len(app_ids)} processed: {kept} kept")
        if target is not None and kept >= target:
            print(f"[API] Reached target {target}, stopping early.")
            break

    print(f"[API] saved {kept}, total processed {idx}.")
    df = pd.DataFrame(records)
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df
