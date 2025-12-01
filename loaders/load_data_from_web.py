from typing import Iterable, Optional, Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

STEAMDB_INFO = "https://steamdb.info/app/{app_id}/info/"
STEAMDB_PRICE = "https://steamdb.info/app/{app_id}/price/"
STEAMDB_PATCH = "https://steamdb.info/app/{app_id}/patchnotes/"
STEAM_COMMUNITY_DISCUSS = "https://steamcommunity.com/app/{app_id}/discussions/"


def _parse_date(text: Optional[str]) -> Optional[pd.Timestamp]:
    if not text:
        return None
    try:
        return pd.to_datetime(text.strip(), errors="coerce")
    except Exception:
        return None


def _fetch_ea_info(app_id: int, session: requests.Session, delay: float) -> Dict[str, object]:
    import time

    url = STEAMDB_INFO.format(app_id=app_id)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    ea_date = None
    for row in soup.select("table.table-info tr"):
        header = row.find("td", class_="table-label")
        if not header:
            continue
        label = header.get_text(strip=True).lower()
        if "early access release date" in label:
            val = row.find("td", class_="table-value")
            if val:
                ea_date = _parse_date(val.get_text(strip=True))
                break
    time.sleep(delay)
    return {"ea_start_date": ea_date}


def _fetch_price_history(app_id: int, session: requests.Session, delay: float) -> Dict[str, object]:
    import time

    url = STEAMDB_PRICE.format(app_id=app_id)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    ea_price = None
    first_discount_rate = None
    first_discount_date = None
    if table:
        rows = table.find_all("tr")
        for row in rows[1:]:
            cells = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cells) < 3:
                continue
            date_text, price_text, discount_text = cells[0], cells[1], cells[2]
            if price_text:
                num = "".join(ch for ch in price_text if (ch.isdigit() or ch in ".,"))
                try:
                    ea_price = float(num.replace(",", ""))
                except ValueError:
                    ea_price = None
            if discount_text and discount_text.endswith("%"):
                try:
                    first_discount_rate = float(discount_text.rstrip("%"))
                except ValueError:
                    first_discount_rate = None
            first_discount_date = _parse_date(date_text)
            break
    time.sleep(delay)
    return {
        "ea_price": ea_price,
        "first_discount_rate": first_discount_rate,
        "first_discount_date": first_discount_date,
    }


def _fetch_patchnotes(app_id: int, session: requests.Session, delay: float) -> Dict[str, object]:
    import time

    url = STEAMDB_PATCH.format(app_id=app_id)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    dates: List[pd.Timestamp] = []
    for row in soup.select("table.table.table-hover tbody tr"):
        date_cell = row.select_one("td:nth-child(1)")
        if not date_cell:
            continue
        dt = _parse_date(date_cell.get_text(strip=True))
        if pd.notna(dt):
            dates.append(dt)
    dates = sorted([d for d in dates if pd.notna(d)])
    update_count = len(dates)
    avg_interval = None
    max_gap = None
    if len(dates) >= 2:
        deltas = pd.Series(dates).diff().dt.days.dropna()
        avg_interval = deltas.mean()
        max_gap = deltas.max()
    time.sleep(delay)
    return {
        "update_count": update_count,
        "avg_update_interval": avg_interval,
        "max_update_gap": max_gap,
    }


def _fetch_community_posts(app_id: int, session: requests.Session, delay: float) -> Dict[str, object]:
    import time

    url = STEAM_COMMUNITY_DISCUSS.format(app_id=app_id)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    topics = soup.select("div.forum_topic")
    count = len(topics)
    time.sleep(delay)
    return {"community_posts": count}


def fetch_meta_web(app_ids: Iterable[int], delay: float = 1.0, limit: Optional[int] = None) -> pd.DataFrame:
    ids_series = pd.Series(app_ids).dropna().astype(int)
    if limit is not None:
        ids_series = ids_series.head(limit)
    session = requests.Session()
    rows = []
    for idx, app_id in enumerate(ids_series.tolist(), start=1):
        base = {"app_id": int(app_id)}
        try:
            base.update(_fetch_ea_info(app_id, session, delay))
            base.update(_fetch_price_history(app_id, session, delay))
            base.update(_fetch_patchnotes(app_id, session, delay))
            base.update(_fetch_community_posts(app_id, session, delay))
        except requests.RequestException:
            pass
        rows.append(base)
        if idx % 10 == 0:
            print(f"[WEB] {idx}/{len(ids_series)} processed")
    return pd.DataFrame(rows)
