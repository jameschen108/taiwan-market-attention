"""處置有價證券清單（LIMITATIONS.md §12.1）。

TWSE 對量價異常標的採分盤交易（5 或 20 分鐘撮合一次）＋ 預收款券。**觸發條件正是
成交量、週轉率、漲跌幅異常**——也就是高關注度。處置一旦生效，後續的成交量與周轉率
被制度性壓縮，而 `turnover_next` **就是 H3b 的應變數**。這是「處理」直接決定
「結果」的反向因果，不是雜訊，因此必須標旗標並做「排除處置週」的穩健性。

來源：TWSE 公布處置有價證券（`announcement/punish`）。回傳含**處置起迄時間**，
不是只有公布日——這是關鍵，處置的影響發生在處置期間而非公布日。

輸出 `data/interim/disposition.csv`：ticker × 處置區間。
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import ROOT

URL = "https://www.twse.com.tw/rwd/zh/announcement/punish"
HEADERS = {"User-Agent": "Mozilla/5.0 (research data collection)"}
RAW = ROOT / "data" / "raw" / "twse_punish"
OUT = ROOT / "data" / "interim" / "disposition.csv"

#: 「113/03/11～113/03/22」；處置期間偶有全形波浪號與半形混用
PERIOD = re.compile(r"(\d{2,3}/\d{1,2}/\d{1,2})\s*[～~\-]\s*(\d{2,3}/\d{1,2}/\d{1,2})")


def _roc(value: str) -> pd.Timestamp | None:
    try:
        y, m, d = (int(x) for x in value.split("/"))
    except ValueError:
        return None
    return pd.Timestamp(year=y + 1911, month=m, day=d)


def fetch_range(start: str, end: str, session: requests.Session,
                max_retries: int = 5) -> list[list]:
    delay = 10.0
    for attempt in range(max_retries):
        try:
            resp = session.get(URL, params={"startDate": start, "endDate": end,
                                            "response": "json"},
                               headers=HEADERS, timeout=90)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"punish {start}~{end}: {exc}") from exc
            time.sleep(delay); delay = min(delay * 2, 300); continue
        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                body = {}
            if body.get("stat") == "OK":
                return body.get("data") or []
            if "無" in str(body.get("stat", "")):
                return []
        if attempt == max_retries - 1:
            raise RuntimeError(f"punish {start}~{end}: HTTP {resp.status_code}")
        time.sleep(delay); delay = min(delay * 2, 300)
    return []


def collect(start_year: int, end_year: int, out_path: Path = OUT,
            raw_dir: Path = RAW) -> pd.DataFrame:
    session = requests.Session()
    rows: list[list] = []
    for year in range(start_year, end_year + 1):
        for q_start, q_end in (("0101", "0331"), ("0401", "0630"),
                               ("0701", "0930"), ("1001", "1231")):
            s, e = f"{year}{q_start}", f"{year}{q_end}"
            cache = raw_dir / f"punish_{s}_{e}.json"
            if cache.exists():
                rows.extend(json.loads(cache.read_text(encoding="utf-8")))
                continue
            chunk = fetch_range(s, e, session)
            raw_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
            rows.extend(chunk)
            print(f"  {s}~{e}: {len(chunk)} 筆", flush=True)
            time.sleep(random.uniform(2.0, 4.0))

    if not rows:
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    out = []
    for r in rows:
        ticker = str(r[2]).strip()
        # 權證、ETN 等非四位數代號一律排除（宇宙全為四位數上市普通股）
        if not re.fullmatch(r"\d{4}", ticker):
            continue
        m = PERIOD.search(str(r[6]))
        if not m:
            continue
        s_date, e_date = _roc(m.group(1)), _roc(m.group(2))
        if s_date is None or e_date is None:
            continue
        out.append({
            "ticker": ticker,
            "name": str(r[3]).strip(),
            "announce_date": _roc(str(r[1]).strip()),
            "start_date": s_date,
            "end_date": e_date,
            "n_days": (e_date - s_date).days + 1,
            "condition": str(r[5]).strip(),
            "措施": "分盤+預收" if "預收" in str(r[8]) else "分盤",
        })
    df = pd.DataFrame(out).sort_values(["ticker", "start_date"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"處置事件 {len(df):,} 筆，{df['ticker'].nunique()} 檔，"
          f"{df['start_date'].min().date()} ~ {df['end_date'].max().date()}")
    return df


def main() -> int:
    collect(2019, 2025)
    return 0


if __name__ == "__main__":
    sys.exit(main())
