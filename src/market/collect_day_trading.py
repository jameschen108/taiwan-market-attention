"""當日沖銷交易標的及成交量值 TWTB4U（LIMITATIONS.md §12.2）。

2020–2024 是台股當沖佔比由約 20% 升至約 40% 的期間。當沖灌大 `volume` 卻不改變
淨部位，而非三大法人訂單失衡的分母 ≈ 2 × volume：

```
ROI = (r_buy − r_sell) / (r_buy + r_sell)，  r_* = volume − inst_*
```

高當沖的個股（正好就是高關注度那些）ROI 被機械性壓向 0，**偏誤方向向下、且偏誤
大小與自變數正相關**。週固定效果吸收得了全市場的時間趨勢，吸收不了橫斷面上的
這一層。

TWSE 的回應是新版 `tables` 結構（兩張表：全市場統計、逐檔明細），不是舊版的
`data` 陣列——舊的解析方式會拿到空清單而**不報錯**，因此本模組明確檢查表名。

輸出 `data/interim/day_trading.csv`：ticker × 交易日 × 當沖股數／買賣金額。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import ROOT

URL = "https://www.twse.com.tw/exchangeReport/TWTB4U"
HEADERS = {"User-Agent": "Mozilla/5.0 (research data collection)"}
RAW = ROOT / "data" / "raw" / "twse_twtb4u"
OUT = ROOT / "data" / "interim" / "day_trading.csv"
DETAIL_TITLE = "當日沖銷交易標的及成交量值"


def _num(value) -> float | None:
    s = str(value).replace(",", "").strip()
    if not s or s in ("--", "---", "-----"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_day(day: pd.Timestamp, session: requests.Session,
              max_retries: int = 5) -> list[list]:
    delay = 10.0
    for attempt in range(max_retries):
        try:
            resp = session.get(URL, params={"date": f"{day:%Y%m%d}",
                                            "selectType": "All", "response": "json"},
                               headers=HEADERS, timeout=90)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"TWTB4U {day:%Y-%m-%d}: {exc}") from exc
            time.sleep(delay); delay = min(delay * 2, 300); continue
        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                body = {}
            if body.get("stat") == "OK":
                for table in body.get("tables") or ():
                    if DETAIL_TITLE in str(table.get("title", "")):
                        return table.get("data") or []
                return []            # 休市日：stat OK 但無明細表
        if attempt == max_retries - 1:
            raise RuntimeError(f"TWTB4U {day:%Y-%m-%d}: HTTP {resp.status_code}")
        time.sleep(delay); delay = min(delay * 2, 300)
    return []


def collect(trading_days_csv: Path, out_path: Path = OUT, tickers: set[str] | None = None,
            raw_dir: Path = RAW, start: str = "2019-01-01",
            end: str = "2024-12-31") -> pd.DataFrame:
    cal = pd.read_csv(trading_days_csv, parse_dates=["date"])
    days = cal.loc[(cal["date"] >= start) & (cal["date"] <= end), "date"]
    session = requests.Session()
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows, n_fetched = [], 0
    for i, day in enumerate(days, 1):
        cache = raw_dir / f"twtb4u_{day:%Y%m%d}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            data = fetch_day(day, session)
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            n_fetched += 1
            time.sleep(random.uniform(1.5, 2.5))
        for r in data:
            ticker = str(r[0]).strip()
            if not re.fullmatch(r"\d{4}", ticker):
                continue
            if tickers is not None and ticker not in tickers:
                continue
            rows.append({
                "ticker": ticker, "date": day,
                "dt_volume": _num(r[3]),
                "dt_buy_value": _num(r[4]),
                "dt_sell_value": _num(r[5]),
                # 「暫停現股賣出後現款買進當沖」註記＝該檔當日被限制當沖
                "dt_restricted": bool(str(r[2]).strip()),
            })
        if i % 50 == 0 or i == len(days):
            print(f"  {i}/{len(days)} 交易日（新抓 {n_fetched}），累計 {len(rows):,} 列",
                  flush=True)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if not df.empty:
        print(f"當沖 {len(df):,} 列，{df['ticker'].nunique()} 檔，"
              f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TWTB4U 當沖收集")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--universe-only", action="store_true",
                    help="只留宇宙內的 267 檔（大幅縮小輸出）")
    args = ap.parse_args(argv)
    uni = None
    if args.universe_only:
        uni = set(pd.read_csv(ROOT / "data" / "external" / "universe.csv",
                              dtype={"ticker": str})["ticker"])
    collect(ROOT / "data" / "interim" / "trading_days.csv", tickers=uni,
            start=args.start, end=args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
