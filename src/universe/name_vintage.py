"""名稱時效性稽核：宇宙清單用的是**現在**的簡稱，樣本卻是 2020–2024。

`data/external/universe.csv` 的 `name_short` 是建清單當下（2026）的公司簡稱。
公司在樣本期內更名時，語料裡寫的是**當時**的名字，比對器只認現在的名字——
該檔在更名前的關注度會被系統性漏掉。

本模組不猜，直接量：掃全語料，統計每個代號旁緊鄰的中文名，與宇宙的 `name_short`
比對。輸出 `audit/name_vintage_check.csv`，逐年拆開以便判斷是否為更名。

這是**宇宙層級的 look-ahead 的一種**（`LIMITATIONS.md` §1 已記載清單本身的
look-ahead，但未記載名稱的）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..audit_corpus import AUDIT, _write_csv
from ..config import ROOT, load_settings, resolve

CJK = r"[一-鿿]"
MIN_HITS = 20
DOMINANCE = 0.30


def scan(root: Path, tickers: dict[str, str]) -> pd.DataFrame:
    """回傳 (ticker, year, name, n)。只數代號與中文名**緊鄰**出現的情形。"""
    pats = {t: re.compile(
        rf"{t}\s*[)\）]?\s*({CJK}{{2,4}})|({CJK}{{2,4}})\s*[\(（]\s*{t}")
        for t in tickers}
    counts: dict[tuple[str, int, str], int] = defaultdict(int)
    for path in sorted(root.glob("stock_*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                raw = line.encode()
                present = [t for t in tickers if t.encode() in raw]
                if not present:
                    continue
                doc = json.loads(line)
                year = datetime.fromisoformat(doc["date_ts"]).year
                text = f"{doc.get('title') or ''}\n{doc.get('content') or ''}"
                for t in present:
                    for m in pats[t].finditer(text):
                        name = m.group(1) or m.group(2)
                        if name:
                            counts[(t, year, name)] += 1
        print(f"  {path.name} 掃描完成", flush=True)
    return pd.DataFrame(
        [{"ticker": t, "year": y, "observed_name": n, "n": c}
         for (t, y, n), c in counts.items()])


def report(df: pd.DataFrame, tickers: dict[str, str]) -> list[dict]:
    rows = []
    for ticker, grp in df.groupby("ticker"):
        official = tickers[ticker]
        total = int(grp["n"].sum())
        if total < MIN_HITS:
            continue
        by_name = grp.groupby("observed_name")["n"].sum().sort_values(ascending=False)
        top, top_n = by_name.index[0], int(by_name.iloc[0])
        n_official = int(by_name.get(official, 0))
        # 觀測到、且佔比夠高、卻不是官方簡稱的名字
        alts = [(nm, int(k)) for nm, k in by_name.items()
                if nm != official and k / total >= DOMINANCE]
        if not alts:
            continue
        yr = (grp[grp["observed_name"].isin([official] + [a for a, _ in alts])]
              .pivot_table(index="year", columns="observed_name", values="n",
                           aggfunc="sum").fillna(0).astype(int))
        rows.append({
            "ticker": ticker,
            "universe_name": official,
            "n_total": total,
            "n_universe_name": n_official,
            "pct_universe_name": round(n_official / total, 3),
            "alt_name": alts[0][0],
            "n_alt_name": alts[0][1],
            "pct_alt_name": round(alts[0][1] / total, 3),
            "by_year": yr.to_json(orient="index", force_ascii=False),
        })
    rows.sort(key=lambda r: -r["pct_alt_name"])
    _write_csv(AUDIT / "name_vintage_check.csv", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="名稱時效性稽核")
    ap.parse_args(argv)
    s = load_settings()
    uni = pd.read_csv(ROOT / "data" / "external" / "universe.csv", dtype={"ticker": str})
    tickers = dict(zip(uni["ticker"], uni["name_short"]))
    print(f"[名稱時效] 掃描 {len(tickers)} 檔")
    df = scan(resolve(s["ptt"]["root"]), tickers)
    rows = report(df, tickers)
    print(f"\n代號旁出現**非**宇宙簡稱、且佔比 ≥ {DOMINANCE:.0%} 的檔數：{len(rows)}")
    for r in rows:
        print(f"  {r['ticker']} 宇宙作「{r['universe_name']}」"
              f"（{r['pct_universe_name']:.0%}），語料多作「{r['alt_name']}」"
              f"（{r['pct_alt_name']:.0%}，{r['n_alt_name']} 次）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
