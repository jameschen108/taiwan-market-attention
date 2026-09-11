"""P2 稽核：配對層的實得結構，以及 v2 核心問題的直接量測。

v2 的第一個動機（`docs/PLAN_V2.md` §2-1）是：v1 把一篇文章的全部關注度壓在
「發文那一刻」的窗口，但一篇週五 22:00 的文章，其留言大量落在週六日。用有時戳的
留言重切窗口後，週末關注度是否被 v1 系統性低估？

本模組**直接量**這件事，不需要任何回歸：同一批留言，一次用母文章時戳指派窗口
（v1 的作法），一次用留言自己的時戳指派（v2 的作法），比較兩者的週末計數。

輸出：
- `audit/ptt_match_structure.csv`　配對結構（match_mode、bulk、effort）
- `audit/ptt_window_displacement.csv`　窗口位移的直接量測
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .audit_corpus import AUDIT, _write_csv
from .config import ROOT, load_settings

INTERIM = ROOT / "data" / "interim"


def _in_main_sample(ts: pd.Series, s: dict) -> pd.Series:
    return (ts >= s["sample"]["main_start"]) & (ts <= f"{s['sample']['main_end']} 23:59:59")


def structure(source: str, s: dict) -> list[dict]:
    a = pd.read_parquet(INTERIM / f"ptt_matches_{source}.parquet")
    main = a[_in_main_sample(a["timestamp"], s)]
    rows: list[dict] = []

    def add(kind: str, key: str, n: int, denom: int) -> None:
        rows.append({"source": source, "kind": kind, "key": key,
                     "n": n, "pct": round(n / denom, 4) if denom else ""})

    add("total", "ticker_article_rows_all", len(a), len(a))
    add("total", "ticker_article_rows_main_sample", len(main), len(a))
    add("total", "n_articles_matched", main["article_id"].nunique(), len(main))
    add("total", "n_tickers", main["ticker"].nunique(), len(main))
    add("total", "n_distinct_authors", main["author_id"].nunique(), len(main))
    for key, n in main["match_mode"].value_counts().items():
        add("match_mode", str(key), int(n), len(main))
    for key, n in main["effort_tier"].value_counts().items():
        add("effort_tier", str(key), int(n), len(main))
    n_bulk = int(main["is_bulk_listing"].sum())
    add("bulk", "rows_flagged", n_bulk, len(main))
    add("bulk", "articles_flagged", int(main.loc[main["is_bulk_listing"], "article_id"].nunique()),
        int(main["article_id"].nunique()))
    print(f"  [{source}] 主樣本 ticker-article {len(main):,}；bulk 列 {n_bulk:,}"
          f"（{n_bulk/len(main):.1%}）；獨立作者 {main['author_id'].nunique():,}")
    return rows


def window_displacement(source: str, s: dict) -> list[dict]:
    """同一批留言，兩種窗口指派法的直接對照。

    只用**非 bulk** 的留言：bulk 文章的留言一次繼承數十至兩百多個 ticker，
    留在裡面會讓兩種指派法的差異被同一批留言放大數十倍。
    """
    c = pd.read_parquet(
        INTERIM / f"ptt_comment_matches_{source}.parquet",
        columns=["ticker", "user_id", "timestamp", "article_timestamp", "is_bulk_listing"])
    c = c[~c["is_bulk_listing"]]
    c = c[_in_main_sample(c["timestamp"].fillna(c["article_timestamp"]), s)]
    d = c.dropna(subset=["timestamp"])

    by_article = d["article_timestamp"].dt.weekday >= 5    # v1 的作法
    by_comment = d["timestamp"].dt.weekday >= 5            # v2 的作法
    n = len(d)
    n_a, n_c = int(by_article.sum()), int(by_comment.sum())

    rows = [
        {"metric": "n_comment_rows_nonbulk", "n": n, "pct": ""},
        {"metric": "weekend_by_article_timestamp_v1", "n": n_a, "pct": round(n_a / n, 4)},
        {"metric": "weekend_by_comment_timestamp_v2", "n": n_c, "pct": round(n_c / n, 4)},
        {"metric": "weekday_article_weekend_comment", "n": int((~by_article & by_comment).sum()),
         "pct": round(float((~by_article & by_comment).mean()), 4)},
        {"metric": "weekend_article_weekday_comment", "n": int((by_article & ~by_comment).sum()),
         "pct": round(float((by_article & ~by_comment).mean()), 4)},
        {"metric": "window_disagreement_total", "n": int((by_article != by_comment).sum()),
         "pct": round(float((by_article != by_comment).mean()), 4)},
        {"metric": "v1_weekend_understatement", "n": n_c - n_a,
         "pct": round((n_c - n_a) / n_a, 4) if n_a else ""},
    ]
    _write_csv(AUDIT / "ptt_window_displacement.csv", rows,
               fieldnames=["metric", "n", "pct"])
    print(f"  週末留言：用文章時戳 {n_a:,}／用留言時戳 {n_c:,}")
    print(f"  → 以文章時戳指派會低估週末留言 {(n_c-n_a)/n_a:.1%}")
    print(f"  平日發文→週末留言 {int((~by_article & by_comment).sum()):,}；"
          f"週末發文→平日留言 {int((by_article & ~by_comment).sum()):,}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P2 配對層稽核")
    ap.add_argument("--sources", default="pttcc")
    args = ap.parse_args(argv)
    s = load_settings()

    print("[1] 配對結構")
    rows: list[dict] = []
    for source in (x.strip() for x in args.sources.split(",") if x.strip()):
        rows += structure(source, s)
    _write_csv(AUDIT / "ptt_match_structure.csv", rows,
               fieldnames=["source", "kind", "key", "n", "pct"])

    if "pttcc" in args.sources:
        print("[2] 窗口位移（v1 vs v2 的指派法）")
        window_displacement("pttcc", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
