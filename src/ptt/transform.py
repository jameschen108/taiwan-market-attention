"""配對層：把語料轉成 ticker × 文章／ticker × 留言 兩張表。

輸出（`data/interim/`）：

| 檔案 | 觀測單位 | 新增於 |
|---|---|---|
| `ptt_matches.parquet` | ticker × 文章 | v1（v2 加 `author_id`） |
| `ptt_comment_matches.parquet` | ticker × 留言 | **v2** |

**留言的 ticker 歸屬是寫死的假設**（PROJECT.md §4.4）：留言繼承母文章配對到的
ticker，不對留言文字另行比對代號。留言短、且「2330」這類數字在留言中同樣有量詞
誤配風險，而排行表數字欄已證實是六類誤配之一。

**`is_bulk_listing` 必須向下傳遞到留言。** 一篇列 242 檔的程式選股文，其留言會
一次繼承 242 個 ticker；不排除的話，汙染比 v1 的文章層更嚴重（留言數是文章數的
百倍量級）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..config import ROOT, load_settings, load_universe_config, resolve
from ..universe.name_matching import Matcher, build_matcher
from .parse import effort_tier, has_comment_metadata, iter_articles, parse_title

INTERIM = ROOT / "data" / "interim"

ARTICLE_COLS = [
    "article_id", "ticker", "match_mode", "timestamp", "category", "is_reply",
    "effort_tier", "author_id", "n_tickers_in_article", "is_bulk_listing",
    "n_comments", "n_push", "n_boo", "n_arrow", "source",
]
COMMENT_COLS = [
    "article_id", "comment_index", "ticker", "user_id", "tag", "timestamp",
    "article_timestamp", "n_tickers_in_article", "is_bulk_listing", "source",
]


def _rows_for_article(art, matcher: Matcher, max_tickers: int, source: str,
                      want_comments: bool):
    """回傳 (文章層列, 留言層列)。無配對的文章不產生列。"""
    found = matcher.match(art.title, art.body)
    if not found:
        return (), ()

    n_tickers = len(found)
    is_bulk = n_tickers > max_tickers
    category, is_reply = parse_title(art.title)
    tier = effort_tier(category, is_reply)

    arows = [{
        "article_id": art.article_id,
        "ticker": ticker,
        "match_mode": mode,
        "timestamp": art.timestamp,
        "category": category,
        "is_reply": is_reply,
        "effort_tier": tier,
        # 空作者維持空（系統發文），下游算獨立帳號數時必須排除，不得當成一個帳號
        "author_id": art.author_id or None,
        "n_tickers_in_article": n_tickers,
        "is_bulk_listing": is_bulk,
        "n_comments": art.n_comments,
        "n_push": art.n_push,
        "n_boo": art.n_boo,
        "n_arrow": art.n_arrow,
        "source": source,
    } for ticker, mode in sorted(found.items())]

    if not want_comments:
        return arows, ()

    crows = [{
        "article_id": art.article_id,
        "comment_index": c.index,
        "ticker": ticker,
        "user_id": c.user_id or None,
        "tag": c.tag,
        # 缺時戳維持 None——這些留言不進任何窗口測度，但仍計入留言總數稽核
        "timestamp": c.timestamp,
        "article_timestamp": art.timestamp,
        "n_tickers_in_article": n_tickers,
        "is_bulk_listing": is_bulk,      # 向下傳遞，見模組說明
        "source": source,
    } for c in art.comments for ticker in sorted(found)]
    return arows, crows


def build(source: str, root: Path, matcher: Matcher, max_tickers: int,
          want_comments: bool, flush_every: int = 200_000):
    """串流建表。留言層可達數千萬列，分批 append 成 parquet 以免爆記憶體。"""
    INTERIM.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    a_path = INTERIM / f"ptt_matches_{source}.parquet"
    c_path = INTERIM / f"ptt_comment_matches_{source}.parquet"
    a_buf: list[dict] = []
    c_buf: list[dict] = []
    a_writer = c_writer = None
    n_art = n_matched = n_arows = n_crows = 0

    def flush(buf, cols, writer, path):
        if not buf:
            return writer
        tbl = pa.Table.from_pandas(
            pd.DataFrame(buf, columns=cols), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, tbl.schema)
        writer.write_table(tbl)
        buf.clear()
        return writer

    for art in iter_articles(source, root, with_comments=want_comments):
        n_art += 1
        arows, crows = _rows_for_article(art, matcher, max_tickers, source, want_comments)
        if arows:
            n_matched += 1
            a_buf.extend(arows)
            c_buf.extend(crows)
            n_arows += len(arows)
            n_crows += len(crows)
        if len(a_buf) >= flush_every:
            a_writer = flush(a_buf, ARTICLE_COLS, a_writer, a_path)
        if len(c_buf) >= flush_every:
            c_writer = flush(c_buf, COMMENT_COLS, c_writer, c_path)
        if n_art % 20000 == 0:
            print(f"    …{n_art:,} 篇，配對 {n_matched:,}，"
                  f"文章列 {n_arows:,}，留言列 {n_crows:,}")

    a_writer = flush(a_buf, ARTICLE_COLS, a_writer, a_path)
    c_writer = flush(c_buf, COMMENT_COLS, c_writer, c_path)
    for w in (a_writer, c_writer):
        if w is not None:
            w.close()

    print(f"  文章 {n_art:,} → 有配對 {n_matched:,}（{n_matched/max(n_art,1):.1%}）"
          f" → ticker-article {n_arows:,}")
    if want_comments:
        print(f"  ticker-comment {n_crows:,}  → {c_path.relative_to(ROOT)}")
    print(f"  → {a_path.relative_to(ROOT)}")
    return {"n_articles": n_art, "n_matched": n_matched,
            "n_article_rows": n_arows, "n_comment_rows": n_crows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PTT 配對層")
    ap.add_argument("--source", default=None, help="pttcc / pttweb；預設取設定檔")
    args = ap.parse_args(argv)

    s = load_settings()
    source = args.source or s["ptt"]["source"]
    root = resolve(s["ptt"]["root"] if source == "pttcc" else s["ptt"]["pttweb_root"])
    max_tickers = int(s["attention"]["max_tickers_per_article"])
    want_comments = bool(s["ptt"]["comments"]["enabled"]) and has_comment_metadata(source)

    print(f"[配對] source={source}  root={root.relative_to(ROOT)}  "
          f"留言層={'是' if want_comments else '否（此來源推文無帳號與時戳）'}")
    matcher = build_matcher(load_universe_config())
    build(source, root, matcher, max_tickers, want_comments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
