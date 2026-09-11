"""分類分布與認知投入分層的實得統計（PLAN_V2 §P1 驗收項）。

分類清單不得假設與 v1 相同，必須以實得語料重新統計後確認。本模組對兩個來源
各跑一次，輸出 `audit/ptt_category_distribution.csv`。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .audit_corpus import AUDIT, _write_csv
from .config import load_settings, resolve
from .ptt.parse import effort_tier, iter_articles, parse_title


def tally(source: str, root: Path, lo: str, hi: str) -> tuple[Counter, Counter, int]:
    cats: Counter = Counter()
    tiers: Counter = Counter()
    n = 0
    for art in iter_articles(source, root, with_comments=False):
        ym = f"{art.timestamp.year:04d}-{art.timestamp.month:02d}"
        if not lo <= ym <= hi:
            continue
        n += 1
        cat, is_reply = parse_title(art.title)
        cats[cat] += 1
        tiers[effort_tier(cat, is_reply)] += 1
    return cats, tiers, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="分類分布統計")
    ap.add_argument("--sources", default="pttcc",
                    help="逗號分隔：pttcc / pttweb（pttweb 需讀 25 萬個小檔，較慢）")
    args = ap.parse_args(argv)

    s = load_settings()
    lo, hi = s["sample"]["main_start"][:7], s["sample"]["main_end"][:7]
    rows: list[dict] = []
    for source in (x.strip() for x in args.sources.split(",") if x.strip()):
        root = resolve(s["ptt"]["root"] if source == "pttcc" else s["ptt"]["pttweb_root"])
        cats, tiers, n = tally(source, root, lo, hi)
        print(f"\n[{source}] {lo}~{hi}  文章 {n:,}  分類 {len(cats)} 種")
        for cat, k in cats.most_common():
            rows.append({"source": source, "kind": "category", "key": cat,
                         "n": k, "pct": round(k / n, 5)})
        for tier in ("high_effort", "mid_effort", "low_effort", "unclassified"):
            k = tiers[tier]
            print(f"  {tier:<13} {k:>8,}  {k/n:6.2%}")
            rows.append({"source": source, "kind": "effort_tier", "key": tier,
                         "n": k, "pct": round(k / n, 5)})
    _write_csv(AUDIT / "ptt_category_distribution.csv", rows,
               fieldnames=["source", "kind", "key", "n", "pct"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
