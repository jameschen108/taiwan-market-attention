"""由語料推導「延伸實體」：找出 v1 手工清單未涵蓋的誤配機制。

v1 的阻擋延伸規則（統一→統一證券、中華電→華電、平台聚集→台聚）是人工抽驗一筆
一筆看出來的。人工只看得到抽到的那 550 筆，看不到的機制就留在資料裡。

本模組改成**資料驅動**：對每個簡稱寫法，掃全語料統計它右邊（左邊）緊接的 1–3 個
字的分布。若某個延伸佔該簡稱出現次數的顯著比例，它很可能是另一個實體的名字，
而不是這檔股票。輸出供人工裁決——**本模組不自動改比對規則**，門檻先鎖死再看結果
的紀律同樣適用於它。

輸出 `audit/name_extension_profile.csv`。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..audit_corpus import AUDIT, _write_csv
from ..config import load_settings, load_universe_config, resolve
from ..ptt.parse import iter_articles
from .name_matching import build_matcher, normalize, strip_ptt_template

# 延伸字若是這些，代表後面接的是句法而非實體名，不算延伸
_STOPS = set("的了是在有和與及或也都很就還把被從對向，。、；：！？（）「」『』…－—\n\r\t ")

#: 純數字／標點的「延伸」不是另一個實體，而是**證券代號緊接簡稱**（「1102亞泥」）
#: ——那是最強的正面證據，不是汙染。必須排除，否則漏網清單會被它塞滿。
_NOT_AN_ENTITY = re.compile(r"^[\d\s,.\-+/%()]*$")


def profile(source: str, root: Path, limit: int | None = None,
            max_ext: int = 3) -> list[dict]:
    cfg = load_universe_config()
    matcher = build_matcher(cfg)
    short = {str(t): spec["name_short"] for t, spec in cfg["tickers"].items()}

    # 只看簡稱寫法（代號沒有「延伸實體」問題，它有自己的欄位偵測規則）
    variants = {v.text: v.ticker for v in matcher.variants}
    if not variants:
        return []
    alt = re.compile("|".join(re.escape(t) for t in
                              sorted(variants, key=len, reverse=True)))

    hits: Counter = Counter()
    after: dict[str, Counter] = defaultdict(Counter)
    before: dict[str, Counter] = defaultdict(Counter)

    for i, art in enumerate(iter_articles(source, root, with_comments=False)):
        if limit and i >= limit:
            break
        text = strip_ptt_template(normalize(f"{art.title}\n{art.body}"))
        for m in alt.finditer(text):
            name = m.group(0)
            hits[name] += 1
            tail = text[m.end():m.end() + max_ext]
            head = text[max(0, m.start() - max_ext):m.start()]
            for k in range(1, max_ext + 1):
                if len(tail) >= k and tail[0] not in _STOPS:
                    after[name][tail[:k]] += 1
                if len(head) >= k and head[-1] not in _STOPS:
                    before[name][head[-k:]] += 1
        if (i + 1) % 20000 == 0:
            print(f"    …{i+1:,} 篇")

    rows: list[dict] = []
    for name, n in hits.most_common():
        ticker = variants[name]
        for side, table in (("after", after[name]), ("before", before[name])):
            for ext, k in table.most_common(8):
                if k < 30 or k / n < 0.02:
                    continue
                if _NOT_AN_ENTITY.match(ext):
                    continue
                rows.append({
                    "ticker": ticker,
                    "name_short": short.get(ticker, "?"),
                    "variant": name,
                    "n_hits": n,
                    "side": side,
                    "extension": ext,
                    "n_ext": k,
                    "share": round(k / n, 4),
                    "extended_text": (name + ext) if side == "after" else (ext + name),
                })
    rows.sort(key=lambda r: -r["share"])
    _write_csv(AUDIT / "name_extension_profile.csv", rows, fieldnames=[
        "ticker", "name_short", "variant", "n_hits", "side", "extension",
        "n_ext", "share", "extended_text"])
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="簡稱延伸實體的資料驅動剖析")
    ap.add_argument("--source", default="pttcc")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    s = load_settings()
    root = resolve(s["ptt"]["root"] if args.source == "pttcc" else s["ptt"]["pttweb_root"])
    print(f"[延伸剖析] source={args.source}")
    rows = profile(args.source, root, args.limit)
    print(f"  高佔比延伸（share ≥ 20%）：")
    for r in rows[:25]:
        if r["share"] >= 0.20:
            print(f"    {r['extended_text']:<12} {r['name_short']}({r['ticker']}) "
                  f"{r['n_ext']:>6,}/{r['n_hits']:>7,} = {r['share']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
