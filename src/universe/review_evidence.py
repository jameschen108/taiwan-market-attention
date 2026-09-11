"""為判讀產生逐筆證據（歸屬正確率抽驗的第二步）。

判讀要回答的是「這篇文章真的在講這一檔嗎」。判讀者需要看到的是**命中的片段與其
上下文**，不是整篇文章——長尾個股的文章動輒數千字，全文丟給判讀者既昂貴又容易
誤判。每一筆輸出：標題、命中片段、命中前後各 120 字的視窗，以及代號／簡稱是否
同時出現。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..config import ROOT, load_settings, load_universe_config, resolve
from ..ptt.index_jsonl import ArticleStore
from .name_matching import build_matcher, normalize, strip_ptt_template

AUDIT = ROOT / "audit"
CONTEXT = 120
MAX_SPANS = 3


def build(source: str, s: dict, tag: str = "") -> pd.DataFrame:
    sample = pd.read_csv(AUDIT / f"ptt_review_sample_{source}{tag}.csv", dtype={"ticker": str})
    cfg = load_universe_config()
    matcher = build_matcher(cfg)
    names = {str(t): [v["text"] for v in spec.get("variants", [])]
             for t, spec in cfg["tickers"].items()}
    short = {str(t): spec["name_short"] for t, spec in cfg["tickers"].items()}
    store = ArticleStore(resolve(s["ptt"]["root"]))

    rows = []
    for r in sample.itertuples():
        doc = store.get(r.article_id)
        if doc is None:
            rows.append({"article_id": r.article_id, "ticker": r.ticker,
                         "evidence": "<找不到語料檔>"})
            continue
        title, body = doc.get("title") or "", doc.get("content") or ""
        text = strip_ptt_template(normalize(f"{title}\n{body}"))

        spans = [("代號", m.matched_text, m.start, m.end)
                 for m in matcher.match_codes(text) if m.ticker == r.ticker]
        spans += [(f"簡稱({m.match_mode})", m.matched_text, m.start, m.end)
                  for m in matcher.match_names(text) if m.ticker == r.ticker]

        windows = []
        for kind, txt, st, en in spans[:MAX_SPANS]:
            lo, hi = max(0, st - CONTEXT), min(len(text), en + CONTEXT)
            frag = text[lo:st] + "【" + text[st:en] + "】" + text[en:hi]
            windows.append(f"[{kind}] …{' '.join(frag.split())}…")

        hit_names = [n for n in names.get(r.ticker, ()) if n in text]
        rows.append({
            "stratum": r.stratum,
            "ticker": r.ticker,
            "name_short": short.get(r.ticker, "?"),
            "article_id": r.article_id,
            "timestamp": r.timestamp,
            "category": r.category,
            "match_mode": r.match_mode,
            "sparsity_tier": r.sparsity_tier,
            "is_bulk_listing": bool(r.is_bulk_listing),
            "n_tickers_in_article": int(r.n_tickers_in_article),
            "title": title[:120],
            "n_hits": len(spans),
            "code_present": r.ticker in text,
            "code_in_title": r.ticker in title,
            "name_present": bool(hit_names),
            "name_in_title": any(n in title for n in hit_names),
            "matched_names": "|".join(hit_names),
            "body_len": len(body),
            "evidence": "\n".join(windows) if windows else "<無法定位命中片段>",
            "auto_verdict": "", "auto_reason": "", "verdict": "", "note": "",
        })

    store.close()
    out = pd.DataFrame(rows)
    path = AUDIT / f"ptt_review_evidence_{source}{tag}.csv"
    out.to_csv(path, index=False)
    print(f"判讀證據 {len(out)} 筆 → {path.relative_to(ROOT)}")
    print(f"  無法定位命中片段 {int((out['n_hits'] == 0).sum())} 筆")
    print(f"  大量清單型貼文 {int(out['is_bulk_listing'].sum())} 筆"
          f"（主規格已排除，仍列入以檢核該旗標）")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="產生判讀證據")
    ap.add_argument("--source", default="pttcc")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    build(args.source, load_settings(), tag=args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
