"""自動初判：把 700 筆抽驗分成「可自動確認」與「需人工裁決」。

**自動初判不取代人工裁決。** 它做兩件事：

1. 對證據明確的案例給出初判，讓人工只需覆核而不必逐筆重讀；
2. 把可疑案例**排到最前面**，讓有限的人力花在真正有爭議的地方。

判準刻意設計成**與比對器獨立**——若只是重跑比對器，每一筆都會是「正確」，那等於
沒判。因此規則用的是比對器沒用到的訊號：代號與簡稱是否互相印證、命中是否在標題、
命中是否只出現一次而內文很長、以及由語料推導的延伸實體剖析。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..audit_corpus import AUDIT, _write_csv
from ..config import ROOT, load_settings, load_universe_config, resolve
from .name_matching import build_matcher, normalize

# 由 audit/name_extension_profile.csv 篩出、且**確認未被現行規則擋掉**的延伸。
# 這份清單由 `leaking_extensions()` 產生，不手工維護。
MIN_EXT_SHARE = 0.08
MIN_EXT_HITS = 40


def leaking_extensions(source: str, root: Path, max_examples: int = 30) -> pd.DataFrame:
    """找出高佔比、但現行比對規則**沒擋住**的延伸實體。

    延伸剖析數的是**原始命中**，其中絕大多數已被阻擋規則處理掉。真正要人工裁決的
    是「剖析出來、卻仍然會被算成關注度」的那些。

    **判定必須用語料中的真實上下文**，不能用人造探測字串。用「華電信 股價」去測
    會漏掉阻擋規則靠的是**前面**那個「中」字——探測字串把它切掉了，於是每一個
    blocked_before 規則都會被誤報成漏網。因此本函式回語料抓真實出現的句子來測。
    """
    prof = pd.read_csv(AUDIT / "name_extension_profile.csv", dtype={"ticker": str})
    cand = (prof[(prof["share"] >= MIN_EXT_SHARE) & (prof["n_ext"] >= MIN_EXT_HITS)]
            .drop_duplicates(subset=["ticker", "extended_text"]))
    if cand.empty:
        return pd.DataFrame()

    wanted = {r.extended_text: r.ticker for r in cand.itertuples()}
    examples: dict[str, list[str]] = {k: [] for k in wanted}
    need = set(wanted)

    from ..ptt.parse import iter_articles
    for art in iter_articles(source, root, with_comments=False):
        if not need:
            break
        text = normalize(f"{art.title}\n{art.body}")
        for ext in list(need):
            pos = text.find(ext)
            if pos < 0:
                continue
            lo, hi = max(0, pos - 60), min(len(text), pos + len(ext) + 60)
            examples[ext].append(" ".join(text[lo:hi].split()))
            if len(examples[ext]) >= max_examples:
                need.discard(ext)

    matcher = build_matcher(load_universe_config())
    rows = []
    for r in cand.itertuples():
        ctxs = examples.get(r.extended_text, [])
        if not ctxs:
            continue
        n_hit = sum(1 for c in ctxs if r.ticker in matcher.match(c, ""))
        rows.append({
            "ticker": r.ticker, "name_short": r.name_short, "variant": r.variant,
            "extended_text": r.extended_text, "side": r.side,
            "n_ext": r.n_ext, "n_hits": r.n_hits, "share": r.share,
            "n_contexts_tested": len(ctxs),
            "n_contexts_matched": n_hit,
            "leak_rate": round(n_hit / len(ctxs), 3),
            "example": ctxs[0][:160],
            "status": ("LEAK：真實上下文中仍會被算成該檔"
                       if n_hit else "已被現行規則擋掉"),
        })
    out = (pd.DataFrame(rows)
           .assign(still_matches=lambda d: d["n_contexts_matched"] > 0)
           .sort_values(["still_matches", "leak_rate", "n_ext"],
                        ascending=[False, False, False]))
    _write_csv(AUDIT / "name_extension_leaks.csv", out.to_dict("records"))
    n_leak = int(out["still_matches"].sum())
    print(f"  高佔比延伸 {len(out)} 種；以語料真實上下文測試後，**仍會汙染**的 {n_leak} 種")
    for r in out[out["still_matches"]].itertuples():
        print(f"    ❌ {r.extended_text:<10} → {r.name_short}({r.ticker})  "
              f"語料佔比 {r.share:>5.1%}（{r.n_ext:>6,} 次）  "
              f"實測漏網 {r.n_contexts_matched}/{r.n_contexts_tested}")
    return out


def _leak_texts(leaks: pd.DataFrame) -> dict[str, list[str]]:
    d: dict[str, list[str]] = {}
    for r in leaks[leaks["still_matches"]].itertuples():
        d.setdefault(r.ticker, []).append(r.extended_text)
    return d


def adjudicate(source: str, s: dict, leaks: pd.DataFrame, tag: str = "") -> pd.DataFrame:
    ev = pd.read_csv(AUDIT / f"ptt_review_evidence_{source}{tag}.csv", dtype={"ticker": str})
    leak_by_ticker = _leak_texts(leaks)

    verdicts, reasons, priority = [], [], []
    for r in ev.itertuples():
        text = normalize(str(r.evidence))
        v, why, pri = "needs_review", "", 2

        # --- 可疑（排最前面）------------------------------------------
        bad = [t for t in leak_by_ticker.get(r.ticker, ()) if t in text]
        if bad:
            v, why, pri = "suspect_wrong", f"命中片段是延伸實體的一部分：{'、'.join(bad)}", 0
        elif "集團" in text and not r.code_present:
            v, why, pri = "suspect_ambiguous", "疑為集團泛稱，非該上市公司本身", 0
        elif (not r.code_present and r.n_hits == 1 and r.body_len > 1500
              and not r.name_in_title):
            v, why, pri = ("suspect_weak",
                           f"僅一處簡稱命中、不在標題、內文 {r.body_len} 字：證據薄弱", 1)

        # --- 可自動確認 ------------------------------------------------
        elif r.code_in_title:
            v, why, pri = "auto_correct", "代號出現在標題（標題非排行表欄位）", 3
        elif r.code_present and r.name_present:
            v, why, pri = "auto_correct", "代號與簡稱互相印證（兩個獨立訊號一致）", 3
        elif r.name_in_title and len(str(r.matched_names).split("|")[0]) >= 3:
            v, why, pri = "auto_correct", "三字以上簡稱出現在標題", 3
        elif r.code_present and r.n_hits >= 2:
            v, why, pri = "auto_correct", "代號在內文出現且命中多處", 3

        verdicts.append(v)
        reasons.append(why)
        priority.append(pri)

    ev["auto_verdict"] = verdicts
    ev["auto_reason"] = reasons
    ev["review_priority"] = priority
    ev = ev.sort_values(["review_priority", "stratum"]).reset_index(drop=True)

    path = AUDIT / f"ptt_review_adjudicated_{source}{tag}.csv"
    ev.to_csv(path, index=False)

    analysis = ev[ev["stratum"] != "bulk"]
    print(f"\n自動初判 {len(ev)} 筆（分析用 {len(analysis)} 筆）→ {path.relative_to(ROOT)}")
    print(f"\n{'初判':<20}{'筆數':>6}{'佔分析用':>10}")
    for v, n in analysis["auto_verdict"].value_counts().items():
        print(f"  {v:<18}{n:>6}{n/len(analysis):>10.1%}")
    n_manual = int((analysis["review_priority"] <= 2).sum())
    print(f"\n需人工裁決（priority ≤ 2）：{n_manual} 筆"
          f"（{n_manual/len(analysis):.0%}），其餘 {len(analysis)-n_manual} 筆可覆核即可")
    return ev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="抽驗的自動初判")
    ap.add_argument("--source", default="pttcc")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    s = load_settings()
    print("[1] 找出仍會汙染的延伸實體（以語料真實上下文測試）")
    leaks = leaking_extensions(args.source, resolve(s["ptt"]["root"]))
    print("\n[2] 逐筆自動初判")
    adjudicate(args.source, s, leaks, tag=args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
