"""P0：v2 語料落地與完整性稽核。

產出四份稽核表，全部落在 `audit/`：

| 檔案 | 內容 |
|---|---|
| `ptt_archive_checksums.csv` | 逐檔 SHA-256、位元組數、列數 |
| `ptt_monthly_coverage.csv` | 逐月文章數／留言數；月度零缺口檢查 |
| `ptt_source_diff.csv` | 2020–2024 重疊區新舊語料 article_id 差集，逐月 |
| `ptt_comment_coverage.csv` | 留言總數、缺時戳數、跨年回推、ticker 可歸屬前的原始覆蓋 |

另把 `date` 與 `date_ts` 相差超過設定容忍值的文章寫入
`ptt_timestamp_discrepancy.csv`。

用法：
    python3 -m src.audit_corpus            # 全部
    python3 -m src.audit_corpus --only checksums,monthly
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .config import ROOT, load_settings, resolve

AUDIT = ROOT / "audit"
PTTWEB_ID_TS = re.compile(r"^M\.(\d{9,11})\.A\.[0-9A-F]+$", re.IGNORECASE)
# 每行 JSON 的第一個鍵即 article_id，供只需 ID 的快掃使用
LINE_ID = re.compile(rb'"article_id"\s*:\s*"([^"]+)"')


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path.relative_to(ROOT)}  ({len(rows)} 列)")


# ---------------------------------------------------------------- checksums

def run_checksums(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("stock_*.jsonl")):
        h = hashlib.sha256()
        n_lines = 0
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
                n_lines += chunk.count(b"\n")
        rows.append({
            "file": path.name,
            "bytes": path.stat().st_size,
            "n_lines": n_lines,
            "sha256": h.hexdigest(),
        })
        print(f"  {path.name}  {n_lines:>7,} 列  {h.hexdigest()[:16]}…")
    _write_csv(AUDIT / "ptt_archive_checksums.csv", rows)
    return rows


# ------------------------------------------------------- 逐月覆蓋與時戳稽核

def run_monthly(root: Path, tol_days: int) -> tuple[list[dict], list[dict], list[dict]]:
    """一次全掃，同時產出逐月覆蓋、時戳歧異、留言覆蓋三張表。"""
    per_month: dict[str, Counter] = defaultdict(Counter)
    discrepancies: list[dict] = []
    comment_stats: dict[str, Counter] = defaultdict(Counter)
    n_articles = 0

    for path in sorted(root.glob("stock_*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                n_articles += 1
                ts = datetime.fromisoformat(doc["date_ts"])
                ym = f"{ts.year:04d}-{ts.month:02d}"
                m = per_month[ym]
                m["n_articles"] += 1
                m["n_comments_declared"] += int(doc.get("n_comments") or 0)
                m["n_push"] += int(doc.get("n_push") or 0)
                m["n_boo"] += int(doc.get("n_boo") or 0)
                m["n_arrow"] += int(doc.get("n_arrow") or 0)
                if doc.get("meta_recovered"):
                    m["n_meta_recovered"] += 1

                header = doc.get("date")
                if header:
                    try:
                        gap = abs((datetime.fromisoformat(header) - ts).days)
                    except ValueError:
                        gap = None
                    if gap is not None and gap > tol_days:
                        discrepancies.append({
                            "article_id": doc["article_id"],
                            "date_header": header,
                            "date_ts": doc["date_ts"],
                            "gap_days": gap,
                            "meta_recovered": bool(doc.get("meta_recovered")),
                        })
                else:
                    m["n_missing_header_date"] += 1

                cs = comment_stats[ym]
                for c in doc.get("comments") or ():
                    cs["n_comments"] += 1
                    if not c.get("time"):
                        cs["n_missing_time"] += 1
                        continue
                    ct = datetime.fromisoformat(c["time"])
                    m_ = cs
                    if ct < ts:
                        # 留言早於文章 → 跨年回推出錯的徵兆
                        m_["n_before_article"] += 1
                    if ct.year != ts.year:
                        m_["n_cross_year"] += 1
                    if not c.get("user_id"):
                        m_["n_missing_user"] += 1

    months = sorted(per_month)
    monthly = [{
        "month": ym,
        **{k: per_month[ym][k] for k in
           ("n_articles", "n_comments_declared", "n_push", "n_boo", "n_arrow",
            "n_meta_recovered", "n_missing_header_date")},
        "n_comments_parsed": comment_stats[ym]["n_comments"],
    } for ym in months]
    _write_csv(AUDIT / "ptt_monthly_coverage.csv", monthly)

    # 月度零缺口檢查
    gaps = _month_gaps(months)
    print(f"  文章總數 {n_articles:,}；月份 {len(months)} 個（{months[0]} ~ {months[-1]}）")
    print(f"  月度缺口：{gaps if gaps else '無'}")
    empty = [r["month"] for r in monthly if r["n_articles"] == 0]
    if empty:
        print(f"  ⚠ 零文章月份：{empty}")

    _write_csv(AUDIT / "ptt_timestamp_discrepancy.csv", discrepancies, fieldnames=[
        "article_id", "date_header", "date_ts", "gap_days", "meta_recovered"])

    coverage = [{
        "month": ym,
        "n_comments": comment_stats[ym]["n_comments"],
        "n_missing_time": comment_stats[ym]["n_missing_time"],
        "n_missing_user": comment_stats[ym]["n_missing_user"],
        "n_cross_year": comment_stats[ym]["n_cross_year"],
        "n_before_article": comment_stats[ym]["n_before_article"],
        "n_usable": comment_stats[ym]["n_comments"] - comment_stats[ym]["n_missing_time"],
    } for ym in months]
    _write_csv(AUDIT / "ptt_comment_coverage.csv", coverage)

    tot = sum(r["n_comments"] for r in coverage)
    miss = sum(r["n_missing_time"] for r in coverage)
    bad = sum(r["n_before_article"] for r in coverage)
    print(f"  留言總數 {tot:,}；缺時戳 {miss:,}（{miss/tot:.4%}）；"
          f"早於母文章 {bad:,}（{bad/tot:.4%}）")
    return monthly, discrepancies, coverage


def _month_gaps(months: list[str]) -> list[str]:
    """回傳首尾之間缺席的月份。"""
    if not months:
        return []
    y0, m0 = map(int, months[0].split("-"))
    y1, m1 = map(int, months[-1].split("-"))
    expected = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        expected.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return sorted(set(expected) - set(months))


# ------------------------------------------------------------ 新舊語料差集

def _pttcc_ids(root: Path) -> dict[str, str]:
    """快掃 article_id（不解析整列 JSON）。回傳 id → 'YYYY-MM'。"""
    out: dict[str, str] = {}
    for path in sorted(root.glob("stock_*.jsonl")):
        with open(path, "rb") as fh:
            for raw in fh:
                m = LINE_ID.search(raw[:200])
                if not m:
                    continue
                aid = m.group(1).decode()
                out[aid] = _ym_from_id(aid)
    return out


def _ym_from_id(article_id: str) -> str:
    m = PTTWEB_ID_TS.match(article_id)
    if not m:
        return "unknown"
    ts = datetime.fromtimestamp(int(m.group(1)))
    return f"{ts.year:04d}-{ts.month:02d}"


def _pttweb_ids(root: Path, lo: str, hi: str) -> dict[str, str]:
    """從檔名取 article_id，不開檔（251,858 個檔）。"""
    out: dict[str, str] = {}
    for path in root.glob("batch-*/M.*.json"):
        aid = path.stem
        ym = _ym_from_id(aid)
        if lo <= ym <= hi:
            out[aid] = ym
    return out


def run_source_diff(pttcc_root: Path, pttweb_root: Path, lo: str, hi: str) -> list[dict]:
    print("  快掃新語料 article_id …")
    new = _pttcc_ids(pttcc_root)
    new = {k: v for k, v in new.items() if lo <= v <= hi}
    print(f"    新語料 {lo}~{hi}：{len(new):,} 篇")
    print("  快掃舊封存 article_id …")
    old = _pttweb_ids(pttweb_root, lo, hi)
    print(f"    舊封存 {lo}~{hi}：{len(old):,} 篇")

    months = sorted(set(new.values()) | set(old.values()))
    per: dict[str, Counter] = defaultdict(Counter)
    for aid, ym in new.items():
        per[ym]["n_new"] += 1
        per[ym]["n_both" if aid in old else "n_new_only"] += 1
    for aid, ym in old.items():
        per[ym]["n_old"] += 1
        if aid not in new:
            per[ym]["n_old_only"] += 1

    rows = []
    for ym in months:
        c = per[ym]
        rows.append({
            "month": ym,
            "n_old": c["n_old"],
            "n_new": c["n_new"],
            "n_both": c["n_both"],
            "n_old_only": c["n_old_only"],     # 官方站已刪文
            "n_new_only": c["n_new_only"],     # 鏡像缺漏
            "old_coverage_of_new": round(c["n_both"] / c["n_new"], 4) if c["n_new"] else "",
            "new_coverage_of_old": round(c["n_both"] / c["n_old"], 4) if c["n_old"] else "",
        })
    _write_csv(AUDIT / "ptt_source_diff.csv", rows)

    t = Counter()
    for r in rows:
        for k in ("n_old", "n_new", "n_both", "n_old_only", "n_new_only"):
            t[k] += r[k]
    print(f"  合計：舊 {t['n_old']:,}／新 {t['n_new']:,}／交集 {t['n_both']:,}")
    print(f"        只在舊（官方已刪文）{t['n_old_only']:,}"
          f"（佔舊 {t['n_old_only']/max(t['n_old'],1):.2%}）")
    print(f"        只在新（鏡像缺漏）  {t['n_new_only']:,}"
          f"（佔新 {t['n_new_only']/max(t['n_new'],1):.2%}）")
    return rows


# ------------------------------------------------- 已刪文的選樣特徵（P0 追加）

TITLE_TAG = re.compile(r"^(?:(?:Re|Fw|RE|FW):\s*)*\[([^\]]{1,6})\]")
RE_PREFIX = re.compile(r"^(?:(?:Re|Fw|RE|FW):\s*)+")


def _profile_pttweb(ids, paths) -> tuple[Counter, dict]:
    cats: Counter = Counter()
    pushes, blens, n_re = [], [], 0
    for aid in ids:
        doc = json.loads(paths[aid].read_text(encoding="utf-8"))
        title = (doc.get("title") or "").strip()
        m = TITLE_TAG.match(title)
        cats[m.group(1) if m else "無分類"] += 1
        if RE_PREFIX.match(title):
            n_re += 1
        pushes.append(len(doc.get("pushes") or ()))
        blens.append(len(doc.get("body") or ""))
    n = max(len(ids), 1)
    pushes.sort()
    blens.sort()
    return cats, {
        "n": len(ids),
        "median_pushes": pushes[len(pushes) // 2] if pushes else 0,
        "mean_pushes": round(sum(pushes) / n, 1),
        "median_body_chars": blens[len(blens) // 2] if blens else 0,
        "pct_reply": round(n_re / n, 4),
    }


def run_deletion_profile(pttcc_root: Path, pttweb_root: Path, lo: str, hi: str,
                         seed: int = 20200101) -> list[dict]:
    """只在舊封存的文章＝官方站已刪文。檢查刪文是否隨機。

    對照組以**逐月等量**抽樣自新舊都有的文章——分類組成隨期間變動（情報類在
    2021 後才變多），不逐月配對會把期間效果誤讀成刪文選樣。
    """
    new_ids = {k for k, v in _pttcc_ids(pttcc_root).items() if lo <= v <= hi}
    paths, by_month = {}, defaultdict(lambda: {"old_only": [], "both": []})
    for path in pttweb_root.glob("batch-*/M.*.json"):
        aid = path.stem
        if not PTTWEB_ID_TS.match(aid):
            continue
        ym = _ym_from_id(aid)
        if lo <= ym <= hi:
            paths[aid] = path
            by_month[ym]["old_only" if aid not in new_ids else "both"].append(aid)

    rng = random.Random(seed)
    treat, ctrl = [], []
    for ym, buckets in by_month.items():
        treat += buckets["old_only"]
        ctrl += rng.sample(buckets["both"], min(len(buckets["old_only"]), len(buckets["both"])))

    c_t, s_t = _profile_pttweb(treat, paths)
    c_c, s_c = _profile_pttweb(ctrl, paths)
    print(f"  已刪文 {s_t['n']:,} 篇；逐月等量對照 {s_c['n']:,} 篇")
    print(f"  已刪文：推文中位 {s_t['median_pushes']}、內文中位 {s_t['median_body_chars']} 字、"
          f"回文佔 {s_t['pct_reply']:.1%}")
    print(f"  對照組：推文中位 {s_c['median_pushes']}、內文中位 {s_c['median_body_chars']} 字、"
          f"回文佔 {s_c['pct_reply']:.1%}")

    rows = []
    for cat in sorted(set(c_t) | set(c_c), key=lambda c: -c_t[c]):
        p_t = c_t[cat] / max(s_t["n"], 1)
        p_c = c_c[cat] / max(s_c["n"], 1)
        rows.append({
            "category": cat,
            "n_deleted": c_t[cat],
            "pct_deleted": round(p_t, 4),
            "n_control": c_c[cat],
            "pct_control": round(p_c, 4),
            "over_representation": round(p_t / p_c, 3) if p_c else "",
        })
    _write_csv(AUDIT / "ptt_deletion_profile.csv", rows)

    _write_csv(AUDIT / "ptt_deletion_summary.csv", [
        {"group": "deleted", **s_t}, {"group": "control_month_matched", **s_c}])
    return rows


def run_deletion_spotcheck(pttcc_root: Path, pttweb_root: Path, lo: str, hi: str,
                           n_each: int = 40, delay: float = 0.5,
                           seed: int = 7) -> list[dict]:
    """向 ptt.cc 抽樣確認：只在舊封存者是否確實 404（而非爬蟲漏抓）。

    需要網路。正向對照（新舊都有者應為 200）是必要的——沒有對照就無法區分
    「文章已刪」與「整個檢查方法壞掉」。
    """
    import time

    import requests

    new_ids = {k for k, v in _pttcc_ids(pttcc_root).items() if lo <= v <= hi}
    old_only, both = [], []
    for path in pttweb_root.glob("batch-*/M.*.json"):
        aid = path.stem
        if PTTWEB_ID_TS.match(aid) and lo <= _ym_from_id(aid) <= hi:
            (both if aid in new_ids else old_only).append(aid)

    rng = random.Random(seed)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (corpus integrity check)"})
    session.cookies.set("over18", "1", domain=".ptt.cc")

    rows = []
    for group, pool in (("old_only", old_only), ("both", both)):
        counts: Counter = Counter()
        for aid in rng.sample(pool, min(n_each, len(pool))):
            try:
                counts[session.get(f"https://www.ptt.cc/bbs/Stock/{aid}.html",
                                   timeout=15, allow_redirects=False).status_code] += 1
            except Exception as exc:                      # noqa: BLE001
                counts[type(exc).__name__] += 1
            time.sleep(delay)
        n = sum(counts.values())
        rows.append({"group": group, "n_checked": n,
                     "n_200": counts.get(200, 0), "n_404": counts.get(404, 0),
                     "other": {k: v for k, v in counts.items() if k not in (200, 404)} or ""})
        print(f"  {group:<9} n={n:<4} {dict(counts)}")
    _write_csv(AUDIT / "ptt_deletion_spotcheck.csv", rows)
    return rows


# --------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v2 語料完整性稽核（PLAN_V2 §5 P0）")
    ap.add_argument("--only", default="checksums,monthly,diff,deletion",
                    help="逗號分隔：checksums / monthly / diff / deletion / spotcheck"
                         "（spotcheck 需連網，預設不跑）")
    args = ap.parse_args(argv)
    steps = {s.strip() for s in args.only.split(",") if s.strip()}

    s = load_settings()
    pttcc = resolve(s["ptt"]["root"])
    pttweb = resolve(s["ptt"]["pttweb_root"])
    tol = int(s["ptt"]["date_audit_tolerance_days"])
    lo = s["sample"]["main_start"][:7]
    hi = s["sample"]["main_end"][:7]
    AUDIT.mkdir(exist_ok=True)

    if "checksums" in steps:
        print("[1] 逐檔 SHA-256")
        run_checksums(pttcc)
    if "monthly" in steps:
        print("[2] 逐月覆蓋、時戳歧異、留言覆蓋")
        run_monthly(pttcc, tol)
    if "diff" in steps:
        print(f"[3] 新舊語料差集（{lo} ~ {hi}）")
        run_source_diff(pttcc, pttweb, lo, hi)
    if "deletion" in steps:
        print("[4] 已刪文的選樣特徵")
        run_deletion_profile(pttcc, pttweb, lo, hi)
    if "spotcheck" in steps:
        print("[5] ptt.cc 現況抽樣（需連網）")
        run_deletion_spotcheck(pttcc, pttweb, lo, hi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
