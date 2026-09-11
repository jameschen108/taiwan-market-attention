"""歸屬正確率抽驗：分層抽樣（PROJECT.md §4「規則可移轉，正確率不可移轉」）。

v1 在舊語料上以 550 筆抽驗把正確率從 77.6% 推到 95%。**規則已移轉到 v2，數字沒有**
——必須以新語料重抽重驗。分層沿用 v1 以便兩輪逐項對照：

| 分層 | 配額 | 為什麼 |
|---|---:|---|
| `tier:*` | 每個稀疏度分層 50 | 誤配率可能隨關注度水準而異；長尾股的名稱空間最擁擠 |
| `collision:*` | 每個碰撞群組成員 20 | 前綴衝突是設計上已知的最高風險 |
| `mode:*` | 每種 match_mode 50 | 代號、一般簡稱、需上下文簡稱的錯誤機制不同 |
| `bulk` | 40 | 檢核 `is_bulk_listing` 旗標本身，不計入正確率分母 |

抽樣**只在主樣本期間**（2020–2024），不含 2019 暖機年。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..config import ROOT, load_settings, load_universe_config
from ..features.attention import sparsity_tier
from ..features.sessions import week_of

INTERIM = ROOT / "data" / "interim"
AUDIT = ROOT / "audit"
SEED = 20200101


def _ticker_tiers(matches: pd.DataFrame, s: dict) -> pd.Series:
    """由配對表直接算每檔的稀疏度分層（不必等完整面板）。

    用**非 bulk** 的配對計算——bulk 貼文是資料傾印，讓它進來會把長尾股誤升為 dense。
    以每檔在樣本期間內「關注度非零的週數」對應 tier 門檻，取該檔的代表性分層。
    """
    m = matches[~matches["is_bulk_listing"]].copy()
    m["week"] = m["timestamp"].map(week_of)
    nonzero_weeks = m.groupby("ticker")["week"].nunique()
    total_weeks = m["week"].nunique()
    # tier 門檻定義在 52 週回顧窗上，此處換算成同一比例尺
    scaled = nonzero_weeks * 52.0 / max(total_weeks, 1)
    spa = s["sparsity"]
    tiers = sparsity_tier(scaled, spa["dense_min_nonzero_weeks"],
                          spa["silent_max_nonzero_weeks"])
    return tiers.reindex(matches["ticker"].unique()).fillna("silent")


def draw(source: str, s: dict, n_per_tier: int = 50, n_per_collision: int = 20,
         n_per_mode: int = 50, n_bulk: int = 40, seed: int = SEED,
         tag: str = "") -> pd.DataFrame:
    matches = pd.read_parquet(INTERIM / f"ptt_matches_{source}.parquet")
    lo, hi = s["sample"]["main_start"], f"{s['sample']['main_end']} 23:59:59"
    matches = matches[(matches["timestamp"] >= lo) & (matches["timestamp"] <= hi)]

    tiers = _ticker_tiers(matches, s)
    matches = matches.assign(sparsity_tier=matches["ticker"].map(tiers))
    cfg = load_universe_config()
    collision_of = {t: g for g, members in cfg["collision_groups"].items()
                    for t in members}
    matches = matches.assign(collision_group=matches["ticker"].map(collision_of))

    analysis = matches[~matches["is_bulk_listing"]]
    rng = seed
    picks: list[pd.DataFrame] = []

    def take(df: pd.DataFrame, n: int, label: str) -> None:
        if df.empty:
            print(f"  ⚠ 分層 {label} 無可抽樣本，SKIP")
            return
        k = min(n, len(df))
        if k < n:
            print(f"  ⚠ 分層 {label} 僅 {k} 筆（配額 {n}）")
        picks.append(df.sample(k, random_state=rng).assign(stratum=label))

    for tier in ("dense", "sparse", "silent"):
        take(analysis[analysis["sparsity_tier"] == tier], n_per_tier, f"tier:{tier}")
    for ticker in sorted(collision_of):
        take(analysis[analysis["ticker"] == ticker], n_per_collision,
             f"collision:{ticker}")
    for mode in ("code", "name", "name_with_context"):
        take(analysis[analysis["match_mode"] == mode], n_per_mode, f"mode:{mode}")
    take(matches[matches["is_bulk_listing"]], n_bulk, "bulk")

    out = (pd.concat(picks, ignore_index=True)
           .drop_duplicates(subset=["article_id", "ticker", "stratum"]))
    AUDIT.mkdir(exist_ok=True)
    path = AUDIT / f"ptt_review_sample_{source}{tag}.csv"
    out.to_csv(path, index=False)
    print(f"\n抽樣 {len(out)} 筆 → {path.relative_to(ROOT)}")
    print(f"  分析用（非 bulk）{int((out['stratum'] != 'bulk').sum())} 筆；"
          f"bulk 旗標檢核 {int((out['stratum'] == 'bulk').sum())} 筆")
    print(f"  獨立文章 {out['article_id'].nunique()}；獨立 ticker {out['ticker'].nunique()}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="歸屬正確率抽驗：分層抽樣")
    ap.add_argument("--source", default="pttcc")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="",
                    help="輸出檔名後綴；修正後的獨立重抽用 --tag _round2")
    args = ap.parse_args(argv)
    draw(args.source, load_settings(), seed=args.seed, tag=args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
