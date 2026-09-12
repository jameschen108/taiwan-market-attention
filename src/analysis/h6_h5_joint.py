"""H6 異質性 ＋ H5 反轉：**同一張表**（PROJECT.md §6.3）。

> **判讀必須聯立。** 單獨的負交互項**不能**區分兩條管道——資訊處理與價格壓力都
> 預測低覆蓋股效果更強。決定性的證據是低覆蓋股的效果**是否在後續週反轉**：
> 反轉 → 價格壓力；不反轉 → 資訊處理。

因此本模組把兩者寫進同一個輸出檔，並由 `_verdict()` 依聯立結果給判讀，
**不提供只輸出其中一半的介面**。

H5 的反轉視野寫死為 t+2 … t+8，並報 **t+2..t+4** 與 **t+2..t+8** 兩個累積。
判讀以累積量為準，單週不算。原論文的反轉視野是一年，v2 樣本僅 260 週且末端受限，
因此只做 8 週並明寫「未檢驗更長視野的反轉」。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ..config import ROOT, load_settings
from .regressions import (BASE_CONTROLS, add_derived, fit_both_inferences,
                          main_sample, results_to_frame, standardize_within)

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output"
X = "abn_attention_weekend"

#: §6.3 的調節變數。預期交互項顯著為負（效果在低覆蓋／小型／低流動性股更強）。
MODERATORS = ["log_att_mean_level_52", "log_market_cap", "turnover", "amihud",
              "foreign_holding_pct"]


def _add_cumulative(d: pd.DataFrame) -> pd.DataFrame:
    """t+2..t+4 與 t+2..t+8 的累積報酬。

    以**加總**而非連乘：週報酬已在 ±5% 量級，加總與連乘差異在小數第四位以下，
    而加總對單週缺值的容忍度可以明確控制——任一週缺值即整段缺值，不以零頂替。
    """
    out = d.copy()
    for lo, hi in ((2, 4), (2, 8)):
        cols = [f"ret_fwd{h}" for h in range(lo, hi + 1) if f"ret_fwd{h}" in out.columns]
        out[f"ret_cum{lo}_{hi}"] = out[cols].sum(axis=1).where(
            out[cols].notna().all(axis=1))
    return out


def _verdict(inter_t: float, rev_t: float, alpha_t: float = 1.96) -> str:
    """聯立判讀規則，寫死於程式（§6.3）。"""
    strong_inter = abs(inter_t) >= alpha_t and inter_t < 0
    if not strong_inter:
        return "交互項不顯著為負 → H6 未獲支持，反轉檢定不具判別力"
    if rev_t <= -alpha_t:
        return "交互項顯著為負 ＋ 後續週顯著反轉 → 支持價格壓力"
    if abs(rev_t) < alpha_t:
        return "交互項顯著為負 ＋ 後續週無反轉 → 支持資訊處理"
    return "交互項顯著為負，但後續週為顯著同向延續 → 兩條管道皆無法解釋，須另尋機制"


def run(panel: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = settings["regression"]["controls"]["required_but_missing"]
    d = _add_cumulative(main_sample(add_derived(panel)))
    mods = [m for m in MODERATORS if m in d.columns]
    std = standardize_within(d, [X, *mods, *BASE_CONTROLS])

    results, verdicts = [], []
    for m in mods:
        inter = f"{X}__x__{m}"
        std[inter] = std[X] * std[m]
        xs = [X, m, inter]
        # H6：異質性
        h6 = fit_both_inferences(std, "ret_oc_next", xs, BASE_CONTROLS,
                                 f"H6 {m}", required_missing=missing)
        # H5：同一組自變數，應變數換成後續週累積 → 反轉檢定
        h5a = fit_both_inferences(std, "ret_cum2_4", xs, BASE_CONTROLS,
                                  f"H5 {m}｜t+2..t+4", required_missing=missing)
        h5b = fit_both_inferences(std, "ret_cum2_8", xs, BASE_CONTROLS,
                                  f"H5 {m}｜t+2..t+8", required_missing=missing)
        results += h6 + h5a + h5b

        for inf in ("twoway_2cluster", "firm_fe_1cluster"):
            g6 = next((r for r in h6 if r.inference == inf), None)
            g5 = next((r for r in h5b if r.inference == inf), None)
            it = g6.tstats.get(inter, np.nan) if g6 and g6.status == "OK" else np.nan
            rt = g5.tstats.get(inter, np.nan) if g5 and g5.status == "OK" else np.nan
            verdicts.append({
                "moderator": m, "inference": inf,
                "interaction_t_next_week": round(it, 3) if it == it else "",
                "interaction_t_cum2_8": round(rt, 3) if rt == rt else "",
                "verdict": _verdict(it, rt) if (it == it and rt == rt) else "無法估計",
            })
    return results_to_frame(results), pd.DataFrame(verdicts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H6 異質性 ＋ H5 反轉（聯立）")
    ap.add_argument("--spec", default="C")
    args = ap.parse_args(argv)
    print(f"[H6+H5] 規格 {args.spec}")
    panel = pd.read_parquet(PROCESSED / f"panel_{args.spec}.parquet")
    coefs, verdicts = run(panel, load_settings())
    OUT.mkdir(parents=True, exist_ok=True)
    coefs.to_csv(OUT / f"T6_joint_reading_{args.spec}.csv", index=False)
    verdicts.to_csv(OUT / f"T6_joint_verdicts_{args.spec}.csv", index=False)
    print(f"  → output/T6_joint_reading_{args.spec}.csv")
    print(f"  → output/T6_joint_verdicts_{args.spec}.csv")
    print()
    print(verdicts.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
