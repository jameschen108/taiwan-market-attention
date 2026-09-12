"""H1 主迴歸與 H8 平行測度（PROJECT.md §6.1、§6.1a、§6.1b）。

```
Ret_{i,t+1} = α + β₁·AbnAtt_weekday + β₂·AbnAtt_weekend
              + δ₁·att_zero_base_weekday + δ₂·att_zero_base_weekend
              + γ'X + firm_FE + week_FE + ε
```

每個規格都跑**兩種推論標準**（主規格的雙向 FE ＋ 雙重 cluster、原論文的個股 FE ＋
僅個股 cluster），並跑**三種報酬定義**（`ret_oc_next` 主規格同論文、`ret_gap_next`
跨週末缺口、`ret_cc_next` 兩段合成）。後兩組不是穩健性而是**機制證據**：價格壓力
預測缺口為正、週內為負；資訊處理預測兩者同號。**論文與 v1 都只有第一組**，因此
後兩組的任何發現都是 v2 的新增，不得寫成「複製了論文的某某結果」。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..config import ROOT, load_settings
from .regressions import (BASE_CONTROLS, INFERENCE, ModelResult, add_derived,
                          fit_both_inferences, main_sample, results_to_frame,
                          standardize_within)

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output"

WINDOW_SPECS = [
    ("H1-1 整週", ["abn_attention_all"]),
    ("H1-2 僅週間", ["abn_attention_weekday"]),
    ("H1-3 僅週末", ["abn_attention_weekend"]),
    ("H1-4 週間＋週末", ["abn_attention_weekday", "abn_attention_weekend"]),
    ("H1-5 交易時段切法", ["abn_attention_intraday", "abn_attention_non_trading"]),
]
RETURNS = [("ret_oc_next", "oc"), ("ret_gap_next", "gap"), ("ret_cc_next", "cc")]

#: H8：主規格的 abn_attention_weekend 在 79% 的列上退化成水準值；獨立帳號數是
#: 唯一有足夠變異的窗口測度（LIMITATIONS.md §11）。每個主表都必須有這一組。
PARALLEL = [
    ("H8-users 獨立帳號數", ["abn_attention_users_weekday",
                             "abn_attention_users_weekend"]),
    ("H8-comments 留言則數", ["abn_attention_comment_weekday",
                              "abn_attention_comment_weekend"]),
]
EFFORTS = ("high_effort", "mid_effort", "low_effort")


def _std_cols(specs) -> list[str]:
    return sorted({x for _, xs in specs for x in xs})


def _skipped(name: str, missing: list[str]) -> list[ModelResult]:
    """測度在該規格**不存在**時，兩種推論標準各寫一列 SKIPPED。

    B 與 A′ 用的是 pttweb，其推文沒有帳號也沒有時戳，`att_users_*` 與
    `att_comment_*` 在那兩個規格上**不可能存在**（PROJECT.md §2.4）。這不是樣本
    不足，是測度不可得——必須寫進表裡，不得靜默略過。
    """
    note = f"此規格的語料無此測度（缺欄位：{';'.join(missing)}）"
    return [ModelResult(name, "SKIPPED", inference=inf, note=note) for inf in INFERENCE]


def run(panel: pd.DataFrame, settings: dict) -> pd.DataFrame:
    missing = settings["regression"]["controls"]["required_but_missing"]
    d = main_sample(add_derived(panel))
    results = []

    # --- H1：四個窗口規格 × 三種報酬定義 ---
    d1 = standardize_within(d, [*_std_cols(WINDOW_SPECS), *BASE_CONTROLS])
    for y, tag in RETURNS:
        for name, xs in WINDOW_SPECS:
            results += fit_both_inferences(d1, y, xs, BASE_CONTROLS,
                                           f"{name}｜{tag}",
                                           required_missing=missing)

    # --- H8：平行測度，主報酬定義。舊語料沒有這些欄位 → SKIPPED ---
    have = [(n, xs) for n, xs in PARALLEL if all(c in d.columns for c in xs)]
    for name, xs in PARALLEL:
        if (name, xs) not in have:
            results += _skipped(f"{name}｜oc", [c for c in xs if c not in d.columns])
    if have:
        d2 = standardize_within(d, [*_std_cols(have), *BASE_CONTROLS])
        for name, xs in have:
            results += fit_both_inferences(d2, "ret_oc_next", xs, BASE_CONTROLS,
                                           f"{name}｜oc", required_missing=missing)

    # --- 基準統計量的穩健性（§6.1）---
    med = ["abn_attention_weekday_medianbase", "abn_attention_weekend_medianbase"]
    if all(c in d.columns for c in med):
        d3 = standardize_within(d, [*med, *BASE_CONTROLS])
        results += fit_both_inferences(d3, "ret_oc_next", med, BASE_CONTROLS,
                                       "H1-R medianbase｜oc", required_missing=missing)

    # --- H2：認知投入分層，**限 dense** ---
    dense = d[d["sparsity_tier"] == "dense"]
    cols = [f"abn_attention_{e}_{w}" for e in EFFORTS for w in ("weekday", "weekend")]
    d4 = standardize_within(dense, [*cols, *BASE_CONTROLS])
    for e in EFFORTS:
        results += fit_both_inferences(
            d4, "ret_oc_next", [f"abn_attention_{e}_weekday", f"abn_attention_{e}_weekend"],
            BASE_CONTROLS, f"H2-{e}｜oc（dense）", required_missing=missing)
    results += fit_both_inferences(
        d4, "ret_oc_next", [f"abn_attention_{e}_weekend" for e in EFFORTS],
        BASE_CONTROLS, "H2-同時 三層週末｜oc（dense）", required_missing=missing)

    return results_to_frame(results)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H1 主迴歸")
    ap.add_argument("--spec", default="C", help="C / B / A_prime")
    args = ap.parse_args(argv)
    path = PROCESSED / f"panel_{args.spec}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在；先跑 src.features.build --spec {args.spec}")
    print(f"[H1] 規格 {args.spec}")
    panel = pd.read_parquet(path)
    df = run(panel, load_settings())
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"T3_h1_main_{args.spec}.csv"
    df.to_csv(out, index=False)
    print(f"  → {out.relative_to(ROOT)}（{len(df)} 列）")
    st = df.drop_duplicates(["model", "inference"])["status"].value_counts()
    print(f"  模型狀態：{st.to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
