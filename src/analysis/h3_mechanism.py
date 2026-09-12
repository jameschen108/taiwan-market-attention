"""H3a／H3b 機制檢定（PROJECT.md §6.2）。

以 `non_inst_roi_next` 與 `turnover_next` 取代應變數，各加落後應變數。

| 結果 | 支持的管道 |
|---|---|
| 週間 → ROI 顯著；週末 → Turnover 顯著 | 資訊處理 |
| 週末 → ROI 顯著 | 價格壓力 |

**兩個應變數都被台股制度混淆，而且方向已量出來**（`LIMITATIONS.md` §12）：

- `turnover_next`：處置股的觸發條件正是量價異常＝高關注度，處置後周轉率被制度性
  壓縮。污染的是**次週**的處置，因此穩健性用 `is_disposition_week_next`。
- `non_inst_roi_next`：當沖灌大 volume 卻不改淨部位，把 ROI 壓向 0。實測
  corr(dt_ratio, |ROI|) = −0.242、corr(dt_ratio, 關注度) = +0.220——**衰減發生在
  自變數高的地方**。因此 `dt_ratio` 必須進控制項，並另做高／低當沖分層。

主表與「已排除混淆」的版本**必須並列**：只報其中一種都會誤導。
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..config import ROOT, load_settings
from .regressions import (BASE_CONTROLS, INFERENCE, ModelResult, add_derived,
                          fit_both_inferences, main_sample, results_to_frame,
                          standardize_within)

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output"
XS = ["abn_attention_weekday", "abn_attention_weekend"]

#: §6.0：**每個主表另加一組以 `att_users_*`（獨立帳號數）為自變數的平行估計。**
#: 主測度的週末窗口在 79.7% 的列上退化成水準值，獨立帳號數的相異值是它的 7 倍
#: （`LIMITATIONS.md` §11）。舊語料的推文沒有帳號，A′／B 上此組必為 SKIPPED。
MEASURES = [("", XS),
            ("｜users", ["abn_attention_users_weekday",
                         "abn_attention_users_weekend"])]


def _estimate(std: pd.DataFrame, xs: list[str], tag: str, missing: list[str],
              has_dt: bool, has_disp: bool) -> list[ModelResult]:
    """單一測度的 H3a ＋ H3b。主表與「已排除混淆」的版本**必須並列**。"""
    results: list[ModelResult] = []

    # --- H3a：非三大法人訂單失衡 ---
    base = [*BASE_CONTROLS, "non_inst_roi_lag1"]
    results += fit_both_inferences(std, "non_inst_roi_next", xs, base,
                                   f"H3a-1 ROI 主表{tag}", required_missing=missing)
    if has_dt:
        results += fit_both_inferences(std, "non_inst_roi_next", xs,
                                       [*base, "dt_ratio"],
                                       f"H3a-2 ROI ＋當沖控制{tag}",
                                       required_missing=missing)
        hi = std["dt_ratio"] > std["dt_ratio"].median()
        for lab, sub in (("高當沖", std[hi]), ("低當沖", std[~hi])):
            results += fit_both_inferences(sub, "non_inst_roi_next", xs, base,
                                           f"H3a-3 ROI（{lab}半數）{tag}",
                                           required_missing=missing)
    else:
        results += [ModelResult(f"H3a-2 ROI ＋當沖控制{tag}", "SKIPPED", inference=i,
                                note="當沖資料不可用（robustness.day_trading）")
                    for i in INFERENCE]

    # --- H3b：異常周轉率 ---
    base_t = [*BASE_CONTROLS, "turnover_lag1"]
    results += fit_both_inferences(std, "turnover_next", xs, base_t,
                                   f"H3b-1 周轉率 主表{tag}", required_missing=missing)
    if has_disp:
        keep = std[std["is_disposition_week_next"].fillna(False) == False]  # noqa: E712
        results += fit_both_inferences(keep, "turnover_next", xs, base_t,
                                       f"H3b-2 周轉率（排除次週處置）{tag}",
                                       required_missing=missing)
    else:
        results += [ModelResult(f"H3b-2 周轉率（排除次週處置）{tag}", "SKIPPED",
                                inference=i,
                                note="處置清單不可用（robustness.disposition_stocks）")
                    for i in INFERENCE]
    return results


def run(panel: pd.DataFrame, settings: dict) -> pd.DataFrame:
    missing = settings["regression"]["controls"]["required_but_missing"]
    d = main_sample(add_derived(panel))
    has_dt = "dt_ratio" in d.columns and d["dt_ratio"].notna().any()
    has_disp = ("is_disposition_week_next" in d.columns
                and d["is_disposition_week_next"].notna().any())
    results: list[ModelResult] = []

    for tag, xs in MEASURES:
        if not all(c in d.columns for c in xs):
            # 舊語料的推文無帳號 → 此測度**不可能存在**，不是樣本不足（§6.1b）
            note = ("此規格的語料無此測度（缺欄位："
                    + ";".join(c for c in xs if c not in d.columns) + "）")
            results += [ModelResult(f"H3 平行測度{tag}", "SKIPPED", inference=i,
                                    note=note) for i in INFERENCE]
            continue
        std = standardize_within(d, [*xs, *BASE_CONTROLS,
                                     *(["dt_ratio"] if has_dt else [])])
        results += _estimate(std, xs, tag, missing, has_dt, has_disp)
    return results_to_frame(results)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H3 機制檢定")
    ap.add_argument("--spec", default="C")
    args = ap.parse_args(argv)
    print(f"[H3] 規格 {args.spec}")
    panel = pd.read_parquet(PROCESSED / f"panel_{args.spec}.parquet")
    df = run(panel, load_settings())
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"T5_h3_mechanism_{args.spec}.csv"
    df.to_csv(out, index=False)
    print(f"  → {out.relative_to(ROOT)}")
    print(f"  模型狀態：{df.drop_duplicates(['model','inference'])['status'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
