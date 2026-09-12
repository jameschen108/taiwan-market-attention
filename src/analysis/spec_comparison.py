"""T14：四方對照的面板層分解（PROJECT.md §6.7）。

```
A  → A′   測度定義效果（§0.1 的四項修正 ＋ bulk 三判準）
A′ → B    期間效果
B  → C    已刪文流失 ＋ 語料測度改變
```

本模組出**兩層**分解：面板層的測度統計（`describe`／`cell_level`），與 P4 跑完後的
**係數層**（`coefficient_decomposition`）。面板層是上半部，係數層是下半部。

**B→C 必須拆成兩欄**（§6.7、`LIMITATIONS.md` §2）。新語料在文章層是舊封存的真子集，
2019–2024 區間少掉 10.5% 的配對列且刪文非隨機，因此中間插入 **B∩C**（舊封存中新
語料也有的那些文章，`src/features/intersect_corpus.py`）：

```
A′ → B      期間效果
B  → B∩C    已刪文流失（同語料同定義，只是拿掉官方站已刪的文章）
B∩C → C     測度改變（同一批文章，換成直爬欄位重算）
```

B∩C **不是第五個規格**，是 T14 的診斷用中繼面板，走 `build_panel(source=...)`，
不進 `specs` 區塊。

規格 A 不在此表：它是 v1 既有的發表結果，用 v1 的程式與定義產出，本專案不重跑，
逐項數字引自 v1 的 `output/`（見 `docs/PLAN_V2.md`）。因此 A→A′ 這一段只能以
v1 發表的少數幾個數字對照，不是逐欄可比——這一點必須在表上寫明。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output"

SPECS = [("A_prime", "A′", "舊語料 pttweb × 全期 × v2 定義"),
         ("B", "B", "舊語料 pttweb × 主樣本期 × v2 定義"),
         ("pttweb_intersect", "B∩C", "舊語料 ∩ 新語料的文章 × 主樣本期"),
         ("C", "C", "新語料 pttcc × 主樣本期 × v2 定義")]

#: 係數層分解的三段。每一段只動一件事。
STEPS = [("A′", "B", "期間效果"),
         ("B", "B∩C", "已刪文流失（非隨機，LIMITATIONS.md §2）"),
         ("B∩C", "C", "測度改變（同一批文章，換直爬欄位重算）")]

#: 取哪幾個係數做分解。H1-4 是主檢定式，三個報酬定義各一次（§5.0、§6.1）。
COEF_TERMS = [("H1-4 週間＋週末｜oc", "abn_attention_weekend", "H1 週末｜ret_oc_next"),
              ("H1-4 週間＋週末｜oc", "abn_attention_weekday", "H1 週間｜ret_oc_next"),
              ("H1-4 週間＋週末｜gap", "abn_attention_weekend", "H1 週末｜ret_gap_next"),
              ("H1-4 週間＋週末｜cc", "abn_attention_weekend", "H1 週末｜ret_cc_next"),
              ("H1-3 僅週末｜oc", "abn_attention_weekend", "H1 僅週末式｜ret_oc_next")]

#: v1 發表的對應數字（規格 A）。只有這幾項可比，其餘欄位 v1 未發表。
V1_PUBLISHED = {
    "n_rows": 123828, "n_tickers": 260, "n_weeks": 506,
    "pct_zero_attention": 0.797, "n_dense_tickers": 36,
    "bulk_excluded_pct_rows": 0.427,
}


def _panel_path(spec: str) -> Path:
    return PROCESSED / f"panel_{spec}.parquet"


def describe(panel: pd.DataFrame) -> dict:
    z = panel["att_all"] == 0
    return {
        "n_rows": len(panel),
        "n_tickers": panel["ticker"].nunique(),
        "n_weeks": panel["week"].nunique(),
        "first_week": str(panel["week"].min().date()),
        "last_week": str(panel["week"].max().date()),
        "pct_zero_attention": round(float(z.mean()), 4),
        "mean_att_all": round(float(panel["att_all"].mean()), 4),
        "mean_att_weekend": round(float(panel["att_weekend"].mean()), 4),
        "pct_zero_att_weekend": round(float((panel["att_weekend"] == 0).mean()), 4),
        "n_dense_tickers": int(panel.loc[panel["sparsity_tier"] == "dense",
                                         "ticker"].nunique()),
        "n_sparse_tickers": int(panel.loc[panel["sparsity_tier"] == "sparse",
                                          "ticker"].nunique()),
        "n_dense_rows": int((panel["sparsity_tier"] == "dense").sum()),
        "pct_tier_missing": round(float(panel["sparsity_tier"].isna().mean()), 4),
        "n_abn_weekend_distinct": int(panel["abn_attention_weekend"].round(6).nunique()),
        "pct_abn_weekend_exactly_zero": round(
            float((panel["abn_attention_weekend"].fillna(np.nan) == 0).mean()), 4),
        "n_with_ret_next": int(panel["ret_next"].notna().sum()),
        "mean_ret_next": round(float(panel["ret_next"].mean()), 6),
        "pct_disposition_weeks": (round(float(panel["is_disposition_week"].mean()), 5)
                                  if panel["is_disposition_week"].notna().any() else ""),
        "pct_dt_ratio_missing": round(float(panel["dt_ratio"].isna().mean()), 4),
        "mean_dt_ratio": (round(float(panel["dt_ratio"].mean()), 4)
                          if panel["dt_ratio"].notna().any() else ""),
    }


def cell_level(spec_a: str, spec_b: str) -> dict:
    """格子層的對照。

    **總量相近不代表是同一批格子。** B 與 C 的面板列數必然相同（都是 ticker×week
    的同一組），零值率也可能碰巧接近；真正要問的是同一個 ticker-week 上兩者是否
    給出同樣的值。少了這一步，T14 的 B→C 那一欄會被總量的巧合誤導。
    """
    cols = ["ticker", "week", "att_all", "att_weekend",
            "abn_attention_weekend", "sparsity_tier"]
    a = pd.read_parquet(_panel_path(spec_a), columns=cols)
    b = pd.read_parquet(_panel_path(spec_b), columns=cols)
    m = a.merge(b, on=["ticker", "week"], suffixes=("_a", "_b"))
    out = {"n_cells": len(m)}
    for col in ("att_all", "att_weekend"):
        x, y = m[f"{col}_a"], m[f"{col}_b"]
        nz = (x > 0) | (y > 0)
        out[f"{col}_corr"] = round(float(np.corrcoef(x, y)[0, 1]), 4)
        out[f"{col}_pct_identical"] = round(float((x == y).mean()), 4)
        out[f"{col}_pct_identical_among_nonzero"] = round(float((x == y)[nz].mean()), 4)
        out[f"{col}_pct_b_lower"] = round(float((y < x).mean()), 4)
        out[f"{col}_pct_b_higher"] = round(float((y > x).mean()), 4)
    ok = m["abn_attention_weekend_a"].notna() & m["abn_attention_weekend_b"].notna()
    out["abn_weekend_corr"] = round(float(np.corrcoef(
        m.loc[ok, "abn_attention_weekend_a"], m.loc[ok, "abn_attention_weekend_b"])[0, 1]), 4)
    out["sparsity_tier_pct_identical"] = round(
        float((m["sparsity_tier_a"] == m["sparsity_tier_b"]).mean()), 4)
    return out


def coefficient_decomposition(out_dir: Path = OUT) -> pd.DataFrame:
    """係數層的三段分解。讀 `output/T3_h1_main_<spec>.csv`，**不重估**。

    **兩種推論標準各一列**（§0.2）：v2 的週末係數在兩者之間跨過 5% 門檻，只報一種
    會讓分解看起來在描述不同的東西。

    缺哪個規格就把那一欄留空並記在 `missing` 欄——不得以「差異為零」頂替。
    """
    tables = {}
    for spec, label, _ in SPECS:
        path = out_dir / f"T3_h1_main_{spec}.csv"
        if path.exists():
            tables[label] = pd.read_csv(path)

    rows = []
    for model, term, item in COEF_TERMS:
        for inf in ("twoway_2cluster", "firm_fe_1cluster"):
            vals, tvals = {}, {}
            for label, df in tables.items():
                hit = df[(df["model"] == model) & (df["term"] == term)
                         & (df["inference"] == inf) & (df["status"] == "OK")]
                if len(hit):
                    vals[label] = float(hit.iloc[0]["coef"])
                    tvals[label] = float(hit.iloc[0]["t"])
            row = {"item": item, "inference": inf,
                   "missing": ";".join(l for _, l, _ in SPECS if l not in vals)}
            for label in vals:
                row[f"coef_{label}"] = round(vals[label], 6)
                row[f"t_{label}"] = round(tvals[label], 3)
            for a, b, why in STEPS:
                row[f"Δ {a}→{b}"] = (round(vals[b] - vals[a], 6)
                                     if a in vals and b in vals else None)
                row[f"Δ {a}→{b} 的意義"] = why
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "T14_coefficient_decomposition.csv", index=False)
    print(f"  → {(out_dir / 'T14_coefficient_decomposition.csv').relative_to(ROOT)}")
    return out


def build(out_dir: Path = OUT) -> pd.DataFrame:
    rows, missing = {}, []
    for spec, label, _ in SPECS:
        path = _panel_path(spec)
        if not path.exists():
            missing.append(spec)
            continue
        rows[label] = describe(pd.read_parquet(path))
    if missing:
        print(f"  ⚠ 未建的規格：{missing}（跑 python3 -m src.features.build --spec <名>）")
    if not rows:
        raise FileNotFoundError("四方對照至少需要一個規格的面板")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "T14_spec_decomposition.csv")
    print(f"  → {(out_dir / 'T14_spec_decomposition.csv').relative_to(ROOT)}")

    # B→C 必須做格子層對照：兩者的面板形狀相同，總量相近可能只是巧合
    if not {"B", "C"} & set(missing):
        cl = cell_level("B", "C")
        pd.DataFrame([cl]).to_csv(out_dir / "T14_cell_level_B_vs_C.csv", index=False)
        print(f"  → {(out_dir / 'T14_cell_level_B_vs_C.csv').relative_to(ROOT)}")
        print(f"    B vs C 格子層：att_all corr={cl['att_all_corr']}、"
              f"完全相同 {cl['att_all_pct_identical']:.1%}；"
              f"tier 相同 {cl['sparsity_tier_pct_identical']:.1%}")
    return df


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="T14 四方對照").parse_args(argv)
    print("[T14] 面板層分解")
    df = build()
    print()
    print(df.to_string())
    print()
    print("[T14b] 係數層分解")
    coef = coefficient_decomposition()
    if not coef.empty:
        show = [c for c in coef.columns
                if c.startswith(("item", "inference", "coef_", "Δ"))
                and not c.endswith("的意義")]
        print(coef[show].to_string(index=False))
    print()
    print("v1 發表值（規格 A，僅供對照，非逐欄可比）：")
    for k, v in V1_PUBLISHED.items():
        print(f"  {k:<28} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
