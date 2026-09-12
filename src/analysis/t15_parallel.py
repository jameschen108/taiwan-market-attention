"""T15 推文平行測度彙總（`docs/PLAN_V2.md` §P4、`PROJECT.md` §6.1b）。

回答 `docs/PLAN_V2.md` §2-1 的那個問題：

> v1 把一篇文章的全部關注度壓在「發文那一刻」的窗口。用有時戳的 1,439 萬則留言
> 重切窗口後，**週末／非交易時段的關注度是否被 v1 系統性低估？**

本模組**不重估任何模型**，只把已產出的表裡「主測度 vs 平行測度」的同位數字並排：

| 來源 | 主測度 | 平行測度 |
|---|---|---|
| `T3_h1_main`   | `H1-4 週間＋週末｜oc` | `H8-users` / `H8-comments` |
| `T5_h3_mechanism` | `H3a-1` / `H3b-1` | 同名 ＋`｜users` |
| `T6_joint_verdicts` | 主測度 發文數 | 平行 獨立帳號數 |
| `T7_verdicts`  | `all` / `weekend` | `comment` |
| `T9_portfolios` | 主測度 發文數 | 平行 獨立帳號數／留言則數 |
| `T13d`         | 發文數 | 留言則數／獨立帳號數 |

**判讀紀律**：平行測度的係數更大、t 值更高，**不等於 v1 的結論錯了**——兩者測的
不是同一件事（發了幾篇 vs 有多少人在講）。可以說的是「週末窗口在哪一種建構下有
足夠變異」，不得寫成「修正了 v1 的低估」。舊語料的推文沒有帳號也沒有時戳，A′ 與 B
上這一整張表必為 SKIPPED（§6.1b）。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ..config import ROOT

OUT = ROOT / "output"

TWO = "twoway_2cluster"


def _coef(df: pd.DataFrame, model: str, term: str, inf: str) -> tuple:
    hit = df[(df["model"] == model) & (df["term"] == term)
             & (df["inference"] == inf) & (df["status"] == "OK")]
    if not len(hit):
        return None, None
    r = hit.iloc[0]
    return float(r["coef"]), float(r["t"])


def _row(source, item, window, main, parallel, measure, note="") -> dict:
    mb, mt = main
    pb, pt = parallel
    return {
        "source": source, "item": item, "window": window,
        "parallel_measure": measure,
        "coef_main": mb, "t_main": mt, "coef_parallel": pb, "t_parallel": pt,
        "abs_t_ratio": (round(abs(pt) / abs(mt), 2)
                        if (mt not in (None, 0) and pt is not None
                            and mt == mt and pt == pt) else None),
        "status": "OK" if pb is not None else "SKIPPED",
        "note": note,
    }


def _read(name: str, spec: str) -> pd.DataFrame | None:
    path = OUT / f"{name}_{spec}.csv"
    return pd.read_csv(path) if path.exists() else None


def build(spec: str = "C") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, missing = [], []

    # --- H1 / H8：主檢定式 ---
    t3 = _read("T3_h1_main", spec)
    if t3 is None:
        missing.append("T3_h1_main")
    else:
        for w, term_main in (("週末", "abn_attention_weekend"),
                             ("週間", "abn_attention_weekday")):
            main = _coef(t3, "H1-4 週間＋週末｜oc", term_main, TWO)
            for label, model, pref in (
                    ("獨立帳號數", "H8-users 獨立帳號數｜oc", "abn_attention_users"),
                    ("留言則數", "H8-comments 留言則數｜oc", "abn_attention_comment")):
                term = f"{pref}_{'weekend' if w == '週末' else 'weekday'}"
                rows.append(_row("T3", "H1 次週報酬 ret_oc_next", w, main,
                                 _coef(t3, model, term, TWO), label))

    # --- H3 機制 ---
    t5 = _read("T5_h3_mechanism", spec)
    if t5 is None:
        missing.append("T5_h3_mechanism")
    else:
        for item, model in (("H3a 非三大法人訂單失衡", "H3a-1 ROI 主表"),
                            ("H3b 異常周轉率", "H3b-1 周轉率 主表")):
            for w in ("週末", "週間"):
                sfx = "weekend" if w == "週末" else "weekday"
                rows.append(_row(
                    "T5", item, w,
                    _coef(t5, model, f"abn_attention_{sfx}", TWO),
                    _coef(t5, f"{model}｜users", f"abn_attention_users_{sfx}", TWO),
                    "獨立帳號數"))

    # --- H6 ＋ H5 聯立判讀：比的是判讀，不是係數 ---
    t6 = _read("T6_joint_verdicts", spec)
    if t6 is None:
        missing.append("T6_joint_verdicts")
    verdict_rows = []
    if t6 is not None and "measure" in t6.columns:
        for m in t6["moderator"].unique():
            sub = t6[(t6["moderator"] == m) & (t6["inference"] == TWO)]
            def pick(meas):
                r = sub[sub["measure"] == meas]
                return (r.iloc[0]["verdict"], r.iloc[0]["interaction_t_next_week"]) \
                    if len(r) else ("（無）", "")
            v_main, t_main = pick("主測度 發文數")
            v_par, t_par = pick("平行 獨立帳號數")
            # 舊語料沒有帳號測度 → SKIPPED，不得寫成「兩個判讀不一致」
            have = v_par != "（無）" and "無此測度" not in str(v_par)
            verdict_rows.append({
                "source": "T6", "moderator": m, "inference": TWO,
                "status": "OK" if have else "SKIPPED",
                "t_main": t_main, "t_parallel": t_par if have else "",
                "verdict_main": v_main, "verdict_parallel": v_par,
                "verdict_agrees": (v_main == v_par) if have else ""})

    # --- H7 起始事件：比的也是判讀 ---
    t7 = _read("T7_verdicts", spec)
    if t7 is None:
        missing.append("T7_verdicts")
    else:
        def h7v(tag):
            r = t7[t7["event"] == tag]
            return (r.iloc[0]["verdict"], r.iloc[0]["n_events_matched"]) if len(r) else ("（無）", "")
        v_main, n_main = h7v("all")
        v_par, n_par = h7v("comment")
        have = "無此測度" not in str(v_par) and v_par != "（無）"
        verdict_rows.append({
            "source": "T7", "moderator": "起始事件（DiD）", "inference": "cluster 於事件週",
            "status": "OK" if have else "SKIPPED",
            "t_main": f"事件 {n_main}", "t_parallel": f"事件 {n_par}" if have else "",
            "verdict_main": v_main, "verdict_parallel": v_par,
            "verdict_agrees": (v_main == v_par) if have else ""})

    # --- §6.6 投資組合 ---
    t9 = _read("T9_portfolios", spec)
    if t9 is None:
        missing.append("T9_portfolios")
    else:
        def port(label):
            r = t9[(t9["portfolio"] == label) & (t9["status"] == "OK")]
            return ((float(r.iloc[0]["mean_ls_gross_weekly"]),
                     float(r.iloc[0]["t_gross"])) if len(r) else (None, None))
        for label in ("平行 獨立帳號數", "平行 留言則數"):
            rows.append(_row("T9", "多空價差（成本前，週）", "週末訊號",
                             port("主測度 發文數｜等權・未篩選"),
                             port(f"{label}｜等權・未篩選"),
                             label.replace("平行 ", ""),
                             note="ex-post universe；成本前價差不得作為主要結論陳述"))

    # --- T13d 測度效度 ---
    t13d = _read("T13d_measure_validity", spec)
    if t13d is None:
        missing.append("T13d_measure_validity")
    else:
        ok = t13d[(t13d["status"] == "OK")
                  & (t13d["sample"] == "主迴歸樣本 dense＋sparse")]
        base = ok[ok["measure"] == "發文數（主測度）"]
        if len(base):
            for meas in ("留言則數", "獨立帳號數"):
                r = ok[ok["measure"] == meas]
                par = ((float(r.iloc[0]["corr_all_weekend"]), np.nan) if len(r)
                       else (None, None))
                rows.append(_row(
                    "T13d", "corr(整週, 週末)（論文 Table 2 ＝ 0.3981）", "週末",
                    (float(base.iloc[0]["corr_all_weekend"]), np.nan),
                    par, meas,
                    note="相關係數，無 t 值" if len(r) else "此規格的語料無此測度"))

    coefs = pd.DataFrame(rows)
    verdicts = pd.DataFrame(verdict_rows)
    if missing:
        print(f"  ⚠ 缺少來源表：{missing}（先跑對應模組）")
    return coefs, verdicts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T15 推文平行測度彙總")
    ap.add_argument("--spec", default="C", help="C / B / A_prime")
    args = ap.parse_args(argv)
    print(f"[T15] 規格 {args.spec}")
    coefs, verdicts = build(args.spec)
    OUT.mkdir(parents=True, exist_ok=True)
    if not coefs.empty:
        coefs.to_csv(OUT / f"T15_parallel_measures_{args.spec}.csv", index=False)
        print(f"  → output/T15_parallel_measures_{args.spec}.csv（{len(coefs)} 列）")
    if not verdicts.empty:
        verdicts.to_csv(OUT / f"T15_parallel_verdicts_{args.spec}.csv", index=False)
        print(f"  → output/T15_parallel_verdicts_{args.spec}.csv（{len(verdicts)} 列）")
    if not coefs.empty:
        print()
        print(coefs[["source", "item", "window", "parallel_measure", "t_main",
                     "t_parallel", "abs_t_ratio", "status"]].round(3).to_string(index=False))
    if not verdicts.empty:
        print()
        print(verdicts[["source", "moderator", "status", "t_main", "t_parallel",
                        "verdict_agrees"]].to_string(index=False))
    print("\n※ 平行測度 t 值較高**不等於** v1 的結論錯了——兩者測的不是同一件事"
          "（發了幾篇 vs 有多少人在講）。見本模組 docstring 的判讀紀律。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
