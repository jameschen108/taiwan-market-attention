"""§6.6 投資組合排序。

規格與 v1 逐項相同（`docs/PLAN_V2.md` §5 P4：「規格完全不動，跑在 C 上」），
只多一組 v2 的平行測度訊號（`abn_attention_users_weekend` 與
`abn_attention_comment_weekend`，§6.1b）。

**報酬用 `ret_oc_next`**（`returns.portfolio_definition`）：週日看到訊號，最早週一
開盤才成交，跨週末缺口對這個策略**不可得**。這一點與 §6.1 主規格一致，也是 §5.0 的
缺口證據線**不能**拿來當策略報酬的理由。

**呈現紀律**（§6.6、`LIMITATIONS.md`）：

- 宇宙為 ex-post 選出，一律標示 ex-post universe，不得宣稱為可實作策略。
- 可排序週若不連續，「每週平均 × 52」只是機械年化，不可解讀為可投資績效。
- **不含成本的多空價差不得作為主要結論陳述。**
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ..config import ROOT, load_settings
from .regressions import main_sample

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output"

#: `returns.portfolio_definition` → 面板欄位。未知值直接 KeyError，不預設回退。
RET_COL = {"open_to_close": "ret_oc_next", "close_to_close": "ret_cc_next"}

#: 主訊號 ＋ v2 的兩條平行測度（§6.1b）。舊語料沒有後兩者。
SIGNALS = [("主測度 發文數", "abn_attention_weekend"),
           ("平行 獨立帳號數", "abn_attention_users_weekend"),
           ("平行 留言則數", "abn_attention_comment_weekend")]


def round_trip_cost(fee_rate: float, fee_discount: float, tax_rate: float,
                    slippage_bps: float) -> float:
    """一次完整換手的單邊成本估計：手續費買賣各一次（可打折）＋ 證交稅 ＋ 兩次滑價。"""
    return fee_rate * fee_discount * 2 + tax_rate + (slippage_bps / 1e4) * 2


def quantile_portfolios(panel: pd.DataFrame, signal: str = "abn_attention_weekend",
                        ret_col: str = "ret_oc_next", n_q: int = 5,
                        weight: str = "equal", tradability_filter: bool = False,
                        min_weekly_value: float = 5_000_000,
                        min_names_per_bucket: int = 5) -> pd.DataFrame:
    """依前週週末關注度分 n 等分，回傳每週各組報酬與多空價差。"""
    cols = ["ticker", "week", signal, ret_col, "market_cap", "value"]
    d = panel[[c for c in cols if c in panel.columns]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[signal, ret_col])
    if tradability_filter:
        d = d[d["value"].fillna(0) >= min_weekly_value]

    rows = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    for week, g in d.groupby("week"):
        if len(g) < n_q * min_names_per_bucket:
            continue
        try:
            g = g.assign(q=pd.qcut(g[signal].rank(method="first"), n_q,
                                   labels=range(1, n_q + 1)))
        except ValueError:
            continue
        rec = {"week": week, "n_names": len(g)}
        long_names: set[str] = set()
        short_names: set[str] = set()
        for q, gq in g.groupby("q", observed=True):
            if weight == "value" and gq["market_cap"].notna().any():
                w = gq["market_cap"].fillna(0)
                r = float((gq[ret_col] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
            else:
                r = float(gq[ret_col].mean())
            rec[f"q{int(q)}"] = r
            rec[f"n_q{int(q)}"] = len(gq)
            if int(q) == n_q:
                long_names = set(gq["ticker"])
            elif int(q) == 1:
                short_names = set(gq["ticker"])
        if f"q{n_q}" in rec and "q1" in rec:
            rec["long_short"] = rec[f"q{n_q}"] - rec["q1"]

        # 實際換手率：與上週相比換掉的名單比例（兩腳平均）。假設每週 100% 換手會
        # 高估成本——訊號本身有持續性。
        def churn(now: set[str], prev: set[str]) -> float:
            return 1.0 if (not prev or not now) else len(now - prev) / len(now)

        rec["turnover_long"] = churn(long_names, prev_long)
        rec["turnover_short"] = churn(short_names, prev_short)
        rec["turnover_avg"] = (rec["turnover_long"] + rec["turnover_short"]) / 2
        prev_long, prev_short = long_names, short_names
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def summarize(port: pd.DataFrame, settings: dict, label: str, signal: str) -> dict:
    """含成本前／成本後的摘要。**成本後為必要欄，不是附錄。**"""
    base = {"portfolio": label, "signal": signal}
    if port.empty or "long_short" not in port.columns:
        return {**base, "status": "SKIPPED", "note": "可排序週不足，無法建組"}

    ls = port["long_short"].dropna()
    cfg = settings["portfolio"]
    cost = round_trip_cost(cfg["fee_rate"], cfg["fee_discount"],
                           cfg["tax_rate"], cfg["slippage_bps"])
    # 成本按**實際換手率**計，兩腳各一次；另保留 100% 換手的保守上界作為對照。
    turnover = port.loc[ls.index, "turnover_avg"].fillna(1.0)
    weekly_cost_series = cost * 2 * turnover
    net = ls - weekly_cost_series
    worst_case_cost = cost * 2

    def stats(s: pd.Series) -> tuple[float, float, float]:
        sd = s.std(ddof=1)
        t = float(s.mean() / (sd / np.sqrt(len(s)))) if sd > 0 else np.nan
        return float(s.mean()), t, float(sd)

    m_g, t_g, sd = stats(ls)
    m_n, t_n, _ = stats(net)
    weeks = pd.to_datetime(port["week"])

    return {
        **base, "status": "OK", "n_weeks": len(ls),
        "weeks_contiguous": bool((weeks.diff().dropna() == pd.Timedelta(days=7)).all()),
        "mean_ls_gross_weekly": m_g, "t_gross": t_g,
        "mean_ls_net_weekly": m_n, "t_net": t_n,
        "weekly_cost_assumed": float(weekly_cost_series.mean()),
        "mean_weekly_turnover": float(turnover.mean()),
        "weekly_cost_full_turnover": worst_case_cost,
        "mean_ls_net_full_turnover": m_g - worst_case_cost,
        "sd_weekly": sd,
        "mechanical_annualized_gross": m_g * 52,
        "mechanical_annualized_net": m_n * 52,
        "universe": "ex-post universe",
        "caveat": ("ex-post universe 之機械年化多空價差；不可解讀為年化績效、"
                   "可投資報酬或策略績效"),
    }


def run_all(panel: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """成本前／成本後、篩選前／篩選後、等權／市值加權、雙重排序，每個訊號各一組。"""
    cfg = settings["portfolio"]
    ret_col = RET_COL[settings["returns"]["portfolio_definition"]]
    specs = [("等權・未篩選", "equal", False),
             ("等權・可交易性篩選後", "equal", True),
             ("市值加權・未篩選", "value", False),
             ("市值加權・可交易性篩選後", "value", True)]
    summaries, series = [], []

    for sig_label, signal in SIGNALS:
        if signal not in panel.columns:
            # 舊語料無留言時戳／帳號 → 此訊號**不可能存在**，不是週數不足（§6.1b）
            summaries.append({"portfolio": "全部", "signal": signal,
                              "status": "SKIPPED",
                              "note": f"此規格的語料無此測度（缺欄位：{signal}）"})
            continue
        for label, weight, filt in specs:
            port = quantile_portfolios(
                panel, signal=signal, ret_col=ret_col, n_q=cfg["n_quantiles"],
                weight=weight, tradability_filter=filt,
                min_weekly_value=cfg["tradability_min_daily_value_twd"] * 5)
            summaries.append(summarize(port, settings, f"{sig_label}｜{label}", signal))
            if not port.empty:
                series.append(port.assign(spec=f"{sig_label}｜{label}", signal=signal))

        # 依市值中位數雙重排序（§6.6）
        if "market_cap" in panel.columns and panel["market_cap"].notna().any():
            med = panel.groupby("week")["market_cap"].transform("median")
            for side, mask in (("小型股", panel["market_cap"] <= med),
                               ("大型股", panel["market_cap"] > med)):
                port = quantile_portfolios(panel[mask], signal=signal,
                                           ret_col=ret_col, n_q=cfg["n_quantiles"])
                summaries.append(summarize(port, settings,
                                           f"{sig_label}｜雙重排序・{side}", signal))
                if not port.empty:
                    series.append(port.assign(spec=f"{sig_label}｜雙重排序・{side}",
                                              signal=signal))

    return (pd.DataFrame(summaries),
            pd.concat(series, ignore_index=True) if series else pd.DataFrame())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§6.6 投資組合")
    ap.add_argument("--spec", default="C", help="C / B / A_prime")
    args = ap.parse_args(argv)
    path = PROCESSED / f"panel_{args.spec}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在；先跑 src.features.build --spec {args.spec}")
    print(f"[T9] 規格 {args.spec}")
    # 與 v1 相同：投資組合建在主迴歸樣本（dense ＋ sparse）上
    summ, series = run_all(main_sample(pd.read_parquet(path)), load_settings())
    OUT.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT / f"T9_portfolios_{args.spec}.csv", index=False)
    print(f"  → output/T9_portfolios_{args.spec}.csv（{len(summ)} 列）")
    if not series.empty:
        series.to_csv(OUT / f"T9_portfolio_weekly_{args.spec}.csv", index=False)
        print(f"  → output/T9_portfolio_weekly_{args.spec}.csv（{len(series)} 列）")
    ok = summ[summ["status"] == "OK"]
    if not ok.empty:
        cols = ["portfolio", "n_weeks", "mean_ls_gross_weekly", "t_gross",
                "mean_ls_net_weekly", "t_net", "mean_weekly_turnover"]
        print()
        print(ok[cols].round(5).to_string(index=False))
        print("\n※ ex-post universe；**不含成本的多空價差不得作為主要結論陳述**（§6.6）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
