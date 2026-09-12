"""H7 起始事件：同週匹配對照的 DiD 事件研究（PROJECT.md §6.5）。

規格與 v1 的 `matched_events.py` **逐項相同**（`docs/PLAN_V2.md` §5 P4：「規格完全
不動，跑在 C 上」）：τ∈[−4, +8]、匹配變數 `ar_m1`／`ar_m2`／`pre_car_34`／
`log(市值)`／`log(周轉率)`、事前報酬 caliper 1%、k=3 近鄰、標準誤群集於事件週。

**為什麼要匹配**：v1 未匹配的事件研究在 τ=−2 就已顯著為正（t = 2.34），也就是
**價格先動、討論才出現**。平行趨勢不成立時，事後的 CAR 無法區分「關注度造成報酬」
與「兩者都由更早的價格衝擊驅動」。同週匹配使市場層級的共同衝擊自動差分掉。

判讀規則寫死於 `_verdict()`（§6.5）：

- 匹配後事前仍顯著 → 反向因果未消除，H7 在本設計下**無法識別**；
- 事前清除、事後顯著 → 支持 H7；
- 事前清除、事後不顯著 → H7 不成立。

`silent` 與 `sparse` 是本假說的樣本：回顧窗全零時 `AbnAtt` 恆為 0，連續型主規格
對它們沒有意義（§2.3），事件設計才是。

**v2 新增一條平行測度事件線**（`comment_is_initiation`，§6.1b）：留言測度的起始
事件數與文章測度相近但不是同一批格子。舊語料無留言時戳，A′／B 兩個規格上此線
**不可能存在**，必須寫成 SKIPPED，不得靜默略過。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import ROOT

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output"

PRE, POST = 4, 8

#: 匹配變數：**事前每一週的異常報酬分開進入**，而非只用 4 週累積。只匹配累積值會讓
#: 組成不同——關注度起始多半由「上一週剛發生的價格變動」觸發，累積量相同但集中在
#: τ=−1 的事件仍會殘留事前趨勢（v1 實測 τ=−1 的 t 值達 4.75）。
MATCH_VARS = ["ar_m1", "ar_m2", "pre_car_34", "log_mktcap", "log_turnover"]

#: 事前報酬的匹配容忍度（絕對值，週報酬）。只用最近鄰不夠：SMD 雖小（< 0.1），
#: 但事件數上千時極小的均值差仍高度顯著，事前趨勢因此殘留。寧可損失事件數。
RETURN_CALIPER = 0.010
RET_MATCH_VARS = ["ar_m1", "ar_m2", "pre_car_34"]

TIERS = ("sparse", "silent")

#: (事件欄, 標籤)。第三條是 v2 的平行測度線（§6.1b）。
EVENT_COLS = [("is_initiation", "all"),
              ("is_initiation_weekend", "weekend"),
              ("comment_is_initiation", "comment")]


def _prepare(panel: pd.DataFrame, tiers: tuple[str, ...]) -> pd.DataFrame:
    d = panel[panel["sparsity_tier"].isin(tiers)].copy()
    d = d.sort_values(["ticker", "week"])
    # 異常報酬 = 個股週報酬 − 同週橫斷面均值（週固定效果的事件研究對應）
    d["ar"] = d["ret"] - d.groupby("week")["ret"].transform("mean")
    d["log_mktcap"] = np.log(d["market_cap"].where(d["market_cap"] > 0))
    d["log_turnover"] = np.log1p(d["turnover"] * 1e4)
    return d


def _windows(d: pd.DataFrame, event_col: str) -> pd.DataFrame:
    """每檔取出所有「有完整 τ∈[−PRE, +POST] 連續週」的位置，每個候選一列。"""
    recs = []
    for ticker, g in d.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        weeks, ar = g["week"].values, g["ar"].values
        is_ev = g[event_col].fillna(False).astype(bool).values
        for i in range(PRE, len(g) - POST):
            # 事件窗必須是連續週，否則 τ 對齊會錯
            span = weeks[i - PRE:i + POST + 1]
            if np.any(np.diff(span).astype("timedelta64[D]").astype(int) != 7):
                continue
            path = ar[i - PRE:i + POST + 1]
            if np.isnan(path).any():
                continue
            recs.append({
                "ticker": ticker, "week": g["week"].iat[i],
                "treated": bool(is_ev[i]),
                "sparsity_tier": g["sparsity_tier"].iat[i],
                "pre_car": float(path[:PRE].sum()),
                "ar_m1": float(path[PRE - 1]),                  # τ = −1
                "ar_m2": float(path[PRE - 2]),                  # τ = −2
                "pre_car_34": float(path[:PRE - 2].sum()),      # τ = −4, −3 累積
                "log_mktcap": g["log_mktcap"].iat[i],
                "log_turnover": g["log_turnover"].iat[i],
                **{f"ar_{tau}": float(path[k])
                   for k, tau in enumerate(range(-PRE, POST + 1))},
            })
    return pd.DataFrame(recs)


def match(cands: pd.DataFrame, k: int = 3,
          caliper: float | None = RETURN_CALIPER) -> pd.DataFrame:
    """同週、同分層的 k-近鄰匹配，加事前報酬 caliper。

    距離為匹配變數在該週該分層內標準化後的歐氏距離。`caliper` 給定時，對照在
    τ=−1、−2 與 τ=−4..−3 上都必須落在容忍度內；否則該事件**無可用對照而被剔除**。
    """
    usable = cands.dropna(subset=MATCH_VARS)
    pairs = []
    for (week, tier), g in usable.groupby(["week", "sparsity_tier"], sort=False):
        treated, controls = g[g["treated"]], g[~g["treated"]]
        if treated.empty or len(controls) < k:
            continue
        mu = g[MATCH_VARS].mean()
        sd = g[MATCH_VARS].std().replace(0, np.nan)
        T = ((treated[MATCH_VARS] - mu) / sd).fillna(0.0).to_numpy()
        C = ((controls[MATCH_VARS] - mu) / sd).fillna(0.0).to_numpy()
        dist = np.sqrt(((T[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))

        if caliper is not None:
            for v in RET_MATCH_VARS:     # 逐一報酬維度施加絕對值 caliper
                tv = treated[v].to_numpy()[:, None]
                cv = controls[v].to_numpy()[None, :]
                dist = np.where(np.abs(tv - cv) > caliper, np.inf, dist)

        order = np.argsort(dist, axis=1)[:, :k]
        for ti in range(len(treated)):
            trow = treated.iloc[ti]
            for ci in order[ti]:
                if not np.isfinite(dist[ti, ci]):
                    continue            # caliper 內無可用對照 → 該事件剔除
                crow = controls.iloc[ci]
                pairs.append({
                    "week": week, "sparsity_tier": tier,
                    "treated_ticker": trow["ticker"],
                    "control_ticker": crow["ticker"],
                    "distance": float(dist[ti, ci]),
                    **{f"t_ar_{tau}": trow[f"ar_{tau}"]
                       for tau in range(-PRE, POST + 1)},
                    **{f"c_ar_{tau}": crow[f"ar_{tau}"]
                       for tau in range(-PRE, POST + 1)},
                    **{f"t_{v}": trow[v] for v in MATCH_VARS},
                    **{f"c_{v}": crow[v] for v in MATCH_VARS},
                })
    return pd.DataFrame(pairs)


def balance_table(cands: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """匹配前後的標準化差異（|SMD| < 0.1 通常視為平衡）。"""
    rows = []
    t_all, c_all = cands[cands["treated"]], cands[~cands["treated"]]
    for v in MATCH_VARS:
        pre_smd = ((t_all[v].mean() - c_all[v].mean())
                   / np.sqrt((t_all[v].var() + c_all[v].var()) / 2))
        if pairs.empty:
            post_smd = np.nan
        else:
            tv, cv = pairs[f"t_{v}"], pairs[f"c_{v}"]
            post_smd = ((tv.mean() - cv.mean())
                        / np.sqrt((tv.var() + cv.var()) / 2))
        rows.append({
            "variable": v,
            "treated_mean": float(t_all[v].mean()),
            "control_mean_unmatched": float(c_all[v].mean()),
            "smd_unmatched": float(pre_smd),
            "control_mean_matched": (float(pairs[f"c_{v}"].mean())
                                     if not pairs.empty else np.nan),
            "smd_matched": float(post_smd),
            "balanced": bool(abs(post_smd) < 0.1) if post_smd == post_smd else False,
        })
    return pd.DataFrame(rows)


def did_car(pairs: pd.DataFrame) -> pd.DataFrame:
    """DiD 事件路徑：每個 τ 的 (treated − control) 平均差與 t 值。

    標準誤在**事件週**層級群集——同一週的多個事件共享市場狀態，視為獨立會低估標準誤。
    """
    if pairs.empty:
        return pd.DataFrame()
    # 先把同一 treated 事件的多個對照平均掉，回到「每事件一列」
    agg = {f"t_ar_{tau}": "first" for tau in range(-PRE, POST + 1)}
    agg.update({f"c_ar_{tau}": "mean" for tau in range(-PRE, POST + 1)})
    ev = pairs.groupby(["week", "treated_ticker"], as_index=False).agg(agg)

    rows = []
    for tau in range(-PRE, POST + 1):
        tmp = pd.DataFrame({"d": ev[f"t_ar_{tau}"] - ev[f"c_ar_{tau}"],
                            "week": ev["week"]}).dropna()
        if tmp.empty:
            continue
        fit = sm.OLS(tmp["d"], np.ones(len(tmp))).fit(
            cov_type="cluster", cov_kwds={"groups": tmp["week"]})
        rows.append({
            "tau": tau, "mean_diff": float(fit.params.iloc[0]),
            "se": float(fit.bse.iloc[0]), "t": float(fit.tvalues.iloc[0]),
            "p": float(fit.pvalues.iloc[0]), "n_events": int(len(tmp)),
            "n_week_clusters": int(tmp["week"].nunique()),
            "treated_mean_ar": float(ev[f"t_ar_{tau}"].mean()),
            "control_mean_ar": float(ev[f"c_ar_{tau}"].mean()),
        })
    out = pd.DataFrame(rows)
    out["car_diff"] = out["mean_diff"].cumsum()
    return out


def initiation_propensity(cands: pd.DataFrame) -> pd.DataFrame:
    """事前報酬是否預測「關注度起始」？這是反向因果強度的直接檢定。"""
    d = cands.dropna(subset=MATCH_VARS).copy()
    if d.empty or d["treated"].nunique() < 2:
        return pd.DataFrame()
    X = sm.add_constant(d[MATCH_VARS])
    try:
        fit = sm.Logit(d["treated"].astype(int), X).fit(disp=0)
    except Exception:                                       # noqa: BLE001
        return pd.DataFrame()
    return pd.DataFrame({
        "term": fit.params.index, "coef": fit.params.values,
        "z": fit.tvalues.values, "p": fit.pvalues.values,
        "odds_ratio": np.exp(fit.params.values),
        "n_obs": len(d), "pseudo_r2": fit.prsquared,
    })


def _verdict(matched: pd.DataFrame) -> str:
    """判讀規則寫死於程式（§6.5）。"""
    if matched.empty:
        return "匹配後無可用事件，無法判讀"
    pre = matched[matched["tau"] < 0]
    post = matched[matched["tau"] >= 1]
    if not bool((pre["t"].abs() < 1.96).all()):
        return "匹配後事前差異仍顯著 → 反向因果未被消除，H7 在本設計下無法識別"
    if bool((post["t"].abs() > 1.96).any()):
        return ("事前差異已消除、事後仍有顯著差異 → 關注度起始帶有超出價格動能的資訊，"
                "支持 H7")
    return ("事前差異已消除、事後亦無顯著差異 → 原始 CAR 型態可由價格動能完全解釋，"
            "H7 不成立")


@dataclass
class Result:
    unmatched: pd.DataFrame
    matched: pd.DataFrame
    balance: pd.DataFrame
    propensity: pd.DataFrame
    clean: pd.DataFrame
    verdict: str
    n_events_raw: int = 0


def run_one(panel: pd.DataFrame, event_col: str,
            tiers: tuple[str, ...] = TIERS, k: int = 3) -> Result:
    d = _prepare(panel, tiers)
    cands = _windows(d, event_col)
    if cands.empty or not cands["treated"].any():
        return Result(*[pd.DataFrame()] * 5, "無可用事件窗", 0)

    # 未匹配基準（供對照）——這正是 v1 平行趨勢失敗的那一條
    rows = []
    for tau in range(-PRE, POST + 1):
        t = cands.loc[cands["treated"], f"ar_{tau}"]
        c = cands.loc[~cands["treated"], f"ar_{tau}"]
        rows.append({"tau": tau, "treated_mean_ar": float(t.mean()),
                     "control_mean_ar": float(c.mean()),
                     "mean_diff": float(t.mean() - c.mean()),
                     "n_events": int(len(t))})
    unmatched = pd.DataFrame(rows)
    unmatched["car_diff"] = unmatched["mean_diff"].cumsum()

    pairs = match(cands, k=k)
    matched = did_car(pairs)

    # 「乾淨」子樣本：事前累積 AR 落在中間三分位（未明顯被價格驅動）
    tr = cands[cands["treated"]]
    lo, hi = tr["pre_car"].quantile([1 / 3, 2 / 3])
    clean_ids = tr[(tr["pre_car"] >= lo) & (tr["pre_car"] <= hi)]
    clean = did_car(pairs.merge(
        clean_ids[["ticker", "week"]].rename(columns={"ticker": "treated_ticker"}),
        on=["treated_ticker", "week"], how="inner")) if not pairs.empty else pd.DataFrame()

    return Result(unmatched, matched, balance_table(cands, pairs),
                  initiation_propensity(cands), clean, _verdict(matched),
                  int(cands["treated"].sum()))


def run(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                      pd.DataFrame, pd.DataFrame]:
    """三條事件線一次跑完，回傳（路徑、平衡、傾向分數、判讀）四張長格式表。"""
    paths, balances, props, verdicts = [], [], [], []
    for col, tag in EVENT_COLS:
        if col not in panel.columns:
            # 舊語料沒有留言時戳 → 此測度**不可能存在**，不是樣本不足（§6.1b）
            verdicts.append({"event": tag, "event_col": col, "status": "SKIPPED",
                             "n_events_raw": 0, "n_events_matched": 0,
                             "n_week_clusters": 0, "n_balanced": 0,
                             "verdict": f"此規格的語料無此測度（缺欄位：{col}）"})
            continue
        res = run_one(panel, col)
        for name, df in (("unmatched", res.unmatched), ("matched", res.matched),
                         ("clean", res.clean)):
            if not df.empty:
                paths.append(df.assign(event=tag, sample=name))
        if not res.balance.empty:
            balances.append(res.balance.assign(event=tag))
        if not res.propensity.empty:
            props.append(res.propensity.assign(event=tag))
        verdicts.append({
            "event": tag, "event_col": col,
            "status": "OK" if not res.matched.empty else "SKIPPED",
            "n_events_raw": res.n_events_raw,
            "n_events_matched": (int(res.matched["n_events"].max())
                                 if not res.matched.empty else 0),
            "n_week_clusters": (int(res.matched["n_week_clusters"].max())
                                if not res.matched.empty else 0),
            "n_balanced": int(res.balance["balanced"].sum()) if not res.balance.empty else 0,
            "verdict": res.verdict,
        })
    cat = lambda fs: (pd.concat(fs, ignore_index=True) if fs else pd.DataFrame())  # noqa: E731
    return cat(paths), cat(balances), cat(props), pd.DataFrame(verdicts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H7 起始事件（同週匹配 DiD）")
    ap.add_argument("--spec", default="C", help="C / B / A_prime")
    args = ap.parse_args(argv)
    path = PROCESSED / f"panel_{args.spec}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在；先跑 src.features.build --spec {args.spec}")
    print(f"[H7] 規格 {args.spec}")
    paths, balance, prop, verdicts = run(pd.read_parquet(path))
    OUT.mkdir(parents=True, exist_ok=True)
    for df, name in ((paths, "T7_event_paths"), (balance, "T7_balance"),
                     (prop, "T7_propensity"), (verdicts, "T7_verdicts")):
        if df.empty:
            continue
        out = OUT / f"{name}_{args.spec}.csv"
        df.to_csv(out, index=False)
        print(f"  → {out.relative_to(ROOT)}（{len(df)} 列）")
    print()
    print(verdicts.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
