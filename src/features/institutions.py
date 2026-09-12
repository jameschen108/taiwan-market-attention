"""台股制度性混淆的面板欄位（LIMITATIONS.md §12）。

原論文的美股設定沒有這三件事，v1 的文件也沒有寫：

| # | 機制 | 對哪個變數 |
|---|---|---|
| 12.1 | **處置股**：觸發條件是量價異常＝高關注度，處置後成交量與周轉率被制度性壓縮 | `turnover_next`＝H3b 的**應變數** |
| 12.2 | **當沖**：灌大 volume 卻不改變淨部位，把 ROI 機械性壓向 0 | `non_inst_roi_next`＝H3a 的應變數 |
| 12.3 | **漲跌幅 10%**：應變數在關注度最高的狀態下被截斷 | `ret_*`（已由 `touched_price_limit` 處理） |

12.1 與 12.2 都是「處理」直接決定「結果」的反向因果，不是雜訊。本模組把兩者化為
面板欄位，供 `robustness.disposition_stocks` 與 `robustness.day_trading` 使用。

**資料缺席時維持缺值並標記，不得靜默略過**——`audit/model_status.csv` 會記錄。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sessions import week_of


def _coverage(path: Path, date_cols: tuple[str, ...]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """收集檔實際涵蓋的日期範圍。

    **範圍外必須是缺值，不是零。** 把「該週沒有當沖／沒有被處置」與「那段期間根本
    沒收資料」混為一談，會讓 A′（2015 起）的 2015–2018 看起來當沖率為 0——那是
    假的零，而且會直接汙染以 `dt_ratio` 為控制變數的規格。
    """
    if not path.exists():
        return None
    d = pd.read_csv(path, usecols=list(date_cols), parse_dates=list(date_cols))
    if d.empty:
        return None
    return d[list(date_cols)].min().min(), d[list(date_cols)].max().max()


def disposition_weeks(path: Path, trading_days: set) -> pd.DataFrame:
    """把處置區間展開成 ticker × 週。

    處置的影響發生在**處置期間**而非公布日，且只在該期間的**交易日**上發生作用，
    因此以交易日曆展開，不用日曆日。回傳含該週落在處置期間的交易日數，讓呼叫端
    能區分「整週處置」與「週中才開始處置」。
    """
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "week", "n_disposition_days"])
    d = pd.read_csv(path, dtype={"ticker": str},
                    parse_dates=["start_date", "end_date"])
    days = pd.Series(sorted(trading_days))
    rows = []
    for r in d.itertuples():
        span = days[(days >= r.start_date.date()) & (days <= r.end_date.date())]
        for day in span:
            rows.append({"ticker": r.ticker, "day": day})
    if not rows:
        return pd.DataFrame(columns=["ticker", "week", "n_disposition_days"])
    # 同一檔的處置區間會重疊（連續兩次處置、或處置期間再被處置），(ticker, day)
    # 必須先去重，否則一週會數出 20 個交易日這種不可能的值
    flat = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "day"])
    flat["week"] = flat["day"].map(week_of)
    return (flat.groupby(["ticker", "week"]).size()
            .rename("n_disposition_days").reset_index())


def weekly_day_trading(path: Path) -> pd.DataFrame:
    """當沖的週聚合。

    `dt_ratio` 的分母用**當沖資料自己的期間內總成交股數**是錯的——TWTB4U 只列
    有當沖的個股，沒當沖的日子該檔不出現。分母必須來自行情面板的 `volume`，
    由呼叫端合併後再算，本函式只負責把分子加總。
    """
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "week", "dt_volume", "dt_buy_value",
                                     "dt_sell_value", "n_dt_restricted_days"])
    d = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["date"])
    d["week"] = d["date"].map(week_of)
    return (d.groupby(["ticker", "week"])
            .agg(dt_volume=("dt_volume", "sum"),
                 dt_buy_value=("dt_buy_value", "sum"),
                 dt_sell_value=("dt_sell_value", "sum"),
                 n_dt_restricted_days=("dt_restricted", "sum"))
            .reset_index())


def attach(panel: pd.DataFrame, disposition_path: Path, day_trading_path: Path,
           trading_days: set) -> tuple[pd.DataFrame, dict]:
    """把兩組制度欄位併入面板，並回傳可用性狀態（供 audit/model_status.csv）。"""
    status = {}

    disp = disposition_weeks(disposition_path, trading_days)
    disp_cov = _coverage(disposition_path, ("start_date", "end_date"))
    if disp.empty or disp_cov is None:
        panel["n_disposition_days"] = np.nan
        panel["is_disposition_week"] = np.nan
        panel["is_disposition_week_next"] = np.nan
        status["disposition_stocks"] = "PENDING：處置清單未收集"
    else:
        panel = panel.merge(disp, on=["ticker", "week"], how="left")
        lo, hi = disp_cov
        # 只在收集範圍內把未命中視為 0；範圍外維持缺值
        in_cov = (panel["week"] >= lo) & (panel["week"] <= hi + pd.Timedelta(days=7))
        panel["n_disposition_days"] = (
            panel["n_disposition_days"].where(~in_cov, panel["n_disposition_days"].fillna(0)))
        panel["is_disposition_week"] = (panel["n_disposition_days"] > 0).where(
            panel["n_disposition_days"].notna())
        n_out = int((~in_cov).sum())
        if n_out:
            status["disposition_coverage"] = (
                f"{lo.date()} ~ {hi.date()}；面板中 {n_out:,} 列落在範圍外，維持缺值")
        # H3b 的應變數是**次週**的周轉率：污染它的是次週的處置，不是本週的。
        # 休市週不得跳過——與 ret_next 用同一條規則。
        panel = panel.sort_values(["ticker", "week"]).reset_index(drop=True)
        g = panel.groupby("ticker", sort=False)
        nxt_ok = g["week"].shift(-1) == panel["week"] + pd.Timedelta(days=7)
        panel["is_disposition_week_next"] = (
            g["is_disposition_week"].shift(-1).where(nxt_ok))
        status["disposition_stocks"] = "AVAILABLE"

    dt = weekly_day_trading(day_trading_path)
    dt_cov = _coverage(day_trading_path, ("date",))
    if dt.empty or dt_cov is None:
        for c in ("dt_volume", "dt_buy_value", "dt_sell_value",
                  "n_dt_restricted_days", "dt_ratio", "dt_value", "dt_value_ratio"):
            panel[c] = np.nan
        status["day_trading"] = "PENDING：TWTB4U 未收集"
    else:
        panel = panel.merge(dt, on=["ticker", "week"], how="left")
        lo, hi = dt_cov
        in_cov = (panel["week"] >= lo) & (panel["week"] <= hi + pd.Timedelta(days=7))
        for c in ("dt_volume", "dt_buy_value", "dt_sell_value",
                  "n_dt_restricted_days"):
            panel[c] = panel[c].where(~in_cov, panel[c].fillna(0.0))
        n_out = int((~in_cov).sum())
        if n_out:
            status["day_trading_coverage"] = (
                f"{lo.date()} ~ {hi.date()}；面板中 {n_out:,} 列落在範圍外，維持缺值")
        # 分母用行情面板的週成交股數；volume 為 0 或缺值時 ratio 維持缺值
        denom = panel["volume"].where(panel["volume"] > 0)
        panel["dt_ratio"] = (panel["dt_volume"] / denom).clip(upper=1.0)
        # 金額基礎另出一個：常被引用的「當沖佔比 ~40%」是**金額**基礎的全市場數字，
        # 與股數基礎在全市場差近一倍（當沖集中在高價股）。兩個都給，避免比錯對象。
        panel["dt_value"] = (panel["dt_buy_value"] + panel["dt_sell_value"]) / 2.0
        if "value" in panel.columns:
            denom_v = panel["value"].where(panel["value"] > 0)
            panel["dt_value_ratio"] = (panel["dt_value"] / denom_v).clip(upper=1.0)
        else:
            # 行情面板未併入時維持缺值，不以股數基礎頂替
            panel["dt_value_ratio"] = np.nan
        status["day_trading"] = "AVAILABLE"

    return panel, status
