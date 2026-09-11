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
    if disp.empty:
        panel["n_disposition_days"] = np.nan
        panel["is_disposition_week"] = np.nan
        panel["is_disposition_week_next"] = np.nan
        status["disposition_stocks"] = "PENDING：處置清單未收集"
    else:
        panel = panel.merge(disp, on=["ticker", "week"], how="left")
        panel["n_disposition_days"] = panel["n_disposition_days"].fillna(0).astype(int)
        panel["is_disposition_week"] = panel["n_disposition_days"] > 0
        # H3b 的應變數是**次週**的周轉率：污染它的是次週的處置，不是本週的。
        # 休市週不得跳過——與 ret_next 用同一條規則。
        panel = panel.sort_values(["ticker", "week"]).reset_index(drop=True)
        g = panel.groupby("ticker", sort=False)
        nxt_ok = g["week"].shift(-1) == panel["week"] + pd.Timedelta(days=7)
        panel["is_disposition_week_next"] = (
            g["is_disposition_week"].shift(-1).where(nxt_ok))
        status["disposition_stocks"] = "AVAILABLE"

    dt = weekly_day_trading(day_trading_path)
    if dt.empty:
        for c in ("dt_volume", "dt_buy_value", "dt_sell_value",
                  "n_dt_restricted_days", "dt_ratio", "dt_value", "dt_value_ratio"):
            panel[c] = np.nan
        status["day_trading"] = "PENDING：TWTB4U 未收集"
    else:
        panel = panel.merge(dt, on=["ticker", "week"], how="left")
        for c in ("dt_volume", "dt_buy_value", "dt_sell_value",
                  "n_dt_restricted_days"):
            panel[c] = panel[c].fillna(0.0)
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
