"""時段窗口指派與週對齊（PROJECT.md §2.1）。

**v2 不改動任何窗口規則**——改了就無法與 v1 對照。唯一的差別在呼叫端：
v1 只能用文章時戳呼叫，v2 的留言帶自己的分鐘級時戳，因此留言以**自己的時戳**
呼叫同一組函式（PROJECT.md §2.4）。

不可用日曆上的週六日判定「非交易時段」——台灣春節、連假、颱風假頻繁，寫死
「週五 13:30 → 週一 09:00」會系統性低估非交易窗口。一律以實際交易日曆判定。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

OPEN_MIN = 9 * 60          # 09:00
CLOSE_MIN = 13 * 60 + 30   # 13:30


def assign_session_window(ts: pd.Timestamp, trading_days: set[dt.date]) -> str:
    """交易日的 09:00–13:30 為 intraday，其餘一律 non_trading。

    刻意歸入 `non_trading` 的兩段台股時間（PROJECT.md §2.1）：
      - 08:30–09:00 開盤前試撮：有揭示試算價、可掛單，但無成交。
      - 13:30 後的盤後定價交易（14:00–14:30）與 2020-10-26 前的盤後零股。
    2020-03-23 起收盤前 5 分鐘（13:25–13:30）為集合競價，仍在 intraday 區間內。

    **生產路徑用的是 `build.assign_windows` 的向量化版本**，本函式為同一規則的
    純量參照實作，由 `tests/test_panel.py::test_window_rules_agree` 守住兩者一致。
    """
    date = ts.date() if isinstance(ts, pd.Timestamp) else ts.date()
    minutes = ts.hour * 60 + ts.minute
    if date in trading_days and OPEN_MIN <= minutes < CLOSE_MIN:
        return "intraday"
    return "non_trading"


def assign_calendar_window(ts: pd.Timestamp) -> str:
    """日曆切法（PROJECT.md §2.1 主規格，與原論文可比）：週六日為 weekend。"""
    return "weekend" if ts.weekday() >= 5 else "weekday"


def next_trading_day(ts: pd.Timestamp, trading_days_sorted: list[dt.date]) -> dt.date | None:
    """該時點之後最近的開市日。

    **這不是窗口指派規則。** 窗口一律用時戳自己的 W-SUN 週（`week_of`）與交易時段
    判定（`assign_session_window`）；本函式只用於 `attention_week` 的期末守門——
    樣本末尾之後已無開市日者無法指派報酬期，回傳 None 並列入排除清單（§1 終點規則）。
    """
    import bisect

    date = ts.date()
    minutes = ts.hour * 60 + ts.minute
    if date in set(trading_days_sorted) and minutes < CLOSE_MIN:
        return date
    idx = bisect.bisect_right(trading_days_sorted, date)
    if idx >= len(trading_days_sorted):
        return None  # 期末無法指派，列入排除清單（PROJECT.md §1 終點規則）
    return trading_days_sorted[idx]


def week_of(date: dt.date | pd.Timestamp) -> pd.Timestamp:
    """星期日錨定的週標籤（W-SUN）。

    使週六／日的關注度在時間上嚴格先於次週一至週五的報酬（PROJECT.md §1）。
    回傳該週的**結束日**（星期日），與 pandas `resample('W-SUN')` 一致。
    """
    ts = pd.Timestamp(date).normalize()
    # 週一=0 … 週日=6；W-SUN 的週結束於星期日
    offset = (6 - ts.weekday()) % 7
    return ts + pd.Timedelta(days=offset)


def attention_week(ts: pd.Timestamp, trading_days_sorted: list[dt.date]) -> pd.Timestamp | None:
    """關注度所屬的「特徵週」。

    以貼文時間所在的 W-SUN 週為準：週六／日的貼文落在以該週日結尾的週，其報酬期為
    下一個日曆週的週一至週五，因此嚴格領先。
    """
    if next_trading_day(ts, trading_days_sorted) is None:
        return None
    return week_of(ts)


def return_week(feature_week: pd.Timestamp) -> pd.Timestamp:
    """特徵週 w 對應的報酬週：下一個 W-SUN 週（週一開盤 → 週五收盤）。"""
    return pd.Timestamp(feature_week) + pd.Timedelta(days=7)


def build_trading_calendar(dates: list[dt.date]) -> dict:
    """由實際成交日建立交易日曆，並標記補班星期六（PROJECT.md §2.1、§6.4）。"""
    days = sorted(set(dates))
    makeup_saturdays = [d for d in days if d.weekday() == 5]
    return {
        "trading_days": set(days),
        "trading_days_sorted": days,
        "makeup_saturdays": makeup_saturdays,
    }


def week_trading_day_count(trading_days: set[dt.date], week_end: pd.Timestamp) -> int:
    """報酬週（週一至週五）的實際交易日數。不足 5 天即為不完整週（PROJECT.md §5）。"""
    monday = pd.Timestamp(week_end) - pd.Timedelta(days=6)
    return sum(
        1 for i in range(5)
        if (monday + pd.Timedelta(days=i)).date() in trading_days
    )
