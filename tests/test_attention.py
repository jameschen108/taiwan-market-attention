"""關注度測度的不變量：不得 look-ahead、缺值維持缺值、門檻照設定檔。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings
from src.features.attention import (abnormal_attention, build_attention_panel,
                                    build_comment_panel, herfindahl,
                                    sparsity_fields, sparsity_tier)
from src.features.sessions import (assign_calendar_window, assign_session_window,
                                   week_of, return_week, week_trading_day_count)


# ------------------------------------------------------------ AbnAtt 契約

def test_abnormal_attention_excludes_current_period():
    """回顧窗嚴格不含當期——含了就是 look-ahead。"""
    counts = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 100])
    out = abnormal_attention(counts, lookback=8, min_periods=8)
    assert out.iloc[:8].isna().all()          # 不足 8 週維持缺值
    # 第 9 期：log1p(100) − log1p(1) > 0，且基準完全不含當期的 100
    assert out.iloc[8] == pytest.approx(np.log1p(100) - np.log1p(1))


def test_abnormal_attention_keeps_zeros_as_real_zeros():
    """PTT 的零是真實的零，log1p 必須保留它而非視為缺值。"""
    counts = pd.Series([0] * 8 + [0])
    out = abnormal_attention(counts, 8, 8, "log1p")
    assert out.iloc[8] == 0.0                  # 一直是零 → 異常值為零，不是 NaN


def test_posonly_variant_treats_zero_as_missing():
    counts = pd.Series([1] * 8 + [0])
    out = abnormal_attention(counts, 8, 8, "log_positive")
    assert np.isnan(out.iloc[8])


def test_sparsity_fields_use_only_past():
    counts = pd.Series([0] * 8 + [5, 0])
    sp = sparsity_fields(counts, lookback_weeks=52, abn_lookback=8)
    # 第 9 期（index 8）本身非零，且前 8 週全零 → 起始事件
    assert bool(sp["is_initiation"].iloc[8]) is True
    assert bool(sp["att_lookback_all_zero"].iloc[8]) is True
    # 第 10 期本身為零 → 不是起始事件（起始要求當期為正）
    assert bool(sp["is_initiation"].iloc[9]) is False


@pytest.mark.parametrize("nonzero,expected", [
    (40, "dense"), (39, "sparse"), (4, "sparse"), (3, "silent"), (0, "silent")])
def test_sparsity_tier_boundaries(nonzero, expected):
    s = pd.Series([float(nonzero)])
    assert sparsity_tier(s, dense_min=40, silent_max=3).iloc[0] == expected


def test_sparsity_tier_missing_stays_missing():
    assert pd.isna(sparsity_tier(pd.Series([np.nan])).iloc[0])


def test_herfindahl():
    assert herfindahl(pd.Series([1, 1, 1, 1])) == pytest.approx(0.25)
    assert herfindahl(pd.Series([3, 1])) == pytest.approx(0.625)
    assert herfindahl(pd.Series([5])) == 1.0        # n=1 恆為 1，是定義不是證據
    assert np.isnan(herfindahl(pd.Series([0, 0])))


# ------------------------------------------------------------ 窗口與週對齊

def test_calendar_window_weekend_is_saturday_sunday():
    assert assign_calendar_window(pd.Timestamp("2021-06-05 10:00")) == "weekend"  # 六
    assert assign_calendar_window(pd.Timestamp("2021-06-06 10:00")) == "weekend"  # 日
    assert assign_calendar_window(pd.Timestamp("2021-06-04 10:00")) == "weekday"  # 五


def test_session_window_uses_real_calendar_not_weekday():
    """國定假日的平日盤中仍是 non_trading——用星期幾判定會系統性低估非交易窗口。"""
    holiday = pd.Timestamp("2021-02-15 10:00")     # 春節，週一但休市
    trading = {pd.Timestamp("2021-06-04").date()}
    assert assign_session_window(holiday, trading) == "non_trading"
    assert assign_session_window(pd.Timestamp("2021-06-04 10:00"), trading) == "intraday"
    # 收盤後（13:30 之後）即使是交易日也是 non_trading
    assert assign_session_window(pd.Timestamp("2021-06-04 14:00"), trading) == "non_trading"


def test_week_is_sunday_anchored_and_weekend_leads_next_week():
    """週六／日的關注度必須嚴格領先次週一至週五的報酬。"""
    saturday = pd.Timestamp("2021-06-05")
    w = week_of(saturday)
    assert w == pd.Timestamp("2021-06-06")         # 該週的星期日
    assert w.weekday() == 6
    r = return_week(w)
    assert r == pd.Timestamp("2021-06-13")
    assert (r - pd.Timedelta(days=6)) > saturday   # 報酬週的週一在關注度之後


def test_incomplete_week_is_detectable():
    monday = pd.Timestamp("2021-06-07")
    days = {(monday + pd.Timedelta(days=i)).date() for i in range(4)}   # 只有四天
    assert week_trading_day_count(days, pd.Timestamp("2021-06-13")) == 4


# ---------------------------------------------------------------- 面板建構

def _weeks(n=12):
    return pd.date_range("2021-01-03", periods=n, freq="W-SUN")


def _article_matches(weeks):
    rows = []
    for i, w in enumerate(weeks):
        for k in range(i % 3):
            rows.append({"ticker": "2330", "week": w, "window": "weekend",
                         "session": "non_trading", "effort": "high_effort",
                         "author_id": f"u{k}"})
    return pd.DataFrame(rows)


def test_build_attention_panel_shape_and_zero_fill():
    weeks = _weeks()
    panel = build_attention_panel(_article_matches(weeks), weeks, ["2330", "1101"],
                                  load_settings())
    assert len(panel) == 2 * len(weeks)
    # 沒有任何配對的 ticker 補零，不是缺值
    assert (panel.loc["1101", "att_all"] == 0).all()
    assert "att_authors_all" in panel and "author_hhi" in panel
    assert "abn_attention_weekend" in panel


def test_author_count_never_exceeds_article_count():
    weeks = _weeks()
    panel = build_attention_panel(_article_matches(weeks), weeks, ["2330"],
                                  load_settings())
    assert (panel["att_authors_all"] <= panel["att_all"]).all()


def test_comment_panel_uses_comment_own_timestamps():
    """留言用自己的時戳切窗口——這是 v2 要檢驗 v1 是否低估週末的關鍵。"""
    weeks = _weeks()
    rows = [{"ticker": "2330", "week": w, "window": "weekend",
             "session": "non_trading", "user_id": f"u{k}"}
            for w in weeks for k in range(5)]
    panel = build_comment_panel(pd.DataFrame(rows), weeks, ["2330"], load_settings())
    assert (panel["att_comment_weekend"] == 5).all()
    assert (panel["att_comment_weekday"] == 0).all()
    assert (panel["att_users_weekend"] == 5).all()
    assert "comment_sparsity_tier" in panel


def test_comment_panel_does_not_overwrite_article_tier():
    """留言層的 tier 另用前綴，不得覆蓋主規格的 sparsity_tier。"""
    weeks = _weeks()
    rows = [{"ticker": "2330", "week": weeks[0], "window": "weekend",
             "session": "non_trading", "user_id": "u1"}]
    panel = build_comment_panel(pd.DataFrame(rows), weeks, ["2330"], load_settings())
    assert "sparsity_tier" not in panel.columns
    assert "comment_sparsity_tier" in panel.columns
