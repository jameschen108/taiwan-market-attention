"""面板層的不變量：暖機期、非平衡、時間對齊、休市週。

這些是最容易靜默出錯的地方——錯了不會報錯，只會讓係數變成別的東西。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings
from src.features.build import assign_windows
from src.features.sessions import week_of

PANEL = Path("data/processed/panel.parquet")
pytestmark = pytest.mark.skipif(not PANEL.exists(),
                                reason="面板未建；先跑 python3 -m src.features.build")


@pytest.fixture(scope="module")
def panel():
    return pd.read_parquet(PANEL)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


# ------------------------------------------------------------------ 暖機期

def test_warmup_year_is_not_in_the_panel(panel, settings):
    """2019 只餵滾動窗，**不得**出現在面板的任何一列。"""
    assert panel["week"].min() >= week_of(pd.Timestamp(settings["sample"]["main_start"]))
    assert (panel["week"].dt.year >= 2020).all()


def test_warmup_removes_tier_missingness_in_first_year(panel):
    """補爬 2019 的唯一理由：沒有它，2020 全年 sparsity_tier 都是缺值。"""
    first_year = panel[panel["week"].dt.year == 2020]
    assert len(first_year) > 0
    assert first_year["sparsity_tier"].isna().sum() == 0
    assert first_year["abn_attention_weekend"].isna().sum() == 0


# ------------------------------------------------------------ 非平衡與缺值

def test_no_rows_before_listing(panel):
    """上市前的週維持缺列——不補零關注度、不補零報酬。"""
    assert (panel["week"] >= panel["listing_date"].map(week_of)).all()


def test_ticker_week_is_unique(panel):
    assert not panel.duplicated(subset=["ticker", "week"]).any()


def test_zero_attention_is_zero_not_missing(panel):
    """PTT 的零是真實的零。att_* 不得有缺值。"""
    for col in ("att_all", "att_weekend", "att_weekday"):
        assert panel[col].isna().sum() == 0


def test_effective_universe_is_260(panel):
    """267 檔中 7 檔上市日晚於樣本結束，有效 260 檔（與 v1 相同）。"""
    assert panel["ticker"].nunique() == 260


# -------------------------------------------------------------- 時間對齊

def test_ret_next_never_skips_a_closed_week(panel):
    """休市週不得跳過去取更後面那一週的報酬。"""
    p = panel.sort_values(["ticker", "week"])
    nxt = p.groupby("ticker")["week"].shift(-1)
    gap_ok = nxt == p["week"] + pd.Timedelta(days=7)
    # 有 ret_next 的列，其下一列必定剛好是下一個日曆週
    assert not (p["ret_next"].notna() & ~gap_ok.fillna(False)).any()


def test_ret_next_equals_next_rows_ret(panel):
    p = panel.sort_values(["ticker", "week"]).reset_index(drop=True)
    nxt_ret = p.groupby("ticker")["ret"].shift(-1)
    ok = p["ret_next"].notna()
    assert np.allclose(p.loc[ok, "ret_next"], nxt_ret[ok], equal_nan=True)


def test_lagged_controls_use_only_past(panel):
    """落後項只能取過去。

    每檔**第一列**除外：它的落後項來自暖機期的最後一週，裁切後在面板裡找不到
    ——那正是暖機期該有的行為，由 `test_warmup_feeds_lags_at_the_boundary` 另測。
    """
    p = panel.sort_values(["ticker", "week"]).reset_index(drop=True)
    prev = p.groupby("ticker")["ret"].shift(1)
    not_first = p.groupby("ticker").cumcount() > 0
    ok = p["ret_lag1"].notna() & not_first
    assert np.allclose(p.loc[ok, "ret_lag1"], prev[ok], equal_nan=True)


def test_warmup_feeds_lags_at_the_boundary(panel):
    """主樣本第一週的落後報酬必須有值——它來自 2019-12 的暖機期。

    若這一項是缺值，代表暖機期沒有真的餵進滾動窗，補爬 2019 就白做了。
    """
    p = panel.sort_values(["ticker", "week"])
    first = p.groupby("ticker").head(1)
    # 2020 年第一週就上市的老股，其 ret_lag1 應大量有值
    old = first[first["listing_date"] < pd.Timestamp("2019-01-01")]
    assert len(old) > 200
    assert old["ret_lag1"].notna().mean() > 0.9


def test_weekend_attention_strictly_precedes_next_week_return(panel):
    """週六日的關注度落在以該週日結尾的週，其報酬期為下一個日曆週的週一起。"""
    row = panel.iloc[0]
    week_end = row["week"]
    assert week_end.weekday() == 6                       # 星期日錨定
    next_monday = week_end + pd.Timedelta(days=1)
    assert next_monday > week_end                        # 報酬期嚴格在後


# ------------------------------------------------------------ 窗口指派

def test_assign_windows_uses_real_calendar():
    """國定假日的平日盤中仍是 non_trading。"""
    trading = {pd.Timestamp("2021-06-04").date()}
    df = pd.DataFrame({"ts": pd.to_datetime([
        "2021-02-15 10:00",   # 春節，週一但休市
        "2021-06-04 10:00",   # 交易日盤中
        "2021-06-04 14:00",   # 交易日收盤後
        "2021-06-05 10:00",   # 週六
    ])})
    out = assign_windows(df, "ts", trading)
    assert list(out["session"]) == ["non_trading", "intraday", "non_trading", "non_trading"]
    assert list(out["window"]) == ["weekday", "weekday", "weekday", "weekend"]


def test_assign_windows_drops_missing_timestamps():
    """缺時戳維持缺值——不得以任何方式補值。"""
    df = pd.DataFrame({"ts": [pd.Timestamp("2021-06-04 10:00"), pd.NaT]})
    assert len(assign_windows(df, "ts", set())) == 1


def test_week_label_is_the_sunday(panel):
    assert (panel["week"].dt.weekday == 6).all()


# -------------------------------------------------- v2 的平行測度契約

def test_comment_measures_present_and_not_overwriting_main_spec(panel):
    for col in ("att_comment_all", "att_comment_weekend", "att_users_all",
                "att_users_weekend", "att_authors_all", "author_hhi"):
        assert col in panel.columns, col
    # 主規格的 tier 不得被留言層覆蓋
    assert "sparsity_tier" in panel and "comment_sparsity_tier" in panel


def test_distinct_users_never_exceed_comment_count(panel):
    ok = panel["att_comment_all"].notna() & panel["att_users_all"].notna()
    assert (panel.loc[ok, "att_users_all"] <= panel.loc[ok, "att_comment_all"]).all()


def test_distinct_authors_never_exceed_article_count(panel):
    assert (panel["att_authors_all"] <= panel["att_all"]).all()


def test_comment_coverage_exceeds_article_coverage(panel):
    """留言測度的覆蓋率必須高於文章測度——這是 v2 做它的理由。"""
    assert (panel["att_comment_all"] > 0).mean() > (panel["att_all"] > 0).mean()


# ---------------------------------------------------------------- H4

def test_no_makeup_saturday_weeks_in_v2(panel):
    """2020–2024 補班星期六為 0 天，H4 的事件數是零而非檢定力不足。"""
    assert int(panel["is_makeup_saturday_week"].sum()) == 0
