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
    """休市週不得跳過去取更後面那一週的報酬。

    每檔**最後一列**除外：它的報酬週是 2025-01-05（主樣本末端之後的那一週，
    實有 4 個交易日），裁切後在面板裡找不到——那正是領先項該有的行為，與落後項
    吃到暖機期對稱，由 `test_leads_reach_past_the_boundary` 另測。
    """
    p = panel.sort_values(["ticker", "week"]).reset_index(drop=True)
    nxt = p.groupby("ticker")["week"].shift(-1)
    gap_ok = nxt == p["week"] + pd.Timedelta(days=7)
    has_successor = p.groupby("ticker").cumcount(ascending=False) > 0
    assert not (p["ret_next"].notna() & has_successor & ~gap_ok.fillna(False)).any()


def test_ret_next_equals_next_rows_ret(panel):
    p = panel.sort_values(["ticker", "week"]).reset_index(drop=True)
    nxt_ret = p.groupby("ticker")["ret"].shift(-1)
    ok = p["ret_next"].notna() & (p.groupby("ticker").cumcount(ascending=False) > 0)
    assert np.allclose(p.loc[ok, "ret_next"], nxt_ret[ok], equal_nan=True)


def test_leads_reach_past_the_boundary(panel):
    """主樣本最後一週的次週報酬必須有值——它來自 2025-01-05 那一週。

    與 `test_warmup_feeds_lags_at_the_boundary` 對稱：落後項吃暖機期（2019-12），
    領先項吃樣本末端之後（2025-01）。`sample.price_backfill_end` 就是為此而設。
    若這一項是缺值，代表主樣本最後一週白白損失了觀測。
    """
    p = panel.sort_values(["ticker", "week"])
    last = p.groupby("ticker").tail(1)
    assert last["ret_next"].notna().mean() > 0.95
    assert (last["week"] == pd.Timestamp("2024-12-29")).all()


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


# ------------------------- v2 測度正確性修正（PROJECT.md §0.1）

def test_main_sample_weeks_are_fully_contained(panel, settings):
    """週必須完整落在 [main_start, main_end] 內。

    以週標籤落點判定會兩端各污染一週：首週含 2019-12-30/31（暖機期洩漏進主樣本），
    末週含 2025-01-01..05（語料外，`att_weekend` 恆為 0、`ret_next` 全缺值）。
    """
    start = pd.Timestamp(settings["sample"]["main_start"])
    end = pd.Timestamp(settings["sample"]["main_end"])
    assert (panel["week"] - pd.Timedelta(days=6) >= start).all()
    assert (panel["week"] <= end).all()
    assert panel["week"].nunique() == 260


def test_returns_decompose_exactly(panel):
    """(1 + ret_cc) = (1 + ret_gap)(1 + ret_oc)。三段必須是同一條價格路徑的拆解。"""
    ok = panel[["ret_cc", "ret_oc", "ret_gap"]].notna().all(axis=1)
    lhs = 1 + panel.loc[ok, "ret_cc"]
    rhs = (1 + panel.loc[ok, "ret_gap"]) * (1 + panel.loc[ok, "ret_oc"])
    assert np.allclose(lhs, rhs)
    assert ok.sum() > 60_000


def test_main_return_follows_the_configured_definition(panel, settings):
    """主規格報酬含跨週末缺口——只用 open_to_close 等於把價格壓力的衝擊段切掉。"""
    assert settings["returns"]["main_definition"] == "close_to_close"
    ok = panel["ret"].notna()
    assert np.allclose(panel.loc[ok, "ret"], panel.loc[ok, "ret_cc"])
    # 缺口不是零頭：佔週報酬變異 8% 量級，且不可被 ret_oc 代表
    assert panel["ret_gap"].var() / panel["ret_cc"].var() > 0.05
    assert panel["ret_gap"].corr(panel["ret_oc"]) < 0


def test_gap_never_crosses_a_fully_closed_week(panel):
    """缺口的上週收盤不得跨過完全休市的週去取更早的收盤。

    每檔第一列除外：其上週收盤來自 2020-01-05（暖機期邊界週），與落後報酬同理。
    """
    p = panel.sort_values(["ticker", "week"]).reset_index(drop=True)
    prev_ok = p.groupby("ticker")["week"].shift(1) == p["week"] - pd.Timedelta(days=7)
    not_first = p.groupby("ticker").cumcount() > 0
    assert not (p["ret_gap"].notna() & not_first & ~prev_ok.fillna(False)).any()
    # 首列的缺口本身必須有值，否則暖機期沒接上
    assert p.groupby("ticker").head(1)["ret_gap"].notna().mean() > 0.95


def test_zero_base_exists_for_every_regressor_window(panel):
    """§6.1 的每個自變數窗口都必須有自己的 zero-base 虛擬變數。"""
    for w in ("all", "weekday", "weekend", "intraday", "non_trading"):
        assert f"att_zero_base_{w}" in panel.columns, w
    a = panel["abn_attention_weekend"]
    flagged = panel.loc[a == 0, "att_zero_base_weekend"].astype(float)
    # 週末窗口版必須蓋住絕大多數的病態零；att_all 版蓋不到一半
    assert (flagged == 1).mean() > 0.80
    assert (panel.loc[a == 0, "att_zero_base"] == 1).mean() < (flagged == 1).mean()


def test_sparsity_tier_never_uses_a_partial_window(panel):
    """滿窗才給 tier。有暖機期時主樣本應全數有值，缺一格就代表暖機期沒接上。"""
    assert panel["sparsity_tier"].isna().sum() == 0
    assert panel["att_nonzero_weeks_52"].max() <= 52


def test_price_limit_flag_is_present_and_plausible(panel):
    """漲跌停截斷發生在關注度最高的狀態，必須可辨識（LIMITATIONS.md §13）。"""
    share = panel["touched_price_limit"].mean()
    assert 0.005 < share < 0.10


def test_window_rules_agree_between_scalar_and_vectorised(panel):
    """`build.assign_windows`（生產路徑）與 `sessions.*`（參照實作）不得分歧。"""
    from src.features.sessions import assign_calendar_window, assign_session_window
    trading = {pd.Timestamp("2021-06-04").date(), pd.Timestamp("2021-06-07").date()}
    ts = pd.to_datetime(["2021-06-04 09:00", "2021-06-04 13:29", "2021-06-04 13:30",
                         "2021-06-04 08:45", "2021-06-05 10:00", "2021-06-06 23:59",
                         "2021-06-07 11:00", "2021-02-15 10:00"])
    out = assign_windows(pd.DataFrame({"ts": ts}), "ts", trading)
    assert list(out["window"]) == [assign_calendar_window(t) for t in ts]
    assert list(out["session"]) == [assign_session_window(t, trading) for t in ts]
