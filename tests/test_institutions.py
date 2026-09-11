"""台股制度性混淆欄位的不變量（LIMITATIONS.md §12）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.institutions import (attach, disposition_weeks,
                                       weekly_day_trading)
from src.features.sessions import week_of

PANEL = Path("data/processed/panel.parquet")


@pytest.fixture
def trading_days():
    # 2021-06-07(一) ~ 2021-06-11(五)，以及次週一 2021-06-14
    return {pd.Timestamp(f"2021-06-{d:02d}").date() for d in (7, 8, 9, 10, 11, 14)}


def _disp_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "disposition.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_disposition_expands_over_trading_days_only(tmp_path, trading_days):
    """處置的影響只在交易日上發生——不得用日曆日展開。"""
    path = _disp_csv(tmp_path, [{"ticker": "2609", "start_date": "2021-06-07",
                                 "end_date": "2021-06-13"}])
    out = disposition_weeks(path, trading_days)
    # 6/07~6/13 共 7 個日曆日，但只有 5 個交易日，且全落在 6/13 那一週
    assert len(out) == 1
    assert out.loc[0, "n_disposition_days"] == 5
    assert out.loc[0, "week"] == week_of(pd.Timestamp("2021-06-11"))


def test_overlapping_disposition_periods_are_deduped(tmp_path, trading_days):
    """同一檔的處置區間會重疊；(ticker, day) 未去重會數出不可能的天數。"""
    path = _disp_csv(tmp_path, [
        {"ticker": "2609", "start_date": "2021-06-07", "end_date": "2021-06-11"},
        {"ticker": "2609", "start_date": "2021-06-09", "end_date": "2021-06-11"},
    ])
    out = disposition_weeks(path, trading_days)
    assert out["n_disposition_days"].max() <= 5


def test_missing_disposition_file_is_pending_not_silent(tmp_path):
    """資料缺席時維持缺值並標記 PENDING，不得靜默略過。"""
    panel = pd.DataFrame({"ticker": ["2330"], "week": [pd.Timestamp("2021-06-13")],
                          "volume": [1000.0]})
    out, status = attach(panel, tmp_path / "nope.csv", tmp_path / "nope2.csv", set())
    assert status["disposition_stocks"].startswith("PENDING")
    assert status["day_trading"].startswith("PENDING")
    assert out["is_disposition_week"].isna().all()
    assert out["dt_ratio"].isna().all()


def test_day_trading_ratio_denominator_comes_from_market_volume(tmp_path):
    """dt_ratio 的分母必須是行情面板的週成交股數。

    TWTB4U 只列**有當沖**的個股，沒當沖的日子該檔根本不出現；用當沖資料自己的
    總量當分母會讓比率恆等於 1。
    """
    dt = tmp_path / "day_trading.csv"
    pd.DataFrame([{"ticker": "2330", "date": "2021-06-08", "dt_volume": 300.0,
                   "dt_buy_value": 1.0, "dt_sell_value": 1.0, "dt_restricted": False}]
                 ).to_csv(dt, index=False)
    panel = pd.DataFrame({"ticker": ["2330"], "week": [week_of(pd.Timestamp("2021-06-08"))],
                          "volume": [1000.0]})
    out, status = attach(panel, tmp_path / "nope.csv", dt, set())
    assert status["day_trading"] == "AVAILABLE"
    assert out.loc[0, "dt_ratio"] == pytest.approx(0.3)


def test_day_trading_ratio_missing_when_volume_is_zero(tmp_path):
    dt = tmp_path / "day_trading.csv"
    pd.DataFrame([{"ticker": "2330", "date": "2021-06-08", "dt_volume": 0.0,
                   "dt_buy_value": 0.0, "dt_sell_value": 0.0, "dt_restricted": False}]
                 ).to_csv(dt, index=False)
    panel = pd.DataFrame({"ticker": ["2330"], "week": [week_of(pd.Timestamp("2021-06-08"))],
                          "volume": [0.0]})
    out, _ = attach(panel, tmp_path / "nope.csv", dt, set())
    assert np.isnan(out.loc[0, "dt_ratio"])


# ------------------------------------------------------------ 面板層

@pytest.mark.skipif(not PANEL.exists(), reason="面板未建")
def test_panel_disposition_fields():
    p = pd.read_parquet(PANEL)
    assert "is_disposition_week" in p and "is_disposition_week_next" in p
    # 一週最多 5 個交易日
    assert p["n_disposition_days"].max() <= 5


@pytest.mark.skipif(not PANEL.exists(), reason="面板未建")
def test_disposition_next_week_never_skips_a_closed_week():
    """污染 turnover_next 的是**次週**的處置；取值不得跳過休市週。

    每檔最後一列除外——它的次週是 2025-01-05，裁切後不在面板裡，與 `ret_next`
    的邊界行為一致（`test_panel.py::test_ret_next_never_skips_a_closed_week`）。
    """
    p = pd.read_parquet(PANEL).sort_values(["ticker", "week"]).reset_index(drop=True)
    nxt = p.groupby("ticker")["week"].shift(-1)
    ok = nxt == p["week"] + pd.Timedelta(days=7)
    has_successor = p.groupby("ticker").cumcount(ascending=False) > 0
    assert not (p["is_disposition_week_next"].notna() & has_successor
                & ~ok.fillna(False)).any()


@pytest.mark.skipif(not PANEL.exists(), reason="面板未建")
def test_disposition_next_week_reaches_past_the_boundary():
    """與 ret_next 對稱：末週的次週處置旗標取自被裁掉的 2025-01-05。"""
    p = pd.read_parquet(PANEL).sort_values(["ticker", "week"])
    last = p.groupby("ticker").tail(1)
    assert last["is_disposition_week_next"].notna().all()


@pytest.mark.skipif(not PANEL.exists(), reason="面板未建")
def test_disposition_weeks_carry_far_higher_attention():
    """§12.1 的機制檢查：處置觸發於量價異常＝高關注度。

    若這一項不成立，`LIMITATIONS.md` §12.1 的論述就失去實證基礎。
    """
    p = pd.read_parquet(PANEL)
    hi = p.loc[p["is_disposition_week"] == True, "att_all"].mean()
    lo = p.loc[p["is_disposition_week"] == False, "att_all"].mean()
    assert hi > 3 * lo
