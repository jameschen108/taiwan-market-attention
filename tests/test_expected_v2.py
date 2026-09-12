"""比對現值與 `tests/expected_v2.py` 的凍結值（`docs/PLAN_V2.md` §P5）。

本檔與其他測試檔**目的不同**：其他檔驗的是邏輯不變量（換資料也不該變），本檔驗的
是**某一版程式跑出來的數字沒有在無人注意的情況下改掉**。

因此失敗的處理順序是固定的：

1. 先問**這個變動是預期的嗎**。改了資料、門檻、測度 → 是，整批會變。
2. 只做了重構卻讓數字動了 → **那是 bug**，不是該重凍的理由。
3. 確認變動正確之後才 `python3 -m src.audit_expected --freeze`。

反過來做（失敗就重凍）會讓這一整組測試變成裝飾品。

輸出檔不存在時整檔 skip——凍結值不能用來假裝某個表跑過了。
"""

from __future__ import annotations

import pytest

from src.audit_expected import collect

try:
    from tests.expected_v2 import EXPECTED
except ImportError:                                         # noqa: BLE001
    EXPECTED = None

pytestmark = pytest.mark.skipif(
    EXPECTED is None,
    reason="尚未凍結；跑 python3 -m src.audit_expected --freeze")

#: 係數的容忍度。同一份資料 ＋ 同一段程式是決定性的，放寬到這個量級只是容忍
#: BLAS 版本差異；真正的變動都在百分之一以上，這個門檻抓得到。
REL = 1e-4


@pytest.fixture(scope="module")
def current():
    return collect()


def _pairs(section: str):
    """凍結值與現值的逐鍵配對。缺鍵與多鍵都要進來，不得靜默跳過。"""
    frozen = EXPECTED.get(section, {}) if EXPECTED else {}
    return sorted(frozen)


# ------------------------------------------------------------------ 面板形狀

@pytest.mark.parametrize("spec", _pairs("panel"))
def test_panel_shape_is_unchanged(current, spec):
    """面板列數、檔數、週數、分層組成。這些一變，底下所有係數都不可比。"""
    assert current["panel"].get(spec) == EXPECTED["panel"][spec]


# -------------------------------------------------------------------- 係數

@pytest.mark.parametrize("key", _pairs("coef"))
def test_coefficient_is_unchanged(current, key):
    want = EXPECTED["coef"][key]
    got = current["coef"].get(key, "（已消失）")
    if want is None:
        # 舊語料沒有留言／帳號測度 → 必須**維持** None，不得變成有值
        assert got is None, f"{key} 原為不可得，現在卻有值：{got}"
        return
    assert isinstance(got, dict), f"{key} 消失或未估出：{got}"
    assert got["n_obs"] == want["n_obs"]
    assert got["coef"] == pytest.approx(want["coef"], rel=REL)
    assert got["t"] == pytest.approx(want["t"], rel=REL)


# ---------------------------------------------------------------- 投資組合

@pytest.mark.parametrize("key", _pairs("portfolio"))
def test_portfolio_is_unchanged(current, key):
    want, got = EXPECTED["portfolio"][key], current["portfolio"].get(key)
    assert got is not None, f"{key} 消失"
    assert got["n_weeks"] == want["n_weeks"]
    for col in ("gross", "t_gross", "net", "t_net"):
        assert got[col] == pytest.approx(want[col], rel=REL), col


def test_every_portfolio_is_net_negative(current):
    """成本後為負是 v2 的實得結果，不是門檻——但它若翻正，是必須被看見的變動。"""
    for key, v in current["portfolio"].items():
        assert v["net"] < 0, f"{key} 的成本後報酬變成正的：{v['net']}"


# -------------------------------------------------------------- H7 事件與判讀

@pytest.mark.parametrize("key", _pairs("events"))
def test_event_counts_and_verdict_are_unchanged(current, key):
    """**判讀字串也凍結**：事件數沒變而判讀變了，代表 t 值跨過了門檻。"""
    assert current["events"].get(key) == EXPECTED["events"][key]


# ------------------------------------------------------------------ 測度效度

@pytest.mark.parametrize("key", _pairs("validity"))
def test_measure_validity_is_unchanged(current, key):
    want, got = EXPECTED["validity"][key], current["validity"].get(key)
    assert got is not None, f"{key} 消失"
    assert got["n_obs"] == want["n_obs"]
    for col in ("corr_all_weekday", "corr_all_weekend"):
        assert got[col] == pytest.approx(want[col], rel=REL), col


def test_window_correlation_pattern_matches_the_paper(current):
    """論文 Table 2：corr(整週, 週間) ≫ corr(整週, 週末)。

    這一條是**型態**不是數值——它若反轉，代表窗口指派壞了，
    與凍結值差多少無關。
    """
    for key, v in current["validity"].items():
        assert v["corr_all_weekday"] > v["corr_all_weekend"], key


# ---------------------------------------------------------- T14 三段分解

@pytest.mark.parametrize("key", _pairs("decomposition"))
def test_decomposition_step_is_unchanged(current, key):
    want, got = EXPECTED["decomposition"][key], current["decomposition"].get(key)
    if want is None:
        assert got is None
        return
    assert got is not None, f"{key} 消失"
    assert got == pytest.approx(want, rel=1e-3, abs=1e-9)


def test_deletion_and_measure_steps_point_in_opposite_directions(current):
    """§6.7 不准把 B→C 併成一項的實質理由：兩個成分方向相反。

    這是**現在的實得結果**，不是設計上的必然——它若同號了，T14 的敘述必須改寫，
    所以要在測試裡看得見。
    """
    item = "H1 週末｜ret_oc_next|twoway_2cluster"
    dele = current["decomposition"][f"{item}|Δ B→B∩C"]
    meas = current["decomposition"][f"{item}|Δ B∩C→C"]
    assert dele > 0 > meas, f"刪文 {dele}、測度 {meas} 不再方向相反"
