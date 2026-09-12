"""P4 後半三個模組的邏輯不變量：H7 事件、§6.6 投資組合、T13 對照。

分類同 `docs/PLAN_V2.md` §P5：本檔只放**與語料無關的邏輯不變量**（判讀規則、
caliper、成本公式、缺測度時的 SKIPPED 契約）。寫死的數字（事件數、係數、多空價差）
不放這裡。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import h7_events as h7
from src.analysis import paper_comparison as pc
from src.analysis import portfolios as pf


# ------------------------------------------------------------ H7 判讀規則

@pytest.mark.parametrize("pre_t,post_t,expect", [
    ([0.5, -1.0, 1.2, -0.3], [0.2] * 8, "H7 不成立"),        # 事前清、事後無
    ([0.5, -1.0, 1.2, -0.3], [0.2] * 7 + [3.0], "支持 H7"),  # 事前清、事後有
    ([0.5, 2.5, 1.2, -0.3], [3.0] * 8, "無法識別"),          # 事前未清 → 壓過事後
])
def test_h7_verdict_rules(pre_t, post_t, expect):
    """判讀規則寫死於程式（PROJECT.md §6.5）：事前未清除時，事後再顯著也不算。"""
    taus = list(range(-4, 0)) + [0] + list(range(1, 9))
    ts = pre_t + [9.9] + post_t          # τ=0 不參與判讀
    m = pd.DataFrame({"tau": taus, "t": ts})
    assert expect in h7._verdict(m)


def test_h7_verdict_on_empty_is_not_a_conclusion():
    assert "無法判讀" in h7._verdict(pd.DataFrame())


# ------------------------------------------------------------ H7 事件窗

def _panel(n_t: int = 6, n_w: int = 30, gap_at: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for t in range(n_t):
        for w in range(n_w):
            if gap_at is not None and w == gap_at:
                continue                 # 挖掉一週 → 該處不得產生事件窗
            rows.append({"ticker": f"T{t}", "week": pd.Timestamp("2020-01-05")
                         + pd.Timedelta(weeks=w), "sparsity_tier": "sparse",
                         "ret": rng.normal(0, 0.03), "market_cap": 1e9 * (t + 1),
                         "turnover": 0.01, "is_initiation": (w == 15)})
    return pd.DataFrame(rows)


def test_windows_require_contiguous_weeks():
    """事件窗必須是連續週，否則 τ 對齊會錯——跨越缺口的位置一律不得產生候選。"""
    full = h7._windows(h7._prepare(_panel(), h7.TIERS), "is_initiation")
    holed = h7._windows(h7._prepare(_panel(gap_at=12), h7.TIERS), "is_initiation")
    assert len(holed) < len(full)
    # 缺口週本身不會出現在任何候選的 τ=0 位置
    missing = pd.Timestamp("2020-01-05") + pd.Timedelta(weeks=12)
    assert missing not in set(holed["week"])


def test_caliper_rejects_controls_beyond_tolerance():
    """caliper 超出容忍度的對照必須被剔除，寧可損失事件數（§6.5）。"""
    base = {"sparsity_tier": "sparse", "log_mktcap": 20.0, "log_turnover": 0.1,
            **{f"ar_{tau}": 0.0 for tau in range(-h7.PRE, h7.POST + 1)}}
    week = pd.Timestamp("2020-06-07")
    cands = pd.DataFrame([
        {**base, "ticker": "A", "week": week, "treated": True,
         "ar_m1": 0.0, "ar_m2": 0.0, "pre_car_34": 0.0},
        # 三個對照的 ar_m1 都遠在 1% 之外
        *[{**base, "ticker": f"C{i}", "week": week, "treated": False,
           "ar_m1": 0.30, "ar_m2": 0.0, "pre_car_34": 0.0} for i in range(3)],
    ])
    assert h7.match(cands, k=3, caliper=h7.RETURN_CALIPER).empty
    assert not h7.match(cands, k=3, caliper=None).empty


def test_missing_measure_is_skipped_not_silently_dropped():
    """舊語料沒有留言測度 → 必須寫成 SKIPPED（§6.1b），不得靜默略過。"""
    _, _, _, verdicts = h7.run(_panel())
    row = verdicts[verdicts["event_col"] == "comment_is_initiation"].iloc[0]
    assert row["status"] == "SKIPPED"
    assert "無此測度" in row["verdict"]


# ------------------------------------------------------------ §6.6 投資組合

def test_round_trip_cost_matches_the_written_formula():
    """手續費買賣各一次（打折）＋ 證交稅一次 ＋ 兩次滑價。"""
    got = pf.round_trip_cost(fee_rate=0.001425, fee_discount=0.6,
                             tax_rate=0.003, slippage_bps=20)
    assert got == pytest.approx(0.001425 * 0.6 * 2 + 0.003 + 0.0020 * 2)


def test_portfolio_return_column_follows_settings():
    """§6.6：週日看到訊號、最早週一開盤成交，因此報酬必須是 open-to-close。"""
    from src.config import load_settings
    assert pf.RET_COL[load_settings()["returns"]["portfolio_definition"]] == "ret_oc_next"
    with pytest.raises(KeyError):
        pf.RET_COL["gap"]                # 未知定義不得有預設回退


def _port_panel(n_t: int = 50, n_w: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for t in range(n_t):
        for w in range(n_w):
            rows.append({"ticker": f"T{t:03d}",
                         "week": pd.Timestamp("2020-01-05") + pd.Timedelta(weeks=w),
                         "sparsity_tier": "sparse",
                         "abn_attention_weekend": rng.normal(),
                         "ret_oc_next": rng.normal(0, 0.03),
                         "market_cap": 1e9 * (t + 1), "value": 1e8})
    return pd.DataFrame(rows)


def test_net_return_is_always_below_gross():
    """成本後為必要欄且必然更低——成本不得為零或負。"""
    port = pf.quantile_portfolios(_port_panel())
    s = pf.summarize(port, {"portfolio": {"fee_rate": 0.001425, "fee_discount": 0.6,
                                          "tax_rate": 0.003, "slippage_bps": 20}},
                     "測試", "abn_attention_weekend")
    assert s["status"] == "OK"
    assert s["mean_ls_net_weekly"] < s["mean_ls_gross_weekly"]
    assert s["universe"] == "ex-post universe"


def test_portfolio_skips_absent_parallel_measures():
    from src.config import load_settings
    summ, _ = pf.run_all(_port_panel(), load_settings())
    skipped = summ[summ["status"] == "SKIPPED"]["signal"].tolist()
    assert "abn_attention_users_weekend" in skipped
    assert "abn_attention_comment_weekend" in skipped


# ------------------------------------------------------------ T13 判讀規則

@pytest.mark.parametrize("p_t,o_t,p_b,o_b,expect", [
    (0.5, 0.5, 0.001, 0.001, "兩者皆不顯著"),
    (3.0, 3.0, 0.01, 0.01, "兩者皆顯著且同號"),
    (3.0, -3.0, 0.01, -0.01, "符號相反"),
    (3.0, 0.5, 0.01, 0.001, "論文顯著、本研究不顯著"),
    (0.5, 3.0, 0.001, 0.01, "本研究顯著、論文不顯著"),
    (None, 3.0, None, 0.01, "無對應估計"),
])
def test_t13_verdict_is_rule_based(p_t, o_t, p_b, o_b, expect):
    """判讀由 |t| > 1.96 與符號算出，不得是看過數字後寫下的字串。"""
    assert expect in pc._verdict(p_t, o_t, p_b, o_b)


def test_t13_reports_both_inference_standards():
    """v1 的週末係數在兩種推論標準之間跨過 5%，**不得只報其中一種**（§0.2）。"""
    assert set(pc.INF_LABEL) == {pc.TWO, pc.ONE}
    from src.analysis.regressions import INFERENCE
    assert set(pc.INF_LABEL) == set(INFERENCE)


def test_measure_validity_skips_measures_the_corpus_lacks():
    rng = np.random.default_rng(11)
    n = 200
    wd = rng.normal(size=n)
    we = rng.normal(size=n)            # 週末窗口與整週的共變遠低於週間
    d = pd.DataFrame({"ticker": ["A"] * n, "sparsity_tier": ["sparse"] * n,
                      "abn_attention_all": wd + 0.3 * we,
                      "abn_attention_weekday": wd,
                      "abn_attention_weekend": we})
    out = pc.measure_validity(d)
    assert set(out[out["status"] == "SKIPPED"]["measure"]) == {"留言則數", "獨立帳號數"}
    ok = out[out["status"] == "OK"]
    assert (ok["corr_all_weekday"] > ok["corr_all_weekend"]).all()


def test_design_differences_reads_period_from_the_panel():
    """A′ 的期間是 `specs` 覆寫值；讀 settings 預設會把全期規格寫成 2020–2024。"""
    d = pd.DataFrame({"ticker": ["A", "B"],
                      "week": pd.to_datetime(["2015-05-03", "2024-12-29"]),
                      "abn_attention_weekend": [0.0, 1.0]})
    row = pc.design_differences(d)
    period = row[row["dimension"] == "期間"].iloc[0]["ours"]
    assert "2015-05" in period and "2024-12" in period


# ------------------------------------------------ T14 係數層分解／T15 彙總

def test_t14_steps_form_a_connected_chain():
    """A′ → B → B∩C → C 必須首尾相接，否則三段差值加起來不等於總差異。"""
    from src.analysis import spec_comparison as sc

    labels = [lab for _, lab, _ in sc.SPECS]
    assert [a for a, _, _ in sc.STEPS] == labels[:-1]
    assert [b for _, b, _ in sc.STEPS] == labels[1:]


def test_t14_splits_b_to_c_into_two_named_steps():
    """§6.7：B→C 不是單一效果，必須拆成已刪文流失與測度改變兩欄。"""
    from src.analysis import spec_comparison as sc

    whys = {b: why for _, b, why in sc.STEPS}
    assert "已刪文流失" in whys["B∩C"]
    assert "測度改變" in whys["C"]


def test_t15_marks_absent_parallel_measure_as_skipped(tmp_path, monkeypatch):
    """舊語料沒有帳號測度時，兩個判讀的比較必須是 SKIPPED，不得是「不一致」。"""
    from src.analysis import t15_parallel as t15

    monkeypatch.setattr(t15, "OUT", tmp_path)
    pd.DataFrame([
        {"measure": "主測度 發文數", "moderator": "amihud",
         "inference": t15.TWO, "interaction_t_next_week": -3.6,
         "verdict": "交互項顯著為負 ＋ 後續週無反轉 → 支持資訊處理"},
    ]).to_csv(tmp_path / "T6_joint_verdicts_B.csv", index=False)

    _, verdicts = t15.build("B")
    row = verdicts[verdicts["moderator"] == "amihud"].iloc[0]
    assert row["status"] == "SKIPPED"
    assert row["verdict_agrees"] == ""
