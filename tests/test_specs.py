"""四方對照規格的契約（PROJECT.md §6.7）。

A → A′ → B → C 的分解只有在**除了語料來源與期間之外什麼都沒動**的前提下才成立。
規格一旦可以順便調別的門檻，對照就失去意義。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_settings
from src.features.build import resolve_spec


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def test_specs_only_override_source_and_sample(settings):
    """規格只能覆寫 source 與 sample——動了別的等於讓對照可以被調參。"""
    for name, cfg in (settings.get("specs") or {}).items():
        assert set(cfg) <= {"source", "sample"}, f"{name} 多覆寫了 {set(cfg) - {'source', 'sample'}}"


def test_unknown_spec_raises(settings):
    with pytest.raises(ValueError):
        resolve_spec(settings, "不存在的規格")


def test_spec_c_equals_the_default(settings):
    """規格 C 就是主規格，不得與預設有任何差異。"""
    d_src, d_smp, _ = resolve_spec(settings, None)
    c_src, c_smp, _ = resolve_spec(settings, "C")
    assert (c_src, c_smp) == (d_src, d_smp)


def test_a_prime_and_b_share_every_threshold_with_c(settings):
    """A′、B、C 三者的差別只在 source 與 sample 的期間三鍵。"""
    _, c_smp, _ = resolve_spec(settings, "C")
    for name in ("B", "A_prime"):
        _, smp, _ = resolve_spec(settings, name)
        differing = {k for k in c_smp if c_smp[k] != smp.get(k)}
        assert differing <= {"main_start", "main_end", "ptt_warmup_start"}, \
            f"{name} 另外動到 {differing}"


def test_a_prime_uses_old_corpus_over_the_full_period(settings):
    """A′ 的定義：**舊語料 × 全期 × v2 的定義**。

    少了 A′，A→B 會同時混了期間效果與 §0.1 的四項測度修正
    （LIMITATIONS.md §14）。
    """
    src, smp, suffix = resolve_spec(settings, "A_prime")
    assert src == "pttweb"
    assert smp["main_start"] == "2015-05-01"
    assert smp["main_end"] == settings["sample"]["main_end"]
    # 暖機期必須回推到封存起點，否則 A′ 的前 52 週 tier 全缺
    assert smp["ptt_warmup_start"] < smp["main_start"]
    assert suffix == "_A_prime"


def test_b_and_c_share_the_same_window(settings):
    """B 與 C 的差別只能是語料來源——期間必須完全一致，否則 B→C 不是純資料效果。"""
    b_src, b_smp, _ = resolve_spec(settings, "B")
    c_src, c_smp, _ = resolve_spec(settings, "C")
    assert b_src != c_src
    for k in ("main_start", "main_end", "ptt_warmup_start"):
        assert b_smp[k] == c_smp[k]


# ------------------------------------------------------- 應變數定義的契約

def test_main_return_definition_is_open_to_close(settings):
    """主規格的應變數必須是 open-to-close。

    論文原文是「abnormal returns (Monday open to Friday close) of week t+1」
    （Table 3a 註 p.16），v1 亦同。曾有一版改成 close_to_close 並在 PROJECT.md
    留下殘留描述，已回退（LIMITATIONS.md §13）。這條測試防止它再漂回去。
    """
    assert settings["returns"]["main_definition"] == "open_to_close"
    assert settings["returns"]["portfolio_definition"] == "open_to_close"


def test_panel_ret_next_matches_the_declared_definition():
    """面板的 ret_next 必須等於設定檔宣告的那個定義，不是另一個。"""
    import numpy as np
    from pathlib import Path
    import pandas as pd
    path = Path("data/processed/panel_C.parquet")
    if not path.exists():
        pytest.skip("面板未建")
    d = pd.read_parquet(path, columns=["ret_next", "ret_oc_next", "ret_cc_next"]).dropna()
    assert np.allclose(d["ret_next"], d["ret_oc_next"])
    assert not np.allclose(d["ret_next"], d["ret_cc_next"])


# ------------------------------------------------ 迴歸層的欄位契約

def test_fit_handles_moderator_that_is_also_a_control():
    """H6 的調節變數本身就是控制變數之一；自變數與控制重疊時不得產生重複欄位。

    不去重會讓 `exog[col]` 回傳 DataFrame 而非 Series，估計崩在
    `AttributeError: 'DataFrame' object has no attribute 'dtype'`。
    """
    import numpy as np
    import pandas as pd
    from src.analysis.regressions import fit

    rng = np.random.default_rng(0)
    n_t, n_w = 40, 60
    rows = []
    for t in range(n_t):
        for w in range(n_w):
            rows.append({"ticker": f"T{t:03d}", "week": pd.Timestamp("2020-01-05")
                         + pd.Timedelta(weeks=w), "sparsity_tier": "sparse",
                         "y": rng.normal(), "x": rng.normal(), "m": rng.normal()})
    d = pd.DataFrame(rows)
    d["x__x__m"] = d["x"] * d["m"]
    r = fit(d, "y", ["x", "m", "x__x__m"], ["m"], "重疊測試")
    assert r.status == "OK", r.note
    assert "x__x__m" in r.params


# ------------------------------------------ 結構不變量：**每個**規格的面板都要過

# `docs/PLAN_V2.md` §P5：「v1 的測試改為以 `ptt.source` 參數化」。下面這些不是
# 數字，是任何一份面板都必須成立的結構性質——只驗 C 的話，A′ 或 B 壞掉不會有人知道。
_BUILT = [s for s in ("C", "B", "A_prime", "pttweb_intersect")
          if Path(f"data/processed/panel_{s}.parquet").exists()]


@pytest.mark.parametrize("spec", _BUILT)
def test_every_built_panel_holds_the_structural_invariants(spec):
    import pandas as pd

    from src.features.sessions import week_of

    d = pd.read_parquet(f"data/processed/panel_{spec}.parquet")

    # 非平衡且明確非平衡：上市前維持缺列，不補零
    assert (d["week"] >= d["listing_date"].map(week_of)).all()
    # ticker×week 唯一
    assert not d.duplicated(subset=["ticker", "week"]).any()
    # PTT 的零是真實的零——att_* 不得有缺值
    for col in ("att_all", "att_weekday", "att_weekend"):
        assert d[col].isna().sum() == 0, f"{spec} 的 {col} 有缺值"
    # 週軸為 Sunday-anchored 且連續
    weeks = pd.Series(sorted(d["week"].unique()))
    assert (weeks.dt.weekday == 6).all(), f"{spec} 的週軸不是星期日錨定"
    assert (weeks.diff().dropna() == pd.Timedelta(days=7)).all(), f"{spec} 的週軸有缺口"
    # 窗口相加等於整週：切法不得漏掉或重複計數
    assert (d["att_weekday"] + d["att_weekend"] == d["att_all"]).all()
    assert (d["att_intraday"] + d["att_non_trading"] == d["att_all"]).all()


@pytest.mark.parametrize("spec", _BUILT)
def test_every_built_panel_respects_its_own_sample_window(spec):
    """期間由規格決定，但**每一份都必須完整落在自己的窗內**（week_containment=full）。

    A′ 的期間是 `specs` 的覆寫值；拿預設的 sample 去驗會誤判。
    """
    import pandas as pd

    from src.config import load_settings

    settings = load_settings()
    # pttweb_intersect 不是 `specs` 裡的規格，它用預設 sample（＝B 的期間）
    _, smp, _ = resolve_spec(settings, spec if spec in (settings.get("specs") or {})
                             else None)
    d = pd.read_parquet(f"data/processed/panel_{spec}.parquet", columns=["week"])
    lo, hi = pd.Timestamp(smp["main_start"]), pd.Timestamp(smp["main_end"])
    assert (d["week"] - pd.Timedelta(days=6)).min() >= lo
    assert d["week"].max() <= hi
