"""主迴歸與機制檢定（PROJECT.md §6.1–§6.2）。

通則：

- 連續變數**在稀疏度分層內標準化**，避免長尾零值把大型股的變異壓扁。
- 主規格為**個股 ＋ 週雙向固定效果、個股與週雙重 cluster**（Petersen 2009）。
- **同時報原論文的推論標準**（僅個股 FE、僅個股 cluster）。v1 的週末係數在兩者
  之間跨過 5% 門檻（t = 1.82 vs 2.36），這件事必須在表上看得見（§0.2、§6.1）。
- **zero-base 虛擬變數逐窗口**：`AbnAtt = 0` 有兩個經濟意義相反的來源，用 `att_all`
  算一個總表旗標只蓋到週末窗口 58.2% 的零基底，逐窗口版蓋到 100%（§2.3）。
- 樣本量或識別條件不足者寫入 `model_status` 並標 `SKIPPED`，**不得勉強輸出係數**。
- 所有結果均為 `diagnostic`：`formal_main_return` 為 False，根因是
  `news_count`、分析師覆蓋、四因子未取得（§7），與 PTT 語料無關。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

MIN_OBS = 500
MIN_ENTITIES = 20
MIN_PERIODS = 20

#: 窗口 → 對應的 zero-base 虛擬變數。**必須與該式的自變數同窗口**。
ZERO_BASE = {
    "abn_attention_all": "att_zero_base_all",
    "abn_attention_weekday": "att_zero_base_weekday",
    "abn_attention_weekend": "att_zero_base_weekend",
    "abn_attention_intraday": "att_zero_base_intraday",
    "abn_attention_non_trading": "att_zero_base_non_trading",
    "abn_attention_users_weekday": "att_zero_base_users_weekday",
    "abn_attention_users_weekend": "att_zero_base_users_weekend",
    "abn_attention_users_intraday": "att_zero_base_users_intraday",
    "abn_attention_users_non_trading": "att_zero_base_users_non_trading",
    "abn_attention_comment_weekday": "att_zero_base_comment_weekday",
    "abn_attention_comment_weekend": "att_zero_base_comment_weekend",
}

#: `listing_age_years` 不可進雙向固定效果規格——它是（週 − 上市日）的線性組合，
#: 會被個股與週固定效果完全吸收。只在無固定效果的橫斷面規格中有識別力。
BASE_CONTROLS = ["ret_lag1", "ret_lag4", "ret_lag25", "log_market_cap",
                 "turnover", "amihud", "foreign_holding_pct"]

INFERENCE = {
    # 名稱: (entity_effects, time_effects, cluster_entity, cluster_time)
    "twoway_2cluster": (True, True, True, True),     # 主規格
    "firm_fe_1cluster": (True, False, True, False),  # 原論文的標準（§0.2）
}


@dataclass
class ModelResult:
    name: str
    status: str
    inference: str = "twoway_2cluster"
    n_obs: int = 0
    n_entities: int = 0
    n_periods: int = 0
    params: dict = field(default_factory=dict)
    tstats: dict = field(default_factory=dict)
    pvalues: dict = field(default_factory=dict)
    stderr: dict = field(default_factory=dict)
    rsquared: float = np.nan
    note: str = ""
    tier_composition: dict = field(default_factory=dict)
    is_diagnostic: bool = True


def standardize_within(df: pd.DataFrame, cols: list[str],
                       group: str = "sparsity_tier") -> pd.DataFrame:
    """在稀疏度分層內標準化（PROJECT.md §6）。

    **虛擬變數不標準化**：`att_zero_base_*` 是 0/1 旗標，標準化後係數不再是
    「零基底相對於非零基底的水準差」，解讀會變成另一件事。
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns or col.startswith("att_zero_base"):
            continue
        g = out.groupby(group, observed=True)[col]
        mu, sd = g.transform("mean"), g.transform("std")
        out[col] = (out[col] - mu) / sd.replace(0, np.nan)
    return out


def available_controls(df: pd.DataFrame, controls: list[str],
                       min_coverage: float = 0.5) -> tuple[list[str], list[str]]:
    """把控制變數分成「可用」與「不可用」。

    缺值政策不可妥協：未取得的控制變數維持缺欄位，**不得補零或補均值**。正式主表
    要求欄位齊備才估計；診斷輸出以可用控制估計，並把被剔除的欄位記入 `note`。
    """
    usable, dropped = [], []
    for col in controls:
        if col in df.columns and df[col].notna().mean() >= min_coverage:
            usable.append(col)
        else:
            dropped.append(col)
    return usable, dropped


def zero_base_for(xs: list[str]) -> list[str]:
    """回傳與這組自變數同窗口的 zero-base 虛擬變數。"""
    return list(dict.fromkeys(ZERO_BASE[x] for x in xs if x in ZERO_BASE))


def fit(df: pd.DataFrame, y: str, xs: list[str], controls: list[str], name: str,
        inference: str = "twoway_2cluster", is_diagnostic: bool = True,
        required_missing: list[str] | None = None) -> ModelResult:
    """單一規格的估計。

    `required_missing` 為 `regression.controls.required_but_missing`；正式模式下
    只要有一項缺就 `SKIPPED`，**不降級輸出**（§7）。
    """
    ent, time, cl_ent, cl_time = INFERENCE[inference]
    if not is_diagnostic and required_missing:
        return ModelResult(name, "SKIPPED", inference,
                           note=f"正式規格要求控制變數齊備，缺：{';'.join(required_missing)}",
                           is_diagnostic=False)

    zb = zero_base_for(xs)
    ctrl, dropped = available_controls(df, [*controls, *zb])
    cols = list(dict.fromkeys(["ticker", "week", "sparsity_tier", y, *xs, *ctrl]))
    d = df[[c for c in cols if c in df.columns]]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[c for c in cols if c not in ("sparsity_tier",)])
    if d.empty:
        return ModelResult(name, "SKIPPED", inference, note="無有效觀測")

    n_obs, n_ent, n_per = len(d), d["ticker"].nunique(), d["week"].nunique()
    if n_obs < MIN_OBS or n_ent < MIN_ENTITIES or n_per < MIN_PERIODS:
        return ModelResult(name, "SKIPPED", inference, n_obs, n_ent, n_per,
                           note=f"樣本不足（門檻 obs≥{MIN_OBS}、個股≥{MIN_ENTITIES}、"
                                f"週≥{MIN_PERIODS}）")

    tiers = (d["sparsity_tier"].value_counts().to_dict()
             if "sparsity_tier" in d.columns else {})
    panel = d.set_index(["ticker", "week"])
    # 自變數與控制變數可能重疊（H6 的調節變數本身就是控制變數之一）。不去重會
    # 選出同名的兩欄，`exog[col]` 變成 DataFrame 而不是 Series，估計直接崩掉。
    exog_cols = list(dict.fromkeys([*xs, *ctrl]))
    exog = panel[exog_cols]
    exog = exog.loc[:, exog.std() > 0]          # 零變異欄位會使 FE 不可識別
    if exog.empty or not set(xs) & set(exog.columns):
        return ModelResult(name, "SKIPPED", inference, n_obs, n_ent, n_per,
                           note="自變數無變異")
    try:
        res = PanelOLS(panel[y], exog, entity_effects=ent, time_effects=time,
                       drop_absorbed=True, check_rank=False).fit(
            cov_type="clustered", cluster_entity=cl_ent, cluster_time=cl_time)
    except Exception as exc:                     # noqa: BLE001
        return ModelResult(name, "FAILED", inference, n_obs, n_ent, n_per,
                           note=f"{type(exc).__name__}: {exc}"[:200])

    note = ("控制變數缺漏（診斷）：" + ";".join(dropped)) if dropped else ""
    return ModelResult(
        name=name, status="OK", inference=inference, n_obs=int(res.nobs),
        n_entities=n_ent, n_periods=n_per,
        params={k: float(v) for k, v in res.params.items()},
        tstats={k: float(v) for k, v in res.tstats.items()},
        pvalues={k: float(v) for k, v in res.pvalues.items()},
        stderr={k: float(v) for k, v in res.std_errors.items()},
        rsquared=float(res.rsquared_within), note=note,
        tier_composition={str(k): int(v) for k, v in tiers.items()},
        is_diagnostic=is_diagnostic)


def fit_both_inferences(df, y, xs, controls, name, **kw) -> list[ModelResult]:
    """同一規格跑兩種推論標準。**不得只報其中一種**（§0.2）。"""
    return [fit(df, y, xs, controls, name, inference=inf, **kw) for inf in INFERENCE]


def results_to_frame(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        base = {"model": r.name, "inference": r.inference, "status": r.status,
                "n_obs": r.n_obs, "n_entities": r.n_entities, "n_periods": r.n_periods,
                "diagnostic": r.is_diagnostic, "note": r.note,
                "tier_composition": ";".join(f"{k}={v}" for k, v in
                                             sorted(r.tier_composition.items()))}
        if r.status != "OK":
            rows.append({**base, "term": "", "coef": np.nan, "se": np.nan,
                         "t": np.nan, "p": np.nan, "r2_within": np.nan})
            continue
        for term in r.params:
            rows.append({**base, "term": term, "coef": r.params[term],
                         "se": r.stderr[term], "t": r.tstats[term],
                         "p": r.pvalues[term], "r2_within": r.rsquared})
    return pd.DataFrame(rows)


def add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["log_market_cap"] = np.log(d["market_cap"].where(d["market_cap"] > 0))
    d["log_att_mean_level_52"] = np.log1p(d["att_mean_level_52"])
    return d


def main_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """主迴歸樣本：`sparsity_tier ∈ {dense, sparse}`。

    `silent` 保留在面板中但自連續型主規格排除——回顧窗全零時 `AbnAtt` 恆為 0，
    連續型設定對它沒有意義，改以事件設計（H7）處理（§2.3）。
    """
    return panel[panel["sparsity_tier"].isin(["dense", "sparse"])]
