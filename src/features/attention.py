"""異常關注度、稀疏度分層、起始事件，以及 v2 的平行測度（PROJECT.md §2）。

PTT 的零是**真實的零**（該週確實沒人討論），不是 Google Trends 那種「低於回報
門檻」的遺漏值，因此用 log1p 而非 mask 後取 log。

v2 新增 §2.4 的平行測度：留言則數、獨立留言帳號數、獨立發文帳號數、發文者集中度。
**這些不取代主規格**——主規格仍為文章數，否則無法與 v1 對照。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR_WINDOWS = ("weekday", "weekend")
SESSION_WINDOWS = ("intraday", "non_trading")
EFFORTS = ("high_effort", "mid_effort", "low_effort")


def abnormal_attention(counts: pd.Series, lookback: int = 8,
                       min_periods: int = 8, transform: str = "log1p") -> pd.Series:
    """AbnAtt = f(Att_w) − mean(f(Att_{w-8..w-1}))。

    回顧窗**嚴格不含當期**，避免 look-ahead。不足 min_periods 維持缺值。
    """
    if transform == "log1p":
        values = np.log1p(counts.astype(float))
    elif transform == "log_positive":
        # 正值限定指標（robustness）：零視為缺值
        values = np.log(counts.astype(float).where(counts > 0))
    else:
        raise ValueError(f"未知 transform: {transform}")
    baseline = values.shift(1).rolling(lookback, min_periods=min_periods).mean()
    return values - baseline


def sparsity_fields(counts: pd.Series, lookback_weeks: int = 52,
                    abn_lookback: int = 8) -> pd.DataFrame:
    """稀疏度欄位（PROJECT.md §2.3）。全部只用**過去**資訊。"""
    counts = counts.astype(float)
    nonzero = (counts > 0).astype(float)
    return pd.DataFrame({
        "att_nonzero_weeks_52": nonzero.shift(1)
            .rolling(lookback_weeks, min_periods=1).sum(),
        "att_mean_level_52": counts.shift(1)
            .rolling(lookback_weeks, min_periods=1).mean(),
        "is_initiation": (
            (counts > 0)
            & (counts.shift(1).rolling(abn_lookback, min_periods=abn_lookback).sum() == 0)
        ),
        "att_lookback_all_zero": (
            counts.shift(1).rolling(abn_lookback, min_periods=abn_lookback).sum() == 0
        ),
    }, index=counts.index)


def sparsity_tier(nonzero_weeks_52: pd.Series, dense_min: int = 40,
                  silent_max: int = 3) -> pd.Series:
    """dense（≥40）／sparse（4–39）／silent（≤3）。門檻寫死於 settings.yaml。"""
    tier = pd.Series(pd.NA, index=nonzero_weeks_52.index, dtype="object")
    valid = nonzero_weeks_52.notna()
    tier[valid & (nonzero_weeks_52 >= dense_min)] = "dense"
    tier[valid & (nonzero_weeks_52 <= silent_max)] = "silent"
    tier[valid & (nonzero_weeks_52 > silent_max) & (nonzero_weeks_52 < dense_min)] = "sparse"
    return tier


def herfindahl(shares: pd.Series) -> float:
    """HHI = Σ sᵢ²，sᵢ 為各帳號佔該 ticker-week 的份額。

    **n = 1 時 HHI 恆為 1**，那是定義的結果而非集中度的證據。呼叫端必須同時帶
    `n_authors`／`n_articles`，並在解讀時排除 n 過小的格子。
    """
    total = shares.sum()
    if total <= 0:
        return np.nan
    p = shares / total
    return float((p ** 2).sum())


def _counts(sub: pd.DataFrame, idx: pd.MultiIndex) -> pd.Series:
    return (sub.groupby(["ticker", "week"]).size()
            .reindex(idx, fill_value=0).astype(float))


def _nunique(sub: pd.DataFrame, col: str, idx: pd.MultiIndex) -> pd.Series:
    """獨立帳號數。空帳號（系統發文）在呼叫前已排除，不得當成一個名為 "" 的帳號。"""
    if sub.empty:
        return pd.Series(0.0, index=idx)
    return (sub.groupby(["ticker", "week"])[col].nunique()
            .reindex(idx, fill_value=0).astype(float))


def build_attention_panel(matches: pd.DataFrame, weeks: pd.DatetimeIndex,
                          tickers: list[str], settings: dict) -> pd.DataFrame:
    """由 ticker-article 配對建出 ticker×week 的關注度面板。

    `matches` 須含 `ticker, week, window, session, effort, author_id`。
    未出現的 ticker-week 補零——**但僅限於該股已上市的週**，上市前的補零由呼叫端
    剔除（PROJECT.md §1）。
    """
    att = settings["attention"]
    spa = settings["sparsity"]

    idx = pd.MultiIndex.from_product([tickers, weeks], names=["ticker", "week"])
    panel = pd.DataFrame(index=idx)

    # 日曆切法與交易時段切法是**兩個不同欄位**，兩者都必須產出，不可互相取代
    panel["att_all"] = _counts(matches, idx)
    for w in CALENDAR_WINDOWS:
        panel[f"att_{w}"] = _counts(matches[matches["window"] == w], idx)
    for w in SESSION_WINDOWS:
        panel[f"att_{w}"] = _counts(matches[matches["session"] == w], idx)
    for e in EFFORTS:
        is_e = matches["effort"] == e
        panel[f"att_{e}"] = _counts(matches[is_e], idx)
        for w in CALENDAR_WINDOWS:
            panel[f"att_{e}_{w}"] = _counts(matches[is_e & (matches["window"] == w)], idx)

    # --- v2 §2.4：獨立發文帳號數與發文者集中度 -------------------------
    authored = matches[matches["author_id"].notna()]
    panel["att_authors_all"] = _nunique(authored, "author_id", idx)
    for w in CALENDAR_WINDOWS:
        panel[f"att_authors_{w}"] = _nunique(
            authored[authored["window"] == w], "author_id", idx)
    if not authored.empty:
        per_author = authored.groupby(["ticker", "week", "author_id"]).size()
        hhi = per_author.groupby(level=["ticker", "week"]).apply(herfindahl)
        panel["author_hhi"] = hhi.reindex(idx)
    else:
        panel["author_hhi"] = np.nan

    return _finalize(panel, idx, att, spa)


def build_comment_panel(comments: pd.DataFrame, weeks: pd.DatetimeIndex,
                        tickers: list[str], settings: dict) -> pd.DataFrame:
    """留言層的平行測度（PROJECT.md §2.4）。

    `comments` 須含 `ticker, week, window, session, user_id`，且**週與窗口是用
    留言自己的時戳算的**，不是母文章的。這正是 v2 要檢驗的事：v1 把一篇週五
    22:00 文章的全部關注度壓在發文那一刻，其留言其實大量落在週六日。

    缺時戳的留言在呼叫前已剔除（缺值維持缺值），差額記於
    `audit/ptt_comment_coverage.csv`。
    """
    att = settings["attention"]
    spa = settings["sparsity"]
    idx = pd.MultiIndex.from_product([tickers, weeks], names=["ticker", "week"])
    panel = pd.DataFrame(index=idx)

    panel["att_comment_all"] = _counts(comments, idx)
    for w in CALENDAR_WINDOWS:
        panel[f"att_comment_{w}"] = _counts(comments[comments["window"] == w], idx)
    for w in SESSION_WINDOWS:
        panel[f"att_comment_{w}"] = _counts(comments[comments["session"] == w], idx)

    users = comments[comments["user_id"].notna()]
    panel["att_users_all"] = _nunique(users, "user_id", idx)
    for w in CALENDAR_WINDOWS:
        panel[f"att_users_{w}"] = _nunique(users[users["window"] == w], "user_id", idx)
    for w in SESSION_WINDOWS:
        panel[f"att_users_{w}"] = _nunique(users[users["session"] == w], "user_id", idx)

    return _finalize(panel, idx, att, spa, base_col="att_comment_all",
                     tier_prefix="comment_")


def _finalize(panel: pd.DataFrame, idx: pd.MultiIndex, att: dict, spa: dict,
              base_col: str = "att_all", tier_prefix: str = "") -> pd.DataFrame:
    """逐檔計算異常值與稀疏度。每檔獨立，不得跨檔滾動。"""
    count_cols = [c for c in panel.columns if c.startswith("att_")]
    out = []
    for _, grp in panel.groupby(level="ticker", sort=False):
        grp = grp.sort_index(level="week")
        block = grp.copy()
        for col in count_cols:
            label = col[len("att_"):]
            block[f"abn_attention_{label}"] = abnormal_attention(
                grp[col], att["lookback_weeks"], att["min_periods"], att["transform"]
            ).values
            if label in ("all", "weekend", "weekday"):
                block[f"abn_attention_{label}_posonly"] = abnormal_attention(
                    grp[col], att["lookback_weeks"], att["min_periods"], "log_positive"
                ).values
        sp = sparsity_fields(grp[base_col], att["sparsity_lookback_weeks"],
                             att["lookback_weeks"])
        for col in sp.columns:
            block[f"{tier_prefix}{col}"] = sp[col].values
        if tier_prefix == "":
            block["is_initiation_weekend"] = (
                block["is_initiation"].values & (grp["att_weekend"].values > 0))
            block["is_initiation_weekday"] = (
                block["is_initiation"].values & (grp["att_weekday"].values > 0))
        out.append(block)

    panel = pd.concat(out)
    nz = panel[f"{tier_prefix}att_nonzero_weeks_52"]
    panel[f"{tier_prefix}sparsity_tier"] = sparsity_tier(
        nz, spa["dense_min_nonzero_weeks"], spa["silent_max_nonzero_weeks"])
    # 回顧窗全零者另設虛擬變數吸收，不得當成 AbnAtt = 0
    panel[f"{tier_prefix}att_zero_base"] = (
        panel[f"{tier_prefix}att_lookback_all_zero"].fillna(False).astype(int))
    return panel
