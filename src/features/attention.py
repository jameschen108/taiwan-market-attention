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
                       min_periods: int = 8, transform: str = "log1p",
                       baseline: str = "median") -> pd.Series:
    """AbnAtt = f(Att_w) − g(f(Att_{w-8..w-1}))，g 為 median（主規格）或 mean。

    回顧窗**嚴格不含當期**，避免 look-ahead。不足 min_periods 維持缺值。

    **basis 為什麼是 median**：原論文的 ASVI 用回顧窗的中位數，理由是不讓窗內的
    單週爆量把基準拉高。本樣本恰好最吃這一點——`att_weekend` 均值 0.13、95% 為零，
    回顧窗典型長相是 {0,0,0,0,0,0,0,1}，mean 給 0.087、median 給 0。
    mean 版保留為穩健性（`abn_attention_*_meanbase`），兩者 corr ≈ 0.94。
    """
    if transform == "log1p":
        values = np.log1p(counts.astype(float))
    elif transform == "log_positive":
        # 正值限定指標（robustness）：零視為缺值
        values = np.log(counts.astype(float).where(counts > 0))
    else:
        raise ValueError(f"未知 transform: {transform}")
    roll = values.shift(1).rolling(lookback, min_periods=min_periods)
    if baseline == "median":
        base = roll.median()
    elif baseline == "mean":
        base = roll.mean()
    else:
        raise ValueError(f"未知 baseline: {baseline}")
    return values - base


def sparsity_fields(counts: pd.Series, lookback_weeks: int = 52,
                    abn_lookback: int = 8, min_periods: int | None = None) -> pd.DataFrame:
    """稀疏度欄位（PROJECT.md §2.3）。全部只用**過去**資訊。

    52 週窗**必須滿窗**（`min_periods = lookback_weeks`）。設 1 的話沒有暖機期也
    不會報錯，只會用殘缺窗算出系統性偏向 `silent` 的錯值——那違反專案的「缺值維持
    缺值」規則，而且會讓「補爬 2019」這個決策失去可驗證的理由。
    """
    counts = counts.astype(float)
    nonzero = (counts > 0).astype(float)
    mp = lookback_weeks if min_periods is None else min_periods
    return pd.DataFrame({
        "att_nonzero_weeks_52": nonzero.shift(1)
            .rolling(lookback_weeks, min_periods=mp).sum(),
        "att_mean_level_52": counts.shift(1)
            .rolling(lookback_weeks, min_periods=mp).mean(),
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
    """逐檔計算異常值與稀疏度。每檔獨立，不得跨檔滾動。

    **逐窗口的 zero-base（PROJECT.md §2.3）**：`AbnAtt = 0` 有兩個意義完全不同的
    來源——「回顧窗全零、當期也零」的長尾股，與「關注度剛好等於常態水準」的台積電。
    分辨它們的虛擬變數必須**與自變數同窗口**。只用 `att_all` 算一個總表旗標是不夠的：
    實測 `abn_attention_weekend` 恰為 0 的列佔 78.2%，其中 75.8 pp 屬前者，而
    `att_all` 版的旗標只蓋到其中 57.3%。
    """
    count_cols = [c for c in panel.columns if c.startswith("att_")]
    lookback, mp = att["lookback_weeks"], att["min_periods"]
    transform, baseline = att["transform"], att.get("baseline", "median")
    sp_mp = att.get("sparsity_min_periods", att["sparsity_lookback_weeks"])
    # 主要自變數所在的窗口都要有自己的 zero-base；base_col 一律包含
    zero_base_cols = [base_col] + [
        c for c in count_cols
        if c[len("att_"):].replace("comment_", "").replace("users_", "")
        in CALENDAR_WINDOWS + SESSION_WINDOWS]

    out = []
    for _, grp in panel.groupby(level="ticker", sort=False):
        grp = grp.sort_index(level="week")
        block = grp.copy()
        for col in count_cols:
            label = col[len("att_"):]
            block[f"abn_attention_{label}"] = abnormal_attention(
                grp[col], lookback, mp, transform, baseline).values
            if label in ("all", "weekend", "weekday"):
                block[f"abn_attention_{label}_posonly"] = abnormal_attention(
                    grp[col], lookback, mp, "log_positive", baseline).values
                # 基準統計量的穩健性：主規格 median，此欄為 mean（PROJECT.md §2）
                other = "mean" if baseline == "median" else "median"
                block[f"abn_attention_{label}_{other}base"] = abnormal_attention(
                    grp[col], lookback, mp, transform, other).values

        sp = sparsity_fields(grp[base_col], att["sparsity_lookback_weeks"],
                             lookback, min_periods=sp_mp)
        for col in sp.columns:
            block[f"{tier_prefix}{col}"] = sp[col].values

        # 逐窗口的 zero-base：回顧窗在**該窗口**全為零
        for col in dict.fromkeys(zero_base_cols):
            label = col[len("att_"):]
            allzero = (grp[col].astype(float).shift(1)
                       .rolling(lookback, min_periods=lookback).sum() == 0)
            block[f"att_zero_base_{label}"] = allzero.astype("Int64").values

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
    # 沿用名稱：att_zero_base 恆等於 base_col 的窗口版，供既有下游引用
    panel[f"{tier_prefix}att_zero_base"] = (
        panel[f"att_zero_base_{base_col[len('att_'):]}"].fillna(0).astype(int))
    return panel
