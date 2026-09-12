"""建立 ticker×week 面板（PROJECT.md §1、§2、§5）。

非平衡且**必須明確非平衡**：上市前的週維持缺列，不得補零關注度、不得補零報酬。
所有領先與落後項依 Sunday-anchored 日曆對齊；完全休市的週保留為一列明確的無報酬觀測。

**v2 的兩項結構差異**：

1. **暖機期**。關注度與稀疏度的滾動窗跨 `ptt_warmup_start` ~ `main_end` 計算，
   面板**最後**才裁到 `main_start` ~ `main_end`。沒有這一步，52 週的
   `sparsity_tier` 在 2020 全年皆為缺值，H6 與 dense/sparse 子樣本會少掉五分之一
   觀測（docs/PLAN_V2.md §3.2）。裁切在最後做，暖機期不會出現在任何表格的觀測數。

2. **留言層平行測度**。`att_comment_*`、`att_users_*` 以留言**自己的時戳**指派窗口
   後併入面板（PROJECT.md §2.4）。主規格的自變數仍是文章數。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, load_settings, load_universe_config
from .attention import build_attention_panel, build_comment_panel
from .imbalance import abnormal_turnover, weekly_non_inst_roi
from .institutions import attach as attach_institutions
from .sessions import week_of

_RET_COL = {"close_to_close": "ret_cc", "open_to_close": "ret_oc"}

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
AUDIT = ROOT / "audit"


def assign_windows(df: pd.DataFrame, ts_col: str, trading_days: set) -> pd.DataFrame:
    """依時戳指派週、日曆窗口、交易時段窗口。

    留言與文章呼叫**同一組函式**，差別只在傳進來的時戳是誰的——這正是 v2 要檢驗
    的事（PROJECT.md §2.4）。
    """
    out = df.dropna(subset=[ts_col]).copy()
    ts = out[ts_col]
    out["week"] = ts.dt.normalize() + pd.to_timedelta((6 - ts.dt.weekday) % 7, unit="D")
    out["window"] = np.where(ts.dt.weekday >= 5, "weekend", "weekday")
    minutes = ts.dt.hour * 60 + ts.dt.minute
    is_trading_day = ts.dt.date.isin(trading_days)
    out["session"] = np.where(is_trading_day & (minutes >= 540) & (minutes < 810),
                              "intraday", "non_trading")
    return out


def _weekly_market(daily: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """日資料 → 週資料：三段式報酬、周轉率、ROI、流動性、漲跌停旗標。

    **報酬拆成三段**（PROJECT.md §5）：

        ret_gap = 本週首個交易日開盤 / 上週最後交易日收盤 − 1   跨週末缺口
        ret_oc  = 本週最後收盤 / 本週首個開盤 − 1                週內
        ret_cc  = 本週最後收盤 / 上週最後收盤 − 1                = 合成，主規格

    缺口不是可忽略的零頭：佔週報酬變異 8.3%、平均 +14.0 bp（週報酬總平均 30.2 bp），
    且與隨後的週內報酬 corr = −0.084——正是價格壓力管道預測的「衝擊 ＋ 反轉」。
    只用 `ret_oc` 估週末係數，等於把衝擊那一段切掉、只留反轉那一段。
    §6.6 投資組合仍用 `ret_oc`：週日看到訊號，最早週一開盤才成交，缺口不可得。
    """
    imb = settings["imbalance"]
    rets = settings["returns"]
    limit = float(rets.get("price_limit_pct", 0.10)) - 0.0015   # 0.0985：含撮合誤差
    df = daily.copy()
    df["week"] = df["date"].map(week_of)
    df = df.sort_values(["ticker", "date"])
    df["_prev_close"] = df.groupby("ticker")["adj_close"].shift(1)
    df["_dret"] = df["adj_close"] / df["_prev_close"] - 1.0
    df["_at_limit"] = df["_dret"].abs() >= limit

    agg = df.groupby(["ticker", "week"]).agg(
        n_trading_days=("date", "size"),
        open_adj=("adj_open", "first"),
        close_adj=("adj_close", "last"),
        volume=("volume", "sum"),
        value=("value", "sum"),
        turnover=("turnover", "sum"),
        shares_outstanding=("shares_outstanding", "last"),
        market_cap=("market_cap", "last"),
        foreign_holding_pct=("foreign_holding_pct", "last"),
        amihud=("amihud", "mean"),
        zero_volume_days=("volume", lambda s: int((s == 0).sum())),
        n_limit_days=("_at_limit", "sum"),
    ).reset_index()

    agg = agg.sort_values(["ticker", "week"])
    g = agg.groupby("ticker", sort=False)
    # 上一個**日曆週**的收盤；跳週（休市整週）時維持缺值，不得跨過去取
    prev_ok = g["week"].shift(1) == agg["week"] - pd.Timedelta(days=7)
    prev_close = g["close_adj"].shift(1).where(prev_ok)

    agg["ret_oc"] = agg["close_adj"] / agg["open_adj"] - 1.0      # 週內（可實作）
    agg["ret_gap"] = agg["open_adj"] / prev_close - 1.0           # 跨週末缺口
    agg["ret_cc"] = agg["close_adj"] / prev_close - 1.0           # 主規格
    bad = agg["open_adj"].isna() | agg["close_adj"].isna()
    agg.loc[bad, ["ret_oc", "ret_gap", "ret_cc"]] = np.nan
    agg["ret"] = agg[_RET_COL[rets.get("main_definition", "close_to_close")]]
    agg["touched_price_limit"] = agg["n_limit_days"] > 0

    roi_parts = []
    for ticker, grp in df.groupby("ticker", sort=False):
        roi = weekly_non_inst_roi(
            grp[["week", "volume", "inst_buy", "inst_sell"]],
            imb["min_daily_volume_shares"], imb["min_valid_days_per_week"])
        if not roi.empty:
            roi_parts.append(pd.DataFrame({"ticker": ticker, "week": roi.index,
                                           "non_inst_roi": roi.values}))
    agg = (agg.merge(pd.concat(roi_parts, ignore_index=True), on=["ticker", "week"],
                     how="left") if roi_parts else agg.assign(non_inst_roi=np.nan))
    return agg


def resolve_spec(settings: dict, spec: str | None) -> tuple[str, dict, str]:
    """把 `specs.<name>` 的覆寫套到 sample 區塊上，回傳 (source, sample, 後綴)。

    **覆寫只允許動 `sample`**：A′ 與 B、C 的差別只在語料來源與期間，其餘門檻
    一律沿用鎖死的設定。允許動別的等於讓四方對照可以被調參，那就失去對照的意義
    （PROJECT.md §6.7），由 `tests/test_specs.py` 守住。
    """
    if spec is None:
        return settings["ptt"]["source"], dict(settings["sample"]), ""
    specs = settings.get("specs") or {}
    if spec not in specs:
        raise ValueError(f"未知規格 {spec!r}；可用：{sorted(specs)}")
    cfg = specs[spec]
    illegal = set(cfg) - {"source", "sample"}
    if illegal:
        raise ValueError(f"規格 {spec} 只允許覆寫 source 與 sample，多了：{sorted(illegal)}")
    smp = {**settings["sample"], **(cfg.get("sample") or {})}
    return cfg["source"], smp, f"_{spec}"


def build_panel(source: str = "pttcc", with_comments: bool = True,
                spec: str | None = None) -> pd.DataFrame:
    settings = load_settings()
    if spec is not None:
        source, smp, spec_suffix = resolve_spec(settings, spec)
        with_comments = source == "pttcc"
        print(f"[規格 {spec}] source={source}  "
              f"{smp['main_start']} ~ {smp['main_end']}  warmup={smp['ptt_warmup_start']}")
    else:
        smp, spec_suffix = dict(settings["sample"]), ""
    exclude_bulk = bool(settings["attention"]["exclude_bulk_listing"])

    uni = pd.read_csv(ROOT / "data" / "external" / "universe.csv", dtype={"ticker": str})
    uni["listing_date"] = pd.to_datetime(uni["listing_date"])
    cal = pd.read_csv(INTERIM / "trading_days.csv", parse_dates=["date"])
    trading_days = {d.date() for d in cal["date"]}

    # --- 週軸：涵蓋暖機期，最後才裁 ---
    warm = pd.Timestamp(smp.get("ptt_warmup_start") or smp["main_start"])
    start, end = pd.Timestamp(smp["main_start"]), pd.Timestamp(smp["main_end"])
    weeks = pd.date_range(week_of(warm), week_of(end), freq="7D")
    tickers = uni["ticker"].tolist()
    containment = smp.get("week_containment", "full")
    main_weeks = [w for w in weeks
                  if (w - pd.Timedelta(days=6)) >= start and w <= end] \
        if containment == "full" else \
        [w for w in weeks if week_of(start) <= w <= week_of(end)]
    print(f"  週軸 {len(weeks)} 週（含暖機期 {week_of(warm).date()} 起）；"
          f"主樣本 {len(main_weeks)} 週 {main_weeks[0].date()} ~ {main_weeks[-1].date()}"
          f"（week_containment={containment}）")

    # --- 文章層 ---
    matches = pd.read_parquet(INTERIM / f"ptt_matches_{source}.parquet")
    n_all = len(matches)
    if exclude_bulk:
        matches = matches[~matches["is_bulk_listing"]]
        print(f"  排除大量清單型貼文 {n_all - len(matches):,} 列"
              f"（{1 - len(matches)/max(n_all,1):.1%}）")
    matches = assign_windows(matches, "timestamp", trading_days)
    matches = matches.rename(columns={"effort_tier": "effort"})
    att = build_attention_panel(matches, weeks, tickers, settings).reset_index()

    # --- 留言層（平行測度）---
    cpath = INTERIM / f"ptt_comment_matches_{source}.parquet"
    if with_comments and cpath.exists():
        comments = pd.read_parquet(
            cpath, columns=["ticker", "user_id", "timestamp", "is_bulk_listing"])
        n_c = len(comments)
        if exclude_bulk:
            comments = comments[~comments["is_bulk_listing"]]
        n_no_ts = int(comments["timestamp"].isna().sum())
        comments = assign_windows(comments, "timestamp", trading_days)
        print(f"  留言 {n_c:,} 列 → 排除 bulk 與缺時戳後 {len(comments):,} 列"
              f"（缺時戳 {n_no_ts:,}）")
        catt = build_comment_panel(comments, weeks, tickers, settings).reset_index()
        att = att.merge(catt, on=["ticker", "week"], how="left")

    # --- 行情層 ---
    daily = pd.read_parquet(INTERIM / "market_daily.parquet")
    market = _weekly_market(daily, settings)

    panel = att.merge(market, on=["ticker", "week"], how="left")
    panel = panel.merge(
        uni[["ticker", "name_short", "sector", "listing_date", "venue", "is_ky"]],
        on="ticker", how="left")

    # --- 非平衡：上市前的週維持缺列 ---
    n_before = len(panel)
    panel = panel[panel["week"] >= panel["listing_date"].map(week_of)].copy()
    n_dropped = n_before - len(panel)

    # --- 交易日曆屬性 ---
    week_days = (pd.Series(sorted(trading_days)).map(week_of)
                 .value_counts().sort_index())
    panel["week_n_trading_days"] = panel["week"].map(week_days).fillna(0).astype(int)
    panel["is_incomplete_week"] = (
        panel["week_n_trading_days"] < settings["returns"]["min_trading_days_per_week"])
    makeup = set(cal.loc[cal["is_makeup_saturday"], "date"].map(week_of))
    panel["is_makeup_saturday_week"] = panel["week"].isin(makeup)

    # --- 領先項：休市週不得跳過 ---
    panel = panel.sort_values(["ticker", "week"]).reset_index(drop=True)
    g = panel.groupby("ticker", sort=False)
    nxt_ok = g["week"].shift(-1) == panel["week"] + pd.Timedelta(days=7)
    # 三段式報酬各領先一週：主規格 ret_next（= ret_cc_next），另出缺口與週內兩段
    # 供 §6.1 的分解規格，以及 §6.6 用可實作的 ret_oc_next
    for src_col, dst in [("ret", "ret_next"), ("ret_cc", "ret_cc_next"),
                         ("ret_oc", "ret_oc_next"), ("ret_gap", "ret_gap_next"),
                         ("non_inst_roi", "non_inst_roi_next"),
                         ("touched_price_limit", "touched_price_limit_next")]:
        panel[dst] = g[src_col].shift(-1).where(nxt_ok)
    panel["abn_turnover"] = g["turnover"].transform(abnormal_turnover)
    panel["turnover_next"] = panel.groupby("ticker", sort=False)["abn_turnover"] \
        .shift(-1).where(nxt_ok)

    # --- 落後項：動能控制 ---
    g = panel.groupby("ticker", sort=False)
    panel["ret_lag1"] = g["ret"].shift(1)
    for lag, name in [(4, "ret_lag4"), (25, "ret_lag25")]:
        panel[name] = g["ret"].transform(
            lambda s, k=lag: s.shift(1).rolling(k, min_periods=max(2, k // 2)).mean())
    panel["non_inst_roi_lag1"] = g["non_inst_roi"].shift(1)
    panel["turnover_lag1"] = g["abn_turnover"].shift(1)

    # --- 反轉檢定用的 t+2..t+8 報酬 ---
    for h in range(2, 9):
        ok_h = g["week"].shift(-h) == panel["week"] + pd.Timedelta(days=7 * h)
        panel[f"ret_fwd{h}"] = g["ret"].shift(-h).where(ok_h)

    # --- 制度斷點 ---
    ic = settings["institutions"]
    panel["regime_continuous_trading"] = panel["week"] >= pd.Timestamp(ic["continuous_trading"])
    panel["regime_odd_lot"] = panel["week"] >= pd.Timestamp(ic["odd_lot_trading"])
    panel["listing_age_years"] = (panel["week"] - panel["listing_date"]).dt.days / 365.25

    # --- 台股制度性混淆（LIMITATIONS.md §12）---
    panel, inst_status = attach_institutions(
        panel, INTERIM / "disposition.csv", INTERIM / "day_trading.csv", trading_days)
    for k, v in inst_status.items():
        print(f"  {k}: {v}")

    # --- 通用詞降級標記 ---
    code_only = set(load_universe_config().get("code_only_tickers", []))
    panel["is_code_only_matched"] = panel["ticker"].isin(code_only)

    # --- 產業內外溢 ---
    own = panel["abn_attention_weekend"]
    sec = panel.groupby(["sector", "week"])["abn_attention_weekend"]
    # 自身為缺值時它本來就不在 sum／count 裡，分母不得再減一
    n_peers = sec.transform("count") - own.notna().astype(int)
    panel["sector_peer_abn_att_weekend"] = (
        (sec.transform("sum") - own.fillna(0)) / n_peers.replace(0, np.nan))
    panel["abn_att_weekend_rel_sector"] = (
        panel["abn_attention_weekend"] - sec.transform("mean"))

    # --- 最後才裁到主樣本：暖機期只餵滾動窗，不進任何表格 ---
    #
    # 週必須**完整**落在 [main_start, main_end] 內（`sample.week_containment: full`）。
    # 以週標籤落點判定會兩端各污染一週：
    #   2020-01-05 涵蓋 2019-12-30~2020-01-05 → 暖機期洩漏進主樣本（違反 §1）
    #   2025-01-05 涵蓋 2024-12-30~2025-01-05 → 語料止於 2024-12-31，該週
    #                                            att_weekend 恆為 0、ret_next 全缺值
    n_with_warmup = len(panel)
    if smp.get("week_containment", "full") == "full":
        in_main = ((panel["week"] - pd.Timedelta(days=6)) >= start) & (panel["week"] <= end)
    else:
        in_main = (panel["week"] >= week_of(start)) & (panel["week"] <= week_of(end))
    panel = panel[in_main]
    panel = panel.reset_index(drop=True)
    print(f"  裁掉暖機期 {n_with_warmup - len(panel):,} 列")

    suffix = spec_suffix or ("" if source == "pttcc" else f"_{source}")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED / f"panel{suffix}.parquet", index=False)
    panel[panel["sparsity_tier"] == "dense"].to_parquet(
        PROCESSED / f"panel_dense{suffix}.parquet", index=False)

    AUDIT.mkdir(exist_ok=True)
    # 資料缺席的穩健性項目必須明確記錄，不得靜默略過（PROJECT.md §6）
    rob = settings.get("robustness", {})
    pd.DataFrame([
        {"item": k, "status": v,
         "config_key": f"robustness.{k}",
         "enabled_in_config": bool(rob.get(k, {}).get("enabled", rob.get(k)))}
        for k, v in inst_status.items()
    ] + [{"item": "h4_makeup_days",
          "status": settings["hypotheses"]["h4_makeup_days"],
          "config_key": "hypotheses.h4_makeup_days",
          "enabled_in_config": False}]
    ).to_csv(AUDIT / f"model_status{suffix}.csv", index=False)

    pd.DataFrame([{
        "spec": spec or "default",
        "source": source,
        "n_rows": len(panel),
        "n_tickers": panel["ticker"].nunique(),
        "n_weeks": panel["week"].nunique(),
        "rows_dropped_pre_listing": n_dropped,
        "rows_dropped_warmup": n_with_warmup - len(panel),
        "first_week": str(panel["week"].min().date()),
        "last_week": str(panel["week"].max().date()),
        "n_dense_rows": int((panel["sparsity_tier"] == "dense").sum()),
        "n_sparse_rows": int((panel["sparsity_tier"] == "sparse").sum()),
        "n_silent_rows": int((panel["sparsity_tier"] == "silent").sum()),
        "n_tier_missing": int(panel["sparsity_tier"].isna().sum()),
        "n_with_ret_next": int(panel["ret_next"].notna().sum()),
        "n_with_ret_gap_next": int(panel["ret_gap_next"].notna().sum()),
        "n_incomplete_weeks": int(panel.drop_duplicates("week")["is_incomplete_week"].sum()),
        "n_weeks_touching_price_limit": int(panel["touched_price_limit"].sum()),
        "ret_definition": settings["returns"].get("main_definition", "close_to_close"),
        "abn_baseline": settings["attention"].get("baseline", "median"),
        "n_with_roi_next": int(panel["non_inst_roi_next"].notna().sum()),
        "pct_zero_attention": round(float((panel["att_all"] == 0).mean()), 4),
        "n_disposition_weeks": (int(panel["is_disposition_week"].sum())
                                if panel["is_disposition_week"].notna().any() else ""),
        "mean_dt_ratio": (round(float(panel["dt_ratio"].mean()), 4)
                          if panel["dt_ratio"].notna().any() else ""),
        "mean_dt_value_ratio": (round(float(panel["dt_value_ratio"].mean()), 4)
                                if panel["dt_value_ratio"].notna().any() else ""),
    }]).to_csv(AUDIT / f"panel_summary{suffix}.csv", index=False)

    print(f"panel {len(panel):,} 列 × {panel['ticker'].nunique()} 檔 × "
          f"{panel['week'].nunique()} 週；上市前剔除 {n_dropped:,} 列")
    return panel


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="建立 ticker×week 面板")
    ap.add_argument("--source", default="pttcc")
    ap.add_argument("--spec", default=None,
                    help="PROJECT.md §6.7 的四方對照規格：C / B / A_prime。"
                         "指定後 --source 由規格決定。")
    args = ap.parse_args()
    build_panel(args.source, with_comments=(args.source == "pttcc"), spec=args.spec)


if __name__ == "__main__":
    main()
