"""T13 與原論文的逐項對照（`docs/PLAN_V2.md` §5 P4）。

原論文：Li, Liu, Ye, Zhao & Zhao, *It Depends on When You Search.* MIS Quarterly
（<https://ssrn.com/abstract=4370525>）。論文端每個數字都標註表號與頁碼，可逐項
回查 PDF；本研究端一律由 `data/processed/panel_<spec>.parquet` 現算，不寫死。

## 三件 v2 特有的事

1. **判讀由規則算出，不沿用 v1 的字串**（`_verdict()`）。v1 的 T13 判讀欄是看過
   v1 的數字之後寫的；v2 的係數不同，照抄會變成對不上號的斷言。
2. **測度效度算三次**（T13d）：發文數、留言則數、獨立帳號數。論文 Table 2 的
   0.9028 / 0.3981 是 v1 最強的一組證據（實得 0.9391 / 0.4027），也最值得用有
   帳號與時戳的新測度重驗。
3. **推論標準必須同時報兩種**（T13b，`PROJECT.md` §0.2、§6.1）。v1 的週末係數在
   兩者之間跨過 5% 門檻；v2 三個規格都重現了這個刀鋒狀態。**不得只報其中一種。**

## 尺度可比性（這一節是本表能不能讀的前提）

論文 Table 3a 註記「All variables are standardized」——**含應變數**。主規格只標準化
自變數與控制變數，因此本模組另跑一組「應變數也標準化」的規格專供對照，並在
`scale` 欄標明。訂單失衡與周轉率兩邊是不同測度（論文為 Boehmer et al. 散戶訂單
失衡、本研究為非三大法人殘差），**係數不可直接比**，比值欄一律留空。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, load_settings
from .regressions import (BASE_CONTROLS, add_derived, fit, main_sample,
                          standardize_within)

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output"

XS = ["abn_attention_weekday", "abn_attention_weekend"]
TWO, ONE = "twoway_2cluster", "firm_fe_1cluster"
INF_LABEL = {TWO: "本研究主規格：個股＋週雙重 cluster",
             ONE: "論文做法：僅 cluster 至個股"}

#: 論文端基準，每筆標註出處（PROJECT.md §0.2 已逐項核對 PDF）。
PAPER = {
    "sample": "S&P 500，2004–2019（訂單失衡部分 2010–2015）",
    "fixed_effects": "僅個股固定效果",
    "clustering": "僅 cluster 至個股",
    "n_obs_return": 255_059,          # Table 3a, p.16
    "corr_all_weekday": 0.9028,       # Table 2, p.13
    "corr_all_weekend": 0.3981,       # Table 2, p.13
    "ret_weekend_beta": 0.0068, "ret_weekend_se": 0.0021,    # Table 3a (4), p.15
    "ret_weekday_beta": 0.0001, "ret_weekday_se": 0.0022,
    "roi_weekday_beta": 0.12311, "roi_weekday_se": 0.04041,  # Table 7 (4), p.21
    "roi_weekend_beta": 0.03185, "roi_weekend_se": 0.0449,
    "turnover_weekend_beta": 0.0049, "turnover_weekend_se": 0.0013,  # Table 8 (4), p.23
    "turnover_weekday_beta": -0.0018, "turnover_weekday_se": 0.0015,
    "portfolio_ew_annual": 2.92,      # Table 5a 敘述, p.17（%/年，成本前）
    "double_sort_small": 3.36,        # Table 6a, p.20（%/年）
    "double_sort_large": 0.53,
}

#: v2 的三組關注度測度。`(標籤, 前綴)`；留言與帳號兩組在舊語料上不存在。
MEASURES = [("發文數（主測度）", "abn_attention"),
            ("留言則數", "abn_attention_comment"),
            ("獨立帳號數", "abn_attention_users")]


def _t(beta: float | None, se: float | None) -> float | None:
    return beta / se if (beta is not None and se) else None


def _stars(t: float | None) -> str:
    if t is None or t != t:
        return ""
    a = abs(t)
    return "***" if a > 2.576 else "**" if a > 1.96 else "*" if a > 1.645 else ""


def _verdict(paper_t: float | None, ours_t: float | None,
             paper_b: float | None = None, ours_b: float | None = None) -> str:
    """型態是否一致——**由規則算出，不是看過數字之後寫下的判斷**。

    顯著門檻 |t| > 1.96。兩邊都顯著時再比符號；係數尺度不可比時符號仍可比。
    """
    if paper_t is None or ours_t is None or paper_t != paper_t or ours_t != ours_t:
        return "— 無對應估計"
    ps, os_ = abs(paper_t) > 1.96, abs(ours_t) > 1.96
    if not ps and not os_:
        return "✅ 型態一致：兩者皆不顯著"
    if ps and os_:
        same = (paper_b is None or ours_b is None
                or (paper_b > 0) == (ours_b > 0))
        return ("✅ 型態一致：兩者皆顯著且同號" if same
                else "❌ 明確不同：兩者皆顯著但**符號相反**")
    return ("❌ 明確不同：論文顯著、本研究不顯著" if ps
            else "❌ 明確不同：本研究顯著、論文不顯著")


def _corr_verdict(paper: float, ours: float, tol: float = 0.15) -> str:
    if ours != ours:
        return "— 無對應估計"
    return ("✅ 幾乎逐位數複製" if abs(ours / paper - 1) <= tol
            else "⚠️ 型態同向但幅度不同")


# ---------------------------------------------------------------------------
# T13d：測度效度（論文 Table 2）
# ---------------------------------------------------------------------------

def measure_validity(panel: pd.DataFrame) -> pd.DataFrame:
    """corr(整週, 週間) 與 corr(整週, 週末)，三組測度 × 兩個樣本各算一次。

    **樣本必須寫在表上**：v1 算在全面板（含 `silent`），而 `silent` 的 `AbnAtt`
    恆為 0，會把相關係數往上推。主迴歸跑的是 dense＋sparse，兩個都報，讀者才知道
    0.94 這個數字是在哪個樣本上得到的。
    """
    rows = []
    for sample_label, d in (("全面板（v1 的基準）", panel),
                            ("主迴歸樣本 dense＋sparse", main_sample(panel))):
        for label, prefix in MEASURES:
            cols = [f"{prefix}_{w}" for w in ("all", "weekday", "weekend")]
            if not all(c in d.columns for c in cols):
                rows.append({"sample": sample_label, "measure": label,
                             "status": "SKIPPED",
                             "note": f"此規格的語料無此測度（缺欄位：{cols[0]}）"})
                continue
            x = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
            c_wd = float(x[cols[0]].corr(x[cols[1]]))
            c_we = float(x[cols[0]].corr(x[cols[2]]))
            rows.append({
                "sample": sample_label, "measure": label, "status": "OK",
                "corr_all_weekday": round(c_wd, 4),
                "corr_all_weekend": round(c_we, 4),
                "gap": round(c_wd - c_we, 4), "n_obs": len(x),
                "paper_weekday": PAPER["corr_all_weekday"],
                "paper_weekend": PAPER["corr_all_weekend"],
                "pattern_matches_paper": bool(c_wd > c_we),
                "verdict_weekday": _corr_verdict(PAPER["corr_all_weekday"], c_wd),
                "verdict_weekend": _corr_verdict(PAPER["corr_all_weekend"], c_we),
                "note": "",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 估計：應變數也標準化，與論文同尺度
# ---------------------------------------------------------------------------

def _standardized(main: pd.DataFrame) -> pd.DataFrame:
    d = standardize_within(main, [*XS, *BASE_CONTROLS, "non_inst_roi_lag1",
                                  "turnover_lag1"])
    for dv in ("ret_oc_next", "non_inst_roi_next", "turnover_next"):
        if dv in d.columns:
            g = d.groupby("sparsity_tier", observed=True)[dv]
            d[f"{dv}_std"] = (d[dv] - g.transform("mean")) / g.transform("std")
    return d


def _fits(d: pd.DataFrame, missing: list[str]) -> dict:
    """三個應變數 × 兩種推論標準。鍵為 `(dv, inference)`。"""
    jobs = [("ret", "ret_oc_next_std", BASE_CONTROLS),
            ("roi", "non_inst_roi_next_std", [*BASE_CONTROLS, "non_inst_roi_lag1"]),
            ("turnover", "turnover_next_std", [*BASE_CONTROLS, "turnover_lag1"])]
    return {(tag, inf): fit(d, y, XS, ctrl, f"T13 {tag}", inference=inf,
                            required_missing=missing)
            for tag, y, ctrl in jobs for inf in (TWO, ONE)}


def inference_sensitivity(fits: dict) -> pd.DataFrame:
    """T13b：同一組資料在兩種推論標準下的結果。**本表最重要的一節。**"""
    rows = []
    for inf, note in ((TWO, "PROJECT.md §0.2；267 檔同時暴露於相同的週別市場衝擊"),
                      (ONE, "原論文 Table 3a 註")):
        r = fits[("ret", inf)]
        if r.status != "OK":
            continue
        for term, zh in (("abn_attention_weekend", "週末"),
                         ("abn_attention_weekday", "週間")):
            t = r.tstats[term]
            rows.append({"inference": INF_LABEL[inf], "window": zh,
                         "beta": r.params[term], "se": r.stderr[term], "t": t,
                         "p": r.pvalues[term], "sig": _stars(t),
                         "n_obs": r.n_obs, "note": note})
    out = pd.DataFrame(rows)
    if not out.empty:
        we = out[out["window"] == "週末"].set_index("inference")
        a, b = INF_LABEL[TWO], INF_LABEL[ONE]
        if a in we.index and b in we.index:
            out.attrs["se_ratio"] = float(we.loc[a, "se"] / we.loc[b, "se"])
            out.attrs["flips_at_5pct"] = bool(we.loc[a, "p"] > 0.05 >= we.loc[b, "p"])
    return out


# ---------------------------------------------------------------------------
# T13a：逐項對照
# ---------------------------------------------------------------------------

def comparison_table(fits: dict, validity: pd.DataFrame,
                     port: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict] = []

    def add(item, p_b, p_se, o_b, o_se, o_t, scale, source, *,
            inference="", comparable=False, note="", verdict=None):
        """`comparable=True` 才算比值。

        兩個估計必須（a）建構相同、（b）尺度相同、（c）論文端顯著，比值才有意義。
        論文係數接近零時（週間 → 報酬 = 0.0001）比值會爆到數十倍，那是分母趨近零
        的算術，不是「效果大數十倍」。這種情況一律留空。
        """
        p_t = _t(p_b, p_se)
        ratio = None
        if comparable and p_b not in (None, 0) and o_b is not None:
            if p_t is None or abs(p_t) > 1.96:
                ratio = o_b / p_b
        rows.append({"item": item, "inference": inference,
                     "paper_estimate": p_b, "paper_se": p_se, "paper_t": p_t,
                     "ours_estimate": o_b, "ours_se": o_se, "ours_t": o_t,
                     "ratio_ours_to_paper": ratio,
                     "verdict": verdict if verdict is not None
                                else _verdict(p_t, o_t, p_b, o_b),
                     "scale": scale, "paper_source": source, "note": note})

    def g(key, term):
        r = fits[key]
        return ((r.params.get(term), r.stderr.get(term), r.tstats.get(term))
                if r.status == "OK" else (None, None, None))

    # --- 測度效度（主測度、主迴歸樣本）---
    v = validity[(validity["measure"] == "發文數（主測度）")
                 & (validity["sample"] == "主迴歸樣本 dense＋sparse")]
    v = v.iloc[0] if len(v) and v.iloc[0]["status"] == "OK" else None
    for zh, pk, ok in (("週間", "corr_all_weekday", "corr_all_weekday"),
                       ("週末", "corr_all_weekend", "corr_all_weekend")):
        ours = float(v[ok]) if v is not None else None
        add(f"相關性 corr(整週, {zh})", PAPER[pk], None, ours, None, None,
            "相關係數", "Table 2, p.13", comparable=True,
            note="樣本為 dense＋sparse；三組測度 × 兩個樣本的完整版見 T13d",
            verdict=(_corr_verdict(PAPER[pk], ours) if ours is not None
                     else "— 無對應估計"))

    # --- H1：兩種推論標準各一列。v1 的 5% 門檻在此翻轉 ---
    for inf in (TWO, ONE):
        for term, zh, pb, pse in (
                ("abn_attention_weekend", "週末",
                 PAPER["ret_weekend_beta"], PAPER["ret_weekend_se"]),
                ("abn_attention_weekday", "週間",
                 PAPER["ret_weekday_beta"], PAPER["ret_weekday_se"])):
            b, s, t = g(("ret", inf), term)
            add(f"H1 {zh}關注度 → 次週報酬", pb, pse, b, s, t,
                "應變數已標準化", "Table 3a 規格(4), p.15", inference=INF_LABEL[inf],
                comparable=(term == "abn_attention_weekend"))

    # --- 機制：測度不同，只比型態與符號 ---
    for tag, term, zh, pb, pse, note in (
            ("roi", "abn_attention_weekday", "週間 → 訂單失衡",
             PAPER["roi_weekday_beta"], PAPER["roi_weekday_se"],
             "不可直接比：論文 DV 為 MROI×100，本研究為非三大法人殘差"),
            ("roi", "abn_attention_weekend", "週末 → 訂單失衡",
             PAPER["roi_weekend_beta"], PAPER["roi_weekend_se"],
             "不可直接比：測度不同；本研究另受當沖混淆（LIMITATIONS.md §12）"),
            ("turnover", "abn_attention_weekend", "週末 → 異常周轉率",
             PAPER["turnover_weekend_beta"], PAPER["turnover_weekend_se"],
             "不可直接比：本研究為 log1p 差分；另受處置股混淆（§12.1）"),
            ("turnover", "abn_attention_weekday", "週間 → 異常周轉率",
             PAPER["turnover_weekday_beta"], PAPER["turnover_weekday_se"],
             "不可直接比：本研究為 log1p 差分")):
        b, s, t = g((tag, TWO), term)
        add(f"機制：{zh}", pb, pse, b, s, t, "應變數已標準化",
            "Table 7 規格(4), p.21" if tag == "roi" else "Table 8 規格(4), p.23",
            inference=INF_LABEL[TWO], note=note)

    # --- 投資組合 ---
    def prow(name):
        if port is None:
            return None
        r = port[(port["portfolio"] == name) & (port["status"] == "OK")]
        return r.iloc[0] if len(r) else None

    ew = prow("主測度 發文數｜等權・未篩選")
    if ew is not None:
        add("投資組合：等權多空（成本前，%/年）", PAPER["portfolio_ew_annual"], None,
            float(ew["mean_ls_gross_weekly"]) * 52 * 100, None, float(ew["t_gross"]),
            "%/年", "Table 5a 敘述, p.17",
            note="ex-post universe 之機械年化；**成本前價差不得作為主要結論陳述**",
            # 論文 Table 5a 只在正文敘述年化價差，未報 t 值或標準誤，因此
            # 「論文那一邊是否顯著」不可得——不得代它假設一個顯著性再來判讀。
            verdict="— 論文未報 t 值，見 note")
        add("投資組合：等權多空（成本後，%/年）", None, None,
            float(ew["mean_ls_net_weekly"]) * 52 * 100, None, float(ew["t_net"]),
            "%/年", "論文無對應項",
            note="手續費打 6 折 ×2 ＋ 證交稅 ＋ 滑價 20bp×2，按實際換手率計",
            verdict="❌ **論文未報告成本後報酬**，無從對照")
    for side, pk in (("小型股", "double_sort_small"), ("大型股", "double_sort_large")):
        r = prow(f"主測度 發文數｜雙重排序・{side}")
        add(f"橫斷面：雙重排序・{side}（成本前，%/年）", PAPER[pk], None,
            float(r["mean_ls_gross_weekly"]) * 52 * 100 if r is not None else None,
            None, float(r["t_gross"]) if r is not None else None,
            "%/年", "Table 6a, p.20",
            note="論文未報 t 值，僅能比型態（小型半邊顯著、大型半邊不顯著）",
            verdict="— 論文未報 t 值，見 note")

    r = fits[("ret", TWO)]
    add("樣本規模（次週報酬迴歸）", PAPER["n_obs_return"], None,
        float(r.n_obs) if r.status == "OK" else None, None, None,
        "觀測數", "Table 3a, p.16",
        note=("本研究樣本期 260 週、260 檔；標準誤因此大於論文，"
              "這是 diagnostic 標記之外的獨立限制"),
        verdict="— 規模對照，非假說檢定")
    return pd.DataFrame(rows)


def design_differences(panel: pd.DataFrame) -> pd.DataFrame:
    """T13c：設計差異登記簿。解讀 T13a 時必須同時看這張。

    期間與檔數由**面板現算**，不讀 `settings.sample`——A′ 的期間是 `specs` 區塊的
    覆寫值，讀預設會把全期規格寫成 2020–2024。
    """
    lo, hi = panel["week"].min(), panel["week"].max()
    zero_we = float((panel["abn_attention_weekend"].fillna(0) == 0).mean())
    rows = [
        ("樣本", "S&P 500（全為大型股）",
         f"{panel['ticker'].nunique()} 檔台股長尾（無金控、缺主要權值股）",
         "本研究樣本整體相當於論文的『小型半邊』"),
        ("期間", "2004–2019（16 年）",
         f"{lo:%Y-%m} ~ {hi:%Y-%m}（{panel['week'].nunique()} 週）",
         "2020–2024 ＝ COVID ＋ 當沖狂熱 ＋ 2023 低點，外部效度比 v1 更窄"),
        ("關注度測度", "Google Trends SVI（連續搜尋量）",
         "PTT 發文數／留言則數／獨立帳號數（極稀疏計數）",
         "**測度性質根本不同**：搜尋是低成本行為，發文是高成本行為"),
        ("主要自變數的變異", "連續搜尋量，無退化問題",
         f"`abn_attention_weekend` 有 {zero_we:.1%} 恰為 0",
         "週末檢定實際上接近稀有事件的虛擬變數檢定（LIMITATIONS.md §11）"),
        ("固定效果", "個股", "個股 ＋ 週（主規格）／個股（論文標準，同列並報）",
         "本研究額外吸收市場層級共同衝擊"),
        ("標準誤", "cluster 至個股", "cluster 至個股與週（主規格）",
         "**結論差異的主要來源**，見 T13b"),
        ("訂單失衡測度", "Boehmer et al. (2021) 散戶訂單失衡",
         "非三大法人殘差（含大戶與其他機構）",
         "測度不同，符號不可直接比"),
        ("制度混淆", "未涉及",
         "處置股壓 turnover_next、當沖機械壓縮 ROI、漲跌停截斷應變數",
         "台股特有，三項全部在 LIMITATIONS.md §12"),
        ("新聞控制", "RavenPack（news count、NIP、MCQ）", "**未取得**",
         "§7 明訂 news_count 為正式主表必要控制項"),
        ("分析師控制", "I/B/E/S（家數、離散度、修正）", "**未取得**", "需 TEJ 授權"),
        ("交易成本", "未報告", "手續費＋證交稅＋滑價，按實際換手率計",
         "§6.6 要求成本後為必要欄"),
        ("H4 自然實驗", "無此設計",
         "**SKIP**：2020–2024 補班星期六交易日為 0 天", "事件數為零，不是檢定力不足"),
    ]
    return pd.DataFrame(rows, columns=["dimension", "paper", "ours", "note"])


# ---------------------------------------------------------------------------

def run(spec: str = "C") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                  pd.DataFrame]:
    settings = load_settings()
    panel = add_derived(pd.read_parquet(PROCESSED / f"panel_{spec}.parquet"))
    validity = measure_validity(panel)
    fits = _fits(_standardized(main_sample(panel)),
                 settings["regression"]["controls"]["required_but_missing"])

    port_path = OUT / f"T9_portfolios_{spec}.csv"
    port = pd.read_csv(port_path) if port_path.exists() else None
    if port is None:
        print(f"  ⚠ {port_path.name} 不存在，投資組合各列留空"
              f"（先跑 python3 -m src.analysis.portfolios --spec {spec}）")

    return (comparison_table(fits, validity, port), inference_sensitivity(fits),
            design_differences(panel), validity)


def _fmt(v, n=4) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{n}f}"


def write_markdown(comp, inf, design, validity, spec: str, out: Path) -> None:
    """人可讀版本。CSV 供程式用，這份供論文與簡報引用。"""
    L = ["# T13 與原論文的逐項對照", "",
         "> 原論文：Li, Liu, Ye, Zhao & Zhao, *It Depends on When You Search.* "
         "MIS Quarterly（<https://ssrn.com/abstract=4370525>）。",
         f"> 論文樣本：{PAPER['sample']}；固定效果：{PAPER['fixed_effects']}；"
         f"標準誤：{PAPER['clustering']}。",
         f"> 本研究端為**規格 {spec}**，由面板現算；判讀欄由 `_verdict()` 依 "
         "|t| > 1.96 與符號算出，不是事後寫下的斷言。",
         "> **所有結果均為 `diagnostic`**（`PROJECT.md` §7）。", "",
         "## T13a 逐項對照", "",
         "| 項目 | 推論標準 | 論文 | (t) | 本研究 | (t) | 比值 | 判讀 | 論文出處 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in comp.itertuples():
        L.append(f"| {r.item} | {r.inference or '—'} | {_fmt(r.paper_estimate)} | "
                 f"{_fmt(r.paper_t, 2)} | {_fmt(r.ours_estimate)} | "
                 f"{_fmt(r.ours_t, 2)} | {_fmt(r.ratio_ours_to_paper, 2)} | "
                 f"{r.verdict} | {r.paper_source} |")
    L += ["", "**尺度說明**：論文 Table 3a 註記「All variables are standardized」含"
          "應變數，因此本表另跑一組「應變數也標準化」的規格。訂單失衡與周轉率兩邊是"
          "不同測度，**係數不可直接比**，比值欄一律留空。", ""]

    L += ["## T13b 推論標準的敏感度（本表最重要的一節）", "",
          "| 推論方式 | 窗口 | β | SE | t | p | | 依據 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in inf.itertuples():
        L.append(f"| {r.inference} | {r.window} | {_fmt(r.beta)} | {_fmt(r.se)} | "
                 f"{_fmt(r.t, 2)} | {_fmt(r.p, 3)} | {r.sig} | {r.note} |")
    if "se_ratio" in inf.attrs:
        flip = "**因此翻轉**" if inf.attrs.get("flips_at_5pct") else "未翻轉"
        L += ["", f"雙重 cluster 的標準誤為個股 cluster 的 "
              f"**{inf.attrs['se_ratio']:.2f} 倍**，5% 門檻{flip}。", "",
              "> **同一份資料、同一個係數，換一個標準誤的算法就跨過或跨不過 5%。**",
              "> 本專案採用較保守的雙重 cluster，理由是 267 檔同時暴露於相同的週別"
              "市場衝擊；這是方法論選擇，不是資料失敗。**兩種都必須報。**", ""]

    L += ["## T13d 測度效度：三組測度各算一次", "",
          "論文 Table 2 為 corr(ASVI, ASVI15) = 0.9028、corr(ASVI, ASVI67) = 0.3981。",
          "v1 實得 0.9391 / 0.4027（全面板）。**樣本必須寫在表上**——`silent` 的 "
          "`AbnAtt` 恆為 0，會把相關係數往上推。", "",
          "| 樣本 | 測度 | corr(整週, 週間) | corr(整週, 週末) | 差 | n | 判讀（週間／週末） |",
          "|---|---|---|---|---|---|---|"]
    for r in validity.itertuples():
        if r.status != "OK":
            L.append(f"| {r.sample} | {r.measure} | — | — | — | — | {r.note} |")
            continue
        L.append(f"| {r.sample} | {r.measure} | {r.corr_all_weekday:.4f} | "
                 f"{r.corr_all_weekend:.4f} | {r.gap:.4f} | {r.n_obs:,} | "
                 f"{r.verdict_weekday}／{r.verdict_weekend} |")

    L += ["", "## T13c 設計差異登記簿", "", "解讀 T13a 時必須同時看這張。", "",
          "| 面向 | 論文 | 本研究 | 說明 |", "|---|---|---|---|"]
    for r in design.itertuples():
        L.append(f"| {r.dimension} | {r.paper} | {r.ours} | {r.note} |")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T13 與原論文對照")
    ap.add_argument("--spec", default="C", help="C / B / A_prime")
    args = ap.parse_args(argv)
    print(f"[T13] 規格 {args.spec}")
    comp, inf, design, validity = run(args.spec)
    OUT.mkdir(parents=True, exist_ok=True)
    for df, name in ((comp, "T13a_paper_comparison"),
                     (inf, "T13b_inference_sensitivity"),
                     (design, "T13c_design_differences"),
                     (validity, "T13d_measure_validity")):
        df.to_csv(OUT / f"{name}_{args.spec}.csv", index=False)
        print(f"  → output/{name}_{args.spec}.csv（{len(df)} 列）")
    md = OUT / f"T13_paper_comparison_{args.spec}.md"
    write_markdown(comp, inf, design, validity, args.spec, md)
    print(f"  → {md.relative_to(ROOT)}")

    n_ok = int(comp["verdict"].str.startswith("✅").sum())
    n_warn = int(comp["verdict"].str.startswith("⚠️").sum())
    n_bad = int(comp["verdict"].str.startswith("❌").sum())
    print(f"\n  T13a {len(comp)} 項：型態一致 {n_ok}、部分不同 {n_warn}、"
          f"明確不同 {n_bad}、無對照 {len(comp) - n_ok - n_warn - n_bad}")
    if "se_ratio" in inf.attrs:
        print(f"  T13b 雙重 cluster 的 SE 為個股 cluster 的 "
              f"{inf.attrs['se_ratio']:.2f} 倍；5% 門檻"
              f"{'因此翻轉' if inf.attrs.get('flips_at_5pct') else '未翻轉'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
