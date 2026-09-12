"""凍結 v2 的實得數字到 `tests/expected_v2.py`（`docs/PLAN_V2.md` §P5）。

P5 把測試分兩類：

- **邏輯不變量**（`tests/test_core_logic.py` 等）：窗口指派、週對齊、碰撞消解。
  與語料無關，換資料不得改變，**永遠不該由本模組產生**。
- **寫死數字**（本模組產生的 `tests/expected_v2.py`）：語料量、面板形狀、係數。
  它們的作用是**抓靜默變動**——改了程式而數字跟著變、卻沒人注意到。

因此凍結必須是**刻意的動作**：

```bash
python3 -m src.audit_expected --freeze      # 只在確認新數字正確之後跑
python3 -m pytest tests/test_expected_v2.py # 平常只驗，不寫
```

不加 `--freeze` 時只比對現值與已凍結值並印出差異，**不寫檔**。這樣「測試失敗了
所以重新凍結」就不會變成無意識的動作——那會讓整組測試失去意義。

凍結檔記錄當時的 commit 與時間，好讓「這批數字是哪一版跑出來的」可回查。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT

AUDIT = ROOT / "audit"
OUT = ROOT / "output"
TESTS = ROOT / "tests"
TARGET = TESTS / "expected_v2.py"

SPECS = ["C", "B", "A_prime"]

#: 凍結哪些係數。`(表, 模型, 項, 標籤)`；每個都取兩種推論標準。
COEFS = [
    ("T3_h1_main", "H1-4 週間＋週末｜oc", "abn_attention_weekend", "h1_weekend_oc"),
    ("T3_h1_main", "H1-4 週間＋週末｜oc", "abn_attention_weekday", "h1_weekday_oc"),
    ("T3_h1_main", "H1-4 週間＋週末｜gap", "abn_attention_weekend", "h1_weekend_gap"),
    ("T3_h1_main", "H1-4 週間＋週末｜cc", "abn_attention_weekend", "h1_weekend_cc"),
    ("T3_h1_main", "H1-5 交易時段切法｜gap", "abn_attention_non_trading",
     "h1_non_trading_gap"),
    ("T3_h1_main", "H1-5 交易時段切法｜gap", "abn_attention_intraday",
     "h1_intraday_gap"),
    ("T3_h1_main", "H1-R 排除次週觸及漲跌停｜gap", "abn_attention_weekend",
     "h1_weekend_gap_nolimit"),
    ("T3_h1_main", "H1-R5 排除漲跌停・交易時段切法｜gap", "abn_attention_non_trading",
     "h1_non_trading_gap_nolimit"),
    ("T3_h1_main", "H1-R medianbase｜oc", "abn_attention_weekend_medianbase",
     "h1_weekend_oc_medianbase"),
    ("T3_h1_main", "H8-users 獨立帳號數｜oc", "abn_attention_users_weekend",
     "h8_users_weekend_oc"),
    ("T5_h3_mechanism", "H3a-1 ROI 主表", "abn_attention_weekday", "h3a_roi_weekday"),
    ("T5_h3_mechanism", "H3a-1 ROI 主表", "abn_attention_weekend", "h3a_roi_weekend"),
    ("T5_h3_mechanism", "H3b-1 周轉率 主表", "abn_attention_weekday",
     "h3b_turnover_weekday"),
    ("T5_h3_mechanism", "H3b-1 周轉率 主表", "abn_attention_weekend",
     "h3b_turnover_weekend"),
    ("T6_joint_reading", "H6 amihud｜主測度 發文數",
     "abn_attention_weekend__x__amihud", "h6_amihud_interaction"),
]

PORTFOLIOS = ["主測度 發文數｜等權・未篩選", "主測度 發文數｜等權・可交易性篩選後",
              "主測度 發文數｜雙重排序・小型股", "主測度 發文數｜雙重排序・大型股"]


def _read(name: str, spec: str) -> pd.DataFrame | None:
    path = OUT / f"{name}_{spec}.csv"
    return pd.read_csv(path) if path.exists() else None


def _round(x, n=8):
    return None if x is None or x != x else round(float(x), n)


def collect() -> dict:
    """由現有的稽核與輸出檔讀出當前值。**不重估任何模型。**"""
    snap: dict = {"panel": {}, "coef": {}, "portfolio": {}, "events": {},
                  "validity": {}, "decomposition": {}}

    # --- 面板形狀 ---
    for spec in SPECS:
        path = AUDIT / f"panel_summary_{spec}.csv"
        if not path.exists():
            continue
        r = pd.read_csv(path).iloc[0]
        snap["panel"][spec] = {
            "n_rows": int(r["n_rows"]), "n_tickers": int(r["n_tickers"]),
            "n_weeks": int(r["n_weeks"]),
            "first_week": str(r["first_week"]), "last_week": str(r["last_week"]),
            "n_dense_rows": int(r["n_dense_rows"]),
            "n_sparse_rows": int(r["n_sparse_rows"]),
            "n_silent_rows": int(r["n_silent_rows"]),
            "n_tier_missing": int(r["n_tier_missing"]),
            "n_with_ret_next": int(r["n_with_ret_next"]),
            "pct_zero_attention": _round(r["pct_zero_attention"], 4),
        }

    # --- 係數 ---
    for spec in SPECS:
        tables = {}
        for table, model, term, label in COEFS:
            if table not in tables:
                tables[table] = _read(table, spec)
            df = tables[table]
            if df is None:
                continue
            for inf in ("twoway_2cluster", "firm_fe_1cluster"):
                hit = df[(df["model"] == model) & (df["term"] == term)
                         & (df["inference"] == inf) & (df["status"] == "OK")]
                key = f"{spec}|{label}|{inf}"
                snap["coef"][key] = ({"coef": _round(hit.iloc[0]["coef"]),
                                      "t": _round(hit.iloc[0]["t"], 4),
                                      "n_obs": int(hit.iloc[0]["n_obs"])}
                                     if len(hit) else None)

    # --- 投資組合 ---
    for spec in SPECS:
        df = _read("T9_portfolios", spec)
        if df is None:
            continue
        for label in PORTFOLIOS:
            hit = df[(df["portfolio"] == label) & (df["status"] == "OK")]
            if not len(hit):
                continue
            r = hit.iloc[0]
            snap["portfolio"][f"{spec}|{label}"] = {
                "n_weeks": int(r["n_weeks"]),
                "gross": _round(r["mean_ls_gross_weekly"]),
                "t_gross": _round(r["t_gross"], 4),
                "net": _round(r["mean_ls_net_weekly"]),
                "t_net": _round(r["t_net"], 4),
            }

    # --- H7 事件數與判讀 ---
    for spec in SPECS:
        df = _read("T7_verdicts", spec)
        if df is None:
            continue
        for r in df.itertuples():
            snap["events"][f"{spec}|{r.event}"] = {
                "status": r.status, "n_events_raw": int(r.n_events_raw),
                "n_events_matched": int(r.n_events_matched),
                "n_week_clusters": int(r.n_week_clusters),
                "verdict": r.verdict,
            }

    # --- 測度效度（論文 Table 2 的對應項）---
    for spec in SPECS:
        df = _read("T13d_measure_validity", spec)
        if df is None:
            continue
        for r in df[df["status"] == "OK"].itertuples():
            snap["validity"][f"{spec}|{r.sample}|{r.measure}"] = {
                "corr_all_weekday": _round(r.corr_all_weekday, 4),
                "corr_all_weekend": _round(r.corr_all_weekend, 4),
                "n_obs": int(r.n_obs),
            }

    # --- T14 係數層三段分解 ---
    path = OUT / "T14_coefficient_decomposition.csv"
    if path.exists():
        df = pd.read_csv(path)
        steps = [c for c in df.columns
                 if c.startswith("Δ") and not c.endswith("的意義")]
        # 欄名是「Δ A′→B」這種非識別字，`itertuples` 會把它改名成 `_5`，
        # `_asdict()` 因此取不到值而全部變成 None——用 records 取。
        for d in df.to_dict(orient="records"):
            for step in steps:
                snap["decomposition"][f"{d['item']}|{d['inference']}|{step}"] = \
                    _round(d.get(step))
    return snap


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unknown"


def load_frozen() -> dict | None:
    if not TARGET.exists():
        return None
    ns: dict = {}
    try:
        exec(compile(TARGET.read_text(encoding="utf-8"), str(TARGET), "exec"), ns)
    except Exception as exc:                                # noqa: BLE001
        # 壞掉的凍結檔不得擋住重新凍結——否則要手動刪檔才救得回來。
        print(f"  ⚠ 既有凍結檔讀不出來（{type(exc).__name__}: {exc}），視為未凍結")
        return None
    return ns.get("EXPECTED")


def diff(current: dict, frozen: dict) -> list[str]:
    """逐鍵比對。**新增與消失都要報**——少了一格通常代表某個模型變成 SKIPPED。"""
    out = []
    for section in sorted(set(current) | set(frozen)):
        cur, old = current.get(section, {}), frozen.get(section, {})
        for key in sorted(set(cur) | set(old)):
            a, b = old.get(key, "（未凍結）"), cur.get(key, "（已消失）")
            if a != b:
                out.append(f"{section}.{key}\n    凍結：{a}\n    現值：{b}")
    return out


def _render(snap: dict) -> str:
    """一格一行的 Python 字面值。

    不用 `json`：JSON 的 `null` 不是合法的 Python 字面值，而 `expected_v2.py` 是被
    import 的模組。也不用 `pprint`：它的換行會把同一格拆成三行，diff 讀起來像亂碼
    ——**這個檔的唯一用途就是看 diff**。
    """
    lines = ["{"]
    for section in sorted(snap):
        lines.append(f"    {section!r}: {{")
        for key in sorted(snap[section]):
            lines.append(f"        {key!r}: {snap[section][key]!r},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def freeze(snap: dict) -> None:
    body = _render(snap)
    header = f'''"""v2 的實得數字（由 `python3 -m src.audit_expected --freeze` 產生，勿手改）。

凍結於 commit `{_git_head()}`，{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC。

這些**不是**不變量，是某一版程式跑出來的結果。`tests/test_expected_v2.py` 比對
現值與這裡的值，作用是抓**靜默變動**：改了程式而數字跟著變、卻沒人注意到。

測試失敗時，正確的順序是先問「這個變動是預期的嗎」，確認之後才重新凍結——
不是反過來。改資料、改門檻、改測度都會讓這裡整批改變，那時重凍是對的；
只改重構卻讓數字動了，那是 bug。
"""

EXPECTED = '''
    TARGET.write_text(header + body + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="凍結／比對 v2 的實得數字")
    ap.add_argument("--freeze", action="store_true",
                    help="寫入 tests/expected_v2.py。只在確認新數字正確之後跑。")
    args = ap.parse_args(argv)

    snap = collect()
    n = sum(len(v) for v in snap.values())
    print(f"[expected_v2] 現值 {n} 格"
          + "（" + "、".join(f"{k} {len(v)}" for k, v in snap.items()) + "）")

    frozen = load_frozen()
    if args.freeze:
        freeze(snap)
        print(f"  → {TARGET.relative_to(ROOT)}（commit {_git_head()}）")
        if frozen:
            d = diff(snap, frozen)
            print(f"  與前一版相差 {len(d)} 格")
        return 0

    if frozen is None:
        print("  尚未凍結；確認數字正確後跑 --freeze")
        return 1
    d = diff(snap, frozen)
    if not d:
        print("  與凍結值完全一致")
        return 0
    print(f"  ⚠ 與凍結值相差 {len(d)} 格：\n")
    for line in d[:40]:
        print(line)
    if len(d) > 40:
        print(f"  …另有 {len(d) - 40} 格")
    return 1


if __name__ == "__main__":
    sys.exit(main())
