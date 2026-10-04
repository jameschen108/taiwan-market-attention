# 重製指南

從原始資料重建 v2 的每一張表。**每一條指令都實跑過**；跑不動的地方寫在 §5，
不以「應該可以」帶過。

研究設計的單一真相來源是 [`PROJECT.md`](PROJECT.md)，限制與命名紀律是
[`LIMITATIONS.md`](LIMITATIONS.md)，階段計劃是 [`docs/PLAN_V2.md`](docs/PLAN_V2.md)。
本文件只講**怎麼跑**。

---

## 0. 前置

```bash
python3 -m pip install -r requirements.txt
```

### 需要哪些資料

| 路徑 | 內容 | 版控 | 取得方式 |
|---|---|---|---|
| `data/pttcc/stock_<年>.jsonl` | 新直爬語料，3.06 GB | ✗ | `~/GItHub/ptt-stock-crawler` |
| `data/pttweb/batch-*/M.*.json` | 舊封存語料（規格 A′／B 需要） | ✗ | v1 專案，唯讀 |
| `data/raw/`、`data/twse/`、`data/external/` | 行情、三大法人、宇宙清單 | ✗ | v1 專案，唯讀 |
| `config/*.yaml` | 所有門檻 | ✓ | 本倉庫 |

`.gitignore` 把上面前三項排除在版控之外：語料體積過大，行情與宇宙是指向 v1 專案的
唯讀來源。**中間產物（`data/interim/`、`data/processed/`）一律可由上述重建**，
不需要備份。

### 兩條紀律

1. **門檻先鎖死再看結果**（`PROJECT.md` §0）。`config/settings.yaml` 裡的每一個數字
   都在跑出任何係數之前定案。跑完之後才調門檻，整份對照就失去意義。
2. **缺值維持缺值**。任何地方都不補零、不補均值。資料缺席時對應的模型標 `SKIPPED`
   並寫進 `audit/model_status*.csv`，不降級輸出。

---

## 1. 語料層（P0–P2）

```bash
# 語料完整性：checksum、月度覆蓋、新舊差集、已刪文剖面
python3 -m src.audit_corpus --only checksums,monthly,diff,deletion
python3 -m src.audit_corpus --only spotcheck        # 需連網，抽驗 40 篇已刪＋40 篇對照

# 配對層：文章 → ticker，留言繼承母文章的 ticker
python3 -m src.ptt.transform --source pttcc
python3 -m src.ptt.transform --source pttweb        # 規格 A′／B 需要
python3 -m src.audit_matching --sources pttcc,pttweb
python3 -m src.audit_categories --sources pttcc
```

產出 `data/interim/ptt_matches_{pttcc,pttweb}.parquet` 與
`ptt_comment_matches_pttcc.parquet`，稽核表在 `audit/`。

歸屬正確率的重驗（550 筆分層抽樣 → 機器判讀）：

```bash
python3 -m src.universe.review_sample --source pttcc
python3 -m src.universe.review_evidence --source pttcc
python3 -m src.universe.auto_verdict --source pttcc
```

---

## 2. 行情層

```bash
python3 -m src.market.normalize
```

由 `data/raw/finmind/`（未還原權值的價、三大法人、除權息）與 `data/twse/t86/`
重建 `data/interim/market_daily.parquet`（683,190 列、268 檔、2014-01 ~ 2025-03）
與 `trading_days.csv`，權值還原報告寫進 `audit/price_adjustment_report.csv`。

> **v2 不重新收集任何行情資料**：既有收集全部蓋過 2020–2024 主樣本與 2019 暖機期。
> `src/market/collect_*.py` 一律 cache-first，重跑不會重打 API。
>
> 這個進入點**在 P5 之前不存在**：`audit/P3_panel.md` 的重跑指令裡寫著它，但跑下去
> 不會報錯也不會做任何事。已補上，並驗證重建結果與既有檔案**逐欄完全一致**。

制度資料（P3.5，`LIMITATIONS.md` §12）：

```bash
python3 -m src.market.collect_disposition                  # 處置有價證券
python3 -m src.market.collect_day_trading --universe-only  # TWTB4U 當沖
```

---

## 3. 面板層（P3）

```bash
for s in C B A_prime; do python3 -m src.features.build --spec $s; done
```

| 規格 | 語料 | 期間 | 產出 |
|---|---|---|---|
| C | pttcc | 2020-01 ~ 2024-12 | `data/processed/panel_C.parquet`（64,980 × 260 × 260） |
| B | pttweb | 2020-01 ~ 2024-12 | `panel_B.parquet` |
| A′ | pttweb | 2015-05 ~ 2024-12 | `panel_A_prime.parquet`（123,333 列、504 週） |

規格只覆寫 `source` 與 `sample`，其餘門檻一律沿用（`tests/test_specs.py` 守住）。

**T14 需要的中繼面板**（B∩C ＝ 舊封存中新語料也有的文章，`LIMITATIONS.md` §14）：

```bash
python3 -m src.features.intersect_corpus
python3 -m src.features.build --source pttweb_intersect
```

B∩C **不是第五個規格**，不進 `config/settings.yaml` 的 `specs` 區塊。

---

## 4. 分析層（P4）

```bash
for s in C B A_prime; do
  python3 -m src.analysis.h1_main          --spec $s   # T3  H1 / H2 / H8 / 穩健性
  python3 -m src.analysis.h3_mechanism     --spec $s   # T5  H3a / H3b
  python3 -m src.analysis.h6_h5_joint      --spec $s   # T6  H6 ＋ H5（同表）
  python3 -m src.analysis.h7_events        --spec $s   # T7  起始事件 DiD
  python3 -m src.analysis.portfolios       --spec $s   # T9  投資組合
  python3 -m src.analysis.paper_comparison --spec $s   # T13 與原論文對照
  python3 -m src.analysis.t15_parallel     --spec $s   # T15 平行測度彙總
done

python3 -m src.analysis.h1_main --spec pttweb_intersect   # T14 係數層要用
python3 -m src.analysis.spec_comparison                   # T14 面板層 ＋ 係數層
```

**順序有兩處相依**，其餘可任意：

- `paper_comparison` 讀 `T9_portfolios_<spec>.csv` → 必須在 `portfolios` 之後。
- `t15_parallel` 讀 T3／T5／T6／T7／T9／T13d → 放最後。
- `spec_comparison` 的係數層讀四個規格的 `T3_h1_main_*.csv` → 四個都要先跑。

跑不動時不會靜默略過：舊語料沒有留言／帳號測度，A′ 與 B 的對應模型一律
`SKIPPED` 並在 `note` 欄寫明「此規格的語料無此測度」。

H4 補班日在 v2 事件數為零，`audit/model_status_*.csv` 記為 `SKIP`。
這是**事件為零**，不是檢定力不足（`LIMITATIONS.md` §3）。

---

## 5. 驗證

```bash
python3 -m pytest -q                      # 327 項
python3 -m src.audit_expected             # 比對 154 格凍結值，不寫檔
```

測試分兩類（`docs/PLAN_V2.md` §P5）：

| 類別 | 檔案 | 換資料時 |
|---|---|---|
| 邏輯不變量 | `test_core_logic` / `test_attention` / `test_panel` / `test_institutions` / `test_specs` / `test_analysis_p4` | **不得改變** |
| 寫死數字 | `tests/expected_v2.py`（154 格） | 整批改變，需重新凍結 |

重新凍結是**刻意的動作**：

```bash
python3 -m src.audit_expected --freeze    # 只在確認新數字正確之後
```

> 失敗時的處理順序是固定的：先問「這個變動是預期的嗎」。改了資料、門檻、測度 →
> 是。**只做了重構卻讓數字動了 → 那是 bug，不是該重凍的理由。**
> 反過來做會讓這一整組測試變成裝飾品。

### 已知跑不動或需要額外條件的項目

| 項目 | 狀況 |
|---|---|
| `audit_corpus --only spotcheck` | 需連網，且 ptt.cc 可能改版 |
| `collect_*` 首次執行 | 需 FinMind 金鑰（`.env`），之後 cache-first |
| 規格 A | v1 的既有發表結果，**本專案不重跑**，數字引自 v1 的 `output/` |
| `data/pttweb`、`data/raw`、`data/twse` | 指向 v1 專案；沒有 v1 就只跑得動規格 C 的一部分 |

---

## 6. 產出對照

| 表 | 檔案 | 說明 |
|---|---|---|
| T3 | `output/T3_h1_main_<spec>.csv` | H1 四窗口 × 三報酬 × 兩推論標準、H2、H8、兩項穩健性 |
| T5 | `output/T5_h3_mechanism_<spec>.csv` | H3a／H3b，含當沖控制與排除處置週 |
| T6 | `output/T6_joint_reading_<spec>.csv`、`T6_joint_verdicts_<spec>.csv` | H6 ＋ H5，**同檔輸出，不得分開報告** |
| T7 | `output/T7_{event_paths,balance,propensity,verdicts}_<spec>.csv` | 起始事件 DiD |
| T9 | `output/T9_portfolios_<spec>.csv`、`T9_portfolio_weekly_<spec>.csv` | 投資組合，成本前後並列 |
| T13 | `output/T13{a,b,c,d}_*_<spec>.csv`、`T13_paper_comparison_<spec>.md` | 與原論文逐項對照 |
| T14 | `output/T14_spec_decomposition.{csv,md}`、`T14_cell_level_B_vs_C.csv`、`T14_coefficient_decomposition.csv` | 四方對照（面板層 ＋ 係數層） |
| T15 | `output/T15_parallel_{measures,verdicts}_<spec>.csv` | 推文平行測度彙總 |

階段稽核：[`audit/P0_corpus_integrity.md`](audit/P0_corpus_integrity.md)、
[`audit/P2_matching.md`](audit/P2_matching.md)、
[`audit/P3_panel.md`](audit/P3_panel.md)、
[`audit/P3_5_institutions.md`](audit/P3_5_institutions.md)、
[`audit/P4_analysis.md`](audit/P4_analysis.md)。

---

## 7. 讀結果之前必須知道的四件事

1. **所有結果均為 `diagnostic`**（`PROJECT.md` §7）。`formal_main_return` 為 False
   的根因是 `news_count`、分析師覆蓋、四因子未取得；換語料不解除任何標記。
2. **推論母體只能寫「本樣本涵蓋之 267 檔個股」**，不得寫「台股」。
3. **每個主表都同時報兩種推論標準**。v2 的 H1 週末係數在兩者之間跨過 5% 門檻，
   只報其中一種會給出相反的結論（`LIMITATIONS.md` §13）。
4. **命名紀律**見 `LIMITATIONS.md` §17：不得寫 ASVI／SVI、散戶淨買超、可投資績效，
   也不得把 v2 新增的缺口證據線寫成「複製了論文的結果」。
