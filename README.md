# 台股長尾關注度（v2）

把 **Li, Liu, Ye, Zhao & Zhao, *"It Depends on When You Search"*（MIS Quarterly、
[SSRN 4370525](https://ssrn.com/abstract=4370525)）** 的「非交易時段關注度預測次週
報酬」研究設計移植到台股，聚焦**長尾個股**——原論文的樣本是 S&P 500（2004–2019），
台灣長尾個股的邊界條件未被檢驗過。刻意偏離原論文之處逐項列於
[`PROJECT.md`](PROJECT.md) §0.2。

v2 是 [`taiwan-attention-long-tail`](../taiwan-attention-long-tail)（v1）用**新語料**
重做：改用 ptt.cc 直爬，換到帶留言帳號與時戳的資料，期間縮短為 2020–2024。

| 文件 | 內容 |
|---|---|
| [`PROJECT.md`](PROJECT.md) | **研究設計與變數定義的單一真相來源**。程式與它不一致時改程式 |
| [`LIMITATIONS.md`](LIMITATIONS.md) | 已知限制。引用任何結果都必須同時引用本文件 |
| [`FINDINGS.md`](FINDINGS.md) | 主要發現。**骨架**：§3（事前／事後界線）、§4（主結果）、§5（窗口定義）、§6（機制）已成稿，其餘待寫 |
| [`REPLICATION.md`](REPLICATION.md) | 從原始資料重建每一張表。每條指令都實跑驗證過 |
| [`docs/PLAN_V2.md`](docs/PLAN_V2.md) | v2 的工作計劃（參考，不是規格） |
| [`audit/`](audit/) | 全部稽核產出 |

---

## 目前狀態

| 階段 | 狀態 |
|---|---|
| P0 語料落地與完整性稽核 | ✅ 完成，見 [`audit/P0_corpus_integrity.md`](audit/P0_corpus_integrity.md) |
| P1 讀取層（雙來源並存） | ✅ 完成 |
| P2 配對層 | ✅ 完成。歸屬正確率**機器抽驗**兩輪：修正前 93.5% → 修正後 **95.5%**，見 [`audit/adjudication/README.md`](audit/adjudication/README.md) |
| P3 特徵層 | ✅ 完成，面板 **64,980 列 × 260 檔 × 260 週**，見 [`audit/P3_panel.md`](audit/P3_panel.md) |
| P3.5 台股制度資料 | ✅ 完成：處置股 156 筆／64 檔、當沖 338,194 列／1,458 交易日，見 [`audit/P3_5_institutions.md`](audit/P3_5_institutions.md) |
| P4 分析層 | ✅ 完成：H1／H2／H3／H5／H6／H7／投資組合／T13／T14／T15 跑在 A′・B・C 三個規格上，H4 明確 SKIP，見 [`audit/P4_analysis.md`](audit/P4_analysis.md) |
| P5 稽核／測試／文件 | ✅ **327 項測試通過**；`expected_v2.py` 已凍結 154 格；`LIMITATIONS.md` 同步至 §17；[`REPLICATION.md`](REPLICATION.md) 建立（21 個 CLI 逐條實跑驗證） |
| P6 交付 | 🚧 [`FINDINGS.md`](FINDINGS.md) 骨架完成（B 版機制主線，9 節）；**§3–§6 已成稿**，其餘 5 節待寫 |

### P4 的三個核心結果

1. **H1 的週末係數在兩種推論標準之間跨過 5% 門檻，三個規格都一樣**——雙重 cluster
   下 t = 1.26～1.63，論文標準下 t = 2.68～3.49。這是推論標準造成的，不是資料。
2. **最強的發現不在主規格而在缺口，而且要用交易時段切法才乾淨**：
   `ret_gap_next`（週五收盤 → 週一開盤）對**非交易時段**關注度的係數在
   3 規格 × 主表／排除漲跌停 × 2 種推論標準的**十二格全部 1% 顯著**，交易時段窗口
   則一格都不顯著。同樣的缺口效果在論文的週間／週末切法下，排除漲跌停後會跌破 5%。
   新的是**應變數的三段拆解**（v1 沒有 `ret_gap_next`），不是切法——交易時段窗口
   v1 就有。不得寫成「複製了論文的某某結果」。
3. **T14 把 B→C 拆成兩欄之後，兩個成分方向相反**：已刪文流失讓週末係數變大
   （+0.000074）、直爬測度讓它變小（−0.000206）。合併成一項會得到「換語料幾乎
   沒影響」的錯誤印象。

### P2 的核心結果

以母文章時戳指派窗口（v1 唯一能做的事）會**低估週末關注度 22.4%**，且位移是
單向的——平日發文→週末留言是反向的 3.9 倍。見 [`audit/P2_matching.md`](audit/P2_matching.md) §3。

### P4 開跑前的規格複核（2026-09-11）

四項測度正確性修正在看到任何係數之前定案，見 [`PROJECT.md`](PROJECT.md) §0.1：
52 週窗 `min_periods` 1 → **52**、週涵蓋改為**完整落在樣本期內**（262 → 260 週）、
bulk 偵測加上全文代號判準、zero-base 虛擬變數改為**逐窗口**（週末窗口覆蓋率
58.2% → 100%）。四項都與原論文的建構無關。

**原論文的建構細節已逐項自 PDF 核對並寫入 §0.2**（ASVI 公式、8 週平均基準、
Monday open → Friday close 的應變數、個股 FE、僅個股 cluster、控制變數集合、
Table 2 相關性）。過程中回退了兩項誤改——異常值基準與報酬定義，兩者都曾被改成與
論文相反的設定，見 [`LIMITATIONS.md`](LIMITATIONS.md) §13。

同時新增四條限制：§11（主要自變數在建構上接近二元）、§12（處置股／當沖／漲跌停
三個台股制度混淆）、§13（原論文的建構細節不得靠推論補）、§14（三方對照需補 A′）。

歸屬正確率重抽驗**已完成**（機器抽驗，95.5%）。判讀者是 LLM 而非人工，殘餘誤差集中在 `name_with_context` 模式（87.9%），限制見 `LIMITATIONS.md` §6.3。

---

## P0 的主要發現

新語料在文章層是舊封存的**真子集**，而且缺的部分不是隨機的。

| | 2020–2024 |
|---|---:|
| 舊封存 pttweb.cc | 167,142 篇 |
| 新直爬 ptt.cc | 143,637 篇（85.9%） |
| 只在舊封存（官方已刪文） | **23,505 篇（14.06%）** |
| 只在新直爬 | **0 篇** |

抽樣向 ptt.cc 實際請求確認：40/40 回 404，正向對照 40/40 回 200。刪文集中在
`請益`（2.28x）與`心得`（1.91x），使 `mid_effort` 只保留 75.1%、`low_effort`
保留 92.7%。

這反轉了計劃的預期：原本要查的是「鏡像站有沒有缺漏」，答案是**沒有**；有完整性
問題的是新語料。詳見 [`audit/P0_corpus_integrity.md`](audit/P0_corpus_integrity.md)。

---

## 資料

| 路徑 | 內容 | 版控 |
|---|---|---|
| `data/pttcc/stock_<年>.jsonl` | 新語料，2019–2024，158,395 篇 / 1,563 萬則留言 | ✗（3.6 GB） |
| `data/pttweb/` → v1 | 舊鏡像封存，供規格 B | ✗（symlink） |
| `data/raw/`、`data/twse/`、`data/external/` → v1 | 行情、三大法人、除權息、減資 | ✗（symlink） |
| `data/universe_267.csv`、`config/universe.yaml` | 研究宇宙 | ✓ |

行情資料涵蓋 2014-01 ~ 2025-03，全部蓋過 v2 期間，**未重新收集**。

## 執行

```bash
pip install -r requirements.txt

# P0：語料完整性稽核
python3 -m src.audit_corpus --only checksums,monthly,diff,deletion
python3 -m src.audit_corpus --only spotcheck        # 需連網

# 分類分布（兩個來源各一次）
python3 -m src.audit_categories --sources pttcc,pttweb

# P2：配對層
python3 -m src.ptt.transform --source pttcc         # 主規格（規格 C）
python3 -m src.ptt.transform --source pttweb        # 規格 A′／B

# P3：特徵層與面板
python3 -m src.features.build --source pttcc

# 測試
python3 -m pytest tests/ -q
```

補爬語料（爬蟲在另一個 repo）：

```bash
python3 ptt_stock_crawler.py --from 2019 --to 2019 -w 8 -d 0.3 -f jsonl -o <本專案>/data/pttcc/stock_2019.jsonl
```

---

## 命名紀律

| 不得寫 | 必須寫 |
|---|---|
| ASVI／SVI | 異常 PTT 關注度 |
| retail order imbalance／散戶淨買超 | 非三大法人訂單失衡 |
| 台股／台灣上市公司 | 本樣本涵蓋之 267 檔個股 |
| 可投資績效 | ex-post universe 的機械年化 |
| PTT 全部討論 | 本語料所涵蓋、且截至爬取時未被刪除的貼文 |

所有結果均為 **diagnostic**：`formal_main_return` 為 False（缺 `news_count`、
分析師覆蓋、四因子等授權資料）。換語料不解除此標記。
