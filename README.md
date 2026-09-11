# 台股長尾關注度（v2）

用 PTT 股板的關注度預測次週報酬，聚焦**長尾個股**——原論文（SVI → 次週報酬）的
研究對象是有分析師覆蓋的大型股，台灣長尾個股的邊界條件未被檢驗過。

v2 是 [`taiwan-attention-long-tail`](../taiwan-attention-long-tail)（v1）用**新語料**
重做：改用 ptt.cc 直爬，換到帶留言帳號與時戳的資料，期間縮短為 2020–2024。

| 文件 | 內容 |
|---|---|
| [`PROJECT.md`](PROJECT.md) | **研究設計與變數定義的單一真相來源**。程式與它不一致時改程式 |
| [`LIMITATIONS.md`](LIMITATIONS.md) | 已知限制。引用任何結果都必須同時引用本文件 |
| [`docs/PLAN_V2.md`](docs/PLAN_V2.md) | v2 的工作計劃（參考，不是規格） |
| [`audit/`](audit/) | 全部稽核產出 |

---

## 目前狀態

| 階段 | 狀態 |
|---|---|
| P0 語料落地與完整性稽核 | ✅ 完成，見 [`audit/P0_corpus_integrity.md`](audit/P0_corpus_integrity.md) |
| P1 讀取層（雙來源並存） | ✅ 完成 |
| P2 配對層 | 🚧 pttcc 跑完（122,562 文章列／1,399 萬留言列），見 [`audit/P2_matching.md`](audit/P2_matching.md)；歸屬正確率重抽驗未做 |
| P3 特徵層 | 🚧 關注度與窗口完成，面板組裝未做 |
| P4 分析層 | ⬜ 未開始 |
| P5 稽核／測試／文件 | 🚧 46 項測試通過；`expected_v2.py` 未建 |

### P2 的核心結果

以母文章時戳指派窗口（v1 唯一能做的事）會**低估週末關注度 22.4%**，且位移是
單向的——平日發文→週末留言是反向的 3.9 倍。見 [`audit/P2_matching.md`](audit/P2_matching.md) §3。
| P6 交付 | ⬜ 未開始 |

**在歸屬正確率重抽驗完成前，任何係數都不得引用**（`LIMITATIONS.md` §6.3）。

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
python3 -m src.ptt.transform --source pttweb        # 規格 A／B

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
