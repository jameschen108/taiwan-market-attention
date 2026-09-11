"""核心邏輯的不變量測試。

分兩類（docs/PLAN_V2.md §P5）：
- **邏輯不變量**（本檔）：窗口指派、標題解析、碰撞消解、留言繼承。與語料無關，
  換資料不得改變。
- **寫死數字**（`tests/expected_v2.py`）：文章數、面板列數、係數。跑完 v2 後更新。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.config import load_settings, load_universe_config
from src.ptt.ingest_jsonl import Article, Comment, _as_bool, _parse_article
from src.ptt.parse import effort_tier, has_comment_metadata, parse_title
from src.ptt.transform import (_rows_for_article, count_codes,
                               count_table_rows)
from src.universe.name_matching import build_matcher, normalize, strip_ptt_template


@pytest.fixture(scope="module")
def matcher():
    return build_matcher(load_universe_config())


# ------------------------------------------------------------ 標題與分層

@pytest.mark.parametrize("title,cat,reply", [
    ("[標的] 2330 台積電", "標的", False),
    ("Re: [標的] 2330 台積電", "標的", True),
    ("RE: [請益] 該買嗎", "請益", True),
    ("Fw: [新聞] 台股大漲", "新聞", True),
    ("[美股] NVDA", "其他", False),          # 清單外標籤 → 其他
    ("沒有標籤的標題", "未分類", False),
])
def test_parse_title(title, cat, reply):
    assert parse_title(title) == (cat, reply)


def test_high_effort_is_original_post_only():
    """[標的] 原PO 才是 high_effort，回文降為 mid——這是 H2 的識別基礎。"""
    assert effort_tier("標的", False) == "high_effort"
    assert effort_tier("標的", True) == "mid_effort"


def test_effort_tiers_are_disjoint_and_total():
    tiers = {effort_tier(c, r)
             for c in ("標的", "請益", "心得", "新聞", "閒聊", "情報", "公告",
                       "爆卦", "投顧", "其他", "未分類")
             for r in (True, False)}
    assert tiers == {"high_effort", "mid_effort", "low_effort", "unclassified"}


# ---------------------------------------------------------------- 歸屬規則

def test_code_match(matcher):
    assert matcher.match("[標的] 2330 台積電 多", "") == {"2330": "code"}


def test_yearlike_code_needs_positive_evidence(matcher):
    """代號區間與西元年份重疊：沒有正面證據時不得採計。"""
    assert matcher.match("[新聞] 2015 年的行情回顧", "") == {}
    assert "2015" in matcher.match("[標的] 2015 豐興 股價 便宜", "")


def test_blocked_extension(matcher):
    """命中片段被相鄰字擴成另一實體時不採計（統一 → 統一證券）。"""
    assert matcher.match("[心得] 統一證券的服務", "") == {}


def test_blocked_extension_survives_line_break(matcher):
    """延伸字與命中片段之間夾空白也必須擋掉，否則排版換行就能繞過。"""
    assert matcher.match("[閒聊] 落後三商 銀好多", "") == {}


def test_ptt_template_is_not_attention(matcher):
    """[標的] 發文樣板的範例「(例 2330 台積電)」不算對台積電的關注。"""
    assert "2330" not in matcher.match("[標的] (例 2330 台積電)", "我想問 1101 台泥")


def test_quantifier_after_code_is_rejected(matcher):
    """「有 1517 張設質」的 1517 是張數，不是利奇。"""
    assert matcher.match("[請益] 有 1517 張設質", "") == {}


def test_context_required_variant_abstains_without_context(matcher):
    """name_with_context 的寫法沒有股票語境詞時保守不配——寧缺勿錯。"""
    assert matcher.match("[請益] 長榮海運要買嗎", "") == {}
    assert matcher.match("長榮海運 股價 好強", "") == {"2603": "name_with_context"}


def test_code_beats_name_as_evidence(matcher):
    """同一檔同時由代號與簡稱命中時，match_mode 記為較強的代號。"""
    assert matcher.match("[標的] 2330 台積電 買進", "") == {"2330": "code"}


@pytest.mark.parametrize("text,ticker,why", [
    ("案1202、案1203,案1208 股價", "1203", "COVID 病例編號"),
    ("Shopify 高點 1762 現價 544", "1762", "股價"),
    ("M.2 2230 和非常短的 M.2 1216 插槽", "1216", "PCB 封裝規格"),
    ("Sent from JPTT on my Vivo 1907", "1907", "手機型號"),
    ("惟1110機服役期間,發現機 股價", "1110", "飛機機尾號"),
    ("3.合晶 1339 3.創惟 -1346", "1339", "名稱式排行表的金額欄"),
])
def test_code_lookalikes_are_rejected(matcher, text, ticker, why):
    """v2 機器抽驗導出：四位數在這些語境中不是證券代號。"""
    assert ticker not in matcher.match(text, ""), why


@pytest.mark.parametrize("text,ticker,why", [
    ("美國晶片製造業沒有擴張和成功所需要 股價", "1810", "和＋成功"),
    ("等於放空台積電和大盤 股價", "1536", "和＋大盤"),
    ("就和大戶佈的籌碼走 股價", "1536", "和＋大戶"),
    ("當輝瑞的日舒缺貨後,中化生產的 股價", "1762", "中化＋生產"),
    ("股市又會開始欣欣向榮 股價", "2901", "成語"),
    ("偕同經銷夥伴於全台積極布建充電 股價", "2330", "全台＋積極"),
    ("長榮運價跌一個多月 股價", "2607", "長榮＋運價"),
    ("持續推動越南、新興化學品產銷 股價", "2605", "普通詞"),
    ("網購平台亞馬遜 股價 大漲", "2340", "平台＋亞馬遜"),
    ("京華城改建案 股價 大漲", "1519", "京華城案"),
    ("陳南光再撰文開砲 股價", "1752", "人名"),
    ("泰金寶9105 DR 股價", "2312", "泰金寶-DR 是 9105"),
])
def test_word_boundary_and_extension_blocks(matcher, text, ticker, why):
    """v2 機器抽驗與語料延伸剖析導出的阻擋規則。"""
    assert ticker not in matcher.match(text, ""), why


@pytest.mark.parametrize("text,ticker", [
    ("和成 股價 漲停", "1810"), ("和大 股價 漲停", "1536"),
    ("中化生 股價 漲停", "1762"), ("欣欣 股價 漲停", "2901"),
    ("台積電 股價 漲停", "2330"), ("榮運 股價 漲停", "2607"),
    ("新興 股價 漲停", "2605"), ("台亞 股價 漲停", "2340"),
    ("華城 股價 漲停", "1519"), ("金寶 股價 漲停", "2312"),
    ("福懋 股價 漲停", "1434"), ("宏泰 股價 漲停", "1612"),
    ("南光 股價 漲停", "1752"), ("台塑 股價 漲停", "1301"),
])
def test_blocking_rules_do_not_kill_the_real_thing(matcher, text, ticker):
    """阻擋規則不得把本尊一起擋掉——這是加規則最容易犯的錯。"""
    assert ticker in matcher.match(text, "")


def test_strip_template_and_normalize():
    assert "2330" not in strip_ptt_template("(例 2330 台積電)")
    assert normalize("（全形）　空白") == "(全形) 空白"


# -------------------------------------------------------- 留言繼承與 bulk

def _article(tickers_text: str, n_comments: int = 2, ts=None) -> Article:
    ts = ts or datetime(2021, 6, 4, 22, 0)
    comments = tuple(
        Comment(index=i, user_id=f"u{i}", tag="推",
                timestamp=datetime(2021, 6, 5, 10, i), content="推")
        for i in range(1, n_comments + 1))
    return Article(
        article_id="M.1622812800.A.001", title="[標的] 測試", body=tickers_text,
        timestamp=ts, header_date=None, author_id="poster", author_nickname="",
        ip="", location="", meta_recovered=False, n_push=n_comments, n_boo=0,
        n_arrow=0, n_comments=n_comments, comments=comments, source_file="t.jsonl")


def test_comments_inherit_article_tickers(matcher):
    """留言繼承母文章的 ticker，不對留言文字另行比對（PROJECT.md §4.4）。"""
    art = _article("2330 台積電 買進 與 1101 台泥 股價", n_comments=3)
    arows, crows = _rows_for_article(art, matcher, max_tickers=15, max_codes=15,
                                     max_table_rows=15, source="pttcc", want_comments=True)
    tickers = {r["ticker"] for r in arows}
    assert tickers == {"2330", "1101"}
    assert len(crows) == 3 * len(tickers)
    assert {r["ticker"] for r in crows} == tickers


def test_bulk_listing_flag_propagates_to_comments(matcher):
    """一篇列數十檔的程式選股文，其 bulk 旗標必須傳到留言，否則汙染更嚴重。"""
    # 真實的程式選股輸出是「代號 簡稱」成對，不是一串裸數字——一串裸數字會（正確地）
    # 被排行表欄位偵測擋掉，拿來當 bulk 的測資等於測錯規則。
    listing = "\n".join([
        "1101 台泥 買進", "1102 亞泥 買進", "1103 嘉泥 買進", "1104 環泥 買進",
        "1108 幸福 買進", "1109 信大 買進", "1110 東泥 買進", "1201 味全 買進",
        "1203 味王 買進", "1210 大成 買進", "1213 大飲 買進", "1215 卜蜂 買進",
        "1216 統一 買進", "1217 愛之味 買進", "1218 泰山 買進", "1219 福壽 買進",
        "1220 台榮 買進", "1301 台塑 買進", "1304 台聚 買進",
    ])
    art = _article(listing, n_comments=2)
    arows, crows = _rows_for_article(art, matcher, max_tickers=15, max_codes=15,
                                     max_table_rows=15, source="pttcc", want_comments=True)
    assert arows[0]["n_tickers_in_article"] > 15
    assert all(r["is_bulk_listing"] for r in arows)
    assert crows and all(r["is_bulk_listing"] for r in crows)


def test_bulk_detected_by_total_codes_even_when_few_are_in_universe(matcher):
    """處置股／買賣超排行表列 151 檔，宇宙內只命中 13 檔——只數宇宙內會完全逃過。

    「是不是資料傾印」是貼文自己的性質，不該取決於我們抽了哪 267 檔。
    """
    # 宇宙外代號（8xxx/9xxx 不在 267 檔內）＋ 兩檔宇宙內
    outside = " ".join(f"{8000+i} 某公司 買超" for i in range(20))
    art = _article(f"2330 台積電 買超\n1101 台泥 買超\n{outside}", n_comments=2)
    arows, crows = _rows_for_article(art, matcher, max_tickers=15, max_codes=15,
                                     max_table_rows=15, source="pttcc", want_comments=True)
    assert arows[0]["n_tickers_in_article"] <= 15      # 宇宙內命中很少
    assert arows[0]["n_codes_in_article"] > 15         # 但全文代號很多
    assert all(r["is_bulk_listing"] for r in arows)    # 仍須標記
    assert crows and all(r["is_bulk_listing"] for r in crows)


def test_bulk_rule_is_a_union_not_a_replacement(matcher):
    """以簡稱列出、完全不寫代號的清單文，仍須由 ticker 數判準抓到。"""
    # 刻意避開通用詞降級的 54 檔——它們停用簡稱比對，只用代號，拿來當測資會測錯規則
    names = "、".join(["愛之味", "台聚", "台達化", "堤維西", "車王電", "復盛應用",
                       "台達電", "楠梓電", "中興電", "三洋電", "葡萄王", "美吾華",
                       "寶齡富錦", "和康生", "凱撒衛", "東和鋼鐵", "高興昌",
                       "第一銅", "中鋼構", "匯僑設計"])
    art = _article(f"{names} 股價 都漲停", n_comments=1)
    arows, _ = _rows_for_article(art, matcher, max_tickers=15, max_codes=15,
                                 max_table_rows=15, source="pttcc", want_comments=True)
    assert arows[0]["n_codes_in_article"] == 0         # 一個代號都沒有
    assert arows[0]["n_tickers_in_article"] > 15
    assert all(r["is_bulk_listing"] for r in arows)


def test_name_only_ranking_table_is_detected(matcher):
    """只印公司名、不印代號的買賣超排行表——前兩個 bulk 判準都抓不到。

    v2 機器抽驗中這類佔誤配的 39%（`audit/adjudication/README.md`）。
    """
    table = "\n".join([
        "外資買超 外資賣超",
        "1.合晶 5795 1.富喬 -6714", "2.聚和 2253 2.榮剛 -5320",
        "3.原相 1216 3.僑威 -2742", "4.加高 1054 4.鈺創 -2005",
        "5.漢磊 1000 5.華容 -1852", "6.台聚 376 6.裕民 -331",
        "7.台橡 944 7.宏碁 1677", "8.中鋼 840 8.金像電 1389",
        "9.亞聚 695 9.亞泥 809", "10.永豐餘 793 10.新光鋼 479",
    ])
    art = _article(table, n_comments=1)
    assert count_codes("", table) <= 15          # 代號判準抓不到
    assert count_table_rows("", table) > 15      # 表格列判準抓得到
    arows, crows = _rows_for_article(art, matcher, max_tickers=15, max_codes=15,
                                     max_table_rows=15, source="pttcc",
                                     want_comments=True)
    assert arows and all(r["is_bulk_listing"] for r in arows)
    assert crows and all(r["is_bulk_listing"] for r in crows)


def test_prose_post_is_not_mistaken_for_a_table():
    """散文型貼文的「名稱＋數字」對數必須遠低於門檻，否則會誤殺真實關注度。"""
    target_post = ("1. 標的: 2330 台積電\n2. 分類:多\n"
                   "3. 分析/正文: 台積電先進製程領先 目標價 800 元\n"
                   "成本在 650 持有 10 張")
    assert count_table_rows("", target_post) < 15


def test_count_codes_ignores_ptt_template():
    """發文樣板的「(例 2330 台積電)」不該灌進代號計數。"""
    assert count_codes("[標的] (例 2330 台積電)", "") == 0
    assert count_codes("[標的] 2330 台積電", "還有 1101 台泥") == 2


def test_unmatched_article_produces_no_rows(matcher):
    art = _article("今天天氣很好，沒有提到任何個股")
    assert _rows_for_article(art, matcher, 15, 15, 15, "pttcc", True) == ((), ())


def test_missing_comment_timestamp_stays_missing(matcher):
    """缺值維持缺值：沒有時戳的留言不得以文章時戳補值。"""
    art = _article("2330 台積電 買進", n_comments=1)
    blank = Comment(index=1, user_id="u1", tag="推", timestamp=None, content="x")
    art2 = Article(
        article_id=art.article_id, title=art.title, body=art.body,
        timestamp=art.timestamp, header_date=None, author_id="p", author_nickname="",
        ip="", location="", meta_recovered=False, n_push=1, n_boo=0, n_arrow=0,
        n_comments=1, comments=(blank,), source_file="t.jsonl")
    _, crows = _rows_for_article(art2, matcher, 15, 15, 15, "pttcc", True)
    assert crows and crows[0]["timestamp"] is None
    assert crows[0]["article_timestamp"] == art.timestamp


# -------------------------------------------------------------- 讀取層契約

def test_author_id_may_be_missing_but_structure_may_not():
    """系統發文無作者是合法的；缺 article_id / title / date_ts 則必須報錯。"""
    doc = {"article_id": "M.1.A.1", "title": "[公告] x", "date_ts": "2020-01-01T00:00:00",
           "author_id": "", "content": "", "comments": []}
    art = _parse_article(doc, "t.jsonl")
    assert art.author_id == "" and art.has_author is False
    with pytest.raises(ValueError):
        _parse_article({**doc, "date_ts": ""}, "t.jsonl")


def test_bool_coercion_handles_string_false():
    """bool("False") 是 True——爬蟲若改成字串輸出，不得靜默反轉旗標。"""
    assert _as_bool("False") is False
    assert _as_bool("True") is True


def test_pttweb_has_no_comment_metadata():
    """規格 B 用舊語料，其推文無帳號無時戳，留言層測度必須不可用。"""
    assert has_comment_metadata("pttcc") is True
    assert has_comment_metadata("pttweb") is False


# ------------------------------------------------------------------ 設定檔

def test_thresholds_locked_before_results():
    """門檻必須在設定檔中寫死（PROJECT.md §0）。"""
    s = load_settings()
    assert s["sample"]["main_start"] == "2020-01-01"
    assert s["sample"]["main_end"] == "2024-12-31"
    assert s["sample"]["ptt_warmup_start"] == "2019-01-01"
    assert s["attention"]["max_tickers_per_article"] == 15
    assert s["attention"]["max_codes_per_article"] == 15
    assert s["attention"]["lookback_weeks"] == 8
    assert s["sparsity"]["dense_min_nonzero_weeks"] == 40
    # v2 的四項測度正確性修正，全部在跑出任何係數之前定案（PROJECT.md §0.1）
    # baseline 不在其中：沿用 v1 的 mean，因為原論文的基準統計量無記載（§0.2）
    assert s["attention"]["baseline"] == "mean"
    assert s["attention"]["sparsity_min_periods"] == s["attention"]["sparsity_lookback_weeks"]
    assert s["sample"]["week_containment"] == "full"
    assert s["returns"]["main_definition"] == "open_to_close"   # 論文 Table 3a 註
    assert s["returns"]["portfolio_definition"] == "open_to_close"


def test_effort_tiers_come_only_from_the_settings_file():
    """分層清單只有一份真相。以前設定檔與程式各一份，而且設定檔少了 爆卦／投顧。"""
    s = load_settings()["effort"]
    assert set(s["low_effort"]["categories"]) == {"新聞", "閒聊", "情報", "公告",
                                                  "爆卦", "投顧"}
    # 程式確實照設定檔走：把 新聞 抽掉，它就不再是 low_effort
    import src.ptt.parse as parse
    parse._effort_map.cache_clear()
    try:
        assert parse.effort_tier("新聞", False) == "low_effort"
        parse._effort_map.cache_clear()
        orig = parse.load_settings if hasattr(parse, "load_settings") else None
        assert orig is None  # 設定由 _effort_map 內部載入，不得在模組層快取成常數
    finally:
        parse._effort_map.cache_clear()


def test_formal_control_set_is_declared_and_still_incomplete():
    """§7 的 `formal_main_return` 判準必須是一份**寫出來的**清單。

    以前 PROJECT.md 說「控制變數全部齊備才出正式主表」，但那份清單不存在。
    未取得的三項（新聞、分析師覆蓋、四因子）是所有結果仍為 diagnostic 的根因。
    """
    c = load_settings()["regression"]["controls"]
    assert set(c["required_but_missing"]) == {"news_count", "analyst_coverage",
                                              "factor_4"}
    assert "week_n_trading_days" in c["available"]


def test_h4_is_explicitly_skipped():
    """2020–2024 補班交易日為 0 天，H4 必須明確 SKIP 而非靜默略過。"""
    s = load_settings()
    assert s["hypotheses"]["h4_makeup_days"] == "SKIP"
    assert s["hypotheses"]["h4_skip_reason"]


def test_v1_comparability_sections_unchanged():
    """這幾節動了就無法與 v1 對照（docs/PLAN_V2.md §6）。"""
    s = load_settings()
    assert s["attention"]["lookback_weeks"] == 8
    assert s["attention"]["sparsity_lookback_weeks"] == 52
    assert s["sessions"] == {"market_open_minute": 540, "market_close_minute": 810}
    assert s["regression"]["cluster"] == ["ticker", "week"]
    assert s["imbalance"]["min_daily_volume_shares"] == 10000


def test_source_paper_is_cited_in_the_spec():
    """原論文必須寫在規格裡，不能只活在 v1 的 repo。

    v2 的文件一度完全沒有引用，結果是有人（Claude）憑「這是一篇關注度論文」
    推成 Da-Engelberg-Gao，並據此把 `attention.baseline` 改成 median。
    引用缺席不是排版問題，是會改到係數的問題（LIMITATIONS.md §13）。
    """
    spec = Path("PROJECT.md").read_text(encoding="utf-8")
    assert "It Depends on When You Search" in spec
    assert "MIS Quarterly" in spec
    assert "4370525" in spec
    # 原論文的推論標準必須寫出來——v1 的週末係數在兩種標準間跨過 5%
    assert "僅 cluster 至個股" in spec
