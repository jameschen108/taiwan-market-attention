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
from src.ptt.transform import _rows_for_article
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
    arows, crows = _rows_for_article(art, matcher, max_tickers=15,
                                     source="pttcc", want_comments=True)
    tickers = {r["ticker"] for r in arows}
    assert tickers == {"2330", "1101"}
    assert len(crows) == 3 * len(tickers)
    assert {r["ticker"] for r in crows} == tickers


def test_bulk_listing_flag_propagates_to_comments(matcher):
    """一篇列數十檔的程式選股文，其 bulk 旗標必須傳到留言，否則汙染更嚴重。"""
    # 真實的程式選股輸出是「代號 簡稱」成對，不是一串裸數字——一串裸數字會（正確地）
    # 被排行表欄位偵測擋掉，拿來當 bulk 的測資等於測錯規則。
    listing = "\n".join([
        "2330 台積電 買進", "1101 台泥 買進", "1102 亞泥 買進", "1103 嘉泥 買進",
        "1104 環泥 買進", "1108 幸福 買進", "1109 信大 買進", "1201 味全 買進",
        "1203 味王 買進", "1210 大成 買進", "1216 統一 買進", "1217 愛之味 買進",
        "1218 泰山 買進", "1219 福壽 買進", "1220 台榮 買進", "1225 福懋油 買進",
        "1227 佳格 買進",
    ])
    art = _article(listing, n_comments=2)
    arows, crows = _rows_for_article(art, matcher, max_tickers=15,
                                     source="pttcc", want_comments=True)
    assert arows[0]["n_tickers_in_article"] > 15
    assert all(r["is_bulk_listing"] for r in arows)
    assert crows and all(r["is_bulk_listing"] for r in crows)


def test_unmatched_article_produces_no_rows(matcher):
    art = _article("今天天氣很好，沒有提到任何個股")
    assert _rows_for_article(art, matcher, 15, "pttcc", True) == ((), ())


def test_missing_comment_timestamp_stays_missing(matcher):
    """缺值維持缺值：沒有時戳的留言不得以文章時戳補值。"""
    art = _article("2330 台積電 買進", n_comments=1)
    blank = Comment(index=1, user_id="u1", tag="推", timestamp=None, content="x")
    art2 = Article(
        article_id=art.article_id, title=art.title, body=art.body,
        timestamp=art.timestamp, header_date=None, author_id="p", author_nickname="",
        ip="", location="", meta_recovered=False, n_push=1, n_boo=0, n_arrow=0,
        n_comments=1, comments=(blank,), source_file="t.jsonl")
    _, crows = _rows_for_article(art2, matcher, 15, "pttcc", True)
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
    assert s["attention"]["lookback_weeks"] == 8
    assert s["sparsity"]["dense_min_nonzero_weeks"] == 40


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
