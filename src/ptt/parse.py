"""PTT 股板文章解析：標題分類、回文判定、認知投入分層。

**雙來源並存**（docs/PLAN_V2.md §P1）。`config/settings.yaml` 的 `ptt.source`
決定讀哪一份語料：

| source | 語料 | 期間 | 留言欄位 |
|---|---|---|---|
| `pttcc` | `data/pttcc/stock_<年>.jsonl` | 2019–2024 | 有 `user_id`、分鐘級時戳 |
| `pttweb` | `data/pttweb/batch-*/M.*.json` | 2015-04 ~ 2025-01 | 只有內容字串 |

兩個讀取器都必須保留：`docs/PLAN_V2.md` §3.4 的規格 B（舊語料、新期間）是把
「期間效果」與「資料效果」拆開的唯一辦法，而規格 B 要用 `pttweb`。

分類清單以新語料重新統計後**沿用 v1 原清單**：新語料中 `effort` 三層涵蓋 96.0%
的文章，清單外的標籤合計不足 0.1%，改動清單只會破壞與 v1 的可對照性。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .ingest_jsonl import Article, Comment
from .ingest_jsonl import iter_articles as _iter_pttcc

# 依實得語料確認；新語料重新統計的分布見 audit/ptt_category_distribution.csv
CATEGORIES = (
    "標的|新聞|請益|心得|閒聊|公告|問卷|其他|情報|創作|爆卦|投顧|討論|活動|贈送"
)
TITLE_PATTERN = re.compile(
    rf"^(?P<prefixes>(?:(?:Re|Fw|RE|FW):\s*)*)\[(?P<category>{CATEGORIES})\]"
)
ANY_TAG = re.compile(r"^(?:(?:Re|Fw|RE|FW):\s*)*\[(?P<category>[^\]]{1,6})\]")
REPLY_PREFIX = re.compile(r"^(?:(?:Re|Fw|RE|FW):\s*)+")

PTTWEB_ID_TS = re.compile(r"^M\.(\d{9,11})\.A\.[0-9A-F]+$", re.IGNORECASE)


def parse_title(title: str) -> tuple[str, bool]:
    """回傳 (category, is_reply)。容許 Re:／Fw: 前綴（回文佔股板貼文大宗）。"""
    title = (title or "").strip()
    m = TITLE_PATTERN.match(title)
    if m:
        return m.group("category"), bool(m.group("prefixes"))
    is_reply = bool(REPLY_PREFIX.match(title))
    if ANY_TAG.match(title):
        return "其他", is_reply
    return "未分類", is_reply


def effort_tier(category: str, is_reply: bool) -> str:
    """認知投入分層（PROJECT.md §3）。

    `high_effort` 嚴格限定 `[標的]` **原PO**（需論述與理由）；`[標的]` 回文歸 mid。
    v2 的推文雖有時戳與帳號，主規格的 effort 分層仍只用文章，以維持與 v1 可對照；
    推文另建平行測度（PROJECT.md §2.4）。
    """
    if category == "標的":
        return "mid_effort" if is_reply else "high_effort"
    if category in ("請益", "心得"):
        return "mid_effort"
    if category in ("新聞", "閒聊", "情報", "公告", "爆卦", "投顧"):
        return "low_effort"
    return "unclassified"


# ------------------------------------------------------------ pttweb 讀取器

def _iter_pttweb(root: Path) -> Iterator[Article]:
    """走訪 data/pttweb/batch-*/M.*.json（v1 鏡像封存）。

    鏡像沒有作者、沒有留言帳號與時戳。這些欄位一律留空／None，**不得以文章時戳
    回推推文時間**——`Comment.timestamp` 維持 None，下游窗口測度自然把它排除。
    """
    root = Path(root)
    batches = sorted(p for p in root.iterdir() if p.is_dir())
    if not batches:
        raise FileNotFoundError(f"找不到 PTT 鏡像封存：{root}")
    for batch in batches:
        for path in sorted(batch.glob("M.*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"PTT 封存解析失敗：{path}") from exc

            aid = doc.get("article_id") or path.stem
            ts = doc.get("timestamp")
            if ts is None:
                m = PTTWEB_ID_TS.match(aid)
                if not m:
                    raise ValueError(f"{path}: 無 timestamp 且 article_id 無法回推")
                ts = int(m.group(1))
            when = datetime.fromtimestamp(int(ts))

            pushes = doc.get("pushes") or ()
            comments = tuple(
                Comment(index=i, user_id="", tag="", timestamp=None, content=str(c))
                for i, c in enumerate(pushes, 1)
            )
            yield Article(
                article_id=aid,
                title=doc.get("title") or "",
                body=doc.get("body") or "",
                timestamp=when,
                header_date=None,
                author_id="",
                author_nickname="",
                ip="",
                location="",
                meta_recovered=False,
                n_push=0,
                n_boo=0,
                n_arrow=0,
                n_comments=len(comments),
                comments=comments,
                source_file=str(path.relative_to(root)),
            )


# ------------------------------------------------------------------ 分派

def iter_articles(
    source: str,
    root: Path,
    years: list[int] | None = None,
    with_comments: bool = True,
) -> Iterator[Article]:
    """依 `ptt.source` 分派到對應讀取器。兩者產出同一個 `Article` 型別。"""
    if source == "pttcc":
        yield from _iter_pttcc(Path(root), years=years, with_comments=with_comments)
    elif source == "pttweb":
        yield from _iter_pttweb(Path(root))
    else:
        raise ValueError(f"未知的 ptt.source：{source!r}（可用：pttcc / pttweb）")


def has_comment_metadata(source: str) -> bool:
    """該來源的推文是否帶帳號與時戳。pttweb 一律為 False。"""
    return source == "pttcc"
