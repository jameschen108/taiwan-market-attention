"""ptt.cc 直爬語料的串流讀取層（v2）。

語料為 `data/pttcc/stock_<年>.jsonl`，共 3.6 GB，**不可整檔載入**：本模組一律
以逐行 generator 供應，呼叫端自行聚合。

**時戳採用規則**（docs/PLAN_V2.md §1）：以 `date_ts` 為唯一時戳來源。`date_ts`
是文章 ID 內含的 unix 時戳（PTT 上不可竄改）渲染成台北當地時間的字串；`date`
來自文章頁面的可編輯表頭，僅供稽核比對，不得用於任何窗口指派。

所有 datetime 一律為**台北當地時間的 naive datetime**——窗口規則（交易時段、
週末）本身就定義在當地時間上，帶 tz 只會在週聚合時徒增轉換風險。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

TAIPEI = timezone(timedelta(hours=8))

#: 文章 ID 形如 M.1577839678.A.745，中段為 unix 時戳
ARTICLE_ID_TS = re.compile(r"^M\.(\d{9,11})\.A\.[0-9A-F]+$", re.IGNORECASE)

#: 結構性必要欄位；缺任一即視為壞列，明確報錯而非靜默略過。
#: `author_id` **不在其中**——看板系統發的選情報導、板務公告本來就無作者
#: （每年 1–3 篇）。依專案規則缺值維持缺值，空作者不得當成一個名為 "" 的帳號。
REQUIRED_FIELDS = ("article_id", "title", "date_ts")


@dataclass(frozen=True, slots=True)
class Comment:
    """一則推文。`timestamp` 為 None 代表原始 `time` 缺值——缺值維持缺值。"""

    index: int
    user_id: str
    tag: str            # 推 / 噓 / →
    timestamp: datetime | None
    content: str


@dataclass(frozen=True, slots=True)
class Article:
    article_id: str
    title: str
    body: str
    timestamp: datetime      # 由 date_ts 而來，台北當地時間
    header_date: datetime | None  # 頁面表頭的 date，僅供稽核
    author_id: str
    author_nickname: str
    ip: str
    location: str
    meta_recovered: bool
    n_push: int
    n_boo: int
    n_arrow: int
    n_comments: int
    comments: tuple[Comment, ...]
    source_file: str

    @property
    def has_author(self) -> bool:
        """系統發文無作者。獨立發文帳號數等測度必須先過這一關。"""
        return bool(self.author_id)

    @property
    def unix_ts(self) -> int:
        """article_id 內含的 unix 時戳；解析不出時由 timestamp 回推。"""
        m = ARTICLE_ID_TS.match(self.article_id)
        if m:
            return int(m.group(1))
        return int(self.timestamp.replace(tzinfo=TAIPEI).timestamp())


def _as_bool(value) -> bool:
    """容忍爬蟲把布林寫成字串。`bool("False")` 是 True，不能直接轉。"""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_comment(raw: dict) -> Comment:
    return Comment(
        index=int(raw.get("index") or 0),
        user_id=(raw.get("user_id") or "").strip(),
        tag=(raw.get("tag") or "").strip(),
        timestamp=_parse_dt(raw.get("time")),
        content=raw.get("content") or "",
    )


def _parse_article(doc: dict, source_file: str) -> Article:
    missing = [f for f in REQUIRED_FIELDS if not doc.get(f)]
    if missing:
        raise ValueError(f"{source_file}: {doc.get('article_id')} 缺必要欄位 {missing}")

    ts = _parse_dt(doc["date_ts"])
    if ts is None:
        raise ValueError(f"{source_file}: {doc['article_id']} 的 date_ts 無法解析")

    return Article(
        article_id=doc["article_id"],
        title=doc.get("title") or "",
        body=doc.get("content") or "",
        timestamp=ts,
        header_date=_parse_dt(doc.get("date")),
        author_id=doc.get("author_id") or "",
        author_nickname=doc.get("author_nickname") or "",
        ip=doc.get("ip") or "",
        location=doc.get("location") or "",
        meta_recovered=_as_bool(doc.get("meta_recovered")),
        n_push=int(doc.get("n_push") or 0),
        n_boo=int(doc.get("n_boo") or 0),
        n_arrow=int(doc.get("n_arrow") or 0),
        n_comments=int(doc.get("n_comments") or 0),
        comments=tuple(_parse_comment(c) for c in (doc.get("comments") or ())),
        source_file=source_file,
    )


def corpus_files(root: Path, years: Sequence[int] | None = None) -> list[Path]:
    """回傳語料檔清單。缺檔明確報錯，不靜默回傳空清單。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"找不到 PTT 語料目錄：{root}")
    files = sorted(root.glob("stock_*.jsonl"))
    if years is not None:
        wanted = {f"stock_{y}.jsonl" for y in years}
        files = [p for p in files if p.name in wanted]
        found = {p.name for p in files}
        if missing := sorted(wanted - found):
            raise FileNotFoundError(f"{root} 缺少語料檔：{missing}")
    if not files:
        raise FileNotFoundError(f"{root} 中沒有 stock_*.jsonl")
    return files


def iter_articles(
    root: Path,
    years: Sequence[int] | None = None,
    with_comments: bool = True,
) -> Iterator[Article]:
    """串流走訪語料。壞列明確報錯（缺值維持缺值，但結構損壞不容忍）。"""
    for path in corpus_files(root, years):
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{lineno} JSON 解析失敗") from exc
                if not with_comments:
                    doc["comments"] = ()
                yield _parse_article(doc, path.name)
