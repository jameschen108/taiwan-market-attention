"""JSONL 語料的隨機存取索引。

抽驗要依 `article_id` 回頭取原文，但語料是 3.6 GB 的 JSONL；每取一篇就全檔掃描
不可行，整檔載入更不可行。本模組建立 `article_id → (檔名, 位元組偏移, 長度)` 的
索引，之後以 `seek` 直接取單篇。

索引落在 `data/interim/pttcc_offsets.csv`，由語料重建，不進版控。
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from ..config import ROOT, load_settings, resolve

INDEX_PATH = ROOT / "data" / "interim" / "pttcc_offsets.csv"
LINE_ID = re.compile(rb'"article_id"\s*:\s*"([^"]+)"')


def build_index(root: Path, out: Path = INDEX_PATH) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["article_id", "file", "offset", "length"])
        for path in sorted(root.glob("stock_*.jsonl")):
            offset = 0
            with open(path, "rb") as src:
                for raw in src:
                    m = LINE_ID.search(raw[:200])
                    if m:
                        w.writerow([m.group(1).decode(), path.name, offset, len(raw)])
                        n += 1
                    offset += len(raw)
            print(f"  {path.name} 索引完成")
    print(f"  → {out.relative_to(ROOT)}（{n:,} 筆）")
    return n


class ArticleStore:
    """以偏移索引隨機存取單篇文章。"""

    def __init__(self, root: Path, index_path: Path = INDEX_PATH) -> None:
        self.root = Path(root)
        if not index_path.exists():
            raise FileNotFoundError(
                f"找不到索引 {index_path}；先跑 python3 -m src.ptt.index_jsonl")
        self._idx: dict[str, tuple[str, int, int]] = {}
        with open(index_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self._idx[row["article_id"]] = (
                    row["file"], int(row["offset"]), int(row["length"]))
        self._handles: dict[str, object] = {}

    def __contains__(self, article_id: str) -> bool:
        return article_id in self._idx

    def get(self, article_id: str) -> dict | None:
        loc = self._idx.get(article_id)
        if loc is None:
            return None
        name, offset, length = loc
        fh = self._handles.get(name)
        if fh is None:
            fh = open(self.root / name, "rb")
            self._handles[name] = fh
        fh.seek(offset)
        return json.loads(fh.read(length).decode("utf-8"))

    def close(self) -> None:
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()


def main() -> int:
    s = load_settings()
    print("[索引] 建立 pttcc 偏移索引")
    build_index(resolve(s["ptt"]["root"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
