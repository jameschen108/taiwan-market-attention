"""B∩C 語料：舊封存中**新語料也有**的那些文章（`PROJECT.md` §6.7）。

§6.7 要求把 B→C 拆成兩欄，不得併為一項：

```
B      舊封存 pttweb 全部文章
B∩C    舊封存 pttweb，但只留 article_id 也出現在新直爬 pttcc 的文章
C      新直爬 pttcc
```

- **B → B∩C ＝ 已刪文流失**：同一份語料、同一套定義，只是把官方站已刪、新語料
  拿不到的那些文章拿掉。差異純粹來自樣本組成，且**非隨機**
  （`audit/P0_corpus_integrity.md`）。
- **B∩C → C ＝ 測度改變**：同一批文章，換成由 ptt.cc 直爬的欄位重算。

輸出走既有的 `build_panel(source=...)` 路徑，**不新增 `specs` 區塊的規格**——四方
對照的規格只允許覆寫 source 與 sample，由 `tests/test_specs.py` 守住。B∩C 是
T14 的診斷用中繼面板，不是第五個規格。
"""

from __future__ import annotations

import sys

import pandas as pd

from ..config import ROOT

INTERIM = ROOT / "data" / "interim"
AUDIT = ROOT / "audit"
SOURCE = "pttweb_intersect"


def build(write: bool = True) -> pd.DataFrame:
    old = pd.read_parquet(INTERIM / "ptt_matches_pttweb.parquet")
    new_ids = set(pd.read_parquet(INTERIM / "ptt_matches_pttcc.parquet",
                                  columns=["article_id"])["article_id"])
    keep = old[old["article_id"].isin(new_ids)].copy()
    keep["source"] = SOURCE

    ts = pd.to_datetime(old["timestamp"])
    window = old[(ts >= "2019-01-01") & (ts < "2025-01-01")]
    stats = pd.DataFrame([{
        "rows_pttweb_all": len(old),
        "rows_pttweb_in_2019_2024": len(window),
        "rows_kept": len(keep),
        "rows_dropped_in_window": int((~window["article_id"].isin(new_ids)).sum()),
        "pct_dropped_in_window": round(
            float((~window["article_id"].isin(new_ids)).mean()), 4),
        "articles_pttweb_in_window": int(window["article_id"].nunique()),
        "articles_kept": int(keep["article_id"].nunique()),
    }])
    if write:
        keep.to_parquet(INTERIM / f"ptt_matches_{SOURCE}.parquet", index=False)
        stats.to_csv(AUDIT / "ptt_intersection_corpus.csv", index=False)
        print(f"  → data/interim/ptt_matches_{SOURCE}.parquet（{len(keep):,} 列）")
        print(f"  → audit/ptt_intersection_corpus.csv")
        print(stats.to_string(index=False))
    return keep


def main() -> int:
    print("[B∩C] 舊封存 ∩ 新直爬的文章集合")
    build()
    print("\n接著跑：python3 -m src.features.build --source pttweb_intersect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
