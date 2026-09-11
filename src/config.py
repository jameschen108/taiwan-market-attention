"""設定載入。所有門檻集中於 config/settings.yaml，程式不得內嵌魔術數字。"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=None)
def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=None)
def load_universe_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "config" / "universe.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str | Path) -> Path:
    """把設定檔中的相對路徑解成專案絕對路徑。"""
    rel = Path(rel)
    return rel if rel.is_absolute() else ROOT / rel
