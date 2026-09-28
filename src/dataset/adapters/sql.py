"""SQL dataset adapters: Spider (official train/dev), BIRD (gated)."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

from ..schema import make_sample


class SpiderAdapter:
    sources = ["spider"]

    def load(self, raw_dir: pathlib.Path,
             max_per_split: int = 0) -> List[Dict[str, Any]]:
        root = raw_dir / "spider"
        out: List[Dict[str, Any]] = []
        for fname in ("train_spider.json", "dev.json"):
            f = root / fname
            if not f.exists():
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            split = "train" if "train" in fname else "dev"
            for i, rec in enumerate(data):
                out.append(self.normalize(rec, f"{split}-{i}", split))
                if max_per_split and i + 1 >= max_per_split:
                    break
        return out

    def normalize(self, rec: Dict[str, Any], sid: str,
                  split: str) -> Dict[str, Any]:
        question = rec.get("question") or ""
        query = rec.get("query") or ""
        if not question or not query:
            return {}
        return make_sample(
            id=f"spider-{sid}", source="spider",
            input_text=question, sql=query,
            metadata={"db_id": rec.get("db_id", ""), "split": split,
                      "sql": rec.get("sql", {}) if isinstance(
                          rec.get("sql"), dict) else {}},
            raw_payload={"question": question, "query": query,
                         "db_id": rec.get("db_id", "")})


class BirdAdapter:
    sources = ["bird"]

    def load(self, raw_dir: pathlib.Path) -> List[Dict[str, Any]]:
        root = raw_dir / "bird"
        if not root.is_dir():
            return []
        out = []
        for f in sorted(root.rglob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for i, rec in enumerate(data if isinstance(data, list) else []):
                s = self.normalize(rec, f"{f.stem}-{i}")
                if s:
                    out.append(s)
        return out

    def normalize(self, rec: Dict[str, Any], sid: str) -> Dict[str, Any]:
        question = rec.get("question") or ""
        query = rec.get("SQL") or rec.get("query") or ""
        if not question or not query:
            return {}
        return make_sample(id=f"bird-{sid}", source="bird",
                           input_text=question, sql=query,
                           metadata={"db_id": rec.get("db_id", "")},
                           raw_payload={"question": question, "SQL": query})
