"""Unified raw-sample schema (Phase 5A Task 3).

One shape for every benchmark sample entering the pipeline. Original data
is NEVER mangled: the untouched record rides along in `raw_payload` and
the typed fields (`trajectory` / `code` / `sql` / `rtl`) are filled per
task type. See data/schema/raw_sample.json for an instantiated exemplar.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

RAW_SAMPLE_KEYS = ["id", "source", "input_text", "trajectory", "code",
                   "sql", "rtl", "metadata", "raw_payload"]


def make_sample(id: str,
                source: str,
                input_text: Optional[str] = None,
                trajectory: Optional[list] = None,
                code: Optional[str] = None,
                sql: Optional[str] = None,
                rtl: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None,
                raw_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    s = {"id": id, "source": source, "input_text": input_text,
         "trajectory": trajectory, "code": code, "sql": sql, "rtl": rtl,
         "metadata": metadata or {}, "raw_payload": raw_payload or {}}
    return s


def to_lifter_input(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Map a unified sample onto the benchmark-lifter input schemas
    (src/lifter/benchmark/*). Pure field mapping — lifting logic is NOT
    duplicated here."""
    src = sample["source"]
    if src in ("toolbench", "toolbench_static", "apibank", "agentbench",
               "xlam"):
        return {"task_id": sample["id"],
                "instruction": sample.get("input_text") or "",
                "trajectory": [{"tool": c.get("tool", ""),
                                 "args": c.get("args", {})}
                                for c in (sample.get("trajectory") or [])]}
    if src in ("humaneval", "mbpp"):
        return {"task_id": sample["id"],
                "prompt": sample.get("input_text") or "",
                "code": sample.get("code") or ""}
    if src in ("spider", "bird"):
        return {"task_id": sample["id"],
                "db_id": (sample.get("metadata") or {}).get("db_id", ""),
                "question": sample.get("input_text") or "",
                "query": sample.get("sql") or ""}
    if src in ("verilogeval", "hdlbits", "rtl"):
        return {"task_id": sample["id"],
                "prompt": sample.get("input_text") or "",
                "rtl": sample.get("rtl") or "",
                "signals": (sample.get("metadata") or {}).get("signals", [])}
    raise ValueError(f"unknown source {src!r}")
