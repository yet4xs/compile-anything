"""BFCL V4 adapter — raw benchmark → EvalSample (evaluation only).

Real file formats (audited 2026-09-29 against the downloaded snapshot):
- case categories are JSONL: {id, question: [[turn messages]],
  function: [function defs]}; multi_turn adds initial_config /
  involved_classes / path / excluded_function; memory adds scenario;
  web_search cases have involved_classes (functions come from classes).
- BFCL_v4_format_sensitivity.json is NOT a case file: it is a single JSON
  doc mapping category -> [case ids] (format-sensitivity subset marker).
- possible_answer/*.json is JSONL: {id, ground_truth, [source]} where
  ground_truth is [{func: {param: [vals]}}] for single-call categories,
  a list of per-turn call-string lists for multi-turn, and final-answer
  strings + subquestion sources for web_search.

Ground truth NEVER enters the model input (tested).
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional

from ..schema import EvalSample

ROOT = pathlib.Path(__file__).resolve().parents[3]
BFCL_DIR = ROOT / "data" / "external_benchmarks" / "bfcl_v4"

LANG = {"simple_java": "java", "simple_javascript": "javascript",
        "simple_python": "python"}
CATEGORY_KIND = {
    "simple": "single", "live_simple": "single",
    "parallel": "parallel", "live_parallel": "parallel",
    "multiple": "multiple", "live_multiple": "multiple",
    "irrelevance": "irrelevance", "live_irrelevance": "irrelevance",
    "live_relevance": "irrelevance",
    "multi_turn_base": "multi_turn", "multi_turn_long_context": "multi_turn",
    "multi_turn_miss_func": "multi_turn", "multi_turn_miss_param":
    "multi_turn",
    "memory": "memory", "web_search": "web_search",
    "format_sensitivity": "format_sensitivity",
}


def _load_jsonl_or_doc(p: pathlib.Path):
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(l) for l in text.splitlines() if l.strip()]


def _category_of(stem: str) -> str:
    return stem.replace("BFCL_v4_", "")


def load(bfcl_dir: pathlib.Path = None) -> List[EvalSample]:
    d = pathlib.Path(bfcl_dir) if bfcl_dir else BFCL_DIR
    # ground truth index: id -> (ground_truth, source)
    gt: Dict[str, dict] = {}
    for pa in sorted((d / "possible_answer").glob("BFCL_v4_*.json")):
        for rec in _load_jsonl_or_doc(pa):
            if isinstance(rec, dict) and rec.get("id"):
                gt[rec["id"]] = rec

    format_sensitive_ids = set()
    samples: List[EvalSample] = []
    for f in sorted(d.glob("BFCL_v4_*.json")):
        cat = _category_of(f.stem)
        if cat == "format_sensitivity":
            doc = json.loads(f.read_text(encoding="utf-8"))
            for ids in doc.values():
                format_sensitive_ids.update(ids)
            continue
        for rec in _load_jsonl_or_doc(f):
            sid = rec.get("id", "")
            g = gt.get(sid, {})
            # instruction: flatten first-turn user messages only — later
            # turns and ground truth stay out of the model input
            turns = rec.get("question") or []
            first_user = ""
            for msg in (turns[0] if turns else []):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    first_user = msg.get("content", "")
                    break
            samples.append(EvalSample(
                benchmark="bfcl_v4",
                case_id=sid,
                instruction=first_user,
                capabilities=rec.get("function"),
                conversation=turns if len(turns) > 1 else None,
                reference_actions=g.get("ground_truth"),
                reference_answer=(g.get("ground_truth")[0]
                                  if cat == "web_search"
                                  and isinstance(g.get("ground_truth"), list)
                                  and g and not isinstance(
                                      g.get("ground_truth")[0], (dict, list))
                                  else None),
                metadata={
                    "category": cat,
                    "kind": CATEGORY_KIND.get(cat, cat),
                    "language": LANG.get(cat, "python"),
                    "initial_config": rec.get("initial_config"),
                    "involved_classes": rec.get("involved_classes"),
                    "scenario": rec.get("scenario"),
                    "source": g.get("source"),
                    "format_sensitive": sid in format_sensitive_ids,
                }))
    return samples


def rejects() -> List[str]:
    """Machine-readable rejection reasons (kept empty by construction —
    every case-file record normalizes; format_sensitivity is an id-list
    doc, not a case, and is recorded in metadata instead)."""
    return []
