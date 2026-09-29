"""ToolBench (full 124k) adapter — evaluation only.

Real record shape (audited): {id, tools: [...], conversations: [
{from: system|user|assistant|tool, value: str}]}. Assistant turns are
ReAct text ("Thought: ... Action: X Action Input: {...}") — calls are
parsed from them; tool turns are environment responses."""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Dict, List

from ..schema import EvalSample

ROOT = pathlib.Path(__file__).resolve().parents[3]
TB_DIR = ROOT / "data" / "external_benchmarks" / "toolbench_full"

_REACT_RE = re.compile(
    r"Action:\s*([A-Za-z0-9_.\-]+)\s*\nAction Input:\s*(\{.*?\})\s*(?=\n|$)",
    re.S)


def _extract_calls(messages: List[Any]) -> List[Dict]:
    calls = []
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict) or m.get("from") != "assistant":
            continue
        for name, argstr in _REACT_RE.findall(m.get("value") or ""):
            try:
                args = json.loads(argstr)
            except json.JSONDecodeError:
                args = {"raw": argstr[:200]}
            calls.append({"name": name,
                          "arguments": args if isinstance(args, dict)
                          else {"raw": argstr[:200]}})
    return calls


def load(limit: int = 0, tb_dir: pathlib.Path = None) -> List[EvalSample]:
    d = pathlib.Path(tb_dir) if tb_dir else TB_DIR
    out: List[EvalSample] = []
    with open(d / "toolbench.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            convs = r.get("conversations") or []
            first_user = next((m.get("value", "") for m in convs
                               if isinstance(m, dict)
                               and m.get("from") == "user"), "")
            out.append(EvalSample(
                benchmark="toolbench_full",
                case_id=str(r.get("id", i)),
                instruction=first_user,
                capabilities=r.get("tools"),
                conversation=convs,
                reference_actions=_extract_calls(convs),
                metadata={"official_metrics": ["pass^1 / win_rate "
                                               "(official ToolBench eval)"],
                          "env_required": "real API replay for official metric"}))
    return out
