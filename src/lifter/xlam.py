"""xLAM -> TaskIR lifter.

Input record schema (xLAM func-calling style):
    {"task_id": str,
     "question": str,                    # natural-language task
     "steps": [str | {"name": str, "arguments": dict}, ...],
     "answer": str}

Steps are strings containing ```json blocks with call arrays
[{"name": "...", "arguments": {...}}] (the xLAM text convention) or already
structured dicts. The lifting core is shared with the benchmark lifters
(src/lifter/chain.py).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..ir.taskir import Module
from .chain import lift_trajectory


def _parse_steps(steps: List[Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for st in steps or []:
        if isinstance(st, dict):
            name = st.get("name") or st.get("tool") or ""
            args = st.get("arguments") or st.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            if name:
                calls.append({"name": name, "arguments": args})
            continue
        text = str(st)
        for m in re.finditer(r"```(?:json)?\s*(.+?)```", text, re.S):
            chunk = m.group(1).strip()
            try:
                payload = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            items = payload if isinstance(payload, list) else [payload]
            for it in items:
                if isinstance(it, dict) and (it.get("name") or it.get("tool")):
                    args = it.get("arguments") or it.get("args") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    calls.append({"name": it.get("name") or it.get("tool"),
                                  "arguments": args})
    return calls


def parse_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": rec.get("task_id") or rec.get("id") or "",
        "question": rec.get("question") or rec.get("query") or "",
        "calls": _parse_steps(rec.get("steps") or []),
        "answer": rec.get("answer") or "",
    }


def lift_record(rec: Dict[str, Any]) -> Module:
    parsed = parse_record(rec)
    mod = lift_trajectory(parsed["question"], parsed["calls"],
                          source="xlam", task_id=parsed["task_id"])
    if parsed["answer"]:
        for n in mod.program.nodes:
            if n.id == "%ans":
                n.params["reference"] = parsed["answer"][:200]
    return mod
