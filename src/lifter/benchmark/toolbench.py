"""ToolBench / API-Bank style tool-use samples -> TaskIR.

Input record schema:

    {"task_id": str,
     "instruction": str,
     "trajectory": [{"tool": "google_flight.search", "args": {...}}, ...],
     "answer": str}

(the xLAM {"question", "steps"} shape is accepted too). Concrete tool
names NEVER enter the TaskIR semantic part — they are lowered via
src/lifter/toolmap.py and preserved in meta.provenance only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ...ir.taskir import Module
from ..chain import lift_trajectory
from .base import BenchmarkLifter, register


def _extract_calls(sample: Dict) -> List[Dict]:
    calls: List[Dict] = []
    for step in sample.get("trajectory") or sample.get("steps") or []:
        if isinstance(step, dict):
            name = step.get("tool") or step.get("name") or ""
            args = step.get("args") or step.get("arguments") or {}
            if name:
                calls.append({"name": name, "arguments": args})
    return calls


@register
class ToolBenchLifter(BenchmarkLifter):
    name = "toolbench"
    dataset = "toolbench/api-bank"

    def can_handle(self, sample: Dict) -> bool:
        return ("instruction" in sample and "trajectory" in sample) or \
               ("question" in sample and "steps" in sample)

    def lift(self, sample: Dict, view: str = "execution") -> Optional[Module]:
        question = sample.get("instruction") or sample.get("question") or ""
        calls = _extract_calls(sample)
        if not question or not calls:
            return None
        return lift_trajectory(question, calls,
                               source=self.name,
                               task_id=sample.get("task_id", ""),
                               view=view)

    def lift_with_reason(self, sample: Dict,
                         view: str = "execution"
                         ) -> Tuple[Optional[Module], Optional[str]]:
        if not (sample.get("instruction") or sample.get("question")):
            return None, "missing instruction"
        if not _extract_calls(sample):
            return None, "empty trajectory"
        return self.lift(sample, view=view), None
