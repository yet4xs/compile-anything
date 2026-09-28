"""Optimizer placeholder — v0.2+.

Planned passes (TaskIR -> TaskIR transformations, run after validation):
  - task fusion: merge adjacent nodes executed by the same executor class
  - dead-code elimination: drop nodes not reaching program.output
  - parallelization: expose width via reordering / MERGE insertion

v0.1 ships only the interface so the pipeline shape is fixed early.
"""
from __future__ import annotations

from typing import List

from ..ir.taskir import Module


class Pass:
    """Base class for TaskIR optimization passes."""
    name: str = "pass"

    def run(self, mod: Module) -> Module:
        raise NotImplementedError


class PassPipeline:
    def __init__(self, passes: List[Pass] | None = None):
        self.passes = passes or []

    def run(self, mod: Module) -> Module:
        for p in self.passes:
            mod = p.run(mod)
        return mod


def available_passes() -> List[str]:
    return []          # none implemented in v0.1
