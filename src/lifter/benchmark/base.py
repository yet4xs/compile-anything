"""Benchmark lifter framework (Phase 4).

Uniform interface for lifting benchmark samples (ToolBench / API-Bank,
HumanEval / MBPP, Spider / BIRD, RTL tasks) into TaskIR:

    class BenchmarkLifter:
        def can_handle(self, sample) -> bool
        def lift(self, sample) -> Optional[Module]      # None = unsupported

`lift_with_reason` additionally returns a machine-readable reason when a
sample cannot be expressed — coverage accounting is a first-class output
of Phase 4 (docs/missing-skills.md), not an afterthought.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ...ir.taskir import Module


class BenchmarkLifter:
    name: str = "benchmark"
    dataset: str = ""

    def can_handle(self, sample: Dict) -> bool:
        raise NotImplementedError

    def lift(self, sample: Dict) -> Optional[Module]:
        raise NotImplementedError

    def lift_with_reason(self, sample: Dict) -> Tuple[Optional[Module], Optional[str]]:
        mod = self.lift(sample)
        if mod is None:
            return None, "unsupported"
        return mod, None


LIFTERS: List[BenchmarkLifter] = []


def register(cls):
    LIFTERS.append(cls())
    return cls


def lift_sample(sample: Dict) -> Tuple[Optional[Module], Optional[str], Optional[str]]:
    """Dispatch by schema sniffing. Returns (module, reason, lifter_name)."""
    for lifter in LIFTERS:
        if lifter.can_handle(sample):
            mod, reason = lifter.lift_with_reason(sample)
            return mod, reason, lifter.name
    return None, "no lifter can_handle", None
