"""Scheduler placeholder — v0.2+.

Will decide, per node: which executor unit runs it and when (instruction
scheduling over heterogeneous functional units). v0.1 keeps nodes unbound;
the simulator executes everything on mock executors.

Planned interface:

    class ExecutorBinding:
        skill: str                 # semantic skill name
        executor: str              # e.g. "lm:qwen2b-instruct", "python:fs"
        est_latency_ms: float
        est_cost: Cost

    class Scheduler:
        def bind(self, mod: Module) -> Dict[str, ExecutorBinding]: ...
        def order(self, mod: Module) -> List[str]: ...   # ready-list order
"""
from __future__ import annotations
