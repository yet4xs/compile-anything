"""RTL / EDA task samples -> TaskIR.

Input record schema:
    {"task_id": str,
     "prompt": str,                        # the RTL task in natural language
     "rtl": str,                           # verilog-ish module snippet
     "signals": ["fifo_count", "depth"],   # relevant signals (optional)
     "failing_check": "assert ...",        # assertion/lint finding (optional)
     "timing": {"slack_ns": 0.3} }         # timing numbers (optional)

Lifting shape (mirrors data/taskir/examples/rtl_debug.json):
    LOAD(design artifacts) -> EXTRACT(signals) [-> CALCULATE(slack) if timing]
    -> SEARCH(known_bug_db) -> CODEGEN(fix) --retry--> VERIFY(lint/assert)
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from ...ir.taskir import Module, Node, Program, Retry
from .base import BenchmarkLifter, register


@register
class RtlLifter(BenchmarkLifter):
    name = "rtl"
    dataset = "rtl/eda"

    def can_handle(self, sample: Dict) -> bool:
        return bool(sample.get("prompt")) and bool(
            sample.get("rtl") or sample.get("code") or
            sample.get("domain") in ("rtl", "eda"))

    def lift(self, sample: Dict, view: str = "execution") -> Optional[Module]:
        prompt = sample.get("prompt") or sample.get("task") or ""
        if not prompt:
            return None
        signals = sample.get("signals") or ["clk", "rst", "data"]
        timing = sample.get("timing") or {}

        nodes = [Node(id="%c0", op="LOAD", inputs=["@task"],
                      params={"source": "design.sv + waveform.vcd",
                              "kind": "design_artifacts"})]
        nodes.append(Node(id="%c1", op="EXTRACT", inputs=["%c0"],
                          params={"fields": signals + ["assertion_failures"]}))
        prev = "%c1"
        if timing:
            nodes.append(Node(id="%c2", op="CALCULATE", inputs=[prev],
                              params={"expr": "setup_slack"}))
            prev = "%c2"
        nodes.append(Node(id="%c3", op="SEARCH", inputs=["@task"],
                          params={"domain": "rtl", "corpus": "known_bug_db"},
                          output_type="List[Rtl]"))
        gen = Node(id="%fix", op="CODEGEN", inputs=[prev, "%c3", "@task"],
                   params={"role": "rtl_fix", "target": "design.sv"})
        ver = Node(id="%verify", op="VERIFY", inputs=["%fix", prev],
                   params={"check": "syntax_and_lint_and_assertion_pass"})
        gen.retry = Retry(max_attempts=3, on="%verify")
        nodes.extend([gen, ver])

        name = re.sub(r"[^0-9A-Za-z_]+", "_", prompt[:40]).strip("_").lower() \
            or f"rtl_{sample.get('task_id', 'task')}"
        prog = Program(name=name, description=prompt,
                       inputs=[{"name": "@task", "type": "Str"}],
                       nodes=nodes, output="%fix")
        return Module(program=prog, meta={
            "name": name,
            "provenance": {"source": self.name,
                           "task_id": str(sample.get("task_id", ""))}})

    def lift_with_reason(self, sample: Dict,
                         view: str = "execution"
                         ) -> Tuple[Optional[Module], Optional[str]]:
        if not (sample.get("prompt") or sample.get("task")):
            return None, "missing prompt"
        return self.lift(sample), None
