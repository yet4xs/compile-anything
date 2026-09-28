"""Shared trajectory-to-TaskIR lifting core.

Used by the xLAM lifter and the benchmark lifters (toolbench / api-bank
style). Policy (spec/skill-isa.md §5):
  - one semantic skill node per tool call (toolmap lowering);
  - typing-aware chaining: link to the previous result when the skill's
    first input accepts it, else bridge with EXTRACT (Any -> open);
  - optional GENERATE(final answer) + VERIFY(grounding) tail with retry.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..ir.taskir import Module, Node, Program, Retry
from ..ir import types as ty
from ..isa import registry
from . import toolmap


def _infer(spec, input_types):
    return spec.output_type(input_types) if input_types else spec.output_type([])


def lift_trajectory(question: str,
                    calls: List[Dict[str, Any]],
                    source: str,
                    task_id: str = "",
                    tail: bool = True) -> Module:
    nodes: List[Node] = []
    prev: Optional[str] = None
    prev_type: Optional[str] = None
    tool_names: List[str] = []

    for i, call in enumerate(calls):
        lowered = toolmap.map_tool(call.get("name", ""), call.get("arguments") or {})
        skill = lowered["skill"]
        spec = registry.get(skill)
        out_t = toolmap.search_output_type(skill, lowered["params"])

        inputs: List[str] = []
        if prev is not None and spec is not None:
            want = spec.input_types[0] if spec.input_types else "Any"
            if ty.is_compatible(want, prev_type or "Any"):
                inputs = [prev]
            else:
                bridge = Node(id=f"%b{i}", op="EXTRACT", inputs=[prev],
                              params={"fields": ["result"]})
                nodes.append(bridge)
                inputs = [bridge.id]
                prev, prev_type = bridge.id, "Any"
        if not inputs and spec is not None:
            want = spec.input_types[0] if spec.input_types else "Any"
            if spec.min_inputs >= 1:
                if ty.is_compatible(want, "Str") or want == "Any":
                    inputs = ["@task"]
                # else: skill tolerates no data input (amount etc. in params)
        nodes.append(Node(id=f"%c{i}", op=skill, inputs=inputs,
                          params=lowered["params"], output_type=out_t))
        new_types = [prev_type or "Any"] if inputs and inputs[0].startswith("%") \
            else (["Str"] if inputs else [])
        prev = nodes[-1].id
        prev_type = out_t or (spec and _infer(spec, new_types)) or "Any"
        tool_names.append(call.get("name", ""))

    if tail:
        gen_inputs = [prev, "@task"] if prev else ["@task"]
        gen = Node(id="%ans", op="GENERATE", inputs=gen_inputs,
                   params={"role": "final_answer",
                           "instruction": question[:200]})
        nodes.append(gen)
        ver = Node(id="%verify", op="VERIFY", inputs=["%ans"],
                   params={"check": "answer_grounded_in_tool_results"})
        nodes.append(ver)
        gen.retry = Retry(max_attempts=2, on="%verify")
        output = gen.id
    else:
        output = prev or ""

    name = re.sub(r"[^0-9A-Za-z_]+", "_", question[:40]).strip("_").lower() \
        or task_id or f"{source}_task"
    prog = Program(name=name, description=question,
                   inputs=[{"name": "@task", "type": "Str"}],
                   nodes=nodes, output=output)
    return Module(program=prog, meta={
        "name": name,
        "provenance": {"source": source, "task_id": task_id,
                       "tools": tool_names},     # raw names live HERE only
    })
