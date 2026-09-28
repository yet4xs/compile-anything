"""xLAM -> TaskIR lifter.

Input record schema (xLAM func-calling style):
    {"task_id": str,
     "question": str,                    # natural-language task
     "steps": [str | {"name": str, "arguments": dict}, ...],
     "answer": str}

Steps are strings containing ```json blocks with call arrays
[{"name": "...", "arguments": {...}}] (the xLAM text convention) or already
structured dicts.

Lifting policy (see spec/skill-isa.md §5):
  - each tool call  -> one semantic skill node (toolmap lowering);
  - call i depends on call i-1 (xLAM trajectories are predominantly linear);
  - terminal: GENERATE(final answer) + VERIFY(grounding), with retry on the
    VERIFY verdict — the canonical "compiled task program" tail;
  - raw tool names are preserved in meta.provenance only.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..ir.taskir import Module, Node, Program, Retry
from ..ir import types as ty
from ..isa import registry
from . import toolmap


def _infer(spec, input_types):
    return spec.output_type(input_types) if input_types else spec.output_type([])


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
    nodes: List[Node] = []
    prev: Optional[str] = None
    prev_type: Optional[str] = None
    tool_names: List[str] = []

    for i, call in enumerate(parsed["calls"]):
        lowered = toolmap.map_tool(call["name"], call["arguments"])
        skill = lowered["skill"]
        spec = registry.get(skill)
        out_t = toolmap.search_output_type(skill, lowered["params"])

        # typing-aware chaining: link to the previous result when the skill's
        # first input accepts it; otherwise bridge with EXTRACT (Any -> open)
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
        nodes.append(Node(
            id=f"%c{i}",
            op=skill,
            inputs=inputs,
            params=lowered["params"],
            output_type=out_t,
        ))
        new_inputs_types = [prev_type or "Any"] if inputs and inputs[0].startswith("%") \
            else (["Str"] if inputs else [])
        prev = nodes[-1].id
        prev_type = out_t or (spec and _infer(spec, new_inputs_types)) or "Any"
        tool_names.append(call["name"])

    gen_inputs = [prev, "@task"] if prev else ["@task"]
    gen = Node(id="%ans", op="GENERATE", inputs=gen_inputs,
               params={"role": "final_answer",
                       "instruction": parsed["question"][:200]})
    if parsed["answer"]:
        gen.params["reference"] = parsed["answer"][:200]
    ver = Node(id="%verify", op="VERIFY", inputs=["%ans"],
               params={"check": "answer_grounded_in_tool_results"})
    gen.retry = Retry(max_attempts=2, on="%verify")
    nodes.extend([gen, ver])

    name = re.sub(r"[^0-9A-Za-z_]+", "_", parsed["question"][:40]).strip("_") or \
        parsed["task_id"] or "xlam_task"
    prog = Program(
        name=name.lower(),
        description=parsed["question"],
        inputs=[{"name": "@task", "type": "Str"}],
        nodes=nodes,
        output="%ans",
    )
    return Module(program=prog, meta={
        "name": prog.name,
        "provenance": {
            "source": "xlam",
            "task_id": parsed["task_id"],
            "tools": tool_names,          # raw names live HERE only
        },
    })
