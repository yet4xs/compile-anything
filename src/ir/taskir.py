"""TaskIR v0.1 core data structures, JSON (canonical) I/O and text printer.

The dataclasses below are the single in-memory representation used by the
validator, lifter, runtime simulator and dataset tooling — every component
reads/writes these, never ad-hoc dicts (LLVM-style: everything speaks IR).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TASKIR_VERSION = "0.1"
ID_RE = re.compile(r"^[%@][A-Za-z0-9_.\-]+$")


@dataclass
class Guard:
    """Predicated execution: run node iff value of `cond` == `expect`."""
    cond: str
    expect: bool = True


@dataclass
class Retry:
    """Re-execute node: on executor error (`on="error"`) or on a downstream
    VERIFY node evaluating to False (`on="%v"`), up to `max_attempts` total."""
    max_attempts: int = 1
    on: str = "error"


@dataclass
class Node:
    id: str
    op: str
    inputs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    after: List[str] = field(default_factory=list)
    guard: Optional[Guard] = None
    retry: Optional[Retry] = None
    output_type: Optional[str] = None
    hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Program:
    name: str
    description: str = ""
    inputs: List[Dict[str, str]] = field(default_factory=list)  # {"name","type"}
    nodes: List[Node] = field(default_factory=list)
    output: str = ""


@dataclass
class Module:
    program: Program
    taskir_version: str = TASKIR_VERSION
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- JSON I/O

def _opt_guard(g: Optional[Guard]) -> Any:
    return None if g is None else {"cond": g.cond, "expect": g.expect}


def _opt_retry(r: Optional[Retry]) -> Any:
    return None if r is None else {"max_attempts": r.max_attempts, "on": r.on}


def _node_to_dict(n: Node) -> Dict[str, Any]:
    return {
        "id": n.id,
        "op": n.op,
        "inputs": list(n.inputs),
        "params": dict(n.params),
        "after": list(n.after),
        "guard": _opt_guard(n.guard),
        "retry": _opt_retry(n.retry),
        "output_type": n.output_type,
        "hints": dict(n.hints),
    }


def module_to_dict(m: Module) -> Dict[str, Any]:
    return {
        "taskir_version": m.taskir_version,
        "meta": dict(m.meta),
        "program": {
            "name": m.program.name,
            "description": m.program.description,
            "inputs": [dict(i) for i in m.program.inputs],
            "nodes": [_node_to_dict(n) for n in m.program.nodes],
            "output": m.program.output,
        },
    }


def _parse_guard(d: Any) -> Optional[Guard]:
    if d is None:
        return None
    return Guard(cond=d["cond"], expect=d.get("expect", True))


def _parse_retry(d: Any) -> Optional[Retry]:
    if d is None:
        return None
    return Retry(max_attempts=int(d.get("max_attempts", 1)), on=d.get("on", "error"))


def _node_from_dict(d: Dict[str, Any]) -> Node:
    return Node(
        id=d["id"],
        op=d["op"],
        inputs=list(d.get("inputs") or []),
        params=dict(d.get("params") or {}),
        after=list(d.get("after") or []),
        guard=_parse_guard(d.get("guard")),
        retry=_parse_retry(d.get("retry")),
        output_type=d.get("output_type"),
        hints=dict(d.get("hints") or {}),
    )


def module_from_dict(d: Dict[str, Any]) -> Module:
    p = d["program"]
    prog = Program(
        name=p.get("name", ""),
        description=p.get("description", ""),
        inputs=[dict(i) for i in p.get("inputs") or []],
        nodes=[_node_from_dict(n) for n in p.get("nodes") or []],
        output=p.get("output", ""),
    )
    return Module(program=prog, taskir_version=d.get("taskir_version", "0.1"),
                  meta=dict(d.get("meta") or {}))


def save_module(m: Module, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(module_to_dict(m), f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_module(path) -> Module:
    with open(path, "r", encoding="utf-8") as f:
        return module_from_dict(json.load(f))


# ---------------------------------------------------------------- text form

def _fmt_value(v: Any) -> str:
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    return json.dumps(v)


def to_text(m: Module) -> str:
    """One-way pretty printer (human / debug / training-target format)."""
    p = m.program
    lines = [f"; TaskIR v0.1  module={m.meta.get('name', p.name)}"]
    if p.description:
        lines.append(f"; task: {p.description}")
    for k, v in (m.meta.get("provenance") or {}).items():
        lines.append(f"; provenance.{k}: {v}")
    if p.inputs:
        lines.append("inputs: " + ", ".join(f"{i['name']}: {i.get('type', 'Any')}"
                                            for i in p.inputs))
    lines.append("")
    for n in p.nodes:
        parts = list(n.inputs)
        parts += [f"{k}={_fmt_value(v)}" for k, v in n.params.items()]
        ann = ""
        if n.after:
            ann += f"  after({', '.join(n.after)})"
        if n.guard:
            ann += f"  guard({n.guard.cond} == {str(n.guard.expect).lower()})"
        if n.retry:
            ann += f"  retry(on={n.retry.on}, max={n.retry.max_attempts})"
        lines.append(f"{n.id} = {n.op}({', '.join(parts)}){ann}")
    lines.append("")
    lines.append(f"return {p.output}")
    return "\n".join(lines) + "\n"
