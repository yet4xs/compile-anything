"""Skeleton -> TaskIR deterministic lowering — Phase 8B-0 Task 6.

Contract:
- assigns SSA ids (%p0, %p1, ...)
- data references: a dependency edge becomes an operand for semantic
  skills, or an `after()` ordering edge for EXEC_ACTION (action params
  stay named arguments)
- preserves tool identity (params.tool) and effect_class (annotation)
- unresolved arguments are REPORTED, never fabricated
- output must pass the official Parser + Validator; a semantically wrong
  skeleton lowers to a wrong-but-legal program (no silent repair)
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from src.compiler.plan_skeleton import (PlanSkeleton, validate_skeleton,
                                        NON_STATE)
from src.ir.parser import parse_text
from src.validator.validator import validate

CONTROL_OPS = {"VERIFY", "SELECT", "MERGE", "EXTRACT", "GENERATE"}


def lower_skeleton(sk: PlanSkeleton, domain: str,
                   registry_schemas: Dict[str, dict],
                   task_input: str = "@task") -> dict:
    """Returns {"taskir": str|None, "unresolved": [...], "errors": [...]}.
    """
    errs = list(validate_skeleton(sk, domain, registry_schemas))
    unresolved = []
    if errs:
        return {"taskir": None, "unresolved": unresolved, "errors": errs}

    lines = ["; TaskIR v0.1  module=skeleton_lowered",
             f"; task: {sk.task_id}",
             f"inputs: {task_input}: Str", ""]
    ssa_of = {}
    last_mutating = None
    for i, n in enumerate(sk.nodes):
        sid = f"%p{i}"
        ssa_of[n.id] = sid
        schema = registry_schemas.get(n.capability_id, {})
        required = [p["name"] for p in schema.get("params", [])
                    if not p.get("optional")]
        args = dict(n.arguments)
        for req in required:
            if req not in args:
                unresolved.append({"node": n.id, "missing_param": req})
        deps = [ssa_of[d] for d in n.depends_on if d in ssa_of]
        parts = []
        ann = ""
        if n.skill == "EXEC_ACTION":
            parts.append(f"action={n.capability_id}")
            for k, v in args.items():
                parts.append(f"{k}={json.dumps(v)}")
            # conservative serialization: mutating actions ordered after
            # the previous mutating action
            if n.effect_class not in NON_STATE and last_mutating:
                ann = f"  after({last_mutating})"
            if n.effect_class not in NON_STATE:
                last_mutating = sid
            ann += f"  hints(tool=\"{n.capability_id}\", effect=\"{n.effect_class}\")"
        else:
            # semantic skill: first dependency becomes the data operand
            operand = deps[0] if deps else task_input
            parts.append(operand)
            for k, v in args.items():
                parts.append(f"{k}={json.dumps(v)}")
            if len(deps) > 1:
                ann = f"  after({', '.join(deps[1:])})"
            ann += (f"  hints(tool=\"{n.capability_id}\", "
                    f"effect=\"{n.effect_class}\")")
        lines.append(f"{sid} = {n.skill}({', '.join(parts)}){ann}")
    out_ssa = ssa_of[sk.nodes[-1].id] if sk.nodes else task_input
    lines += ["", f"return {out_ssa}"]
    text = "\n".join(lines) + "\n"

    if unresolved:
        return {"taskir": None, "unresolved": unresolved,
                "errors": [f"unresolved x{len(unresolved)}"], "draft": text}
    # official gate
    try:
        mod = parse_text(text)
        rep = validate(mod)
        if not rep.valid:
            return {"taskir": None, "unresolved": [],
                    "errors": [f"validator: {rep.errors[:2]}"],
                    "draft": text}
    except Exception as e:
        return {"taskir": None, "unresolved": [],
                "errors": [f"parser: {e}"], "draft": text}
    return {"taskir": text, "unresolved": [], "errors": []}
