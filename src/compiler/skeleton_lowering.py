"""Skeleton -> TaskIR deterministic lowering — Phase 8B-0 Task 6,
hardened in 8B-1A Task 2.

Contract:
- assigns SSA ids (%p0, %p1, ...)
- DATA dependencies (depends_on): the first type-COMPATIBLE dep becomes
  the operand of a semantic skill; incompatible outputs fall back to
  ORDERING (after()) with the task input as operand — the coercion is
  REPORTED in `coerced_ordering_deps`, never silent
- ORDERING dependencies (order_on) and mutating serialization use after()
- preserves tool identity: EXEC_ACTION(action=...) for mutations;
  hints(tool=...) for semantic-skill nodes (the binding contract the
  evaluator reads for non-mutating tools)
- unresolved arguments are REPORTED, never fabricated
- output must pass the official Parser + Validator; a semantically wrong
  skeleton lowers to a wrong-but-legal program (no silent repair)
"""
from __future__ import annotations

from typing import Dict

from src.compiler.plan_skeleton import (PlanSkeleton, validate_skeleton,
                                        hard_errors, NON_STATE)
from src.isa.registry import get as get_skill_spec
from src.ir.parser import parse_text
from src.ir.taskir import _fmt_value
from src.ir.types import is_compatible, normalize
from src.validator.validator import validate


def _dep_output_type(domain: str, dep_tool: str) -> str:
    from src.eval.tau3_skill_oracle import get_tool_oracle
    o = get_tool_oracle(domain, dep_tool)
    spec = get_skill_spec(o["semantic_skill"]) \
        if o["semantic_skill"] else None
    if spec:
        try:
            return normalize(str(spec.output_type([])))
        except Exception:
            return "Any"
    return "Any"


def _operand_slot_type(skill: str) -> str:
    spec = get_skill_spec(skill) if skill else None
    if spec and spec.input_types:
        return spec.input_types[0]
    return "Any"


def lower_skeleton(sk: PlanSkeleton, domain: str,
                   registry_schemas: Dict[str, dict],
                   task_input: str = "@task", strict: bool = False) -> dict:
    """Returns {"taskir", "unresolved", "errors", "coerced_ordering_deps",
    "warnings"}; taskir is None on any blocking failure.

    strict=True (8B-1A.1): type-incompatible data dependencies FAIL the
    compilation (semantic_loss) instead of being coerced to ordering with
    a @task operand; required-schema params must come from the registry
    (params=None marks schema_unknown and skips param checks). The
    non-strict mode remains available for diagnostics only."""
    all_errs = validate_skeleton(sk, domain, registry_schemas)
    warnings = [e for e in all_errs if e.startswith("warning:")]
    errs = hard_errors(all_errs)
    unresolved = []
    coerced = []
    if errs:
        return {"taskir": None, "unresolved": unresolved, "errors": errs,
                "coerced_ordering_deps": coerced, "warnings": warnings}

    lines = ["; TaskIR v0.1  module=skeleton_lowered",
             f"; task: {sk.task_id}",
             f"inputs: {task_input}: Str", ""]
    ssa_of = {}
    tool_of = {}
    # per-node output type derived from the SKELETON's declared skill
    # (registry-independent; oracle refinement only when available)
    skill_out = {}
    for n in sk.nodes:
        spec = get_skill_spec(n.skill) if n.skill else None
        try:
            skill_out[n.id] = normalize(str(spec.output_type([])))                 if spec else "Any"
        except Exception:
            skill_out[n.id] = "Any"
    last_mutating = None
    field_ssa: Dict[tuple, str] = {}  # (producer_id, field) → EXTRACT_FIELD SSA
    for i, n in enumerate(sk.nodes):
        sid = f"%p{i}"
        ssa_of[n.id] = sid
        tool_of[n.id] = n.capability_id
        schema = registry_schemas.get(n.capability_id, {})
        # effect_class: prefer registry; fall back to skeleton node value;
        # NEVER None (None causes spurious serialization of read-only tools)
        if not n.effect_class:
            n.effect_class = schema.get("effect_class", "read_only")
        if not n.effect_class:
            n.effect_class = "read_only"
        required = [p["name"] for p in (schema.get("params") or [])
                    if not p.get("optional")]
        args = dict(n.arguments)
        for req in required:
            if req not in args:
                unresolved.append({"node": n.id, "missing_param": req})
        # v0.2: separate literal args from runtime_refs
        param_refs = {}
        literal_args = {}
        for k, v in args.items():
            if isinstance(v, dict) and v.get("kind") == "runtime_ref_multi":
                # v0.3: list param whose elements come from different
                # producers/paths — one EXTRACT_FIELD per element, then a
                # MAKE_LIST to build the value that gets bound.
                elems = v.get("elements") or []
                if not elems:
                    return {"taskir": None, "unresolved": unresolved,
                            "errors": [f"runtime_ref_multi {k}: empty"],
                            "coerced_ordering_deps": [], "warnings": warnings}
                ext_ids = []
                for e in elems:
                    producer = e.get("producer", "")
                    field_name = e.get("field", "")
                    if producer not in ssa_of:
                        return {"taskir": None, "unresolved": unresolved,
                                "errors": [f"runtime_ref_multi {k}: producer "
                                           f"{producer!r} not yet defined"],
                                "coerced_ordering_deps": [],
                                "warnings": warnings}
                    key = (producer, field_name)
                    if key not in field_ssa:
                        ext_id = f"%ext{len(field_ssa)}"
                        lines.append(f'{ext_id} = EXTRACT_FIELD('
                                     f'{ssa_of[producer]}, '
                                     f'field="{field_name}")')
                        field_ssa[key] = ext_id
                    ext_ids.append(field_ssa[key])
                list_id = f"%lst{len(field_ssa)}"
                lines.append(f"{list_id} = MAKE_LIST(" +
                             ", ".join(ext_ids) + ")")
                param_refs[k] = list_id
            elif isinstance(v, dict) and v.get("kind") == "runtime_ref":
                producer = v.get("producer", "")
                field_name = v.get("field", "")
                if producer not in ssa_of:
                    return {"taskir": None, "unresolved": unresolved,
                            "errors": [f"runtime_ref {k}: producer "
                                       f"{producer!r} not yet defined"],
                            "coerced_ordering_deps": [],
                            "warnings": warnings}
                key = (producer, field_name)
                if key not in field_ssa:
                    ext_id = f"%ext{len(field_ssa)}"
                    lines.append(f'{ext_id} = EXTRACT_FIELD('
                                 f'{ssa_of[producer]}, '
                                 f'field="{field_name}")')
                    field_ssa[key] = ext_id
                param_refs[k] = field_ssa[key]
            else:
                literal_args[k] = v
        data_deps = [d for d in n.depends_on if d in ssa_of]
        order_deps = [d for d in n.order_on if d in ssa_of]
        parts = []
        afters = [ssa_of[d] for d in order_deps]
        if n.skill == "EXEC_ACTION":
            # DATA dependencies are OPERANDS: actual value passing
            # (EXEC_ACTION is variadic-Any). Dropping them or demoting
            # them to ordering would silently erase dataflow.
            data_operands = [ssa_of[d] for d in data_deps]
            spec = get_skill_spec("EXEC_ACTION")
            if data_operands and spec and spec.max_inputs is not None                     and len(data_operands) > spec.max_inputs:
                return {"taskir": None, "unresolved": unresolved,
                        "errors": [f"unresolved_data_dependency: {n.id} "
                                   f"needs {len(data_operands)} data "
                                   f"operands, EXEC_ACTION expresses "
                                   f"{spec.max_inputs}"],
                        "coerced_ordering_deps": [],
                        "warnings": warnings}
            parts.append(f"action={n.capability_id}")
            parts.extend(data_operands)
            for k, v in literal_args.items():
                parts.append(f"{k}={_fmt_value(v)}")
            # ORDER deps + conservative EFFECT serialization
            if n.effect_class not in NON_STATE and last_mutating:
                afters.append(last_mutating)
            if n.effect_class not in NON_STATE:
                last_mutating = sid
            hints = (f'hints(tool="{n.capability_id}", '
                     f'effect="{n.effect_class}")')
            ann = ("  after(" + ", ".join(dict.fromkeys(afters)) + ")"
                   if afters else "") + f"  {hints}"
            # v0.2: runtime parameter bindings
            if param_refs:
                ann += "  bind(" + ", ".join(
                    f"{k}={v}" for k, v in param_refs.items()) + ")"
        else:
            # first type-compatible data dep becomes the operand;
            # incompatible deps are coerced to ORDERING (reported)
            operand = task_input
            want = _operand_slot_type(n.skill)
            used_dep = None
            for d in data_deps:
                got = skill_out.get(d, "Any")
                if got == "Any" and domain:
                    got = _dep_output_type(domain, tool_of[d])
                if used_dep is None and is_compatible(want, got):
                    operand = ssa_of[d]
                    used_dep = ssa_of[d]
                elif strict:
                    return {"taskir": None, "unresolved": unresolved,
                            "errors": [f"semantic_loss: type-incompatible "
                                       f"data dep {n.id}->{d} "
                                       f"(want {want}, got {got})"],
                            "coerced_ordering_deps": [],
                            "warnings": warnings}
                else:
                    coerced.append({"node": n.id, "dep": d,
                                    "want": want, "got": got})
                    afters.append(ssa_of[d])
            if strict and used_dep is None and want not in ("Any", "Str"):
                # no compatible producer: the value must come from the
                # runtime/conversation, which the plan cannot supply
                return {"taskir": None, "unresolved": unresolved,
                        "errors": [f"unresolved: runtime input required "
                                   f"for {n.id} slot type {want}"],
                        "coerced_ordering_deps": [],
                        "warnings": warnings}
            parts.append(operand)
            for k, v in args.items():
                parts.append(f"{k}={_fmt_value(v)}")
            hints = (f'hints(tool="{n.capability_id}", '
                     f'effect="{n.effect_class}")')
            ann = ("  after(" + ", ".join(dict.fromkeys(afters)) + ")"
                   if afters else "") + f"  {hints}"
        lines.append(f"{sid} = {n.skill}({', '.join(parts)}){ann}")
    out_ssa = ssa_of[sk.nodes[-1].id] if sk.nodes else task_input
    lines += ["", f"return {out_ssa}"]
    text = "\n".join(lines) + "\n"

    if unresolved:
        return {"taskir": None, "unresolved": unresolved,
                "errors": [f"unresolved x{len(unresolved)}"],
                "coerced_ordering_deps": coerced, "warnings": warnings,
                "draft": text}
    try:
        mod = parse_text(text)
        # strict mode: pass the capability registry for typed validation
        from src.compiler.capability_schema import normalize_registry
        norm_reg = normalize_registry(registry_schemas)
        v2_reg = {}
        for t, entry in norm_reg.items():
            if entry.get("input_schema") or entry.get("output_schema"):
                v2_reg[t] = {
                    "input_schema": entry.get("input_schema") or {},
                    "output_schema": entry.get("output_schema") or {},
                    "effect_class": entry.get("effect_class"),
                }
        if strict and v2_reg:
            rep = validate(mod, capability_registry=v2_reg)
        else:
            rep = validate(mod)
        if not rep.valid:
            return {"taskir": None, "unresolved": [],
                    "errors": [f"validator: {rep.errors[:2]}"],
                    "coerced_ordering_deps": coerced,
                    "warnings": warnings, "draft": text}
    except Exception as e:
        return {"taskir": None, "unresolved": [],
                "errors": [f"parser: {e}"],
                "coerced_ordering_deps": coerced, "warnings": warnings,
                "draft": text}
    return {"taskir": text, "unresolved": [], "errors": [],
            "coerced_ordering_deps": coerced, "warnings": warnings}
