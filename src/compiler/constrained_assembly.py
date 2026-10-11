"""Post-hoc SSA constraint assembly (Phase 7C-0.1).

Replaces the regex-based logic in constrained_decoder.py v1. The ONLY TaskIR
parsing entry is src/ir/parser.py::parse_text(); constraints here operate on
parsed Node/Program objects (AST level), never on raw text.

Honest framing: this is post-hoc constrained ASSEMBLY — the model emits a
free-form completion once, then this pass parses, checks, and repairs it into
a validator-clean module. It is not token-level constrained decoding.

Levels (cumulative):
  1 grammar   — official ID_RE + duplicate definitions. Never mutates nodes;
               must be lossless on validator-valid input (D1 fidelity).
  2 typed SSA — undefined refs auto-fixed to a type-compatible definition
               (is_compatible), arity truncated to spec.max_inputs, program
               output retargeted to the last node when dangling. All
               mutations logged as corrections.
  3 control   — after/guard/retry annotations must survive assembly intact;
               a change is a rejection, not a silent fix.
"""
import copy
from typing import Dict, List

from src.ir.parser import parse_text, TaskIRSyntaxError  # noqa: F401 (re-export)
from src.ir.taskir import Module, Node, ID_RE
from src.ir.types import is_compatible, normalize
from src.isa.registry import get as get_skill
from src.validator.validator import validate


def node_key(n: Node):
    """Identity of a node for AST-equality (id, op, inputs, params, after)."""
    return (n.id, n.op, tuple(n.inputs),
            tuple(sorted((k, str(v)) for k, v in n.params.items())),
            tuple(n.after or []))


def module_ast_equal(a: Module, b: Module) -> bool:
    """True iff two modules carry the identical node sequence and output."""
    if a.program.output != b.program.output:
        return False
    if [n.id for n in a.program.nodes] != [n.id for n in b.program.nodes]:
        return False
    return all(node_key(x) == node_key(y)
               for x, y in zip(a.program.nodes, b.program.nodes))


def apply_constraints(module: Module, level: int) -> Dict:
    """Apply level 1-3 constraints to a parsed module.

    Returns {"module": repaired copy (input never mutated),
             "valid": bool, "corrections": [...], "rejections": [...]}.
    """
    if level < 0 or level > 3:
        raise ValueError(f"level must be 0..3, got {level}")
    mod = copy.deepcopy(module)
    corrections: List[Dict] = []
    rejections: List[Dict] = []
    prog = mod.program

    defined = {inp["name"] for inp in prog.inputs}

    # ---- Level 1: grammar (checks only, no mutation) ----
    if level >= 1:
        for node in prog.nodes:
            if not ID_RE.match(node.id):
                rejections.append({"node": node.id, "reason": "bad_id_format"})
            if node.id in defined:
                rejections.append({"node": node.id, "reason": "duplicate_def"})
            defined.add(node.id)

    # ---- Level 2: typed SSA ----
    if level >= 2:
        defined = {inp["name"] for inp in prog.inputs}
        type_map = {inp["name"]: inp.get("type", "Any") for inp in prog.inputs}

        for node in prog.nodes:
            spec = get_skill(node.op)

            # undefined refs -> type-compatible definition (nearest upstream)
            fixed_inputs = []
            for i, ref in enumerate(node.inputs):
                if ref in defined:
                    fixed_inputs.append(ref)
                    continue
                cands = []
                if spec and spec.input_types:
                    want = spec.input_types[min(i, len(spec.input_types) - 1)]
                    cands = [d for d in sorted(defined)
                             if is_compatible(want, type_map.get(d, "Any"))]
                if not cands:
                    cands = sorted(defined)
                if cands:
                    fixed_inputs.append(cands[-1])
                    corrections.append({"node": node.id, "kind": "undef_ref",
                                        "old": ref, "new": cands[-1]})
                else:
                    fixed_inputs.append(ref)
                    rejections.append({"node": node.id,
                                       "reason": f"unfixable_ref:{ref}"})

            # arity truncation to the skill's max_inputs
            if spec and spec.max_inputs and len(fixed_inputs) > spec.max_inputs:
                corrections.append({"node": node.id, "kind": "arity",
                                    "old": len(fixed_inputs),
                                    "new": spec.max_inputs})
                fixed_inputs = fixed_inputs[:spec.max_inputs]

            node.inputs = fixed_inputs

            # propagate types via the official rule (never string surgery)
            if spec:
                in_types = [type_map.get(r, "Any") for r in node.inputs]
                type_map[node.id] = normalize(str(spec.output_type(in_types)))
            else:
                type_map[node.id] = "Any"
            defined.add(node.id)

        # dangling program output -> retarget to last node
        if prog.output and prog.output not in defined:
            node_ids = [n.id for n in prog.nodes]
            if node_ids:
                corrections.append({"node": "__output__",
                                    "kind": "output_retarget",
                                    "old": prog.output, "new": node_ids[-1]})
                prog.output = node_ids[-1]

    # ---- Level 3: control preservation ----
    if level >= 3:
        orig = {n.id: n for n in module.program.nodes}
        for node in prog.nodes:
            src = orig.get(node.id)
            if src is None:
                continue
            if (src.after or []) != (node.after or []):
                rejections.append({"node": node.id, "reason": "after_changed"})
            if (src.guard is None) != (node.guard is None):
                rejections.append({"node": node.id, "reason": "guard_changed"})
            if (src.retry is None) != (node.retry is None):
                rejections.append({"node": node.id, "reason": "retry_changed"})

    try:
        valid = validate(mod).valid and not rejections
    except Exception:
        valid = False

    return {"module": mod, "valid": valid,
            "corrections": corrections, "rejections": rejections}


def parse_and_assemble(text: str, level: int) -> Dict:
    """parse_text() + apply_constraints in one step.

    Returns result dict; parse failure yields {"module": None, "valid": False,
    "error": "parse:<msg>"} so callers can count parse vs constraint failures
    separately.
    """
    try:
        mod = parse_text(text)
    except (TaskIRSyntaxError, Exception) as e:
        return {"module": None, "valid": False,
                "corrections": [], "rejections": [],
                "error": f"parse:{type(e).__name__}:{str(e)[:120]}"}
    res = apply_constraints(mod, level)
    res["error"] = None
    return res
