"""Phase 8D-2A.2: Effect-safe dependency refinement pass.

Removes redundant DATA dependencies from neural-generated TaskIR while
preserving observable semantics. An edge is SAFE to remove only when ALL:
1. The consumer doesn't use any value from the producer (no param_ref)
2. ALL required params are bound via named mechanisms (param_refs/literals)
3. BOTH endpoints are pure read_only (no mutation, no communication,
   no external side effect)
4. No alternative ORDER/EFFECT constraint is needed

Communication (send_alert, notifications) is an OBSERVABLE SIDE EFFECT.
Removing ordering constraints on communication can allow notifications
to be sent BEFORE the actions they describe. This is a business logic
error even if the program still validates.

UNPROVEN edges (where safety cannot be established) are NOT removed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.ir.parser import parse_text
from src.ir.taskir import Node, Module
from src.validator.validator import validate


@dataclass
class RewriteCertificate:
    edge_from: str
    edge_to: str
    edge_kind: str  # DATA_OPERAND | ORDER
    reason: str
    producer_consumed: bool  # does consumer actually use producer output?
    effect_risk: str  # "none" | "mutation_ordering" | "unknown"


@dataclass
class RefinementResult:
    original: Module
    refined: Module
    certificates: List[RewriteCertificate] = field(default_factory=list)
    edges_removed: int = 0
    validation_pass: bool = False
    unchanged: bool = True


def _has_real_value_consumption(consumer: Node, producer_id: str,
                                 ext_chain: Dict[str, str]) -> bool:
    """Check if consumer uses a value from producer through NAMED binding
    (param_refs only). Positional inputs are NOT counted here because we're
    deciding whether to remove them — counting them would be tautological."""
    for ref in consumer.param_refs.values():
        r = ref
        while r in ext_chain:
            r = ext_chain[r]
        if r == producer_id:
            return True
    return False


def _is_safe_to_reorder(node: Node, cap_registry: Optional[dict] = None) -> bool:
    """True only for pure read_only nodes. Communication, mutation, and
    unknown effects are all UNSAFE to reorder — they have observable
    consequences whose ordering may carry business meaning."""
    eff = (node.hints or {}).get("effect", "")
    if eff == "read_only":
        return True
    if eff and eff != "read_only":
        return False
    # Check registry if available
    tool = node.params.get("action") or (node.hints or {}).get("tool")
    if cap_registry and tool in cap_registry:
        return cap_registry[tool].get("effect_class", "read_only") == "read_only"
    # Conservative: if unknown, NOT safe
    return False


# Backward-compatible alias (but stricter than before)
def _is_read_only(node: Node, cap_registry: Optional[dict] = None) -> bool:
    return _is_safe_to_reorder(node, cap_registry)


def _all_params_bound(node: Node, cap_registry: Optional[dict],
                       ext_chain: Dict[str, str]) -> bool:
    """Check if all required tool params are provided via param_refs
    or literal params (not positional operands)."""
    tool = node.params.get("action") or (node.hints or {}).get("tool")
    if not tool or not cap_registry or tool not in cap_registry:
        return False
    meta = cap_registry[tool]
    input_schema = meta.get("input_schema")
    if not isinstance(input_schema, dict):
        return False
    for param_name in input_schema:
        if param_name in node.param_refs:
            continue
        if param_name in node.params:
            continue
        return False
    return True


def refine_dependencies(module: Module,
                         cap_registry: Optional[dict] = None) -> RefinementResult:
    """Remove redundant DATA dependencies from a Module.

    A dependency (A → B) is redundant when:
    - B doesn't consume any value from A (no param_ref or operand)
    - Both A and B are read-only (no EFFECT ordering needed)
    - Removing doesn't create a cycle
    """
    import copy
    refined = copy.deepcopy(module)
    certificates = []
    removed = 0

    # Build EXTRACT_FIELD chain map
    ext_chain = {}
    for n in refined.program.nodes:
        if n.op == "EXTRACT_FIELD" and n.inputs:
            ext_chain[n.id] = n.inputs[0]

    # Build effect class lookup
    node_effects = {}
    for n in refined.program.nodes:
        node_effects[n.id] = _is_read_only(n, cap_registry)

    # For each non-EXTRACT_FIELD node, check each input operand
    for node in refined.program.nodes:
        if node.op == "EXTRACT_FIELD":
            continue

        # Check positional inputs (operands)
        new_inputs = []
        for inp in node.inputs:
            # Resolve through EXTRACT_FIELD
            src = inp
            while src in ext_chain:
                src = ext_chain[src]

            consumed = _has_real_value_consumption(node, src, ext_chain)
            both_safe = node_effects.get(node.id, False) and \
                         node_effects.get(src, True)
            params_satisfied = _all_params_bound(node, cap_registry,
                                                   ext_chain)

            # SAFE removal requires ALL THREE:
            # 1. No value consumption (param_refs don't reference producer)
            # 2. Both endpoints are pure read_only (no comm/mutation)
            # 3. All required params bound via named mechanisms
            if not consumed and both_safe and params_satisfied:
                certificates.append(RewriteCertificate(
                    edge_from=src, edge_to=node.id,
                    edge_kind="DATA_OPERAND",
                    reason="both endpoints read_only; no value consumption; "
                            "all params bound via param_refs",
                    producer_consumed=False,
                    effect_risk="none"))
                removed += 1
                # Don't add this input back
            elif not consumed and not both_safe:
                # UNSAFE: at least one endpoint has observable effects.
                # Record certificate but KEEP the edge.
                certificates.append(RewriteCertificate(
                    edge_from=src, edge_to=node.id,
                    edge_kind="DATA_OPERAND",
                    reason="CANDIDATE but UNSAFE: endpoint has observable "
                            "effects (mutation/communication/unknown)",
                    producer_consumed=False,
                    effect_risk="UNSAFE_effect_ordering"))
                new_inputs.append(inp)  # MUST keep UNSAFE edges
            else:
                new_inputs.append(inp)
        node.inputs = new_inputs

        # Check after() edges — explicit ORDER constraints are ALWAYS kept.
        # Users (or the planner) explicitly requested this ordering; even
        # between read_only nodes, removing a user-specified ORDER changes
        # the program's semantics. No automatic removal.
        # (All after() edges are preserved as-is.)
        pass  # node.after is never modified

    # Validate refined module
    try:
        rep = validate(refined)
        validation_pass = rep.valid
    except Exception:
        validation_pass = False

    unchanged = removed == 0
    return RefinementResult(
        original=module, refined=refined,
        certificates=certificates, edges_removed=removed,
        validation_pass=validation_pass, unchanged=unchanged)


def refine_taskir_text(taskir_text: str,
                        cap_registry: Optional[dict] = None) -> dict:
    """Convenience wrapper: parse → refine → re-serialize."""
    from src.ir.taskir import to_text
    try:
        mod = parse_text(taskir_text)
    except Exception as e:
        return {"refined": None, "removed": 0, "certificates": [],
                "error": str(e)}
    result = refine_dependencies(mod, cap_registry)
    return {
        "refined": to_text(result.refined),
        "removed": result.edges_removed,
        "certificates": [{"from": c.edge_from, "to": c.edge_to,
                           "kind": c.edge_kind, "reason": c.reason,
                           "consumed": c.producer_consumed,
                           "effect_risk": c.effect_risk}
                          for c in result.certificates],
        "validation_pass": result.validation_pass,
    }
