"""Constrained Patch ISA for TaskIR repair.

Patches are JSON objects that describe minimal modifications to a TaskIR program.
The LLM generates patches, NOT entire programs. Each patch is validated against:
1. Scope: only repairs nodes within the SSA slice's repairable set
2. SSA: definitions must precede uses (or be inserted before)
3. Types: operand types must be compatible with Skill ISA signatures
4. Acyclicity: patches must not introduce dependency cycles
5. Boundary: frozen nodes are never modified
6. Effect safety: guard/retry relationships are preserved (placeholder until effect system lands)
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any, Tuple
from src.ir.taskir import Module, Node
from src.repair.ssa_slicer import SSASlice
from src.validator.validator import validate
from src.isa.registry import get as get_skill

# Allowed patch operations
PATCH_OPS = {
    "REPLACE_INPUT",       # Change one input reference of a node
    "REPLACE_NODE",        # Replace a node entirely (new op + params)
    "INSERT_BEFORE",       # Insert a new node before an anchor node
    "UPDATE_ANNOTATION",   # Update a node's params/hints
    "RENAME_SSA",          # Rename a node's SSA ID
    "UPDATE_OUTPUT",       # Change which node is the program output
}

REQUIRED_FIELDS = {
    "REPLACE_INPUT":   ["node_id", "position", "new_reference"],
    "REPLACE_NODE":    ["node_id", "new_op", "new_params"],
    "INSERT_BEFORE":   ["anchor_id", "new_id", "new_op", "new_params", "new_inputs"],
    "UPDATE_ANNOTATION": ["node_id", "field", "value"],
    "RENAME_SSA":      ["old_id", "new_id"],
    "UPDATE_OUTPUT":   ["new_output"],
}


@dataclass
class PatchResult:
    """Result of applying a patch to a module."""
    success: bool
    module: Optional[Module]              # deep-copied + patched module
    errors: List[str]
    changed_nodes: List[str]
    tokens_used: int = 0                  # estimated tokens for generation


def validate_patch_schema(patch: Dict) -> List[str]:
    """Check patch is a valid JSON patch with required fields."""
    errors = []
    if not isinstance(patch, dict):
        return ["patch must be a JSON object"]
    if "op" not in patch:
        return ["patch must have 'op' field"]
    op = patch["op"]
    if op not in PATCH_OPS:
        return [f"unknown op '{op}', allowed: {PATCH_OPS}"]
    for field_name in REQUIRED_FIELDS.get(op, []):
        if field_name not in patch:
            errors.append(f"missing required field '{field_name}' for {op}")
    return errors


def check_scope(patch: Dict, slice_info: SSASlice) -> List[str]:
    """Check patch only modifies nodes within the repairable scope."""
    errors = []
    op = patch.get("op", "")

    node_id = patch.get("node_id") or patch.get("anchor_id") or patch.get("old_id")
    if node_id and node_id in slice_info.frozen:
        errors.append(f"cannot modify frozen node '{node_id}'")

    new_output = patch.get("new_output")
    if op == "UPDATE_OUTPUT" and new_output and new_output in slice_info.frozen:
        errors.append(f"cannot set output to frozen node '{new_output}'")

    if slice_info.overflow:
        errors.append("slice_overflow: repair scope exceeds budget")

    return errors


def check_ssa_order(module: Module) -> List[str]:
    """Check SSA definitions appear before uses (after patch application)."""
    errors = []
    seen = set()
    for inp in module.program.inputs:
        seen.add(f"@{inp['name'].lstrip('@')}")
    for node in module.program.nodes:
        for inp in node.inputs:
            if inp not in seen:
                errors.append(f"SSA violation: '{inp}' used before defined in '{node.id}'")
        seen.add(node.id)
    return errors


def check_types(module: Module) -> List[str]:
    """Check node inputs match Skill ISA signatures."""
    errors = []
    for node in module.program.nodes:
        spec = get_skill(node.op)
        if not spec:
            continue
        # Check min/max inputs
        if spec.min_inputs and len(node.inputs) < spec.min_inputs:
            errors.append(f"type: '{node.id}' ({node.op}) needs >= {spec.min_inputs} inputs, got {len(node.inputs)}")
        if spec.max_inputs and len(node.inputs) > spec.max_inputs:
            errors.append(f"type: '{node.id}' ({node.op}) allows <= {spec.max_inputs} inputs, got {len(node.inputs)}")
    return errors


def check_acyclicity(module: Module) -> List[str]:
    """Check no dependency cycles exist after patch."""
    node_ids = {n.id for n in module.program.nodes}
    graph = {n.id: [i for i in n.inputs if i in node_ids] for n in module.program.nodes}

    visited = set()
    in_stack = set()

    def dfs(node_id: str) -> bool:
        if node_id in in_stack:
            return False
        if node_id in visited:
            return True
        in_stack.add(node_id)
        for dep in graph.get(node_id, []):
            if not dfs(dep):
                return False
        in_stack.remove(node_id)
        visited.add(node_id)
        return True

    for nid in node_ids:
        if not dfs(nid):
            return [f"cycle detected involving '{nid}'"]
    return []


def apply_patch(module: Module, patch: Dict, slice_info: SSASlice) -> PatchResult:
    """Apply a single patch to a deep copy of the module, then validate."""
    import copy

    # Schema check
    schema_errors = validate_patch_schema(patch)
    if schema_errors:
        return PatchResult(False, None, schema_errors, [])

    # Scope check
    scope_errors = check_scope(patch, slice_info)
    if scope_errors:
        return PatchResult(False, None, scope_errors, [])

    mod = copy.deepcopy(module)
    nodes = mod.program.nodes
    node_map = {n.id: n for n in nodes}
    changed = []

    op = patch["op"]

    try:
        if op == "REPLACE_INPUT":
            nid = patch["node_id"]
            pos = int(patch["position"])
            new_ref = patch["new_reference"]
            if nid not in node_map:
                return PatchResult(False, None, [f"node '{nid}' not found"], [])
            if pos >= len(node_map[nid].inputs):
                return PatchResult(False, None, [f"position {pos} out of range"], [])
            old = node_map[nid].inputs[pos]
            node_map[nid].inputs[pos] = new_ref
            changed.append(nid)

        elif op == "REPLACE_NODE":
            nid = patch["node_id"]
            if nid not in node_map:
                return PatchResult(False, None, [f"node '{nid}' not found"], [])
            node = node_map[nid]
            node.op = patch["new_op"]
            node.params = patch.get("new_params", {})
            changed.append(nid)

        elif op == "INSERT_BEFORE":
            anchor = patch["anchor_id"]
            if anchor not in node_map:
                return PatchResult(False, None, [f"anchor '{anchor}' not found"], [])
            new_node = Node(
                id=patch["new_id"],
                op=patch["new_op"],
                inputs=patch.get("new_inputs", []),
                params=patch.get("new_params", {}),
            )
            idx = next(i for i, n in enumerate(nodes) if n.id == anchor)
            nodes.insert(idx, new_node)
            node_map[new_node.id] = new_node
            changed.append(new_node.id)

        elif op == "UPDATE_ANNOTATION":
            nid = patch["node_id"]
            if nid not in node_map:
                return PatchResult(False, None, [f"node '{nid}' not found"], [])
            field_name = patch["field"]
            value = patch["value"]
            if field_name == "params":
                node_map[nid].params = value
            elif field_name in ("output_type", "hints"):
                setattr(node_map[nid], field_name, value)
            else:
                return PatchResult(False, None, [f"unknown annotation field '{field_name}'"], [])
            changed.append(nid)

        elif op == "RENAME_SSA":
            old_id = patch["old_id"]
            new_id = patch["new_id"]
            if old_id not in node_map:
                return PatchResult(False, None, [f"node '{old_id}' not found"], [])
            node = node_map.pop(old_id)
            node.id = new_id
            node_map[new_id] = node
            for n in nodes:
                n.inputs = [new_id if i == old_id else i for i in n.inputs]
            if mod.program.output == old_id:
                mod.program.output = new_id
            changed.append(new_id)

        elif op == "UPDATE_OUTPUT":
            new_out = patch["new_output"]
            if new_out not in node_map:
                return PatchResult(False, None, [f"output target '{new_out}' not found"], [])
            mod.program.output = new_out
            changed.append(new_out)

    except Exception as e:
        return PatchResult(False, None, [f"patch application error: {e}"], [])

    # Post-patch checks
    ssa_errors = check_ssa_order(mod)
    type_errors = check_types(mod)
    cycle_errors = check_acyclicity(mod)

    all_errors = ssa_errors + type_errors + cycle_errors
    if all_errors:
        return PatchResult(False, mod, all_errors, changed)

    # Full validator
    try:
        report = validate(mod)
        if not report.valid:
            remaining = [f"{e.code}: {e.msg}" for e in report.errors]
            return PatchResult(False, mod, remaining, changed)
    except Exception as e:
        return PatchResult(False, mod, [f"validator crash: {e}"], [])

    return PatchResult(True, mod, [], changed)
