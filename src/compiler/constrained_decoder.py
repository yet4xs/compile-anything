"""[DEPRECATED — Phase 7C-0.1] Use src/compiler/constrained_assembly.py instead.

This v1 module parsed model output with a custom regex instead of the official
src/ir/parser.py::parse_text(), which corrupted valid programs (D1 was NOT
lossless) and rejected legal numeric SSA ids like %1. Its "decode time"
framing was also inaccurate: constraints were applied AFTER a full free-form
completion, i.e. post-hoc constrained assembly, not token-level constrained
decoding.

Retained for history; do not import in new code.
---
SSA-Constrained TaskIR Generation — prevents invalid programs at decode time.

Three constraint levels:
  Level 1 (Grammar): TaskIR text format, SSA ID format, return structure
  Level 2 (Typed SSA): def-before-use, Skill ISA operand types/arity, no duplicate IDs
  Level 3 (Control/Effect): after/guard/retry static legality
"""
from __future__ import annotations
import re, json
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field

from src.ir.taskir import Module, Program, Node
from src.validator.validator import validate
from src.isa.registry import get as get_skill, REGISTRY

VALID_OPS = set(REGISTRY.keys()) | {"VERIFY", "SELECT"}
POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}


@dataclass
class SSAState:
    """Tracks the current SSA symbol table during generation."""
    defined: Set[str] = field(default_factory=set)
    node_types: Dict[str, str] = field(default_factory=dict)  # id -> output_type
    node_ops: Dict[str, str] = field(default_factory=dict)     # id -> op
    program_inputs: Set[str] = field(default_factory=set)

    def define(self, node_id: str, op: str, output_type: Optional[str] = None):
        self.defined.add(node_id)
        self.node_ops[node_id] = op
        if output_type:
            self.node_types[node_id] = output_type

    def is_defined(self, ref: str) -> bool:
        return ref in self.defined or ref in self.program_inputs

    def type_of(self, ref: str) -> Optional[str]:
        return self.node_types.get(ref)


@dataclass
class ConstrainedStep:
    """One step of constrained generation."""
    accepted: bool
    node: Optional[Node] = None
    reason: str = ""
    corrected_inputs: Optional[List[str]] = None  # if we auto-fixed inputs


class SSAConstraintEngine:
    """Validates and auto-corrects TaskIR node generation at the operation level."""

    def __init__(self, module_inputs: List[Dict[str, str]], level: int = 3):
        """
        Args:
            module_inputs: [{"name": "@query", "type": "Str"}, ...]
            level: 1=grammar, 2=typed SSA, 3=control/effect
        """
        self.level = level
        self.ssa = SSAState()
        self.nodes: List[Node] = []
        for inp in module_inputs:
            name = inp["name"]
            self.ssa.program_inputs.add(name)

    def reset(self, module_inputs: List[Dict[str, str]]):
        self.ssa = SSAState()
        self.nodes = []
        for inp in module_inputs:
            self.ssa.program_inputs.add(inp["name"])

    def check_grammar(self, node_id: str, op: str) -> List[str]:
        """Level 1: Grammar checks."""
        errors = []
        if not re.match(r'^%[a-zA-Z_]\w*$', node_id):
            errors.append(f"invalid SSA ID format: '{node_id}' (must be %name)")
        if op not in VALID_OPS:
            errors.append(f"unknown operation: '{op}' (must be one of {sorted(VALID_OPS)[:10]}...)")
        if node_id in self.ssa.defined:
            errors.append(f"duplicate SSA ID: '{node_id}' already defined")
        return errors

    def check_typed_ssa(self, inputs: List[str], op: str) -> Tuple[List[str], Optional[List[str]]]:
        """Level 2: Typed SSA checks. Returns (errors, corrected_inputs).

        If auto-fix is possible (replace undefined ref with type-compatible node),
        returns corrected_inputs instead of an error.
        """
        errors = []
        corrected = list(inputs)
        spec = get_skill(op)
        fixed = False

        for i, ref in enumerate(inputs):
            if not self.ssa.is_defined(ref):
                # Try auto-fix: find type-compatible defined node
                if spec and spec.input_types:
                    expected = spec.input_types[min(i, len(spec.input_types)-1)]
                    candidates = [nid for nid in sorted(self.ssa.defined)
                                  if expected in (self.ssa.type_of(nid) or "Any", "Any")]
                    if not candidates:
                        candidates = sorted(self.ssa.program_inputs | self.ssa.defined)
                    if candidates:
                        corrected[i] = candidates[0]
                        fixed = True
                else:
                    # No type info, use any defined node
                    candidates = sorted(self.ssa.program_inputs | self.ssa.defined)
                    if candidates:
                        corrected[i] = candidates[0]
                        fixed = True

        if not fixed:
            for i, ref in enumerate(inputs):
                if not self.ssa.is_defined(ref):
                    errors.append(f"undefined reference: '{ref}'")

        # Arity check
        if spec:
            if spec.min_inputs and len(inputs) < spec.min_inputs:
                errors.append(f"'{op}' needs >= {spec.min_inputs} inputs, got {len(inputs)}")
            if spec.max_inputs and len(inputs) > spec.max_inputs:
                # Auto-fix: truncate
                corrected = corrected[:spec.max_inputs]
                fixed = True

        return errors, (corrected if fixed else None)

    def check_control(self, node: Node) -> List[str]:
        """Level 3: Control/effect checks."""
        errors = []
        if node.guard:
            cond = node.guard.cond
            if not self.ssa.is_defined(cond):
                errors.append(f"guard references undefined: '{cond}'")
        if node.retry:
            retry_on = node.retry.on
            if not self.ssa.is_defined(retry_on):
                errors.append(f"retry references undefined: '{retry_on}'")
            if retry_on == node.id:
                errors.append(f"retry references self: '{node.id}'")
        return errors

    def submit(self, node_id: str, op: str, inputs: List[str], params: Dict,
               output_type: Optional[str] = None, **kwargs) -> ConstrainedStep:
        """Submit a candidate node. Returns accepted/corrected/rejected."""
        # Level 1: Grammar
        if self.level >= 1:
            grammar_errors = self.check_grammar(node_id, op)
            if grammar_errors:
                return ConstrainedStep(False, reason="; ".join(grammar_errors))

        # Level 2: Typed SSA (with auto-fix)
        corrected_inputs = inputs
        if self.level >= 2:
            ssa_errors, fixed_inputs = self.check_typed_ssa(inputs, op)
            if fixed_inputs is not None:
                corrected_inputs = fixed_inputs
            elif ssa_errors:
                return ConstrainedStep(False, reason="; ".join(ssa_errors),
                                       corrected_inputs=inputs)

        # Create node
        node = Node(
            id=node_id, op=op, inputs=corrected_inputs, params=params,
            output_type=output_type, **kwargs)

        # Level 3: Control/effect
        if self.level >= 3:
            control_errors = self.check_control(node)
            if control_errors:
                return ConstrainedStep(False, node=node, reason="; ".join(control_errors))

        # Accept: register in SSA
        spec = get_skill(op)
        if spec and spec.output_rule:
            ot = spec.output_rule.replace("FIXED:", "").replace("SAME_AS:0", "Any")
        else:
            ot = output_type
        self.ssa.define(node_id, op, ot)
        self.nodes.append(node)
        return ConstrainedStep(True, node=node, corrected_inputs=corrected_inputs)

    def set_output(self, node_id: str) -> Tuple[bool, str]:
        """Set program output. Returns (success, error_msg)."""
        if not self.ssa.is_defined(node_id):
            return False, f"output references undefined node: '{node_id}'"
        return True, ""

    def build_module(self, name: str, output: str) -> Optional[Module]:
        """Build the final module. Returns None if invalid."""
        prog = Program(
            name=name,
            inputs=[{"name": n, "type": "Any"} for n in sorted(self.ssa.program_inputs)],
            nodes=self.nodes,
            output=output,
        )
        mod = Module(program=prog)
        try:
            if validate(mod).valid:
                return mod
        except Exception:
            pass
        return None


def parse_taskir_text_incremental(text: str, constraint_engine: SSAConstraintEngine,
                                   inputs: List[Dict[str, str]]) -> Tuple[List[ConstrainedStep], Optional[Module]]:
    """Parse TaskIR text and apply constraints at each node definition.

    Returns list of steps + the final module if valid.
    """
    constraint_engine.reset(inputs)
    steps = []

    lines = text.strip().split("\n")
    last_node_id = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue

        # Parse node definition: %id = OP(params...) [inputs]
        m = re.match(r'(%\w+)\s*=\s*(\w+)\s*\((.*?)\)\s*(?:\[(.*?)\])?', line)
        if not m:
            continue

        node_id = m.group(1)
        op = m.group(2)
        param_str = m.group(3) or ""
        input_str = m.group(4) or ""

        # Parse inputs
        inputs_list = [x.strip() for x in input_str.split(",") if x.strip()] if input_str else []

        # Parse params (simple key=value pairs)
        params = {}
        for pm in re.finditer(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|[\w.-]+)', param_str):
            key, val = pm.group(1), pm.group(2).strip("'\"")
            params[key] = val

        # Submit through constraint engine
        step = constraint_engine.submit(node_id, op, inputs_list, params)
        steps.append(step)
        if step.accepted:
            last_node_id = node_id

    # Try to set output
    if last_node_id:
        ok, _ = constraint_engine.set_output(last_node_id)
        if ok:
            module = constraint_engine.build_module("generated", last_node_id)
            return steps, module

    return steps, None
