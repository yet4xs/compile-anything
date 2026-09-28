"""TaskIR validator v0.1 — the IR verification pass.

Enforces the invariants of spec/taskir-spec.md §6:
  V1 structure, V2 def-use, V3 DAG acyclicity, V4 types,
  V5 skill availability, V6 control flow; plus dead-code warnings.

Any dataset record must pass validate() before it enters data/taskir/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..ir.taskir import ID_RE, Module, Node
from ..ir import types as ty
from ..isa import registry


@dataclass
class Issue:
    code: str
    severity: str            # "error" | "warning"
    msg: str
    node: Optional[str] = None


@dataclass
class Report:
    issues: List[Issue] = field(default_factory=list)
    inferred_types: Dict[str, str] = field(default_factory=dict)  # node id -> type

    @property
    def valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def err(self, code, msg, node=None):
        self.issues.append(Issue(code, "error", msg, node))

    def warn(self, code, msg, node=None):
        self.issues.append(Issue(code, "warning", msg, node))

    def summary(self) -> str:
        n_err, n_warn = len(self.errors), len(self.warnings)
        if not self.issues:
            return "valid, no warnings"
        parts = [f"{n_err} error(s)" if n_err else None,
                 f"{n_warn} warning(s)" if n_warn else None]
        head = f"invalid: {'; '.join(p for p in parts if p)}" if n_err \
            else f"valid with {'; '.join(p for p in parts if p)}"
        return head


def _dep_edges(node: Node) -> List[str]:
    """References that create graph edges (data + control + guard).
    Retry edges are temporal back-edges and are excluded from cycle checking."""
    edges = list(node.inputs) + list(node.after)
    if node.guard is not None:
        edges.append(node.guard.cond)
    return edges


def validate(mod: Module) -> Report:
    rep = Report()
    prog = mod.program

    # ---- V1 structure ------------------------------------------------------
    if mod.taskir_version != "0.1":
        rep.warn("VERSION", f"taskir_version={mod.taskir_version!r}, spec'd: 0.1")
    if not prog.name:
        rep.err("STRUCT", "program.name is empty")

    globals_declared: Dict[str, str] = {}
    for g in prog.inputs:
        gname = g.get("name", "")
        if not ID_RE.match(gname or "") or not gname.startswith("@"):
            rep.err("STRUCT", f"bad input name {gname!r} (must be '@name')")
        elif gname in globals_declared:
            rep.err("STRUCT", f"duplicate input {gname}")
        else:
            globals_declared[gname] = ty.normalize(g.get("type", "Any"))

    nodes = prog.nodes
    defined: Dict[str, Node] = {}
    for i, n in enumerate(nodes):
        if not ID_RE.match(n.id or "") or not n.id.startswith("%"):
            rep.err("STRUCT", f"bad node id {n.id!r} (must be '%name')", n.id)
        if n.id in defined or n.id in globals_declared:
            rep.err("STRUCT", f"duplicate id {n.id}", n.id)
            continue
        defined[n.id] = n
        if not isinstance(n.op, str) or not n.op:
            rep.err("STRUCT", "node.op must be a non-empty string", n.id)
        if not isinstance(n.params, dict):
            rep.err("STRUCT", "node.params must be a dict", n.id)
        if n.retry is not None and not isinstance(n.retry.max_attempts, int):
            rep.err("STRUCT", "retry.max_attempts must be an int", n.id)

    def type_of(ref: str) -> Optional[str]:
        if ref in globals_declared:
            return globals_declared[ref]
        if ref in rep.inferred_types:
            return rep.inferred_types[ref]
        return None

    # ---- V2/V4/V5/V6 per-node pass (canonical order) -----------------------
    verify_ids = {n.id for n in nodes if n.op == "VERIFY"}

    for n in nodes:
        if n.id not in defined:        # already reported (duplicate/struct)
            continue
        spec = registry.get(n.op)

        # V5 skill availability
        if spec is None:
            rep.err("SKILL_UNKNOWN", f"op {n.op!r} not in Skill ISA registry", n.id)
            rep.inferred_types[n.id] = "Any"
            continue

        # V2 def-use (strict ordering for data/after/guard)
        for ref in n.inputs:
            if ref in globals_declared:
                continue
            if ref not in defined:
                rep.err("UNDEFINED_REF", f"input {ref!r} is never defined", n.id)
            elif list(defined).index(ref) >= list(defined).index(n.id):
                rep.err("USE_BEFORE_DEF", f"input {ref!r} used before defined", n.id)
        for ref in n.after:
            if ref not in defined and ref not in globals_declared:
                rep.err("UNDEFINED_REF", f"after {ref!r} is never defined", n.id)
            elif ref in defined and list(defined).index(ref) >= list(defined).index(n.id):
                rep.err("USE_BEFORE_DEF", f"after {ref!r} used before defined", n.id)
        if n.guard is not None:
            g = n.guard
            if g.cond not in defined and g.cond not in globals_declared:
                rep.err("UNDEFINED_REF", f"guard.cond {g.cond!r} undefined", n.id)
            elif g.cond in defined and list(defined).index(g.cond) >= list(defined).index(n.id):
                rep.err("USE_BEFORE_DEF", f"guard.cond {g.cond!r} used before defined", n.id)
            if not isinstance(g.expect, bool):
                rep.err("CONTROL", "guard.expect must be a JSON bool", n.id)

        # arity (V6)
        nn = len(n.inputs)
        if nn < spec.min_inputs:
            rep.err("ARITY", f"{n.op} expects >= {spec.min_inputs} inputs, got {nn}", n.id)
        if spec.max_inputs is not None and nn > spec.max_inputs:
            rep.err("ARITY", f"{n.op} expects <= {spec.max_inputs} inputs, got {nn}", n.id)

        # V4 types
        input_types: List[str] = []
        for j, ref in enumerate(n.inputs):
            actual = type_of(ref)
            if actual is None:
                input_types.append("Any")   # cascade guard; error already reported
                continue
            input_types.append(actual)
            if j < len(spec.input_types):
                declared = spec.input_types[j]
                if not ty.is_compatible(declared, actual):
                    rep.err("TYPE_MISMATCH",
                            f"input #{j+1} ({ref}: {actual}) incompatible with "
                            f"{n.op} signature ({declared})", n.id)
        inferred = spec.output_type(input_types) if len(input_types) == nn else "Any"
        if n.output_type is not None:
            declared = ty.normalize(n.output_type)
            if not ty.is_compatible(inferred, declared):
                rep.err("TYPE_MISMATCH",
                        f"explicit output_type {declared!r} incompatible with "
                        f"inferred {inferred!r}", n.id)
            inferred = declared
        rep.inferred_types[n.id] = inferred

        # V6 control-flow semantics
        if n.guard is not None and n.guard.cond in rep.inferred_types:
            ct = rep.inferred_types[n.guard.cond]
            if not ty.is_compatible("Bool", ct):
                rep.err("TYPE_MISMATCH",
                        f"guard.cond {n.guard.cond} must be Bool, got {ct}", n.id)
        if n.op == "SELECT":
            if len(input_types) == 3 and nn == 3:
                if not ty.is_compatible("Bool", input_types[0]):
                    rep.err("TYPE_MISMATCH", "SELECT cond must be Bool", n.id)
                if not ty.is_compatible(input_types[1], input_types[2]):
                    rep.err("TYPE_MISMATCH",
                            f"SELECT branches differ: {input_types[1]} vs {input_types[2]}",
                            n.id)
        if n.retry is not None:
            if n.retry.max_attempts < 1:
                rep.err("CONTROL", "retry.max_attempts must be >= 1", n.id)
            if n.retry.on != "error":
                if n.retry.on not in defined:
                    rep.err("UNDEFINED_REF",
                            f"retry.on {n.retry.on!r} is never defined", n.id)
                elif n.retry.on not in verify_ids:
                    rep.err("CONTROL",
                            f"retry.on {n.retry.on!r} must reference a VERIFY node",
                            n.id)
                else:
                    if not _depends_on(defined[n.retry.on], n.id, defined):
                        rep.err("CONTROL",
                                f"retry.on VERIFY {n.retry.on!r} does not depend on "
                                f"this node; retry would be meaningless", n.id)

    # ---- V3 DAG acyclicity (data + after + guard edges) ---------------------
    color: Dict[str, int] = {}

    def dfs(nid: str) -> bool:
        color[nid] = 1
        node = defined.get(nid)
        if node is not None:
            for ref in _dep_edges(node):
                if ref in defined:
                    if color.get(ref) == 1:
                        return False
                    if color.get(ref, 0) == 0 and not dfs(ref):
                        return False
        color[nid] = 2
        return True

    for nid in defined:
        if color.get(nid, 0) == 0:
            if not dfs(nid):
                rep.err("CYCLE", f"dependency cycle through {nid!r}", nid)
                break

    # ---- output check (V6) ---------------------------------------------------
    if prog.output:
        if prog.output not in defined:
            rep.err("OUTPUT", f"program.output {prog.output!r} is not a defined node")
    else:
        rep.err("OUTPUT", "program.output is empty")

    # ---- W dead code ----------------------------------------------------------
    live = set()

    def mark_live(nid: str):
        if nid in live:
            return
        live.add(nid)
        node = defined.get(nid)
        if node is None:
            return
        for ref in _dep_edges(node):
            mark_live(ref)
        if node.retry is not None and node.retry.on in defined:
            mark_live(node.retry.on)

    if prog.output in defined:
        mark_live(prog.output)
    for nid in defined:
        if nid not in live:
            rep.warn("DEAD_NODE", f"node never reaches output", nid)
    used_globals = set()
    for n in defined.values():
        used_globals.update(r for r in _dep_edges(n) if r in globals_declared)
    for gname in globals_declared:
        if gname not in used_globals:
            rep.warn("UNUSED_INPUT", f"program input {gname} is never used")

    return rep


def _depends_on(node: Node, target: str, defined: Dict[str, Node],
                seen: Optional[set] = None) -> bool:
    """Does `node` transitively depend (data/after/guard) on `target`?"""
    if seen is None:
        seen = set()
    if node.id in seen:
        return False
    seen.add(node.id)
    for ref in _dep_edges(node):
        if ref == target:
            return True
        if ref in defined and _depends_on(defined[ref], target, defined, seen):
            return True
    return False


def validate_file(path) -> Report:
    from ..ir.taskir import load_module
    return validate(load_module(path))
