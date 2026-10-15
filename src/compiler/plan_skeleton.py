"""Semantic Plan Skeleton — Phase 8B-0 Task 5, hardened in 8B-1A Task 2.

The skeleton is what a neural planner is allowed to decide: which
capabilities, which skill semantics, dependencies, required arguments.
Everything else (SSA ids, typing, TaskIR syntax) is the compiler's job in
skeleton_lowering.py. Validation checks the skeleton against the runtime
CapabilityIR registry; nothing here rewrites a wrong plan.

8B-1A invariants:
- depends_on may only reference nodes defined EARLIER (no silent forward
  references; a topological order must be explicit)
- EXEC_ACTION never silently bypasses skill/effect compatibility: oracle
  skill mismatches are surfaced as warnings (identity-preserving calls of
  read tools are legal but flagged)
- mutating capabilities MUST keep EXEC_ACTION identity
- order_on separates ORDERING dependencies from DATA dependencies
  (depends_on)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.eval.tau3_skill_oracle import get_tool_oracle

NON_STATE = ("read_only", "communication", "agent_internal")


@dataclass
class SkeletonNode:
    id: str
    capability_id: str
    skill: str
    depends_on: List[str] = field(default_factory=list)   # DATA deps
    order_on: List[str] = field(default_factory=list)     # ORDERING deps
    arguments: Dict = field(default_factory=dict)
    effect_class: Optional[str] = None     # filled from oracle/registry

    # arguments values may be:
    #   literal: any JSON value (compile-time constant)
    #   runtime_ref: {"kind": "runtime_ref", "producer": "p0",
    #                 "field": "user_id"}
    #   unresolved: None (missing; lowering will report)


@dataclass
class PlanSkeleton:
    task_id: str
    nodes: List[SkeletonNode] = field(default_factory=list)

    def node(self, nid):
        return next((n for n in self.nodes if n.id == nid), None)


def skeleton_from_dict(d: dict) -> PlanSkeleton:
    sk = PlanSkeleton(task_id=d.get("task_id", ""))
    for n in d.get("nodes", []):
        sk.nodes.append(SkeletonNode(
            id=n["id"], capability_id=n.get("capability_id", ""),
            skill=n.get("skill", ""),
            depends_on=list(n.get("depends_on", [])),
            order_on=list(n.get("order_on", [])),
            arguments=dict(n.get("arguments", {}) or {})))
    return sk


def skeleton_to_dict(sk: PlanSkeleton) -> dict:
    return {"task_id": sk.task_id,
            "nodes": [{"id": n.id, "capability_id": n.capability_id,
                       "skill": n.skill, "depends_on": list(n.depends_on),
                       "order_on": list(n.order_on),
                       "arguments": dict(n.arguments)}
                      for n in sk.nodes]}


def validate_skeleton(sk: PlanSkeleton, domain: str,
                      registry_schemas: Dict[str, dict]) -> List[str]:
    """Errors (empty list = valid). Skill mismatches are appended as
    'warning:' entries — legal but surfaced, never silently bypassed."""
    errs = []
    seen = set()
    for idx, n in enumerate(sk.nodes):
        if n.id in seen:
            errs.append(f"duplicate node id {n.id}")
        seen.add(n.id)
        o = get_tool_oracle(domain, n.capability_id)             if domain else {"semantic_skill": None, "effect_class": None}
        schema = registry_schemas.get(n.capability_id)
        if o["semantic_skill"] is None and not schema:
            errs.append(f"{n.id}: capability '{n.capability_id}' not in "
                        f"runtime registry")
            continue
        n.effect_class = o["effect_class"]
        if schema:
            expected = o["semantic_skill"]
            if expected is None:
                # non-tau3 tool (corpus registry): identity/skill kept
                # as declared; effect unknown -> flagged, not guessed
                errs.append(f"warning: {n.id}: '{n.capability_id}' has no "
                            f"effect oracle (skill kept as declared)")
                continue
            if n.skill == "EXEC_ACTION":
                # identity-preserving: legal, but a read-tool call through
                # EXEC_ACTION is flagged (its oracle skill differs)
                if expected and expected != "EXEC_ACTION":
                    errs.append(f"warning: {n.id}: EXEC_ACTION call of "
                                f"'{n.capability_id}' whose oracle skill is "
                                f"{expected} (identity kept, mismatch "
                                f"flagged)")
            elif expected and n.skill != expected:
                errs.append(f"{n.id}: skill {n.skill} incompatible with "
                            f"'{n.capability_id}' (oracle: {expected})")
            if n.effect_class not in NON_STATE and n.skill != "EXEC_ACTION":
                errs.append(f"{n.id}: mutating capability "
                            f"'{n.capability_id}' must use EXEC_ACTION to "
                            f"preserve tool identity")
    # dependency references: defined AND earlier (no forward refs)
    for idx, n in enumerate(sk.nodes):
        earlier = {x.id for x in sk.nodes[:idx]}
        for ref in list(n.depends_on) + list(n.order_on):
            if ref not in {x.id for x in sk.nodes}:
                errs.append(f"{n.id}: depends_on/order_on undefined "
                            f"node {ref}")
            elif ref not in earlier:
                errs.append(f"{n.id}: forward reference {ref} — nodes must "
                            f"be topologically ordered")
    # cycle check (DFS over combined edges)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n.id: WHITE for n in sk.nodes}

    def dfs(nid):
        color[nid] = GRAY
        node = sk.node(nid)
        for dep in list(node.depends_on) + list(node.order_on):
            if color.get(dep) == GRAY:
                errs.append(f"cycle via {nid}->{dep}")
            elif color.get(dep) == WHITE:
                dfs(dep)
        color[nid] = BLACK

    for n in sk.nodes:
        if color[n.id] == WHITE:
            dfs(n.id)
    return errs


def hard_errors(errs: List[str]) -> List[str]:
    """Errors that block lowering (warnings excluded)."""
    return [e for e in errs if not e.startswith("warning:")]
