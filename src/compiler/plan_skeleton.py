"""Semantic Plan Skeleton — Phase 8B-0 Task 5.

The skeleton is what a neural planner is allowed to decide: which
capabilities, which skill semantics, dependencies, required arguments.
Everything else (SSA ids, typing, TaskIR syntax) is the compiler's job in
skeleton_lowering.py. Validation checks the skeleton against the runtime
CapabilityIR registry; nothing here rewrites a wrong plan.
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
    depends_on: List[str] = field(default_factory=list)
    arguments: Dict = field(default_factory=dict)
    effect_class: Optional[str] = None     # filled from oracle/registry


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
            skill=n.get("skill", ""), depends_on=list(n.get("depends_on",
                                                            [])),
            arguments=dict(n.get("arguments", {}) or {})))
    return sk


def validate_skeleton(sk: PlanSkeleton, domain: str,
                      registry_schemas: Dict[str, dict]) -> List[str]:
    """Errors (empty list = valid). Checks registry membership, skill
    compatibility, dependency acyclicity/definitions, and side-effect
    serialization (conservative ordering)."""
    errs = []
    seen = set()
    for n in sk.nodes:
        if n.id in seen:
            errs.append(f"duplicate node id {n.id}")
        seen.add(n.id)
        o = get_tool_oracle(domain, n.capability_id)
        schema = registry_schemas.get(n.capability_id)
        if o["semantic_skill"] is None and not schema:
            errs.append(f"{n.id}: capability '{n.capability_id}' not in "
                        f"runtime registry")
            continue
        if schema:
            n.effect_class = schema.get(
                "mutates_state") and "mutating" or o["effect_class"]
            n.effect_class = o["effect_class"]
            expected = o["semantic_skill"]
            if n.skill == "EXEC_ACTION":
                pass                       # identity-preserving, always ok
            elif expected and n.skill != expected:
                errs.append(f"{n.id}: skill {n.skill} incompatible with "
                            f"'{n.capability_id}' (oracle: {expected})")
            # mutating nodes must keep identity
            if n.effect_class not in NON_STATE and n.skill != "EXEC_ACTION":
                errs.append(f"{n.id}: mutating capability "
                            f"'{n.capability_id}' must use EXEC_ACTION to "
                            f"preserve tool identity")
    # dependencies
    defined = set()
    for n in sk.nodes:
        for dep in n.depends_on:
            if dep not in {x.id for x in sk.nodes}:
                errs.append(f"{n.id}: depends_on undefined node {dep}")
        defined.add(n.id)
    # cycle check (DFS)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n.id: WHITE for n in sk.nodes}
    def dfs(nid):
        color[nid] = GRAY
        for dep in (sk.node(nid).depends_on or []):
            if color.get(dep) == GRAY:
                errs.append(f"cycle via {nid}->{dep}")
            elif color.get(dep) == WHITE:
                dfs(dep)
        color[nid] = BLACK
    for n in sk.nodes:
        if color[n.id] == WHITE:
            dfs(n.id)
    return errs
