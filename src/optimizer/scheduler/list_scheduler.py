"""Resource-constrained list scheduler for TaskIR programs (v0.1).

Input : a validated TaskIR module (a DAG in canonical order)
Output: a feasible schedule — start/end times per node under a bounded
        executor pool per resource class.

Model
-----
Edges respected (same set the validator's V3 uses):
    data (node.inputs) + control-only (node.after) + guard conditions
    (node.guard.cond). Retry edges are temporal back-edges and are NOT
    scheduling dependencies (see docs/runtime-review.md Q3).
When the effect-token extension lands (docs/effect-system-proposal.md),
`effect_in` references join `_edges()` — that is the "effect respect" rule:
same-class actions serialize through the chain.

Executors are grouped by resource class (api / python / lm / db / runtime),
each class has a configurable pool size (default 1). A ready node starts at
    start = max(earliest-after-deps, class pool free time).
Priority = latency-weighted longest path to any sink (classic list
scheduling critical-path heuristic); ties break by program order.

Known simplifications (documented, by design for v0.1):
  - both guarded branches are scheduled optimistically; the runtime may
    still skip a branch, so the schedule is an upper-bound envelope;
  - retry re-executions are not modeled (profiling-driven expected-latency
    analysis is future scheduler work).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...ir.taskir import Module, Node
from ...runtime import executors as exec_mod
from ...isa.registry import SkillSpec


@dataclass
class ScheduledItem:
    node: str
    op: str
    resource_class: str
    executor: str
    start_ms: float
    end_ms: float

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class Schedule:
    program: str
    items: List[ScheduledItem] = field(default_factory=list)
    makespan_ms: float = 0.0
    resource_limits: Dict[str, int] = field(default_factory=dict)
    ok: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def order(self) -> List[str]:
        return [i.node for i in sorted(self.items, key=lambda x: (x.start_ms,
                                                                  x.end_ms))]

    def to_dict(self) -> Dict:
        return {
            "program": self.program,
            "ok": self.ok,
            "makespan_ms": self.makespan_ms,
            "resource_limits": self.resource_limits,
            "notes": self.notes,
            "schedule": [i.to_dict() for i in
                         sorted(self.items, key=lambda x: (x.start_ms, x.end_ms))],
        }


def _edges(node: Node) -> List[str]:
    """Scheduling dependencies. Effect tokens (v0.2 IR) will join here."""
    refs = list(node.inputs) + list(node.after)
    if node.guard is not None:
        refs.append(node.guard.cond)
    return refs


def _spec(op: str) -> Optional[SkillSpec]:
    return exec_mod.spec_of(op)


class ListScheduler:
    def __init__(self, module: Module,
                 pools: Optional[Dict[str, int]] = None):
        self.mod = module
        self.pools = dict(pools or {})      # class -> pool size (default 1)

    # ------------------------------------------------------------------ api
    def schedule(self) -> Schedule:
        nodes: List[Node] = self.mod.program.nodes
        by_id = {n.id: n for n in nodes}
        sched = Schedule(program=self.mod.program.name,
                         resource_limits=dict(self.pools))

        # latency-weighted priority: longest path to any sink
        prio: Dict[str, float] = {}

        def priority(nid: str) -> float:
            if nid in prio:
                return prio[nid]
            n = by_id[nid]
            spec = _spec(n.op)
            lat = spec.cost.latency_ms if spec else 5.0
            succ = [priority(u) for u in by_id
                    if nid in _edges(by_id[u])]
            prio[nid] = lat + (max(succ) if succ else 0.0)
            return prio[nid]

        for n in nodes:
            priority(n.id)

        # event-driven list scheduling
        done_at: Dict[str, float] = {}
        class_free: Dict[str, List[float]] = {}      # class -> free times
        remaining = {n.id: set(r for r in _edges(n) if r in by_id)
                     for n in nodes}

        while remaining:
            ready = [nid for nid, deps in remaining.items() if not deps]
            if not ready:
                sched.notes.append("dependency cycle: unscheduled "
                                   + ", ".join(sorted(remaining)))
                return sched
            # highest critical-path priority first; program order breaks ties
            nid = max(ready, key=lambda x: (priority(x), -nodes.index(by_id[x])))
            node = by_id[nid]
            spec = _spec(node.op)
            rclass = spec.resource_class if spec else "python"
            lat = spec.cost.latency_ms if spec else 5.0
            pool = self.pools.get(rclass, 1)
            free = class_free.setdefault(rclass, [0.0] * pool)

            deps_end = max((done_at[d] for d in _edges(node) if d in done_at),
                           default=0.0)
            slot = min(range(pool), key=lambda i: free[i])
            start = max(deps_end, free[slot])
            end = start + lat
            free[slot] = end
            done_at[nid] = end

            sched.items.append(ScheduledItem(
                node=nid, op=node.op, resource_class=rclass,
                executor=(spec.executors[0] if spec and spec.executors
                          else f"{rclass}:mock"),
                start_ms=round(start, 3), end_ms=round(end, 3)))

            del remaining[nid]
            for deps in remaining.values():
                deps.discard(nid)

        sched.items.sort(key=lambda i: (i.start_ms, i.end_ms))
        sched.makespan_ms = max((i.end_ms for i in sched.items), default=0.0)
        sched.ok = True
        self._verify(by_id, sched)
        return sched

    # -------------------------------------------------------------- checks
    @staticmethod
    def _verify(by_id: Dict[str, Node], sched: Schedule) -> None:
        """Internal sanity: deps end before consumer starts."""
        end_of = {i.node: i.end_ms for i in sched.items}
        for i in sched.items:
            for ref in _edges(by_id[i.node]):
                if ref in end_of and end_of[ref] > i.start_ms + 1e-9:
                    sched.ok = False
                    sched.notes.append(
                        f"violation: {ref} ends {end_of[ref]} after "
                        f"{i.node} starts {i.start_ms}")
                    return
