"""TaskIR runtime simulator v0.1.

Demand-driven (pull-based) interpreter over the SSA node list:
  - data/control/guard dependencies evaluated on demand, memoized;
  - guards skip nodes whose predicate does not hold;
  - retry: on executor error, or on a downstream VERIFY flipping to False
    (which invalidates the node's value and all transitive dependents —
    rollback-lite — then re-executes up to max_attempts);
  - deterministic given the same seed and optional fail_plan injection.

Produces an execution trace and cost totals (see src/cost/model.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ir.taskir import Module, Node
from ..ir import types as ty
from . import executors as exec_mod
from .executors import _digest


class _Skipped:
    """Sentinel value for nodes whose guard predicate did not hold."""
    def __repr__(self):
        return "<skipped>"


SKIPPED = _Skipped()


class RuntimeFailure(Exception):
    pass


@dataclass
class TraceEvent:
    seq: int
    node: str
    op: str
    attempt: int
    status: str                 # ok | skipped | error | retry_exhausted
    latency_ms: float
    tokens_in: int
    tokens_out: int
    flops: float
    resource_class: str
    energy_j: float = 0.0
    memory_mb: float = 0.0
    value_digest: str = ""
    note: str = ""
    superseded: bool = False   # historical event invalidated by a rollback

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ExecutionResult:
    module: Module
    events: List[TraceEvent] = field(default_factory=list)
    status: str = "completed"                     # completed | failed
    output_id: str = ""
    output_digest: str = ""
    values: Dict[str, Any] = field(default_factory=dict)
    inferred_types: Dict[str, str] = field(default_factory=dict)
    # cost totals
    seq_latency_ms: float = 0.0
    critical_path_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    flops: float = 0.0
    lm_calls: int = 0
    api_calls: int = 0
    node_calls: int = 0
    retries: int = 0
    skipped: int = 0
    energy_j: float = 0.0
    peak_memory_mb: float = 0.0   # max single-node footprint; no overlap modeled

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("module", "values", "events")}
        d["events"] = [e.to_dict() for e in self.events]
        return d


class Simulator:
    def __init__(self, module: Module, seed: str = "sim",
                 fail_plan: Optional[Dict[str, List[str]]] = None,
                 jitter: float = 0.1):
        self.mod = module
        self.seed = seed
        # fail_plan: node id -> list of per-attempt outcomes, e.g.
        #   {"%4": ["error"]}                first attempt errors
        #   {"%5": ["verify_false"]}         first attempt of VERIFY returns False
        self.fail_plan = fail_plan or {}
        self.jitter = jitter
        self.nodes: Dict[str, Node] = {n.id: n for n in module.program.nodes}
        self.memo: Dict[str, Any] = {}
        self.attempts: Dict[str, int] = {}
        self.events: List[TraceEvent] = []
        self._rng = random.Random(f"{seed}|jitter")
        self._in_progress: set = set()

    # ------------------------------------------------------------------ run
    def run(self) -> ExecutionResult:
        res = ExecutionResult(module=self.mod)
        prog = self.mod.program
        res.output_id = prog.output
        try:
            out = self._eval(prog.output)
            # VERIFY-driven rollback phase (checkpoint/restore style):
            # demand every VERIFY referenced by a retry policy, then for each
            # verdict==False with remaining budget, invalidate the retried
            # node's value and all transitive dependents and re-execute.
            # This runs OUTSIDE the demand chain on purpose: probing the
            # verify inline inside _eval would demand nodes that are still
            # on the evaluation stack (A -> B -> C(VERIFY), retry on A) and
            # be misreported as a cycle.
            for n in prog.nodes:
                if n.retry and n.retry.on != "error" and n.retry.on in self.nodes:
                    self._eval(n.retry.on)
            while True:
                cand = [n for n in prog.nodes
                        if n.retry and n.retry.on != "error"
                        and n.retry.on in self.nodes
                        and self.memo.get(n.retry.on) is False
                        and self.attempts.get(n.id, 0) < n.retry.max_attempts]
                if not cand:
                    break
                target = cand[0]                      # program order: deterministic
                cleared = self._invalidate_dependents(target.id)
                self._eval(prog.output)
                # re-evaluate ALL cleared nodes in program order so that the
                # post-rollback memory state is consistent everywhere (a
                # dependent that is not output-reachable must not keep a
                # stale trace event / no value at all)
                for n in prog.nodes:
                    if n.id in cleared:
                        self._eval(n.id)
                self._eval(target.retry.on)
            for n in prog.nodes:
                if n.retry and n.retry.on != "error" and n.retry.on in self.nodes \
                        and self.memo.get(n.retry.on) is False \
                        and self.attempts.get(n.id, 0) >= n.retry.max_attempts:
                    self._mark_retry_exhausted(n.id)
            res.output_digest = _digest(self.memo.get(prog.output, out))
        except RuntimeFailure as e:
            res.status = "failed"
            res.output_digest = f"<failed: {e}>"

        res.events = list(self.events)
        res.values = {k: (v if isinstance(v, _Skipped) else v)
                      for k, v in self.memo.items()}
        ok_events = [e for e in self.events if e.status in ("ok", "retry_exhausted")]
        res.seq_latency_ms = round(sum(e.latency_ms for e in self.events
                                       if e.status != "skipped"), 3)
        res.critical_path_ms = round(self._critical_path_ms(), 3)
        res.tokens_in = sum(e.tokens_in for e in self.events if e.status != "skipped")
        res.tokens_out = sum(e.tokens_out for e in self.events if e.status != "skipped")
        res.flops = sum(e.flops for e in self.events if e.status != "skipped")
        res.lm_calls = sum(1 for e in self.events
                           if e.status != "skipped" and e.resource_class == "lm")
        res.api_calls = sum(1 for e in self.events
                            if e.status != "skipped" and e.resource_class in ("api", "db"))
        res.node_calls = sum(1 for e in self.events if e.status != "skipped")
        res.retries = sum(max(0, self.attempts.get(n.id, 1) - 1)
                          for n in self.nodes.values() if n.retry is not None)
        res.skipped = sum(1 for e in self.events if e.status == "skipped")
        res.energy_j = round(sum(e.energy_j for e in self.events
                                 if e.status != "skipped"), 4)
        res.peak_memory_mb = max((e.memory_mb for e in self.events
                                  if e.status != "skipped"), default=0.0)
        return res

    # ------------------------------------------------------------------ eval
    def _eval(self, nid: str) -> Any:
        if nid in self.memo:
            return self.memo[nid]
        if nid in self._in_progress:
            raise RuntimeFailure(f"dependency cycle detected at runtime: {nid}")
        if not nid.startswith("%"):
            # program input (@global): synthesize a concrete mock value from
            # its declared type so consumers (FILTER/MIN/...) have data
            return self._global_value(nid)

        node = self.nodes[nid]
        self._in_progress.add(nid)

        try:
            if node.guard is not None:
                cond = self._eval(node.guard.cond)
                if isinstance(cond, _Skipped):
                    # predication propagation: the condition value itself was
                    # skipped, so this node's predicate is unknown -> skip
                    self._event(node, attempt=0, status="skipped", latency=0.0,
                                value=SKIPPED,
                                note=f"guard cond {node.guard.cond} skipped "
                                     f"(predication propagation)")
                    self.memo[nid] = SKIPPED
                    return SKIPPED
                if bool(cond) != node.guard.expect:
                    ev = self._event(node, attempt=0, status="skipped",
                                     latency=0.0, value=SKIPPED,
                                     note=f"guard {node.guard.cond} != {node.guard.expect}")
                    self.memo[nid] = SKIPPED
                    return SKIPPED

            max_att = node.retry.max_attempts if node.retry else 1

            while True:
                attempt = self.attempts.get(nid, 0) + 1
                self.attempts[nid] = attempt
                inputs = [self._eval(r) for r in node.inputs]
                if node.op == "SELECT":
                    # SELECT is the branch reconvergence point: the non-taken
                    # branch is legitimately skipped by its guard; only a
                    # skipped CHOSEN branch is a runtime inconsistency.
                    if isinstance(inputs[0], _Skipped):
                        # unknown condition -> predication propagation
                        self._event(node, attempt=0, status="skipped",
                                    latency=0.0, value=SKIPPED,
                                    note="SELECT cond skipped "
                                         "(predication propagation)")
                        self.memo[nid] = SKIPPED
                        return SKIPPED
                    pass
                else:
                    for ref, v in zip(node.inputs, inputs):
                        if isinstance(v, _Skipped):
                            raise RuntimeFailure(
                                f"{nid} consumes skipped value {ref} "
                                f"(guard mismatch upstream)")
                for ref in node.after:
                    self._eval(ref)

                # failure injection / execution
                injected = self.fail_plan.get(nid, [])
                if attempt <= len(injected) and injected[attempt - 1] == "verify_false" \
                        and node.op == "VERIFY":
                    value, err = False, None
                else:
                    value, err = exec_mod.execute(node, inputs, attempt,
                                                  seed=self.seed)
                    if err is None and node.op == "SELECT" \
                            and isinstance(value, _Skipped):
                        err = "SELECT chosen branch was skipped (guard inconsistency)"
                lat = self._latency(node)
                if err is None and attempt <= len(injected) \
                        and injected[attempt - 1] == "error":
                    err = "injected error"

                if err is not None:
                    self._event(node, attempt=attempt, status="error",
                                latency=lat, value=None, note=err)
                    if attempt < max_att:
                        continue                       # retry on error
                    raise RuntimeFailure(f"{nid} failed after {attempt} attempt(s): {err}")

                self.memo[nid] = value
                self._event(node, attempt=attempt, status="ok", latency=lat,
                            value=value)
                break

            return self.memo[nid]
        finally:
            self._in_progress.discard(nid)

    # -------------------------------------------------------------- helpers
    def _global_value(self, nid: str) -> Any:
        from . import executors as em
        prog = self.mod.program
        decl = next((g for g in prog.inputs if g.get("name") == nid), None)
        ty_s = ty.normalize(decl.get("type", "Str")) if isinstance(decl, dict) \
            else "Str"
        rng = em._rng(self.seed, "global", nid)
        head, args = ty.parse(ty_s)
        if head in ("List", "Set"):
            return em._make_items(rng, "product", rng.randint(3, 6))
        if head == "Map":
            return {"key_a": round(rng.uniform(1, 99), 2),
                    "key_b": round(rng.uniform(1, 99), 2)}
        if ty_s in ("Str", "Any", "Json", "Unknown"):
            return f"<{nid.lstrip('@')} input text>"
        if ty_s == "Float":
            return round(rng.uniform(1, 100), 2)
        if ty_s == "Int":
            return rng.randint(1, 100)
        if ty_s == "Bool":
            return True
        if ty_s == "Table":
            rows = em._make_items(rng, "product", 3)
            return {"columns": list(rows[0].keys()), "rows": rows}
        return f"<{nid.lstrip('@')} input>"

    def _latency(self, node: Node) -> float:
        spec = exec_mod.spec_of(node.op)
        base = spec.cost.latency_ms if spec else 5.0
        if self.jitter:
            base *= 1.0 + self._rng.uniform(-self.jitter, self.jitter)
        return round(base, 3)

    def _event(self, node: Node, attempt: int, status: str, latency: float,
               value: Any, note: str = "") -> TraceEvent:
        spec = exec_mod.spec_of(node.op)
        c = spec.cost if spec else None
        ev = TraceEvent(
            seq=len(self.events), node=node.id, op=node.op, attempt=attempt,
            status=status, latency_ms=latency,
            tokens_in=(c.tokens_in if status == "ok" and c else 0),
            tokens_out=(c.tokens_out if status == "ok" and c else 0),
            flops=(c.flops if status == "ok" and c else 0.0),
            resource_class=(spec.resource_class if spec else "python"),
            energy_j=(c.energy_j if status == "ok" and c else 0.0),
            memory_mb=(c.memory_mb if status == "ok" and c else 0.0),
            value_digest=_digest(value),
            note=note,
        )
        self.events.append(ev)
        return ev

    def _mark_retry_exhausted(self, nid: str) -> None:
        for e in reversed(self.events):
            if e.node == nid:
                if e.status == "ok":
                    e.status = "retry_exhausted"
                    e.note = "verify still false after max_attempts"
                break

    def _invalidate_dependents(self, nid: str) -> set:
        """Clear memoized values for `nid` and every transitive dependent
        (retry budgets in self.attempts are kept). Returns the cleared ids."""
        users: Dict[str, List[str]] = {k: [] for k in self.nodes}
        for n in self.nodes.values():
            for ref in list(n.inputs) + list(n.after):
                if ref in users:
                    users[ref].append(n.id)
            if n.guard is not None and n.guard.cond in users:
                users[n.guard.cond].append(n.id)

        stack, dead = [nid], {nid}
        while stack:
            cur = stack.pop()
            for u in users.get(cur, []):
                if u not in dead:
                    dead.add(u)
                    stack.append(u)
        for d in dead:
            self.memo.pop(d, None)
            # prior events of cleared nodes are HISTORY now (kept for the
            # trace); invariants and reports must reason on final state only
            for e in self.events:
                if e.node == d:
                    e.superseded = True
        return dead

    def _critical_path_ms(self) -> float:
        """Longest-latency path over executed (non-skipped) nodes, using
        data + after + guard edges. This is the wall-clock lower bound under
        unlimited parallelism."""
        best: Dict[str, float] = {}

        def path_of(nid: str) -> float:
            if nid in best:
                return best[nid]
            node = self.nodes.get(nid)
            if node is None:
                return 0.0
            lat = next((e.latency_ms for e in reversed(self.events)
                        if e.node == nid and e.status != "skipped"), 0.0)
            preds = [path_of(r) for r in list(node.inputs) + list(node.after)]
            if node.guard is not None:
                preds.append(path_of(node.guard.cond))
            best[nid] = lat + (max(preds) if preds else 0.0)
            return best[nid]

        return path_of(self.mod.program.output) if self.mod.program.output in self.nodes \
            else 0.0
