"""TaskIR semantic fuzzer — adversarial validation of IR semantics.

Pipeline per program:
    random typed builder (valid by construction)
        -> validator   MUST accept        (else: validator false-rejection)
        -> simulator   only DEFINED failures allowed
        -> runtime invariants checked on the trace

Defined runtime failures (whitelisted, documented semantics):
    - "consumes skipped value ..."          unguarded consumer of a skipped value
    - "SELECT chosen branch was skipped"    inconsistent branch predication
    - "failed after N attempt(s): ..."      executor error after retry budget
    - "dependency cycle detected at runtime" (defense-in-depth; unreachable
      for validated programs since V2 enforces acyclic canonical order)

Anything else (TypeError/KeyError/RecursionError/...) is a bug and gets
saved to data/fuzz_failures/failure_XXXXX.json together with the program.

Invariants checked:
    IV1 execution respects the DAG: every dep's last-ok event precedes its
       consumer's last-ok event (SELECT's non-taken branch exempt);
    IV2 per-node attempts are 1..k in order (skips emit no attempts);
    IV3 determinism: same seed + fail plan -> identical event sequence;
    IV4 completed => output value is not <skipped>;
    IV5 totals sanity (tokens/flops/energy >= 0, node_calls consistent);
    IV6 Bool-typed ops (VERIFY/COMPARE) really produce bools.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.taskir import Guard, Module, Node, Program, Retry, module_to_dict  # noqa: E402
from src.validator.validator import validate  # noqa: E402
from src.runtime.simulator import Simulator, SKIPPED  # noqa: E402

DEFINED_FAILURE_MARKERS = (
    "consumes skipped value",
    "SELECT chosen branch was skipped",
    "failed after ",
    "dependency cycle detected at runtime",
)

DOMAINS = ["flight", "hotel", "news", "product", "stock", "weather"]


class Builder:
    """Emits structurally valid random programs (types tracked by category).

    Categories: str, bool, list, float, int, any — coarser than the real type
    system, always compatible with what the validator will infer."""

    def __init__(self, rng: random.Random, idx: int):
        self.rng = rng
        self.idx = idx
        self.nodes: list = []
        self.values: list = [("@task", "str")]   # (id, category)
        self.counter = 0

    # ------------------------------------------------------------------ utils
    def _nid(self) -> str:
        self.counter += 1
        return f"%n{self.counter}"

    def _pick(self, categories):
        cands = [v for v in self.values if v[1] in categories]
        return self.rng.choice(cands)[0] if cands else None

    def _add(self, op, inputs, category, params=None, **kw) -> str:
        nid = self._nid()
        node = Node(id=nid, op=op, inputs=inputs, params=params or {}, **kw)
        # random guard from an earlier bool value
        if self.rng.random() < 0.15:
            bools = [v for v in self.values if v[1] == "bool"]
            if bools:
                node.guard = Guard(cond=self.rng.choice(bools)[0],
                                   expect=self.rng.random() < 0.7)
        # random control-only dependency
        if self.rng.random() < 0.10:
            prev = [v for v in self.values if v[0].startswith("%")]
            if prev:
                node.after = [self.rng.choice(prev)[0]]
        self.nodes.append(node)
        self.values.append((nid, category))
        return nid

    # ------------------------------------------------------------------ rules
    def build(self) -> Module:
        n_steps = self.rng.randint(3, 12)
        for _ in range(n_steps):
            self._step()
        self._attach_verify_retries()
        # output: bias towards later nodes
        cands = [v[0] for v in self.values if v[0].startswith("%")]
        out = self.rng.choice(cands[-max(1, len(cands) // 2):])
        prog = Program(name=f"fuzz_{self.idx:05d}",
                       description=f"fuzz program {self.idx}",
                       inputs=[{"name": "@task", "type": "Str"}],
                       nodes=self.nodes, output=out)
        return Module(program=prog,
                      meta={"name": prog.name,
                            "provenance": {"source": "fuzz"}})

    def _step(self):
        r = self.rng
        rules = [
            (0.14, self._r_search), (0.05, self._r_fetch), (0.04, self._r_load),
            (0.12, self._r_list_op), (0.07, self._r_agg),
            (0.14, self._r_generate), (0.05, self._r_str_lm),
            (0.05, self._r_extract), (0.09, self._r_verify),
            (0.05, self._r_compare), (0.03, self._r_calc),
            (0.09, self._r_select), (0.04, self._r_merge),
            (0.04, self._r_action),
        ]
        x, rule = 0.0, None
        pick = r.random() * sum(w for w, _ in rules)
        for w, fn in rules:
            x += w
            if pick <= x:
                rule = fn
                break
        rule = rule or self._r_generate
        rule()

    def _r_search(self):
        q = self._pick(("str", "any")) or "@task"
        self._add("SEARCH", [q], "list", {"domain": self.rng.choice(DOMAINS)})

    def _r_fetch(self):
        s = self._pick(("str",)) or "@task"
        self._add("FETCH", [s], "str", {"url": "input_url"})

    def _r_load(self):
        self._add("LOAD", ["@task"], "any", {"source": "data.bin"})

    def _r_list_op(self):
        src = self._pick(("list",))
        if not src:
            return self._r_search()
        op = self.rng.choice(["FILTER", "SORT", "DEDUP"])
        self._add(op, [src], "list",
                  {"predicate": "keep"} if op == "FILTER"
                  else {"key": "price", "order": "asc"} if op == "SORT" else {})

    def _r_agg(self):
        src = self._pick(("list",))
        if not src:
            return self._r_search()
        if self.rng.random() < 0.5:
            # item types are CONCRETE (e.g. Entity), not Any: they do not
            # satisfy Float/Str slots (exact-match type system)
            self._add(self.rng.choice(["ARGMIN", "ARGMAX", "MIN", "MAX"]),
                      [src], "item", {"key": "price"})
        else:
            self._add("COUNT", [src], "int")

    def _r_generate(self):
        ins = [v[0] for v in self.rng.sample(self.values,
                                             k=min(len(self.values),
                                                   self.rng.randint(1, 3)))]
        self._add("GENERATE", ins, "str", {"role": "answer"})

    def _r_str_lm(self):
        s = self._pick(("str",))
        if not s:
            return self._r_generate()
        op = self.rng.choice(["SUMMARIZE", "TRANSLATE"])
        self._add(op, [s], "str", {"lang": "fr"} if op == "TRANSLATE" else {})

    def _r_extract(self):
        src = self._pick(("any", "list", "str"))
        if src is None:
            return self._r_load()
        if self.rng.random() < 0.5:
            self._add("EXTRACT", [src], "any", {"fields": ["value"]})
        else:
            self._add("EXTRACT_ENTITIES", [src], "list")

    def _r_verify(self):
        ins = [v[0] for v in self.rng.sample(self.values,
                                             k=min(len(self.values),
                                                   self.rng.randint(1, 2)))]
        self._add("VERIFY", ins, "bool", {"check": "ok"})

    def _r_compare(self):
        a = self._pick(("any", "float", "int", "str")) or "@task"
        b = self._pick(("any", "float", "int", "str")) or "@task"
        self._add("COMPARE", [a, b], "bool", {"op": ">"})

    def _r_calc(self):
        # CALCULATE declares Float inputs; the type system is exact-match,
        # so only genuinely-Float values qualify ("item" types like Entity
        # and Int do NOT)
        cands = [v[0] for v in self.values if v[1] == "float"]
        ins = self.rng.sample(cands, k=min(len(cands), self.rng.randint(0, 2))) \
            if cands else []
        if self.rng.random() < 0.5 and not ins:
            self._add("CONVERT", [], "float", {"from": "USD", "to": "EUR"})
        else:
            self._add("CALCULATE", ins, "float", {"expr": "x*1.1"})

    def _r_select(self):
        c = self._pick(("bool",))
        if not c:
            return self._r_verify()
        vals = [v for v in self.values if v[0].startswith("%")]
        if len(vals) < 2:
            return self._r_generate()
        a = self.rng.choice(vals)
        pool = [v for v in vals if v[0] != a[0]
                and (v[1] == a[1] or v[1] == "any" or a[1] == "any")]
        if not pool:
            return self._r_generate()
        b = self.rng.choice(pool)
        self._add("SELECT", [c, a[0], b[0]], a[1])

    def _r_merge(self):
        lists = [v[0] for v in self.values if v[1] == "list"]
        if len(lists) < 2:
            return self._r_search()
        a, b = self.rng.sample(lists, 2)
        self._add(self.rng.choice(["MERGE", "JOIN"]), [a, b], "list")

    def _r_action(self):
        if self.rng.random() < 0.5:
            s = self._pick(("str",)) or "@task"
            self._add("SEND", [s], "any", {"channel": "email"})
        else:
            self._add("EXEC_ACTION", [], "any", {"action": "sync"})

    # ------------------------------------------------------------------ retry
    def _deps_of(self, nid: str) -> set:
        node = next(n for n in self.nodes if n.id == nid)
        refs = list(node.inputs) + list(node.after)
        if node.guard is not None:
            refs.append(node.guard.cond)
        return {r for r in refs if r.startswith("%")}

    def _closure(self, nid: str) -> set:
        seen, stack = set(), [nid]
        while stack:
            cur = stack.pop()
            for d in self._deps_of(cur):
                if d not in seen:
                    seen.add(d)
                    stack.append(d)
        return seen

    def _attach_verify_retries(self):
        verifies = [n for n in self.nodes if n.op == "VERIFY"]
        for v in verifies:
            if self.rng.random() < 0.35:
                upstream = [n for n in self.nodes
                            if n.id != v.id and n.retry is None
                            and n.id in self._closure(v.id)]
                if upstream:
                    d = self.rng.choice(upstream)
                    d.retry = Retry(max_attempts=self.rng.choice([2, 3]), on=v.id)
        for n in self.nodes:
            if n.retry is None and self.rng.random() < 0.10:
                n.retry = Retry(max_attempts=self.rng.randint(1, 3), on="error")


def make_fail_plan(mod: Module, rng: random.Random) -> dict:
    plan = {}
    for n in mod.program.nodes:
        if n.op == "VERIFY" and rng.random() < 0.25:
            plan[n.id] = ["verify_false"]
        elif rng.random() < 0.06:
            plan[n.id] = ["error"]
    return plan


# ------------------------------------------------------------------ invariants
def check_invariants(mod: Module, res) -> list:
    problems = []
    nodes = {n.id: n for n in mod.program.nodes}

    def deps(n):
        refs = list(n.inputs) + list(n.after)
        if n.guard is not None:
            refs.append(n.guard.cond)
        return [r for r in refs if r in nodes]

    last_ok = {e.node: e.seq for e in res.events
               if e.status in ("ok", "retry_exhausted") and not e.superseded}

    if res.status == "completed":
        # IV4: a skipped output is DEFINED semantics when the output node is
        # predicated (own guard, or SELECT with a skipped cond); it is a bug
        # when an unpredicated output ends up skipped.
        out_node = nodes.get(mod.program.output)
        out_val = res.values.get(mod.program.output)
        if isinstance(out_val, SKIPPED.__class__):
            predicated = (out_node is not None
                          and (out_node.guard is not None or out_node.op == "SELECT"))
            has_skip_event = any(e.node == mod.program.output
                                 and e.status == "skipped" for e in res.events)
            if not predicated or not has_skip_event:
                problems.append("IV4: completed but output value is <skipped> "
                                "without predication")
        # IV1
        for n in nodes.values():
            if n.id not in last_ok:
                continue
            cond_val = None
            if n.op == "SELECT":
                cond_val = res.values.get(n.inputs[0])
            for pos, ref in enumerate(n.inputs):
                if ref not in nodes:
                    continue
                if n.op == "SELECT" and pos >= 1:
                    chosen = 1 if cond_val is True else 2
                    if pos != chosen:
                        continue           # non-taken branch may be skipped
                if ref not in last_ok:
                    problems.append(
                        f"IV1: {n.id} executed but input {ref} never produced "
                        f"a value (skipped/never ran)")
                elif last_ok[ref] >= last_ok[n.id]:
                    problems.append(
                        f"IV1: order violation {ref} (seq {last_ok[ref]}) "
                        f"not before {n.id} (seq {last_ok[n.id]})")
            for ref in n.after:
                if ref in last_ok and last_ok[ref] >= last_ok[n.id]:
                    problems.append(f"IV1: after-order violation {ref} !< {n.id}")
            if n.guard is not None and n.guard.cond in last_ok \
                    and last_ok[n.guard.cond] >= last_ok[n.id]:
                problems.append(
                    f"IV1: guard-order violation {n.guard.cond} !< {n.id}")

    # IV2: attempts per node are 1..k in order
    per_node: dict = {}
    for e in res.events:
        if e.status != "skipped":
            per_node.setdefault(e.node, []).append(e.attempt)
    for nid, atts in per_node.items():
        if atts != list(range(1, len(atts) + 1)):
            problems.append(f"IV2: attempts for {nid} not 1..k in order: {atts}")

    # IV5
    if res.tokens_in < 0 or res.tokens_out < 0 or res.flops < 0 or res.energy_j < 0:
        problems.append("IV5: negative cost totals")
    n_exec = sum(1 for e in res.events if e.status != "skipped")
    if res.node_calls != n_exec:
        problems.append(f"IV5: node_calls {res.node_calls} != executed events {n_exec}")

    # IV6
    for n in nodes.values():
        if n.op in ("VERIFY", "COMPARE") and n.id in res.values:
            v = res.values[n.id]
            if not isinstance(v, bool) and not isinstance(v, SKIPPED.__class__):
                problems.append(f"IV6: {n.op} {n.id} produced non-bool {type(v).__name__}")

    return problems


def event_signature(res):
    return [(e.node, e.op, e.status, e.attempt, e.value_digest, e.latency_ms)
            for e in res.events]


def run_one(mod: Module, seed: str, fail_plan: dict):
    """Returns (result, undefined_failure_detail_or_None)."""
    try:
        return Simulator(mod, seed=seed, fail_plan=fail_plan).run(), None
    except RuntimeFailure as e:               # shouldn't escape run(), but be safe
        return None, f"RuntimeFailure escaped run(): {e}"
    except Exception:
        return None, traceback.format_exc(limit=8)


def fuzz(n: int, seed: int, outdir: pathlib.Path) -> int:
    rng = random.Random(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("failure_*.json"):
        old.unlink()
    failures = []
    stats = {"generated": 0, "valid": 0, "completed": 0, "failed_defined": 0}

    for i in range(1, n + 1):
        mod = Builder(rng, i).build()
        stats["generated"] += 1
        fail_plan = make_fail_plan(mod, rng)
        payload = module_to_dict(mod)
        record = {"program_index": i, "fail_plan": fail_plan, "module": payload}

        rep = validate(mod)
        if not rep.valid:
            record["phase"] = "validate"
            record["detail"] = [f"{e.code}@{e.node}: {e.msg}" for e in rep.errors]
            failures.append(record)
            continue
        stats["valid"] += 1

        res, undef = run_one(mod, seed=f"fuzz{i}", fail_plan=fail_plan)
        if undef is not None:
            record["phase"] = "simulate"
            record["detail"] = undef
            failures.append(record)
            continue
        if res.status == "failed":
            if not any(m in res.output_digest for m in DEFINED_FAILURE_MARKERS):
                record["phase"] = "simulate-undefined-failure"
                record["detail"] = res.output_digest
                failures.append(record)
                continue
            stats["failed_defined"] += 1
        else:
            stats["completed"] += 1

        problems = check_invariants(mod, res)
        if problems:
            record["phase"] = "invariant"
            record["detail"] = problems
            failures.append(record)
            continue

        # IV3 determinism (re-run with identical inputs)
        res2, undef2 = run_one(mod, seed=f"fuzz{i}", fail_plan=fail_plan)
        if undef2 is not None or event_signature(res) != event_signature(res2):
            record["phase"] = "determinism"
            record["detail"] = undef2 or "event sequences differ between runs"
            failures.append(record)

    for k, rec in enumerate(failures, 1):
        (outdir / f"failure_{k:05d}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"fuzz: {stats['generated']} generated | {stats['valid']} valid | "
          f"{stats['completed']} completed | {stats['failed_defined']} "
          f"failed-with-defined-reason | {len(failures)} FAILURES")
    if failures:
        print(f"failure artifacts -> {outdir}")
        for rec in failures[:5]:
            print(f"  [{rec['program_index']}] {rec['phase']}: "
                  f"{str(rec['detail'])[:140]}")
    return len(failures)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "data" / "fuzz_failures"))
    args = ap.parse_args()
    n_fail = fuzz(args.n, args.seed, pathlib.Path(args.out))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
