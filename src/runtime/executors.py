"""Mock executors for the TaskIR runtime simulator.

Each semantic skill gets a deterministic executor that synthesizes a typed
value from its inputs (hash-seeded, so runs are reproducible). Real executor
binding is scheduler/runtime work (v0.2+); here we only need correct shapes,
plausible latencies and honest accounting.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, List, Optional, Tuple

from ..ir.taskir import Node
from ..isa import SkillSpec
from ..isa.registry import CONTROL_OPS, REGISTRY

_DOMAIN_FIELDS = {
    "flight": ["carrier", "price", "depart_time", "arrive_time"],
    "hotel": ["name", "price_per_night", "stars", "distance_km"],
    "weather": ["city", "date", "condition", "temp_c"],
    "news": ["title", "source", "date", "category"],
    "maps": ["route", "distance_km", "duration_min"],
    "product": ["name", "price", "rating", "stock"],
    "paper": ["title", "year", "citations", "venue"],
    "stock": ["symbol", "price", "change_pct", "volume"],
    "email": ["subject", "from", "date", "snippet"],
    "person": ["name", "affiliation", "role"],
    "repo": ["name", "stars", "language", "updated"],
}


def _rng(*parts: Any) -> random.Random:
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _digest(v: Any, limit: int = 90) -> str:
    s = repr(v).replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _domain_of(node: Node, inputs: List[Any]) -> str:
    d = node.params.get("domain")
    if isinstance(d, str) and d in _DOMAIN_FIELDS:
        return d
    if inputs and isinstance(inputs[0], list) and inputs[0] \
            and isinstance(inputs[0][0], dict):
        keys = set(inputs[0][0].keys())
        for dom, fields in _DOMAIN_FIELDS.items():
            if keys.issuperset(set(fields[:2])):
                return dom
    return "product"


def _make_items(rng: random.Random, domain: str, n: int) -> List[dict]:
    fields = _DOMAIN_FIELDS.get(domain)
    if fields is None:                      # e.g. QUERY_DB table="market"
        fields = ["name", "value", "date", "category"]
    items = []
    for i in range(n):
        item = {}
        for j, f in enumerate(fields):
            if "price" in f or f in ("stars", "citations", "volume", "rating"):
                item[f] = round(rng.uniform(20, 900), 2)
            elif f in ("temp_c", "distance_km", "duration_min", "change_pct"):
                item[f] = round(rng.uniform(-5, 120), 1)
            else:
                item[f] = f"{domain}_{f}_{i}{chr(65 + (i + j) % 26)}"
        items.append(item)
    return items


def execute(node: Node, inputs: List[Any], attempt: int,
            seed: str) -> Tuple[Any, Optional[str]]:
    """Run the mock executor for `node`. Returns (value, error)."""
    op, params, rng = node.op, node.params, _rng(seed, node.id, attempt)
    err: Optional[str] = None

    if op == "SEARCH":
        domain = _domain_of(node, inputs)
        value: Any = _make_items(rng, domain, rng.randint(3, 6))
    elif op in ("FETCH",):
        value = f"<fetched content from {params.get('url', 'source')} (len={rng.randint(800, 4000)})>"
    elif op == "QUERY_DB":
        rows = _make_items(rng, str(params.get("table", "generic")), rng.randint(3, 8))
        value = {"columns": list(rows[0].keys()), "rows": rows}
    elif op == "FILTER":
        base = inputs[0] if inputs and isinstance(inputs[0], list) else []
        kept = [x for x in base
                if _rng(seed, node.id, attempt, repr(x)).random() < 0.6]
        # mock guarantee: never empty (execution failures are injection-only)
        value = kept or base[:1]
    elif op in ("TRANSFORM",):
        base = inputs[0] if inputs and isinstance(inputs[0], list) else []
        value = [dict(x, _transformed=str(params.get("op", "norm"))) if isinstance(x, dict)
                 else x for x in base]
    elif op == "EXTRACT":
        src = inputs[0] if inputs else {}
        fields = params.get("fields") or []
        if isinstance(src, dict):
            value = {k: src.get(k) for k in fields} if fields else src
        elif isinstance(src, list) and src and isinstance(src[0], dict):
            value = [{k: x.get(k) for k in fields} for x in src] if fields else src
        else:
            value = src
    elif op == "DEDUP":
        base = inputs[0] if inputs and isinstance(inputs[0], list) else []
        value = list({repr(x): x for x in base}.values())
    elif op == "SORT":
        base = inputs[0] if inputs and isinstance(inputs[0], list) else []
        key = params.get("key")
        try:
            value = sorted(base, key=(lambda x: x.get(key, 0)) if key and base and
                          isinstance(base[0], dict) else None,
                           reverse=params.get("order") == "desc")
        except TypeError:
            value = base
    elif op == "JOIN":
        a = inputs[0] if inputs and isinstance(inputs[0], list) else []
        b = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], list) else []
        value = a + b
    elif op == "MERGE":
        a = inputs[0] if inputs else []
        b = inputs[1] if len(inputs) > 1 else []
        value = (a + b) if isinstance(a, list) and isinstance(b, list) else (a or b)
    elif op in ("ARGMIN", "ARGMAX", "MIN", "MAX"):
        base = inputs[0] if inputs and isinstance(inputs[0], list) else []
        key = params.get("key")
        if not base:
            value, err = None, f"{op} of empty list"
        else:
            try:
                sel = (min if op in ("ARGMIN", "MIN") else max)(
                    base, key=(lambda x: x.get(key, 0)) if key and
                    isinstance(base[0], dict) else None)
                value = sel
            except TypeError:
                value = base[0]
    elif op == "SUM":
        nums = [float(x) for x in (inputs[0] or []) if isinstance(x, (int, float))]
        value = round(sum(nums), 2)
    elif op == "AVG":
        nums = [float(x) for x in (inputs[0] or []) if isinstance(x, (int, float))]
        value = round(sum(nums) / len(nums), 2) if nums else 0.0
    elif op == "COUNT":
        base = inputs[0] if inputs and isinstance(inputs[0], list) else []
        value = len(base)
    elif op == "CALCULATE":
        vals = [x for x in inputs if isinstance(x, (int, float))]
        value = round(sum(vals) * rng.uniform(0.8, 1.2)
                      if not params.get("expr") else rng.uniform(1.0, 999.0), 4)
    elif op == "COMPARE":
        value = rng.random() > 0.35
    elif op == "CONVERT":
        amt = inputs[0] if inputs and isinstance(inputs[0], (int, float)) else 100.0
        rate = _rng(str(params.get("from", "USD")), str(params.get("to", "EUR"))
                    ).uniform(0.7, 1.4)
        value = round(float(amt) * rate, 2)
    elif op == "GENERATE":
        src = "; ".join(_digest(x, 40) for x in inputs)
        value = (f"[{params.get('role', 'answer')}] Based on {src or 'the task'}: "
                 f"synthesized result #{attempt} (deterministic mock).")
    elif op == "SUMMARIZE":
        value = f"<summary of {_digest(inputs[0] if inputs else '', 40)} in {len(str(params)) + 3} points>"
    elif op == "TRANSLATE":
        value = f"<{params.get('lang', 'en')} translation of {_digest(inputs[0] if inputs else '', 40)}>"
    elif op == "CLASSIFY":
        value = rng.choice(["relevant", "not_relevant", "positive", "negative"])
    elif op == "EXTRACT_ENTITIES":
        value = [{"name": f"entity_{i}", "type": rng.choice(["org", "person", "loc"])}
                 for i in range(rng.randint(1, 4))]
    elif op == "CODEGEN":
        value = f"def solution_{node.id.strip('%')}():\n    return {rng.randint(1, 99)}"
    elif op == "PLAN":
        value = {"steps": [f"step_{i}" for i in range(rng.randint(2, 5))]}
    elif op == "SEND":
        value = {"status": "sent", "channel": params.get("channel", "email")}
    elif op == "EXEC_ACTION":
        value = {"status": "ok", "action": params.get("action", "generic")}
    elif op == "LOAD":
        value = {"source": params.get("source", node.inputs and node.inputs[0]),
                 "rows": _make_items(rng, "product", rng.randint(2, 5))}
    elif op == "SAVE":
        value = {"status": "saved", "bytes": rng.randint(100, 5000)}
    elif op == "VERIFY":
        value = True     # default verdict; failure injection flips this
    elif op == "SELECT":
        cond = bool(inputs[0]) if inputs else False
        value = inputs[1] if cond else (inputs[2] if len(inputs) > 2 else None)
    else:
        # unknown op should have been caught by the validator; be defensive
        value, err = None, f"no mock executor for op {op!r}"

    return value, err


def spec_of(op: str) -> Optional[SkillSpec]:
    return CONTROL_OPS.get(op) or REGISTRY.get(op)
