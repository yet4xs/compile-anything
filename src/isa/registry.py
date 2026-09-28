"""Skill ISA registry v0.1 — the authoritative instruction table.

Mirrors spec/skill-isa.md. Each SkillSpec declares a typed signature, a
nominal cost model and candidate executors. Control instructions (VERIFY,
SELECT) live here too so the validator/simulator handle them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..ir import types as ty

# ---------------------------------------------------------------- cost model
FLOPS_PER_TOKEN_2B = 4e9      # 2 FLOPs/param/token, 2B executor
FLOPS_PER_TOKEN_70B = 1.4e11  # 70B single-shot baseline
BIG_MODEL_BASELINE = {        # nominal "one big model answers directly" reference
    "params": 70e9,
    "tokens_in": 1200,
    "tokens_out": 600,
    "latency_ms": 2500.0,
    "energy_j": 900.0,         # amortized serving energy (nominal)
    "memory_mb": 140000.0,     # ~8x H100-80GB class deployment
}


@dataclass
class Cost:
    latency_ms: float = 5.0
    tokens_in: int = 0
    tokens_out: int = 0
    flops: float = 1e6        # nominal for python-class ops
    calls: int = 1
    energy_j: float = 0.005   # nominal per invocation (python-class default)
    memory_mb: float = 30.0   # resident footprint of the executing unit


def _lm_cost(latency_ms: float, t_in: int, t_out: int) -> Cost:
    return Cost(latency_ms=latency_ms, tokens_in=t_in, tokens_out=t_out,
                flops=(t_in + t_out) * FLOPS_PER_TOKEN_2B,
                energy_j=0.02 * (t_in + t_out),   # 2B executor, nominal
                memory_mb=4600.0)                 # 2B fp16 weights + KV cache


def _api_cost(latency_ms: float) -> Cost:
    return Cost(latency_ms=latency_ms, flops=0.0, energy_j=0.005, memory_mb=0.0)


@dataclass
class SkillSpec:
    name: str
    cls: str
    desc: str = ""
    input_types: List[str] = field(default_factory=list)
    min_inputs: int = 1
    max_inputs: Optional[int] = None          # None = variadic
    output_rule: str = "FIXED:Any"            # FIXED:t | SAME_AS:i | ITEM_OF:i
    resource_class: str = "python"
    cost: Cost = field(default_factory=Cost)
    executors: List[str] = field(default_factory=list)

    def output_type(self, input_types: List[str]) -> str:
        rule = self.output_rule
        if rule.startswith("FIXED:"):
            return rule[len("FIXED:"):].strip()
        kind, idx = rule.split(":", 1)
        i = int(idx)
        actual = input_types[i] if i < len(input_types) else "Any"
        actual = ty.normalize(actual)
        if kind == "SAME_AS":
            return actual
        if kind == "ITEM_OF":
            return ty.item_of(actual)
        raise ValueError(f"bad output rule: {rule}")


def _s(name, cls, input_types, output_rule, resource_class, cost, desc="",
       min_inputs=None, max_inputs=None, executors=None) -> SkillSpec:
    return SkillSpec(
        name=name, cls=cls, desc=desc, input_types=input_types,
        min_inputs=(len(input_types) if min_inputs is None else min_inputs),
        max_inputs=(None if max_inputs == "variadic" else
                    (max_inputs if max_inputs is not None else len(input_types))),
        output_rule=output_rule, resource_class=resource_class, cost=cost,
        executors=executors or [],
    )


_L = ["List[Any]"]

REGISTRY: Dict[str, SkillSpec] = {s.name: s for s in [
    # -- io ---------------------------------------------------------------
    _s("LOAD", "io", ["Str"], "FIXED:Any", "python", Cost(20, flops=1e6),
       "load a file/dataset given a source spec", executors=["python:fs"]),
    _s("SAVE", "io", ["Any"], "FIXED:Any", "python", Cost(20, flops=1e6),
       "persist a value", executors=["python:fs"]),
    # -- retrieval ----------------------------------------------------------
    _s("SEARCH", "retrieval", ["Str", "Any"], "FIXED:List[Any]", "api",
       _api_cost(120), "semantic search over a domain (params.domain)",
       min_inputs=1, max_inputs=2,
       executors=["tool:search_api", "python:retrieval_mock"]),
    _s("FETCH", "retrieval", ["Str"], "FIXED:Str", "api", _api_cost(150),
       "fetch a URL/resource", executors=["python:http"]),
    _s("QUERY_DB", "retrieval", ["Str"], "FIXED:Table", "db", _api_cost(40),
       "structured query (params.table)", executors=["db:sql"]),
    # -- transform ----------------------------------------------------------
    _s("FILTER", "transform", _L, "SAME_AS:0", "python", Cost(5),
       "keep items matching params.predicate", executors=["python:listcomp"]),
    _s("TRANSFORM", "transform", _L, "SAME_AS:0", "python", Cost(10),
       "map params.op over items", executors=["python:listcomp"]),
    _s("EXTRACT", "transform", ["Any"], "FIXED:Any", "python", Cost(8),
       "project params.fields", executors=["python:dict"]),
    _s("DEDUP", "transform", _L, "SAME_AS:0", "python", Cost(3), "unique items"),
    _s("SORT", "transform", _L, "SAME_AS:0", "python", Cost(4),
       "sort by params.key, params.order", executors=["python:sorted"]),
    _s("JOIN", "transform", ["List[Any]", "List[Any]"], "FIXED:List[Any]",
       "python", Cost(8), "join two lists"),
    _s("MERGE", "transform", ["Any", "Any"], "SAME_AS:0", "python", Cost(2),
       "merge two parallel-branch values of the same type"),
    # -- compute -------------------------------------------------------------
    _s("ARGMIN", "compute", _L, "ITEM_OF:0", "python", Cost(1),
       "item with min params.key"),
    _s("ARGMAX", "compute", _L, "ITEM_OF:0", "python", Cost(1),
       "item with max params.key"),
    _s("MIN", "compute", _L, "ITEM_OF:0", "python", Cost(1), "min by params.key"),
    _s("MAX", "compute", _L, "ITEM_OF:0", "python", Cost(1), "max by params.key"),
    _s("SUM", "compute", ["List[Float]"], "FIXED:Float", "python", Cost(1), "sum"),
    _s("AVG", "compute", ["List[Float]"], "FIXED:Float", "python", Cost(1), "mean"),
    _s("COUNT", "compute", _L, "FIXED:Int", "python", Cost(1), "length"),
    _s("CALCULATE", "compute", ["Float"], "FIXED:Float", "python", Cost(1),
       "evaluate params.expr over variadic numeric inputs",
       min_inputs=0, max_inputs="variadic"),
    _s("COMPARE", "compute", ["Any", "Any"], "FIXED:Bool", "python", Cost(1),
       "params.op comparison of two values"),
    _s("CONVERT", "compute", ["Float"], "FIXED:Float", "api", _api_cost(80),
       "unit/currency conversion (params.from, params.to; amount via input "
       "or params.amount when literal)", min_inputs=0),
    # -- lm ------------------------------------------------------------------
    _s("GENERATE", "lm", ["Any"], "FIXED:Str", "lm", _lm_cost(600, 800, 300),
       "natural-language generation from inputs (params.role)",
       min_inputs=1, max_inputs="variadic", executors=["lm:qwen2b-instruct"]),
    _s("SUMMARIZE", "lm", ["Str"], "FIXED:Str", "lm", _lm_cost(400, 2000, 300),
       executors=["lm:qwen2b-instruct"]),
    _s("TRANSLATE", "lm", ["Str"], "FIXED:Str", "lm", _lm_cost(500, 1000, 1000),
       "params.lang", executors=["lm:qwen2b-instruct"]),
    _s("CLASSIFY", "lm", ["Any"], "FIXED:Str", "lm", _lm_cost(200, 600, 5),
       executors=["lm:qwen2b-classifier"]),
    _s("EXTRACT_ENTITIES", "lm", ["Any"], "FIXED:List[Entity]", "lm",
       _lm_cost(300, 1000, 200),
       "input: text or list of text-bearing records",
       executors=["lm:qwen2b-ner"]),
    _s("CODEGEN", "lm", ["Any"], "FIXED:Str", "lm", _lm_cost(900, 1000, 600),
       min_inputs=1, max_inputs="variadic", executors=["lm:qwen2b-coder"]),
    _s("PLAN", "lm", ["Str"], "FIXED:Json", "lm", _lm_cost(500, 600, 400),
       executors=["lm:qwen2b-instruct"]),
    # -- action ----------------------------------------------------------------
    _s("SEND", "action", ["Str"], "FIXED:Any", "api", _api_cost(100),
       "send a message (params.channel)", executors=["tool:messaging_api"]),
    _s("EXEC_ACTION", "action", ["Any"], "FIXED:Any", "api", _api_cost(150),
       "generic external action (params.action); lifter fallback",
       min_inputs=0, max_inputs="variadic", executors=["tool:*"]),
]}

# -- control instructions (IR-level; see taskir-spec.md §4.3) ----------------
CONTROL_OPS: Dict[str, SkillSpec] = {s.name: s for s in [
    SkillSpec(name="VERIFY", cls="control",
              desc="produce Bool verdict on inputs (params.check)",
              input_types=["Any"], min_inputs=1, max_inputs=None,
              output_rule="FIXED:Bool", resource_class="lm",
              cost=_lm_cost(50, 400, 1), executors=["lm:qwen2b-verifier"]),
    SkillSpec(name="SELECT", cls="control",
              desc="dataflow phi: pick input 1 or 2 by Bool cond (input 0)",
              input_types=["Bool", "Any", "Any"], min_inputs=3, max_inputs=3,
              output_rule="SAME_AS:1", resource_class="runtime",
              cost=Cost(0.1, flops=0.0)),
]}

ALL_OPS: Dict[str, SkillSpec] = {**REGISTRY, **CONTROL_OPS}


def get(op: str) -> Optional[SkillSpec]:
    return ALL_OPS.get(op)


def is_control(op: str) -> bool:
    return op in CONTROL_OPS
