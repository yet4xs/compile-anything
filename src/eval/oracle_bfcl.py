"""BFCL deterministic oracle (Task 13) — ground truth → TaskIR.

PURPOSE: representability audit ONLY. The produced TaskIR lives under
data/external_benchmarks/derived_oracle/ and is BLOCKED from training by
src/dataset/external_guard.py. This module never feeds SFT.

Mapping policy (deterministic, BFCL-specific rules — deliberately
independent of the frozen src/lifter/toolmap.py):
  read/search/get/find/query/list/retrieve  -> SEARCH / QUERY_DB
  send/mail/push                           -> SEND
  book/reserve/purchase/pay                -> EXEC_ACTION(book/pay)
  cancel/delete/remove                     -> EXEC_ACTION(cancel)
  create/add/insert/update/set/toggle/...  -> EXEC_ACTION(mutate)
  calculate/compute/evaluate/convert       -> CALCULATE / CONVERT
  cd/mkdir/mv/rm/ls/cat (multi-turn shell) -> EXEC_ACTION(fs)
  anything else                            -> unknown -> partial

Representability classes:
  full          all GT calls mapped; plan validates
  partial       multi_turn_state (inter-turn state) / unknown tools /
                environment_required (web_search real search)
  none          memory (needs memory ops; ISA gap)
  irrelevance   expected behavior = answer WITHOUT calls -> a
                GENERATE-only program (representable)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..ir.taskir import Module, Node, Program
from ..validator.validator import validate

READ = r"^(get|find|search|query|list|retrieve|lookup|show|view|check|fetch)"
SEND = r"(send|mail|push|notify)"
BOOK = r"(book|reserv|purchase|pay|checkout|buy)"
CANCEL = r"(cancel|delete|remove|refund|drop)"
MUTATE = r"(create|add|insert|update|set|toggle|enable|disable|grant|revoke|change|modify|edit|upload|post|reset|reboot|toggle|make|write|open|close|start|stop)"
CALC = r"(calculat|comput|evaluate|convert|currency|exchange)"
FS = r"^(cd|mkdir|mv|rm|ls|cat|pwd|cp|touch|grep|echo)$"


def map_function(name: str) -> Tuple[str, str]:
    """(skill, kind) — kind: read|send|book|cancel|mutate|calc|fs|unknown"""
    n = (name or "").strip().lower()
    if re.match(FS, n.split("(")[0].split(".")[0].split("_for")[0]):
        return "EXEC_ACTION", "fs"
    if re.match(READ, n):
        return "SEARCH", "read"
    if re.search(SEND, n):
        return "SEND", "send"
    if re.search(BOOK, n):
        return "EXEC_ACTION", "book"
    if re.search(CANCEL, n):
        return "EXEC_ACTION", "cancel"
    if re.search(CALC, n):
        return "CALCULATE", "calc"
    if re.search(MUTATE, n):
        return "EXEC_ACTION", "mutate"
    return "EXEC_ACTION", "unknown"


def _calls_from_ground_truth(gt) -> List[Dict]:
    """BFCL possible_answer ground_truth -> ordered call dicts."""
    calls: List[Dict] = []
    if gt is None:
        return calls
    items = gt
    if isinstance(gt, list) and gt and isinstance(gt[0], list):
        # multi-turn: list of per-turn call lists (strings like
        # "cd(folder='document')")
        for turn in gt:
            for c in turn if isinstance(turn, list) else []:
                if isinstance(c, str):
                    m = re.match(r"([A-Za-z0-9_]+)\((.*)\)", c)
                    if m:
                        calls.append({"name": m.group(1),
                                      "arguments": {"raw": m.group(2)}})
                elif isinstance(c, dict):
                    for k, v in c.items():
                        calls.append({"name": k, "arguments": v or {}})
        return calls
    for c in items if isinstance(items, list) else []:
        if isinstance(c, str):
            m = re.match(r"([A-Za-z0-9_]+)\((.*)\)", c)
            if m:
                calls.append({"name": m.group(1),
                              "arguments": {"raw": m.group(2)}})
        elif isinstance(c, dict):
            for k, v in c.items():
                calls.append({"name": k, "arguments": v or {}})
    return calls


def _program_from_calls(instruction: str, calls: List[Dict],
                        unmapped: List[str],
                        sequential: bool = False) -> Optional[Module]:
    """Build a TaskIR program from ground-truth calls.

    BFCL parallel/multiple calls are INDEPENDENT (no dataflow between
    them): each takes @task directly (or no input for zero-input skills).
    Sequential structure (multi-turn) is expressed with `after` edges —
    ordering WITHOUT fabricated dataflow — which is exactly the honest
    representation of inter-turn state."""
    from ..ir import types as ty
    from ..isa import registry
    nodes: List[Node] = []
    for i, c in enumerate(calls):
        skill, _ = map_function(c["name"])
        spec = registry.get(skill)
        params = {"action": c["name"]} if skill == "EXEC_ACTION" else {}
        params.update({k: v for k, v in (c.get("arguments") or {}).items()
                       if isinstance(v, (str, int, float, bool))})
        inputs: List[str] = []
        if spec is not None and spec.min_inputs >= 1:
            want = spec.input_types[0] if spec.input_types else "Any"
            if ty.is_compatible(want, "Str"):
                inputs = ["@task"]          # independent: query from task
        node = Node(id=f"%c{i}", op=skill, inputs=inputs, params=params)
        if sequential and i > 0:
            node.after = [nodes[i - 1].id]  # ordering, not dataflow
        nodes.append(node)
    if not nodes:                      # irrelevance: answer without calls
        nodes = [Node(id="%ans", op="GENERATE", inputs=["@task"],
                      params={"role": "final_answer"})]
        return Module(program=Program(
            name="bfcl_oracle", description=instruction,
            inputs=[{"name": "@task", "type": "Str"}],
            nodes=nodes, output="%ans"))
    return Module(program=Program(
        name="bfcl_oracle", description=instruction,
        inputs=[{"name": "@task", "type": "Str"}],
        nodes=nodes, output=nodes[-1].id))


def oracle(sample) -> Dict:
    """Returns {status: full|partial|none, module?, reasons: [...]}."""
    cat = sample.metadata.get("category", "")
    kind = sample.metadata.get("kind", cat)
    reasons: List[str] = []

    if kind == "memory":
        return {"status": "none",
                "reasons": ["unsupported_skill:memory_ops"]}
    if kind == "web_search":
        return {"status": "partial",
                "reasons": ["environment_required:real_web_search"],
                "module": _program_from_calls(
                    sample.instruction, [{"name": "web_search",
                                          "arguments": {}}], [])}

    calls = _calls_from_ground_truth(sample.reference_actions)
    unmapped = [c["name"] for c in calls
                if map_function(c["name"])[1] == "unknown"]
    if kind == "irrelevance":
        mod = _program_from_calls(sample.instruction, [], [])
        return {"status": "full" if validate(mod).valid else "none",
                "module": mod, "reasons": []}
    if not calls and kind != "irrelevance":
        return {"status": "partial",
                "reasons": ["no_ground_truth_calls"]}

    sequential = kind == "multi_turn"
    mod = _program_from_calls(sample.instruction, calls, unmapped,
                              sequential=sequential)
    rep = validate(mod)
    if not rep.valid:
        return {"status": "none",
                "reasons": [f"invalid_taskir:{rep.errors[0].code}"]}
    if kind == "multi_turn":
        reasons.append("multi_turn_state:inter_turn_state_not_expressed")
    if unmapped:
        reasons.append(
            f"unknown_tool_semantics:{len(unmapped)}/{len(calls)}")
    return {"status": "partial" if reasons else "full",
            "module": mod, "reasons": reasons}
