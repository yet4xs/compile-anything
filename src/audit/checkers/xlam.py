"""Trajectory checkers — xLAM / ToolBench-Static / ToolBench (Tasks 2/3).

Deterministic verification of plan_target against the RAW tool-call
trajectory: action coverage, order, semantic mapping (vs recorded
provenance), and argument fidelity per-skill.
"""
from __future__ import annotations

import json
from typing import Dict, List

from ..semantic_label import (SemanticAuditResult, flag, annotate,
                              instruction_plan_consistency, value_preserved)

# ops inserted by compiler policy/bridging, not by the trajectory
NON_ACTION_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

# Task 3: key argument fields per skill (checked for preservation)
KEY_FIELDS = {
    "SEARCH": ["domain", "query"],
    "SEND": ["channel", "query"],
    "CONVERT": ["from", "to", "amount"],
    "QUERY_DB": ["table", "query"],
    "CALCULATE": ["expr"],
    "TRANSLATE": ["lang"],
    "EXEC_ACTION": ["action"],
}


def audit_trajectory(record: dict, raw_calls: List[Dict]) -> SemanticAuditResult:
    """record: v3 corpus record; raw_calls: [{tool/name, args/arguments}]."""
    res = SemanticAuditResult(sample_id=record.get("id", ""),
                              source=record.get("source", ""),
                              tier=record.get("quality_tier", ""))
    nodes = record["plan_json"]["program"]["nodes"]
    lowering = record.get("lowering") or []
    action_nodes = [n for n in nodes if n["op"] not in NON_ACTION_OPS]
    ops = [n["op"] for n in nodes]

    # ---- action coverage -------------------------------------------
    n_raw = len(raw_calls or [])
    n_ir = len(action_nodes)
    res.checks["action_coverage"] = {"raw_calls": n_raw, "ir_actions": n_ir}
    if n_ir != n_raw:
        flag(res, "MISSING_ACTION" if n_ir < n_raw else "EXTRA_ACTION",
             f"raw={n_raw} ir={n_ir}")

    # ---- order + semantic mapping (via lowering provenance) ---------
    # lowering[] is in raw call order; the i-th action node must carry the
    # i-th lowered skill
    res.checks["action_order"] = "linear-chain"
    mismatches = []
    for i, low in enumerate(lowering):
        if i >= len(action_nodes):
            break
        if action_nodes[i]["op"] != low.get("skill"):
            mismatches.append(
                f"node{i}={action_nodes[i]['op']} vs lowering={low.get('skill')}")
    if mismatches:
        flag(res, "WRONG_SKILL", "; ".join(mismatches[:3]))
    if len(lowering) != n_ir:
        flag(res, "MISSING_ACTION",
             f"lowering entries {len(lowering)} != action nodes {n_ir}")

    # ---- argument fidelity (Task 3) ---------------------------------
    lost = []
    preserved = total = 0
    for i, call in enumerate(raw_calls or []):
        if i >= len(action_nodes):
            break
        node = action_nodes[i]
        skill = node["op"]
        blob = json.dumps(node.get("params", {}), ensure_ascii=False)
        args = call.get("args") or call.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        # check every scalar raw argument value survives somewhere in the
        # node params (normalized containment, not exact match)
        for k, v in args.items():
            if not isinstance(v, (str, int, float)) or v in ("", None):
                continue
            total += 1
            if value_preserved(v, blob):
                preserved += 1
            else:
                lost.append(f"{skill}.{k}={str(v)[:30]}")
    rate = round(preserved / total, 4) if total else 1.0
    res.checks["argument_preservation_rate"] = rate
    res.evidence["argument_loss_examples"] = lost[:5]
    if total and rate < 0.999:
        flag(res, "ARGUMENT_LOSS", f"rate={rate} lost={lost[:3]}")

    # ---- instruction consistency (Task 4) ------------------------------
    # trajectory sources: the label's contract is faithfulness to the raw
    # calls; keyword gaps become GROUND_TRUTH_AMBIGUOUS annotations, not
    # suspect (the gap is between instruction and the benchmark's own
    # reference, not in our lowering)
    instruction_plan_consistency(res, record.get("instruction", ""), ops,
                                 mode="annotate")

    # mapping provenance consistency (Task 2.1 semantic mapping)
    kinds = [c.get("mapping_kind") for c in lowering]
    res.evidence["mapping_kinds"] = kinds

    if res.status != "suspect":
        res.status = "verified"
        res.checks["verdict_basis"] = ("raw trajectory count/order/skills/"
                                       "arguments all matched")
    return res
