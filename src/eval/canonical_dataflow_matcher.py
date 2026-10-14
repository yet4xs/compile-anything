"""Phase 8D-1B.1: Canonical Task Graph Matching.

Node-rename-invariant, reorder-invariant semantic comparison of dataflow
programs. Four independent metrics:
  A. Canonical Call & Binding Exact — tool instances + literals +
     runtime_ref (producer, field, consumer, param) triples, allowing
     legal reordering of independent nodes
  B. Exact DATA/ORDER/EFFECT DAG — edge sets must match exactly
     (extra DATA edges counted as errors)
  C. Mock Trace Semantic Match — actual execution traces compared
  D. End-to-End Strict — A + B + C all correct
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ir.parser import parse_text


# ── Canonical extraction from both representations ──

def canonical_from_skeleton(nodes: List[dict]) -> dict:
    """Extract canonical representation from skeleton node list."""
    calls = []
    edges = set()  # (producer_call_idx, field, consumer_call_idx, param)
    for i, n in enumerate(nodes):
        literals = {}
        rt_refs = {}
        for k, v in (n.get("arguments") or {}).items():
            if isinstance(v, dict) and v.get("kind") == "runtime_ref":
                rt_refs[k] = (v.get("producer", ""),
                              v.get("field", ""))
            else:
                literals[k] = v
        calls.append({
            "idx": i,
            "id": n.get("id", f"p{i}"),
            "tool": n.get("capability_id", ""),
            "literals": literals,
            "rt_refs": rt_refs,
        })
    # resolve rt_refs to indices
    id_to_idx = {c["id"]: c["idx"] for c in calls}
    for c in calls:
        for pk, (prod_id, field) in c["rt_refs"].items():
            prod_idx = id_to_idx.get(prod_id, -1)
            edges.add((prod_idx, field, c["idx"], pk))
    # DATA edges from depends_on
    data_edges = set()
    for i, n in enumerate(nodes):
        for d in n.get("depends_on", []):
            j = id_to_idx.get(d, -1)
            if j >= 0:
                data_edges.add((j, i))
    return {"calls": calls, "bindings": edges, "data_edges": data_edges}


def canonical_from_taskir(taskir_text: str) -> Optional[dict]:
    """Extract canonical representation from TaskIR v0.2 text."""
    if not taskir_text:
        return None
    try:
        mod = parse_text(taskir_text)
    except Exception:
        return None

    # Map EXTRACT_FIELD intermediaries
    ext_map = {}  # ssa_id → (producer_ssa, field_name)
    for n in mod.program.nodes:
        if n.op == "EXTRACT_FIELD" and n.inputs:
            ext_map[n.id] = (n.inputs[0],
                              str(n.params.get("field", "")))

    # Build calls (skip EXTRACT_FIELD)
    calls = []
    ssa_to_call = {}  # ssa_id → call index
    for n in mod.program.nodes:
        if n.op == "EXTRACT_FIELD":
            continue
        tool = n.params.get("action") or (n.hints or {}).get("tool", "")
        literals = {k: v for k, v in n.params.items() if k != "action"}
        # Resolve param_refs through EXTRACT_FIELD chain
        rt_refs = {}
        for pk, ref in n.param_refs.items():
            if ref in ext_map:
                prod_ssa, field = ext_map[ref]
                rt_refs[pk] = (prod_ssa, field)
            elif ref in ssa_to_call or ref in ext_map:
                rt_refs[pk] = (ref, "")  # direct ref without field
        calls.append({
            "idx": len(calls), "id": n.id, "tool": tool,
            "literals": literals, "rt_refs": rt_refs,
            "raw_ssa": n.id,
        })
        ssa_to_call[n.id] = len(calls) - 1

    # Resolve rt_refs producer SSAs to call indices
    bindings = set()
    for c in calls:
        for pk, (prod_ssa, field) in c["rt_refs"].items():
            # Resolve through EXTRACT_FIELD chain
            while prod_ssa in ext_map:
                prod_ssa, field = ext_map[prod_ssa]
            prod_idx = ssa_to_call.get(prod_ssa, -1)
            bindings.add((prod_idx, field, c["idx"], pk))

    # DATA edges: from inputs + after
    data_edges = set()
    for c in calls:
        # We need to look at the original node for inputs/after
        pass
    # Re-iterate nodes for edges
    for n in mod.program.nodes:
        if n.op == "EXTRACT_FIELD":
            continue
        ci = ssa_to_call.get(n.id, -1)
        if ci < 0:
            continue
        for ref in n.inputs:
            # Resolve through EXTRACT_FIELD
            r = ref
            while r in ext_map:
                r, _ = ext_map[r]
            ri = ssa_to_call.get(r, -1)
            if ri >= 0 and ri != ci:
                data_edges.add((ri, ci))
        for ref in (n.after or []):
            ri = ssa_to_call.get(ref, -1)
            if ri >= 0 and ri != ci:
                data_edges.add((ri, ci))

    return {"calls": calls, "bindings": bindings, "data_edges": data_edges}


# ── Canonical matching (node-reorder invariant) ──

def _call_signature(call: dict) -> tuple:
    """Signature for matching: tool + sorted literals."""
    lits = tuple(sorted((k, str(v)) for k, v in call["literals"].items()))
    return (call["tool"], lits)


def _binding_signature(binding: tuple, calls: list) -> tuple:
    """Semantic binding signature independent of node indices."""
    prod_idx, field, cons_idx, param = binding
    prod_tool = calls[prod_idx]["tool"] if 0 <= prod_idx < len(calls) else "?"
    cons_tool = calls[cons_idx]["tool"] if 0 <= cons_idx < len(calls) else "?"
    prod_lits = _call_signature(calls[prod_idx]) if 0 <= prod_idx < len(calls) else ("?", ())
    cons_lits = _call_signature(calls[cons_idx]) if 0 <= cons_idx < len(calls) else ("?", ())
    # Use tool + literals to disambiguate same-tool instances
    return (prod_tool, prod_lits[1][:3], field, cons_tool, cons_lits[1][:3], param)


def canonical_call_binding_match(ref: dict, pred: dict) -> dict:
    """Metric A: Canonical Call & Binding Exact.
    Allows legal reordering; matches by (tool, literals) not position."""
    result = {"calls_match": False, "bindings_match": False,
              "n_ref_calls": len(ref["calls"]),
              "n_pred_calls": len(pred["calls"]),
              "n_ref_bindings": len(ref["bindings"]),
              "n_pred_bindings": len(pred["bindings"]),
              "n_correct_bindings": 0,
              "extra_bindings": 0}

    # Match calls by (tool, sorted literals) multiset
    ref_sigs = sorted(_call_signature(c) for c in ref["calls"])
    pred_sigs = sorted(_call_signature(c) for c in pred["calls"])
    result["calls_match"] = ref_sigs == pred_sigs

    # Match bindings by semantic signature
    ref_b = {_binding_signature(b, ref["calls"]) for b in ref["bindings"]}
    pred_b = {_binding_signature(b, pred["calls"]) for b in pred["bindings"]}
    inter = len(ref_b & pred_b)
    result["n_correct_bindings"] = inter
    result["extra_bindings"] = len(pred_b - ref_b)
    result["bindings_match"] = (ref_b == pred_b)

    return result


def canonical_dag_match(ref: dict, pred: dict) -> dict:
    """Metric B: Exact DATA/ORDER/EFFECT DAG.
    Extra DATA edges are errors. Uses semantic node identity."""
    result = {"data_exact": False, "extra_data": 0, "missing_data": 0,
              "data_precision": 0.0, "data_recall": 0.0, "data_f1": 0.0}

    # Build semantic node identity map
    def node_map(canonical: dict) -> dict:
        """Map call index → (tool, sorted_literals) for identity."""
        m = {}
        for c in canonical["calls"]:
            m[c["idx"]] = _call_signature(c)
        return m

    ref_nodes = node_map(ref)
    pred_nodes = node_map(pred)

    # Convert edges to semantic form (node signatures, not indices)
    ref_edges = set()
    for i, j in ref["data_edges"]:
        if i in ref_nodes and j in ref_nodes:
            ref_edges.add((ref_nodes[i], ref_nodes[j]))

    pred_edges = set()
    for i, j in pred["data_edges"]:
        if i in pred_nodes and j in pred_nodes:
            pred_edges.add((pred_nodes[i], pred_nodes[j]))

    inter = len(ref_edges & pred_edges)
    result["extra_data"] = len(pred_edges - ref_edges)
    result["missing_data"] = len(ref_edges - pred_edges)
    result["data_exact"] = (ref_edges == pred_edges)
    if pred_edges:
        result["data_precision"] = round(inter / len(pred_edges), 4)
    if ref_edges:
        result["data_recall"] = round(inter / len(ref_edges), 4)
    if inter:
        p, r = result["data_precision"], result["data_recall"]
        result["data_f1"] = round(2 * p * r / (p + r) if (p + r) else 0, 4)
    return result


def canonical_full_match(ref: dict, pred: dict) -> dict:
    """Compute all four metrics."""
    a = canonical_call_binding_match(ref, pred)
    b = canonical_dag_match(ref, pred)
    # Metrics C (trace) and D (strict) require execution — computed
    # separately by the caller with runtime access
    return {"A_call_binding": a, "B_dag": b}


# ── High-level evaluation entry point ──

def evaluate_canonical(ref_skeleton_nodes: list,
                       pred_taskir_text: str) -> Optional[dict]:
    """Full canonical evaluation of one prediction vs reference."""
    ref = canonical_from_skeleton(ref_skeleton_nodes)
    pred = canonical_from_taskir(pred_taskir_text)
    if pred is None:
        return None
    return canonical_full_match(ref, pred)
