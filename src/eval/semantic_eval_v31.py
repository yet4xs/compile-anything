"""Phase 8C-1 Task 2: semantic evaluator v3.1 — strict joint.

Fixes over v3:
- Validator Pass is now a REQUIRED component of Strict Joint
- DATA/ORDER dependency matching uses canonical node indices
- Effect: only scored when reliable labels exist; otherwise reported
  as unsupported (never auto-passed)
- Extra params now make args_ok False for that call
- Strict Joint = ValidatorPass AND identity AND order AND args AND
  DATA-DAG AND ORDER-DAG (when DAG has edges to check)
"""
from __future__ import annotations

import json
from typing import Dict, List

from src.ir.parser import parse_text
from src.validator.validator import validate

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}


def _norm(v) -> str:
    return str(v).strip().lower()


def extract_from_taskir(taskir_text: str):
    """Extract ordered calls + edges from TaskIR text."""
    if not taskir_text:
        return None
    try:
        mod = parse_text(taskir_text)
    except Exception:
        return None
    calls = []
    ssa_index = {}
    for n in mod.program.nodes:
        ssa_index[n.id] = len(calls)
        tool = None
        if n.op == "EXEC_ACTION" and n.params.get("action"):
            tool = _norm(n.params["action"])
        else:
            t = (n.hints or {}).get("tool")
            if t:
                tool = _norm(t)
        params = {k: v for k, v in n.params.items() if k != "action"}
        data_deps = []
        for r in n.inputs:
            if r in ssa_index:
                data_deps.append(ssa_index[r])
        order_deps = []
        for a in (n.after or []):
            if a in ssa_index:
                order_deps.append(ssa_index[a])
        calls.append({
            "idx": len(calls), "op": n.op, "tool": tool,
            "params": params,
            "data_deps": data_deps, "order_deps": order_deps,
        })
    return calls


def evaluate_v31(taskir_text: str, rec: dict) -> dict:
    """Strict joint semantic evaluation."""
    ref_nodes = rec["target_skeleton"]["nodes"]
    reg = rec.get("capability_registry") or {}
    out = {
        "n_ref": len(ref_nodes),
        "parser_pass": False, "validator_pass": False,
        "identity_exact": False, "order_exact": False,
        "args_exact": False, "data_dag_exact": None,
        "order_dag_exact": None, "effect_ok": None,
        "effect_supported": False,
        "strict_joint": False,
        "extra_calls": 0, "extra_params": 0,
        "missing_params": 0, "wrong_values": 0,
        "call_details": [],
    }
    calls = extract_from_taskir(taskir_text)
    if calls is None:
        return out
    out["parser_pass"] = True

    # Validator Pass (required for strict joint)
    try:
        mod = parse_text(taskir_text)
        out["validator_pass"] = validate(mod).valid
    except Exception:
        out["validator_pass"] = False

    # --- per-call matching ---
    n = min(len(ref_nodes), len(calls))
    out["extra_calls"] = max(0, len(calls) - len(ref_nodes))
    identity_all = (len(calls) == len(ref_nodes))
    args_all = True
    for i in range(n):
        rn, pc = ref_nodes[i], calls[i]
        ref_tool = _norm(rn["capability_id"])
        id_ok = (pc["tool"] == ref_tool)
        if not id_ok:
            identity_all = False
        detail = {"pos": i, "tool_ok": id_ok, "args": {}}

        ref_args = rn["arguments"] or {}
        meta = reg.get(rn["capability_id"], {})
        required = meta.get("required") or []
        keys = sorted(set(list(ref_args.keys()) + list(required))
                      if ref_args else list(required))
        if not keys:
            detail["args"] = {"status": "no_verifiable_args"}
            out["call_details"].append(detail)
            continue
        missing = [k for k in keys if k not in pc["params"]]
        wrong = [k for k in keys if k in pc["params"]
                 and _norm(pc["params"][k]) != _norm(ref_args.get(k, ""))]
        extra = [k for k in pc["params"] if k not in ref_args]
        out["missing_params"] += len(missing)
        out["wrong_values"] += len(wrong)
        out["extra_params"] += len(extra)
        if missing or wrong or extra:
            args_all = False
            detail["args"] = {"missing": missing, "wrong": wrong,
                              "extra": extra}
        else:
            detail["args"] = {"status": "exact"}
        out["call_details"].append(detail)
    out["identity_exact"] = identity_all
    out["order_exact"] = (len(calls) == len(ref_nodes))
    out["args_exact"] = args_all

    # --- DAG matching (canonical indices) ---
    ref_data = set()
    ref_order = set()
    for i, rn in enumerate(ref_nodes):
        for d in rn.get("depends_on", []):
            try:
                j = int(d.replace("p", "")) if d.startswith("p") else \
                    int(d.replace("s", "")) if d.startswith("s") else -1
                ref_data.add((j, i))
            except ValueError:
                pass
        for d in rn.get("order_on", []):
            try:
                j = int(d.replace("p", "")) if d.startswith("p") else \
                    int(d.replace("s", "")) if d.startswith("s") else -1
                ref_order.add((j, i))
            except ValueError:
                pass
    pred_data = set()
    pred_order = set()
    for c in calls:
        for d in c["data_deps"]:
            pred_data.add((d, c["idx"]))
        for d in c["order_deps"]:
            pred_order.add((d, c["idx"]))
    # DAG only checked when reference HAS edges (dep-free tasks = None)
    if ref_data:
        out["data_dag_exact"] = (ref_data == pred_data)
    if ref_order:
        out["order_dag_exact"] = (ref_order == pred_order)

    # --- effect (only when reliable labels exist) ---
    effect_labels = [reg.get(n["capability_id"], {}).get("effect")
                     for n in ref_nodes
                     if reg.get(n["capability_id"], {}).get("effect")]
    if effect_labels and all(effect_labels):
        out["effect_supported"] = True
        out["effect_ok"] = all(effect_labels)
    # else: effect_supported stays False, effect_ok stays None

    # --- Strict Joint ---
    dag_ok = True
    if out["data_dag_exact"] is False:
        dag_ok = False
    if out["order_dag_exact"] is False:
        dag_ok = False
    out["strict_joint"] = bool(
        out["validator_pass"]
        and out["identity_exact"]
        and out["order_exact"]
        and out["args_exact"]
        and dag_ok
        and (out["effect_ok"] is not False))  # None (unsupported) ok
    return out
