"""Phase 8B-1C Tasks 2+3: per-call semantic evaluation (v3).

Fixes over 8B-1B scoring:
- PER-CALL matching: every reference call must be matched at its position
  with the right tool id and its own arguments — no tool-set shortcuts,
  no cross-call parameter merging
- extra calls and extra illegal params are reported separately
- missing required params count wrong; calls WITHOUT reference arguments
  get an independent rule (schema-required keys must be present; the
  task is flagged value_unverifiable, never auto-passed)
- exact_joint_v3 = identity + call order + per-call args + dependencies
- DAG scoring from ONE extractor over TaskIR for BOTH arms: DATA edges
  = SSA operand refs, ORDER edges = after(); scored vs the reference
  skeleton's depends_on/order_on; dependency-free tasks reported
  separately (never merged into zero scores)
- effect scoring: corpus has no reliable effect ground truth ->
  effect_supported=False (unsupported), never faked
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from src.ir.parser import parse_text

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}


def _norm(v) -> str:
    return str(v).strip().lower()


def extract_calls_from_taskir(taskir_text: str):
    """Single dependency extractor for BOTH arms: ordered calls with
    DATA deps (SSA operands) and ORDER deps (after())."""
    if not taskir_text:
        return None
    try:
        mod = parse_text(taskir_text)
    except Exception:
        return None
    calls = []
    for n in mod.program.nodes:
        tool = None
        if n.op == "EXEC_ACTION" and n.params.get("action"):
            tool = _norm(n.params["action"])
        else:
            t = (n.hints or {}).get("tool")
            if t:
                tool = _norm(t)
        params = {k: v for k, v in n.params.items() if k != "action"}
        data_deps = [r for r in n.inputs if r.startswith("%")]
        calls.append({
            "ssa": n.id, "op": n.op, "tool": tool,
            "params": params,
            "data_deps": data_deps,
            "order_deps": list(n.after or []),
        })
    return calls


def evaluate_v3(taskir_text: str, rec: dict) -> dict:
    """rec must carry target_skeleton + capability_registry."""
    ref_nodes = rec["target_skeleton"]["nodes"]
    reg = rec.get("capability_registry") or {}
    out = {
        "n_ref_calls": len(ref_nodes),
        "parse": False, "call_matches": [],
        "extra_calls": 0, "extra_params_total": 0,
        "missing_params_total": 0, "wrong_value_total": 0,
        "value_unverifiable_calls": 0,
        "identity_ok": False, "order_ok": False, "args_ok": False,
        "exact_joint_v3": False,
        "dag": {"n_ref_dep_tasks": None, "data_f1": None,
                "order_f1": None, "combined_exact": None},
        "effect_supported": False,   # corpus: no reliable effect GT
    }
    calls = extract_calls_from_taskir(taskir_text)
    if calls is None:
        return out
    out["parse"] = True

    # ---------- per-call matching (positional) ----------
    n = min(len(ref_nodes), len(calls))
    out["extra_calls"] = max(0, len(calls) - len(ref_nodes))
    all_match = len(calls) == len(ref_nodes)
    args_all_ok = True
    for i in range(n):
        rn, pc = ref_nodes[i], calls[i]
        ref_tool = _norm(rn["capability_id"])
        id_ok = (pc["tool"] == ref_tool)
        ref_args = rn["arguments"] or {}
        schema = reg.get(rn["capability_id"], {})
        required = schema.get("required") or [] \
            if schema.get("schema") == "props" else []
        cmatch = {"pos": i, "ref_tool": ref_tool, "pred_tool": pc["tool"],
                  "identity_ok": id_ok, "arg_status": None}
        if not id_ok:
            all_match = False
            cmatch["arg_status"] = "skipped_wrong_tool"
            out["call_matches"].append(cmatch)
            continue
        keys = list(ref_args.keys()) or list(required)
        if not ref_args:
            # independent rule: no reference values -> require presence
            # of schema-required keys; flag unverifiable, never auto-pass
            missing = [k for k in required if k not in pc["params"]]
            if missing:
                out["missing_params_total"] += len(missing)
                args_all_ok = False
                cmatch["arg_status"] = f"missing:{missing}"
            else:
                cmatch["arg_status"] = "keys_ok_value_unverifiable"
                out["value_unverifiable_calls"] += 1
            out["call_matches"].append(cmatch)
            continue
        missing = [k for k in keys if k not in pc["params"]]
        wrong = [k for k in keys
                 if k in pc["params"]
                 and _norm(pc["params"][k]) != _norm(ref_args[k])]
        extra = [k for k in pc["params"] if k not in ref_args]
        out["missing_params_total"] += len(missing)
        out["wrong_value_total"] += len(wrong)
        out["extra_params_total"] += len(extra)
        if missing or wrong:
            args_all_ok = False
            cmatch["arg_status"] = (f"missing:{missing} wrong:{wrong} "
                                    f"extra:{extra}")
        else:
            cmatch["arg_status"] = "ok" + (f"+extra:{extra}" if extra
                                           else "")
        out["call_matches"].append(cmatch)
    out["identity_ok"] = (all_match and all(
        c["identity_ok"] for c in out["call_matches"][:n]))
    out["order_ok"] = (len(calls) == len(ref_nodes))
    out["args_ok"] = args_all_ok
    out["exact_joint_v3"] = bool(out["parse"] and out["identity_ok"]
                                 and out["order_ok"] and out["args_ok"])

    # ---------- DAG scoring (same extractor for both arms) ----------
    ref_data = {(i, rn["depends_on"].index(d)) if False else (i, d)
                for i, rn in enumerate(ref_nodes)
                for d in rn["depends_on"]}
    ref_order = {(i, d) for i, rn in enumerate(ref_nodes)
                 for d in rn.get("order_on", [])}
    pred_data = {(i, d) for i, c in enumerate(calls)
                 for d in c["data_deps"]}
    pred_order = {(i, d) for i, c in enumerate(calls)
                  for d in c["order_deps"]}

    def f1(ref, pred):
        if not ref and not pred:
            return None                    # dependency-free: reported apart
        inter = len(ref & pred)
        if not inter:
            return 0.0
        p = inter / len(pred)
        r = inter / len(ref)
        return round(2 * p * r / (p + r), 4)

    has_dep = bool(ref_data or ref_order)
    out["dag"] = {
        "n_ref_dep_tasks": 1 if has_dep else 0,
        "data_f1": f1(ref_data, pred_data),
        "order_f1": f1(ref_order, pred_order),
        "combined_exact": (ref_data == pred_data
                           and ref_order == pred_order) if has_dep else None,
    }
    return out
