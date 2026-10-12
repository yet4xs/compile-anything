"""TaskIR semantic evaluation — Phase 8A.1 corrected protocol (v2).

Evaluator-integrity fixes over 8A (advisor review):
1. E2E DENOMINATORS: parse failures, validator failures and empty plans
   all score ZERO in E2E metrics; conditional variants (scored over
   structurally-valid programs only) are reported side by side, never
   mixed. This removes the conditional-mean artifact (C: 0.68 steps but
   0.82 ordered recall).
2. PLAN METRICS: ordered Precision / Recall / F1 (LCS-based) plus Exact
   Ordered Plan Match and Predicted/Reference step ratio — LCS recall
   alone would reward padding.
3. STRICT BINDING: no auto-bindable by skill class. A tool counts as
   IR-bound only when an EXEC_ACTION node carries its identity, the id
   exists in the runtime registry, and required schema params are
   present. IR Binding F1 measures TaskIR identity retention; Tool
   Selection F1 (resolver set vs reference set) is reported separately.
4. ARGUMENT VALUE ACCURACY: denominator = ALL reference-call parameters
   of matched tools; missing keys score zero.
5. ARGUMENT-AWARE REFERENCE NORMALIZATION: only consecutive, IDENTICAL
   (same tool AND same arguments) read-only calls are merged as provably
   redundant; mutating / communication / agent-internal calls are never
   merged; different arguments are never merged. Raw trace, conservative
   plan and a commutable-op DAG (independent adjacent reads) are all
   reported.

The 8A evaluator (taskir_semantic_eval.py) stays frozen for reproducing
8A numbers; v2 is the protocol going forward.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from src.ir.parser import parse_text
from src.validator.validator import validate
from src.eval.tau3_skill_oracle import get_tool_oracle
from src.eval.tau3_semantic_oracle_data import COARSE

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
NON_STATE_CLASSES = ("read_only",)          # mergeable-if-identical only


def _f1(inter, npred, nref):
    p = inter / npred if npred else 0.0
    r = inter / nref if nref else 0.0
    return (round(2 * p * r / (p + r), 4) if (p + r) else 0.0), \
        round(p, 4), round(r, 4)


def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _norm(v):
    return str(v).strip().lower()


# ------------------------------------------------- reference normalization

def normalize_ref_plan(domain: str, ref_actions: List[dict]) -> dict:
    """Argument-aware conservative normalization.

    Merge rule: consecutive calls with the SAME tool and IDENTICAL
    arguments whose effect_class is read_only (a provably redundant
    re-fetch). Everything else — mutating, communication, agent-internal,
    or same-tool-different-args — is kept in original order.
    Also derives a commutable-op DAG (reporting only): adjacent
    independent read-only ops with disjoint arguments.
    """
    raw = []
    for a in ref_actions:
        name = a.get("name", "")
        if not name:
            continue
        raw.append({"tool": name,
                    "args": a.get("arguments") or {},
                    "oracle": get_tool_oracle(domain, name)})
    plan = []
    for item in raw:
        o = item["oracle"]
        dup = (plan and plan[-1]["tool"] == item["tool"]
               and plan[-1]["args"] == item["args"])
        if dup and o["effect_class"] in NON_STATE_CLASSES:
            plan[-1]["merged_count"] += 1
            continue
        plan.append({"tool": item["tool"], "args": dict(item["args"]),
                     "effect_class": o["effect_class"],
                     "merged_count": 1})
    commutable = []
    for i in range(len(plan) - 1):
        a, b = plan[i], plan[i + 1]
        disjoint = not (set(map(str, a["args"])) &
                        set(map(str, b["args"])))
        if (a["effect_class"] == "read_only"
                and b["effect_class"] == "read_only" and disjoint):
            commutable.append((i, i + 1))
    return {"raw_n": len(raw), "plan": plan,
            "commutable_pairs": commutable}


def ref_skill_plan(domain: str, ref_actions: List[dict]):
    np_ = normalize_ref_plan(domain, ref_actions)
    skills = [get_tool_oracle(domain, p["tool"])["semantic_skill"]
              for p in np_["plan"]]
    return skills, np_


# ------------------------------------------------------------- evaluation

def evaluate_sample_v2(domain: str, pred_text: str,
                       ref_actions: List[dict],
                       tool_schemas: Dict[str, dict],
                       selected_tools: Optional[List[str]] = None) -> dict:
    """Corrected five-layer evaluation. selected_tools = tool ids the
    RESOLVER chose (Tool Selection F1, independent of the IR)."""
    res = {"domain": domain}

    l0 = {"parse_ok": False, "valid": False, "error": None}
    mod = None
    try:
        mod = parse_text(pred_text)
        l0["parse_ok"] = True
        rep = validate(mod)
        l0["valid"] = rep.valid
        if not rep.valid:
            l0["error"] = "; ".join(str(e) for e in rep.errors[:2])[:200]
    except Exception as e:
        l0["error"] = f"{type(e).__name__}: {e}"[:200]
    res["l0"] = l0
    struct_ok = l0["parse_ok"] and l0["valid"]

    ref_skills, norm_plan = ref_skill_plan(domain, ref_actions)
    ref_plan = norm_plan["plan"]
    scorable_tools = [p["tool"] for p in ref_plan
                      if get_tool_oracle(domain, p["tool"])["semantic_skill"]
                      is not None]
    user_side = sorted({p["tool"] for p in ref_plan
                        if get_tool_oracle(domain, p["tool"])["semantic_skill"]
                        is None})
    res["partial_infeasible"] = bool(user_side)
    res["user_side_refs"] = user_side
    res["ref_raw_n"] = norm_plan["raw_n"]
    res["ref_normalized_n"] = len(ref_plan)

    pred_seq = [n.op for n in mod.program.nodes
                if n.op not in POLICY_OPS] if mod else []

    # ---------- L2: conditional + E2E, plan P/R/F1, exact match ----------
    inter = sum((Counter(pred_seq) & Counter(ref_skills)).values())
    cf_f1, cf_p, cf_r = _f1(inter, len(pred_seq), len(ref_skills))
    lcs = _lcs_len(pred_seq, ref_skills)
    plan_p = round(lcs / len(pred_seq), 4) if pred_seq else 0.0
    plan_r = round(lcs / len(ref_skills), 4) if ref_skills else 0.0
    plan_f1 = (round(2 * plan_p * plan_r / (plan_p + plan_r), 4)
               if (plan_p + plan_r) else 0.0)
    pc = [COARSE.get(o, "other") for o in pred_seq]
    rc = [COARSE.get(o, "other") for o in ref_skills]
    cinter = sum((Counter(pc) & Counter(rc)).values())
    cf2, _, _ = _f1(cinter, len(pc), len(rc))
    res["l2"] = {
        "conditional_skill_f1": cf_f1 if struct_ok else None,
        "e2e_skill_f1": cf_f1 if struct_ok else 0.0,
        "coarse_skill_f1": cf2 if struct_ok else 0.0,
        "plan_precision": plan_p if struct_ok else 0.0,
        "plan_recall": plan_r if struct_ok else 0.0,
        "plan_f1": plan_f1 if struct_ok else 0.0,
        "exact_ordered_match": bool(pred_seq == ref_skills) if struct_ok
        else False,
    }

    # ---------- L1: strict IR binding + tool selection ----------
    exec_nodes = [n for n in (mod.program.nodes if mod else [])
                  if n.op == "EXEC_ACTION"]
    pred_bound = {}
    for n in exec_nodes:
        act = _norm(n.params.get("action", ""))
        if act and act not in pred_bound:
            pred_bound[act] = n
    bound_ok = set()
    for tool, n in pred_bound.items():
        schema = tool_schemas.get(tool)
        if not schema:
            continue                      # not in the runtime registry
        required = [pp["name"] for pp in schema.get("params", [])
                    if not pp.get("optional")]
        if all(k in n.params for k in required):
            bound_ok.add(tool)
    ref_set = set(scorable_tools)
    b_f1, b_p, b_r = _f1(len(bound_ok & ref_set),
                         len(pred_bound), len(ref_set))
    res["l1_ir_binding"] = {
        "pred_bound_ids": sorted(pred_bound),
        "registry_schema_ok": sorted(bound_ok),
        "ir_binding_f1": b_f1 if struct_ok else 0.0,
        "ir_binding_p": b_p if struct_ok else 0.0,
        "ir_binding_r": b_r if struct_ok else 0.0,
        "exact_tool_set": bool(set(pred_bound) == ref_set) if struct_ok
        else False,
    }
    if selected_tools is not None:
        sel = set(selected_tools)
        s_f1, s_p, s_r = _f1(len(sel & ref_set), len(sel), len(ref_set))
        res["l1_tool_selection"] = {"selected": sorted(sel), "f1": s_f1,
                                    "p": s_p, "r": s_r}
    else:
        res["l1_tool_selection"] = None

    # ---------- L3: args with ALL-ref-params denominator ----------
    total_req = correct = 0
    coverage_vals = []
    for p in ref_plan:
        tool = p["tool"]
        if tool not in bound_ok:
            continue
        ref_args = p["args"]
        node_params = {k: v for k, v in pred_bound[tool].params.items()
                       if k != "action"}
        keys = list(ref_args.keys())
        if not keys:
            continue
        total_req += len(keys)
        correct += sum(1 for k in keys
                       if k in node_params
                       and _norm(node_params[k]) == _norm(ref_args[k]))
        coverage_vals.append(
            len([k for k in keys if k in node_params]) / len(keys))
    res["l3"] = {
        "arg_value_acc_all_ref_params":
            (round(correct / total_req, 4)
             if total_req else None) if struct_ok else
            (0.0 if total_req else None),
        "param_coverage":
            (round(sum(coverage_vals) / len(coverage_vals), 4)
             if coverage_vals else None) if struct_ok else
            (0.0 if coverage_vals else None),
        "pred_steps": len(pred_seq),
        "ref_steps": len(ref_skills),
        "step_ratio": (round(len(pred_seq) / len(ref_skills), 4)
                       if ref_skills else None),
    }

    # ---------- L4: effects (strict) ----------
    effect_ok = effect_total = 0
    downgrades = []
    for p in ref_plan:
        tool = p["tool"]
        o = get_tool_oracle(domain, tool)
        if o["semantic_skill"] is None:
            continue
        effect_total += 1
        mut = o["effect_class"] not in (
            "read_only", "communication", "agent_internal")
        if tool in bound_ok:
            effect_ok += 1                # identity + registry + schema
        elif not mut and o["semantic_skill"] in set(pred_seq):
            effect_ok += 1                # read realized semantically
            # (semantic realization does NOT identify which read API)
        elif mut and not exec_nodes:
            downgrades.append(tool)
    res["l4"] = {
        "effect_consistency": (round(effect_ok / effect_total, 4)
                               if effect_total else None),
        "downgrades": downgrades,
        "unsafe_retry": [
            {"node": n.id, "tool": _norm(n.params.get("action", ""))}
            for n in (mod.program.nodes if mod else [])
            if n.retry and n.op == "EXEC_ACTION"
            and get_tool_oracle(
                domain, _norm(n.params.get("action", "")))["effect_class"]
            in ("state_creation", "state_transition_irreversible")],
        "unsupported": (["multi-step transactional mutation ordering "
                         "not expressible in TaskIR v0.1"]
                        if sum(1 for p in ref_plan
                               if p["effect_class"] in (
                                   "state_creation",
                                   "state_transition_irreversible")) > 1
                        else []),
    }
    res["executable_task_success"] = None
    return res
