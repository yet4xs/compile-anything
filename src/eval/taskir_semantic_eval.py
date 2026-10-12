"""Phase 8A Task 3: five-layer TaskIR semantic evaluation.

L0 Structural  — parse + validator.
L1 Tool Sel.   — concrete tool ids (EXEC_ACTION action= / bound params)
                 vs reference tool set; P/R/F1 + exact set; partial-
                 infeasible flags for user-side reference tools.
L2 Skill Sem.  — predicted op sequence vs VERIFIED oracle skill sequence;
                 exact and coarse ontology, multiset P/R/F1 + ordered
                 sequence match (dual-reported, never mixed).
L3 Args & Deps — required-param coverage under the runtime schema,
                 argument VALUE accuracy vs reference arguments,
                 ordered plan recall (LCS) and depth; SSA legality is L0's
                 validator domain, only reported for context.
L4 Effect      — read/mutate consistency per matched tool (downgrade =
                 mutating ref tool realized as a read-only skill node),
                 bindability, unsafe-retry flags, unsupported expressiveness.

Executable Task Success is NOT computed here (no trusted sandbox) and must
not be inferred from L0-L4.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from src.ir.parser import parse_text
from src.validator.validator import validate
from src.eval.tau3_skill_oracle import get_tool_oracle
from src.eval.tau3_semantic_oracle_data import COARSE

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
READ_ONLY_SKILLS = {"FETCH", "SEARCH", "CALCULATE", "QUERY_DB", "LOAD"}


def _f1(inter, npred, nref):
    p = inter / npred if npred else 0.0
    r = inter / nref if nref else 0.0
    return round((2 * p * r / (p + r)) if (p + r) else 0.0, 4), \
        round(p, 4), round(r, 4)


def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y
                        else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _norm(v):
    return str(v).strip().lower()


def pred_tool_bindings(mod) -> List[dict]:
    """Concrete tool bindings expressible in TaskIR v0.1: EXEC_ACTION nodes
    (action=identity) plus params whose value equals a known tool name."""
    out = []
    for n in mod.program.nodes:
        if n.op == "EXEC_ACTION":
            out.append({"node": n.id, "tool": _norm(n.params.get("action", "")),
                        "params": {k: v for k, v in n.params.items()
                                   if k != "action"}})
    return out


def evaluate_sample(domain: str, pred_text: str,
                    ref_actions: List[dict],
                    tool_schemas: Dict[str, dict]) -> dict:
    """ref_actions: [{name, arguments}, ...] from evaluation_criteria.
    tool_schemas: {tool_name: {params: [...], mutates_state, tool_type}}
    from the runtime registry."""
    res = {"domain": domain}
    # ---------------- L0 ----------------
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

    # reference plan (verified oracle). tau3 references are MULTI-TURN
    # execution traces (e.g. 10 consecutive search_direct_flight calls);
    # TaskIR is a PLAN. Normalize by collapsing CONSECUTIVE duplicate
    # tools (iteration loops) while preserving first-occurrence order and
    # separate call sites. Raw counts are kept for transparency.
    raw_names = [a.get("name", "") for a in ref_actions if a.get("name")]
    ref_names = [n for i, n in enumerate(raw_names)
                 if i == 0 or n != raw_names[i - 1]]
    res["ref_steps_raw"] = len(raw_names)
    ref_plan = [get_tool_oracle(domain, n) for n in ref_names]
    ref_skill_seq = [o["semantic_skill"] for o in ref_plan
                     if o["semantic_skill"]]
    # strict: only oracle-known agent tools are scorable
    scorable = [n for n, o in zip(ref_names, ref_plan)
                if o["semantic_skill"] is not None]
    user_side = [n for n, o in zip(ref_names, ref_plan)
                 if o["semantic_skill"] is None]
    res["partial_infeasible"] = bool(user_side)
    res["user_side_refs"] = user_side

    # ---------------- L1 ----------------
    bindings = pred_tool_bindings(mod) if mod else []
    pred_tools = [b["tool"] for b in bindings if b["tool"]]
    scorable_set = set(scorable)
    matched = set(pred_tools) & scorable_set
    f1, p, r = _f1(len(matched), len(set(pred_tools)), len(scorable_set))
    l1 = {"pred_tools": sorted(set(pred_tools)),
          "ref_tools": sorted(scorable_set),
          "matched": sorted(matched),
          "precision": p, "recall": r, "f1": f1,
          "exact_set": set(pred_tools) == scorable_set,
          "n_pred": len(set(pred_tools)), "n_ref": len(scorable_set)}
    # semantic coverage: ref tools realized only as semantic-skill nodes
    if mod:
        pred_ops = Counter(n.op for n in mod.program.nodes)
        sem_cov = 0
        for n, o in zip(ref_names, ref_plan):
            sk = o["semantic_skill"]
            if n not in matched and sk and sk in pred_ops:
                sem_cov += 1
        l1["ref_covered_semantically_only"] = sem_cov
    res["l1"] = l1

    # ---------------- L2 ----------------
    pred_seq = [n.op for n in mod.program.nodes
                if n.op not in POLICY_OPS] if mod else []
    ref_seq = ref_skill_seq            # already normalized (see above)
    inter = sum((Counter(pred_seq) & Counter(ref_seq)).values())
    ef1, ep, er = _f1(inter, len(pred_seq), len(ref_seq))
    pc = [COARSE.get(o, "other") for o in pred_seq]
    rc = [COARSE.get(o, "other") for o in ref_seq]
    cinter = sum((Counter(pc) & Counter(rc)).values())
    cf1, cp, cr = _f1(cinter, len(pc), len(rc))
    res["l2"] = {
        "pred_seq": pred_seq, "ref_seq": ref_seq,
        "exact_f1": ef1, "exact_p": ep, "exact_r": er,
        "exact_sequence_match": pred_seq == ref_seq,
        "coarse_f1": cf1, "coarse_p": cp, "coarse_r": cr,
        "coarse_sequence_match": pc == rc,
    }

    # ---------------- L3 ----------------
    l3 = {"required_param_coverage": None, "arg_value_acc": None,
          "ordered_recall": None, "ordered_precision": None,
          "pred_steps": len(pred_seq), "ref_steps": len(ref_seq)}
    if mod and ref_seq:
        lcs = _lcs_len(pred_seq, ref_seq)
        l3["ordered_recall"] = round(lcs / len(ref_seq), 4) if ref_seq else None
        l3["ordered_precision"] = round(lcs / len(pred_seq), 4) if pred_seq else None
    if mod and matched:
        covs, accs = [], []
        for b in bindings:
            tool = b["tool"]
            if tool not in matched or tool not in tool_schemas:
                continue
            schema = tool_schemas[tool]
            required = [pp["name"] for pp in schema.get("params", [])
                        if not pp.get("optional")]
            have = set(b["params"].keys())
            if required:
                covs.append(len([x for x in required if x in have])
                            / len(required))
            ref_args = next((a.get("arguments", {}) for a in ref_actions
                             if _norm(a.get("name", "")) == tool), {})
            keys = set(b["params"].keys()) & set(ref_args.keys())
            if keys:
                accs.append(sum(1 for k in keys
                                if _norm(b["params"][k]) == _norm(ref_args[k]))
                            / len(keys))
        if covs:
            l3["required_param_coverage"] = round(sum(covs) / len(covs), 4)
        if accs:
            l3["arg_value_acc"] = round(sum(accs) / len(accs), 4)
    res["l3"] = l3

    # ---------------- L4 ----------------
    l4 = {"effect_consistency": None, "downgrades": [],
          "bindability": None, "unsafe_retry": [], "unsupported": []}
    if mod:
        node_by_tool = {b["tool"]: b for b in bindings if b["tool"]}
        pred_has_exec = any(b["tool"] for b in bindings)
        pred_op_set = set(pred_seq)
        if scorable:
            ok = 0
            for n, o in zip(ref_names, ref_plan):
                if o["semantic_skill"] is None:
                    continue
                mut = o["effect_class"] not in (
                    "read_only", "communication", "agent_internal")
                if n in matched:
                    ok += 1                 # identity-bound
                elif not mut and o["semantic_skill"] in pred_op_set:
                    ok += 1                 # read realized as its skill node
                else:
                    # effect downgrade: a MUTATING reference realized by a
                    # program with no mutating op at all
                    if mut and not pred_has_exec:
                        l4["downgrades"].append(n)
            l4["effect_consistency"] = round(ok / len(scorable), 4)
        if scorable:
            bindable = sum(1 for n in scorable
                           if n in matched
                           or get_tool_oracle(domain, n)["semantic_skill"]
                           in READ_ONLY_SKILLS)
            l4["bindability"] = round(bindable / len(scorable), 4)
        for n in mod.program.nodes:
            if n.retry and n.op == "EXEC_ACTION":
                act = _norm(n.params.get("action", ""))
                o = get_tool_oracle(domain, act)
                if o["effect_class"] in ("state_creation",
                                         "state_transition_irreversible"):
                    l4["unsafe_retry"].append(
                        {"node": n.id, "tool": act,
                         "reason": f"retry on {o['effect_class']} action"})
        if len([o for o in ref_plan
                if o["effect_class"] in ("state_creation",
                                          "state_transition_irreversible")]) > 1:
            l4["unsupported"].append(
                "multi-step transactional mutation ordering/rollback is "
                "not expressible in TaskIR v0.1 (recorded, not passed)")
    res["l4"] = l4

    res["executable_task_success"] = None    # requires trusted sandbox
    return res
