"""Phase 8A.1 + 8B-0 offline scoring with the corrected v2 evaluator.

Scores (no overwriting of raw predictions):
  - the frozen 8A 720 records (semantic_details_rescored.jsonl texts)
    under conditions A_G0/A_G1/A_G2/B_G0/C_G0(old=oracle labels)/D_refplan
    -> results/phase8a/semantic_layers_corrected.json
  - the new grid (A2, C_learned_labels, P3, P4) + P1 oracle lowering
    and P0 (=A_G0) -> results/phase8b/hierarchical_planning_feasibility.json
Grouping: depth buckets (1 / 2-3 / 4-6 / 7+ normalized ref steps) and
read-only vs side-effect tasks.
"""
import json, os, sys
from collections import defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.eval.taskir_semantic_eval_v2 import (
    evaluate_sample_v2, normalize_ref_plan, ref_skill_plan)
from src.compiler.plan_skeleton import SkeletonNode, PlanSkeleton
from src.compiler.skeleton_lowering import lower_skeleton
from src.eval.tau3_skill_oracle import get_tool_oracle
from src.eval.tau3_tool_registry import build_registry, runtime_pool

DOMS = ["airline", "banking_knowledge", "retail", "telecom"]

refs = {}
for d in DOMS:
    data = json.load(open(
        f"third_party/tau3-bench/data/tau2/domains/{d}/tasks.json",
        encoding="utf-8"))
    for t in data:
        acts = (t.get("evaluation_criteria") or {}).get("actions") or []
        if acts:
            refs[f"{d}-{t['id']}"] = (d, t["user_scenario"]["instructions"],
                                      [{"name": a["name"],
                                        "arguments": a.get("arguments") or {}}
                                       for a in acts if a.get("name")])

reg = build_registry()
pool = runtime_pool(reg)
schemas = {d: {t["name"]: t for t in ts} for d, ts in pool.items()}


def oracle_skeleton_and_lower(d, ref_actions):
    skills, np_ = ref_skill_plan(d, ref_actions)
    sk = PlanSkeleton(task_id="oracle")
    for i, p in enumerate(np_["plan"]):
        o = get_tool_oracle(d, p["tool"])
        sk.nodes.append(SkeletonNode(
            id=f"p{i}", capability_id=p["tool"],
            skill=o["semantic_skill"] or "EXEC_ACTION",
            depends_on=[f"p{i-1}"] if i else [],
            arguments=dict(p["args"])))
    return lower_skeleton(sk, d, schemas[d])


def group_of(r):
    ref_n = r["ref_normalized_n"]
    if ref_n <= 1:
        depth = "1"
    elif ref_n <= 3:
        depth = "2-3"
    elif ref_n <= 6:
        depth = "4-6"
    else:
        depth = "7+"
    mut = any(get_tool_oracle(r["domain"], t)["effect_class"] not in
              ("read_only", "communication", "agent_internal")
              for t in r["user_side_refs"] + []) or r.get("has_mutation", False)
    return depth


def agg(rows):
    n = max(1, len(rows))
    def m(path):
        a, b = path
        vals = [x[a].get(b) for x in rows
                if isinstance(x.get(a), dict)
                and x[a].get(b) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {"n": len(rows),
            "valid%": round(100 * sum(x["l0"]["valid"] for x in rows) / n, 1),
            "e2e_skill_f1": m(("l2", "e2e_skill_f1")),
            "cond_skill_f1": m(("l2", "conditional_skill_f1")),
            "plan_f1": m(("l2", "plan_f1")),
            "plan_p": m(("l2", "plan_precision")),
            "plan_r": m(("l2", "plan_recall")),
            "exact_ordered%": round(
                100 * sum(x["l2"]["exact_ordered_match"] for x in rows) / n, 1),
            "ir_binding_f1": m(("l1_ir_binding", "ir_binding_f1")),
            "tool_sel_f1": m(("l1_tool_selection", "f1")),
            "arg_acc": m(("l3", "arg_value_acc_all_ref_params")),
            "effect": m(("l4", "effect_consistency")),
            "step_ratio": m(("l3", "step_ratio")),
            "pred_steps": m(("l3", "pred_steps")),
            "ref_steps": m(("l3", "ref_steps"))}


def main():
    # ---------- Part 1: corrected 8A conditions ----------
    rows8a = [json.loads(l) for l in open(
        "results/phase8a/semantic_details.jsonl",
        encoding="utf-8") if l.strip()]
    out8a = {}
    by_cond = defaultdict(list)
    for r in rows8a:
        cid = r["case_id"]
        if cid not in refs:
            continue
        d, _instr, rr = refs[cid]
        sel = None                     # 8A run didn't store selections
        v2 = evaluate_sample_v2(d, r.get("text", ""), rr, schemas[d], sel)
        v2["condition"] = r["condition"]
        v2["case_id"] = cid
        v2["domain"] = d
        # D rebuilt from the SAME conservative normalized plan
        if r["condition"] == "D_refplan":
            low = oracle_skeleton_and_lower(d, rr)
            v2 = evaluate_sample_v2(d, low.get("taskir", ""), rr,
                                    schemas[d], sel)
            v2["condition"] = "D_refplan_v2"
            v2["case_id"] = cid
            v2["domain"] = d
            v2["d_lower_errors"] = low["errors"][:2]
        by_cond[v2["condition"]].append(v2)
    for c, rs in sorted(by_cond.items()):
        out8a[c] = agg(rs)
        # depth grouping
        out8a[c + "__by_depth"] = {
            g: agg([r for r in rs if group_of(r) == g])
            for g in ("1", "2-3", "4-6", "7+")}

    with open("results/phase8a/semantic_layers_corrected.json", "w") as f:
        json.dump(out8a, f, indent=2, ensure_ascii=False)

    # ---------- Part 2: 8B-0 arms ----------
    import os as _os
    if not _os.path.exists("results/phase8a/grid_texts.jsonl"):
        print("grid not ready - 8A.1 corrected tables saved; rerun for 8B")
        return
    grid = [json.loads(l) for l in open(
        "results/phase8a/grid_texts.jsonl", encoding="utf-8")
            if l.strip()]
    by_arm = defaultdict(list)
    for r in grid:
        cid = r["case_id"]
        if cid not in refs:
            continue
        d, _instr, rr = refs[cid]
        sel = r.get("selected")
        v2 = evaluate_sample_v2(d, r.get("text", ""), rr, schemas[d], sel)
        v2["arm"] = r["arm"]
        v2["case_id"] = cid
        v2["domain"] = d
        v2["latency_s"] = r.get("latency_s")
        if r["arm"] == "P2_skeleton":
            v2["skeleton_parse_ok"] = r.get("skeleton_parse_ok")
        by_arm[r["arm"]].append(v2)
    # P1 oracle skeleton -> lowering (deterministic, CPU)
    p1_rows = []
    for cid, (d, _i, rr) in refs.items():
        if cid not in {r["case_id"] for r in grid}:
            continue
        low = oracle_skeleton_and_lower(d, rr)
        v2 = evaluate_sample_v2(d, low.get("taskir", ""), rr, schemas[d],
                                None)
        v2["arm"] = "P1_oracle_skeleton"
        v2["case_id"] = cid
        v2["domain"] = d
        v2["lower_errors"] = low["errors"][:2]
        p1_rows.append(v2)
    by_arm["P1_oracle_skeleton"] = p1_rows
    # P0 = A_G0 texts from 8A
    p0 = [r for r in rows8a if r["condition"] == "A_G0"
          and r["case_id"] in {x["case_id"] for x in grid}]
    p0_rows = []
    for r in p0:
        d, _i, rr = refs[r["case_id"]]
        v2 = evaluate_sample_v2(d, r.get("text", ""), rr, schemas[d], None)
        v2["arm"] = "P0_composer_G0"
        v2["case_id"] = r["case_id"]
        v2["domain"] = d
        p0_rows.append(v2)
    by_arm["P0_composer_G0"] = p0_rows

    out8b = {}
    for arm in ("P0_composer_G0", "P1_oracle_skeleton", "P2_skeleton",
                "P3_identity", "P4_length", "A2_G0",
                "C_learned_labels_G0"):
        rs = by_arm.get(arm)
        if not rs:
            continue
        out8b[arm] = agg(rs)
        out8b[arm + "__by_depth"] = {
            g: agg([r for r in rs if group_of(r) == g])
            for g in ("1", "2-3", "4-6", "7+")}
        if arm == "P2_skeleton":
            out8b[arm]["skeleton_json_ok%"] = round(
                100 * sum(1 for r in rs if r.get("skeleton_parse_ok"))
                / len(rs), 1)
        lats = [r["latency_s"] for r in rs if r.get("latency_s")]
        if lats:
            out8b[arm]["latency_p50"] = sorted(lats)[len(lats) // 2]

    with open("results/phase8b/hierarchical_planning_feasibility.json",
              "w") as f:
        json.dump(out8b, f, indent=2, ensure_ascii=False)

    print("=== 8A.1 corrected conditions ===")
    for c in ("A_G0", "A_G1", "A_G2", "B_G0", "C_G0", "D_refplan_v2"):
        if c in out8a:
            s = out8a[c]
            print(f"{c:<14} valid={s['valid%']:>5}% e2eF1={s['e2e_skill_f1']}"
                  f" planF1={s['plan_f1']} exact={s['exact_ordered%']}%"
                  f" bind={s['ir_binding_f1']} arg={s['arg_acc']}"
                  f" eff={s['effect']} ratio={s['step_ratio']}")
    print("\n=== 8B-0 arms ===")
    for a in ("P0_composer_G0", "P1_oracle_skeleton", "P2_skeleton",
              "P3_identity", "P4_length"):
        if a in out8b:
            s = out8b[a]
            print(f"{a:<20} valid={s['valid%']:>5}% e2eF1={s['e2e_skill_f1']}"
                  f" planF1={s['plan_f1']} exact={s['exact_ordered%']}%"
                  f" bind={s['ir_binding_f1']} ratio={s['step_ratio']}")
    print("\nSCORING-DONE")


if __name__ == "__main__":
    main()
