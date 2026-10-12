"""Phase 8B-1A Task 1: audit WHY the 28.3% of P1 oracle skeletons fail.

Per-task classification along the compilation chain:
  user_side_ref            - reference uses tools the agent cannot call
  unknown_skill            - tool not in the hand-verified oracle
  skeleton_validation      - validate_skeleton errors
  missing_required_params  - lowering reports unresolved args
  parser / validator       - lowered TaskIR fails official gates
  isa_expressiveness       - effect/annotation not representable
Reports the advisor's four success rates; no failure is dropped.
"""
import json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import SkeletonNode, PlanSkeleton
from src.compiler.skeleton_lowering import lower_skeleton
from src.compiler.plan_skeleton import validate_skeleton
from src.eval.tau3_skill_oracle import get_tool_oracle
from src.eval.tau3_tool_registry import build_registry, runtime_pool
from src.eval.taskir_semantic_eval_v2 import ref_skill_plan

DOMS = ["airline", "banking_knowledge", "retail", "telecom"]
SAMPLING_SEED = 20261013
import random

reg = build_registry()
pool = runtime_pool(reg)
schemas = {d: {t["name"]: t for t in ts} for d, ts in pool.items()}


def load_tasks():
    exclude = set()
    try:
        cs = json.load(open("results/phase7/online_cross_source.json"))
        for r in cs.get("details", []):
            exclude.add(r["case_id"])
    except FileNotFoundError:
        pass
    tasks = []
    for d in DOMS:
        data = json.load(open(
            f"third_party/tau3-bench/data/tau2/domains/{d}/tasks.json",
            encoding="utf-8"))
        usable = []
        for t in data:
            acts = (t.get("evaluation_criteria") or {}).get("actions") or []
            if not acts:
                continue
            cid = f"{d}-{t['id']}"
            if cid in exclude:
                continue
            usable.append({"case_id": cid, "domain": d,
                           "ref_actions": [
                               {"name": a["name"],
                                "arguments": a.get("arguments") or {}}
                               for a in acts if a.get("name")]})
        rng = random.Random(SAMPLING_SEED)
        rng.shuffle(usable)
        tasks.extend(usable[:30])
    return tasks


def classify(t):
    d = t["domain"]
    skills, np_ = ref_skill_plan(d, t["ref_actions"])
    oracles = [get_tool_oracle(d, p["tool"]) for p in np_["plan"]]
    if any(o["semantic_skill"] is None for o in oracles):
        # user-side / unknown tool in reference
        names = [p["tool"] for p, o in zip(np_["plan"], oracles)
                 if o["semantic_skill"] is None]
        return ("user_side_ref", names)
    sk = PlanSkeleton(task_id=t["case_id"])
    for i, p in enumerate(np_["plan"]):
        o = get_tool_oracle(d, p["tool"])
        sk.nodes.append(SkeletonNode(
            id=f"p{i}", capability_id=p["tool"],
            skill=o["semantic_skill"] or "EXEC_ACTION",
            depends_on=[f"p{i-1}"] if i else [],
            arguments=dict(p["args"])))
    errs = validate_skeleton(sk, d, schemas[d])
    if errs:
        return ("skeleton_validation", errs[:2])
    low = lower_skeleton(sk, d, schemas[d])
    if low.get("unresolved"):
        return ("missing_required_params",
                [u["missing_param"] for u in low["unresolved"]][:4])
    if low.get("errors"):
        e0 = str(low["errors"][0])
        if e0.startswith("parser"):
            return ("taskir_parser", e0[:80])
        if e0.startswith("validator"):
            return ("taskir_validator", e0[:120])
        return ("isa_expressiveness", e0[:80])
    if low.get("taskir") is None:
        return ("unknown", [])
    return ("ok", None)


def main():
    tasks = load_tasks()
    results = []
    for t in tasks:
        cat, detail = classify(t)
        results.append({"case_id": t["case_id"], "domain": t["domain"],
                        "category": cat, "detail": detail})
    counts = Counter(r["category"] for r in results)
    n = len(results)
    ok = counts["ok"]

    def rate(sub):
        m = [r for r in results if sub(r)]
        return (round(100 * sum(1 for r in m if r["category"] == "ok")
                      / len(m), 1) if m else None, len(m))

    no_user, n_no_user = rate(lambda r: r["category"] != "user_side_ref")
    # within-expressiveness: no user-side, no unknown skill
    express, n_express = rate(lambda r: r["category"] not in
                              ("user_side_ref", "unknown_skill"))
    # complete params: no missing_required_params category (and ok-adjacent)
    complete, n_complete = rate(
        lambda r: r["category"] not in ("missing_required_params",
                                        "user_side_ref"))

    out = {
        "n": n,
        "overall_success_pct": round(100 * ok / n, 1),
        "failure_categories": dict(counts),
        "success_within_expressiveness_pct": express,
        "n_within_expressiveness": n_express,
        "success_complete_params_pct": complete,
        "n_complete_params": n_complete,
        "success_no_user_side_pct": no_user,
        "n_no_user_side": n_no_user,
        "per_task": results,
    }
    os.makedirs("results/phase8b", exist_ok=True)
    with open("results/phase8b/p1_failure_audit.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"n={n} overall ok={100*ok/n:.1f}%")
    print("categories:", dict(counts))
    print(f"within-expressiveness: {express}% (n={n_express})")
    print(f"complete-params: {complete}% (n={n_complete})")
    print(f"no-user-side: {no_user}% (n={n_no_user})")
    # representative failure details
    for cat in counts:
        if cat == "ok":
            continue
        ex = next(r for r in results if r["category"] == cat)
        print(f"  e.g. {cat}: {ex['case_id']} {str(ex['detail'])[:90]}")


if __name__ == "__main__":
    main()
