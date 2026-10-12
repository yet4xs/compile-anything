"""Phase 8A Task 5 (exploratory): rescore the stored 7C-2.1 tau3 outputs
(G0/G1/G2, 100 tasks) with the NEW five-layer semantic evaluator.

EXPLORATORY ONLY: these tasks fed 7C-2.1 evaluation (not this scorer's
development, but they are not a fresh confirmation set). The fresh set is
the 120-task run in eval_phase8a_semantics.py.
"""
import json, os, sys
from collections import defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.eval.taskir_semantic_eval import evaluate_sample
from src.eval.tau3_tool_registry import build_registry, runtime_pool

DOMS = ["airline", "banking_knowledge", "retail", "telecom"]

# tasks.json reference actions by case_id
refs = {}
for d in DOMS:
    data = json.load(open(
        f"third_party/tau3-bench/data/tau2/domains/{d}/tasks.json",
        encoding="utf-8"))
    for t in data:
        acts = (t.get("evaluation_criteria") or {}).get("actions") or []
        if acts:
            refs[f"{d}-{t['id']}"] = {
                "domain": d,
                "ref_actions": [{"name": a["name"],
                                 "arguments": a.get("arguments") or {}}
                                for a in acts if a.get("name")]}

reg = build_registry()
pool = runtime_pool(reg)
schemas = {d: {t["name"]: t for t in ts} for d, ts in pool.items()}

cs = json.load(open("results/phase7/online_cross_source.json"))
rows = defaultdict(list)
matched_caseids = 0
for r in cs.get("details", []):
    cid = r["case_id"]
    if cid not in refs:
        continue
    matched_caseids += 1
    info = refs[cid]
    res = evaluate_sample(info["domain"], r["text"],
                          info["ref_actions"], schemas[info["domain"]])
    res["arm"] = r["arm"]
    res["case_id"] = cid
    rows[r["arm"]].append(res)
print(f"rescored {matched_caseids}/300 records "
      f"({len(rows.get('G0', []))} tasks x arms)")

def agg(rs):
    n = max(1, len(rs))
    def m(path):
        a, b = path
        vals = [x[a][b] for x in rs
                if a in x and x[a].get(b) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {"n": len(rs),
            "valid%": round(100 * sum(x["l0"]["valid"] for x in rs) / n, 1),
            "L1_f1": m(("l1", "f1")),
            "L2_exact_f1": m(("l2", "exact_f1")),
            "L2_seq%": round(100 * sum(x["l2"]["exact_sequence_match"]
                                       for x in rs) / n, 1),
            "L3_ordRec": m(("l3", "ordered_recall")),
            "L3_argAcc": m(("l3", "arg_value_acc")),
            "L4_effect": m(("l4", "effect_consistency")),
            "L4_bind": m(("l4", "bindability")),
            "pred/ref steps": (m(("l3", "pred_steps")),
                               m(("l3", "ref_steps")))}

out = {"exploratory": True,
       "note": "7C-2.1 stored outputs rescored with the 8A evaluator; "
               "not a fresh confirmation set",
       "arms": {}}
for arm in ("G0", "G1", "G2"):
    if rows.get(arm):
        a = agg(rows[arm])
        out["arms"][arm] = a
        print(f"{arm}: {a}")

os.makedirs("results/phase8a", exist_ok=True)
with open("results/phase8a/exploratory_rescore.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("saved results/phase8a/exploratory_rescore.json")
