"""Phase 8A final scoring: rescore the full run's per-sample texts with the
NORMALIZED evaluator (the run started before the trace-normalization fix)
and recompute condition summaries + module attribution from the rescored
rows. Raw run outputs are not modified.
"""
import json, os, sys
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.eval.taskir_semantic_eval import evaluate_sample
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
            refs[f"{d}-{t['id']}"] = (d, [
                {"name": a["name"], "arguments": a.get("arguments") or {}}
                for a in acts if a.get("name")])

reg = build_registry()
pool = runtime_pool(reg)
schemas = {d: {t["name"]: t for t in ts} for d, ts in pool.items()}

rows = []
for line in open("results/phase8a/semantic_details.jsonl", encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    cid = r["case_id"]
    if cid not in refs:
        continue
    d, rr = refs[cid]
    res = evaluate_sample(d, r.get("text", ""), rr, schemas[d])
    res["condition"] = r["condition"]
    res["case_id"] = cid
    res["domain"] = d
    rows.append(res)
print(f"rescored {len(rows)} records")

CONDS = ["A_G0", "A_G1", "A_G2", "B_G0", "C_G0", "D_refplan"]


def agg(rs):
    n = max(1, len(rs))
    def m(a, b):
        vals = [x[a][b] for x in rs
                if a in x and x[a].get(b) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {"n": len(rs),
            "valid%": round(100 * sum(x["l0"]["valid"] for x in rs) / n, 1),
            "parse%": round(100 * sum(x["l0"]["parse_ok"] for x in rs) / n, 1),
            "L1_f1": m("l1", "f1"),
            "L2_exact_f1": m("l2", "exact_f1"),
            "L2_seq%": round(100 * sum(x["l2"]["exact_sequence_match"]
                                       for x in rs) / n, 1),
            "L3_ordRec": m("l3", "ordered_recall"),
            "L3_argAcc": m("l3", "arg_value_acc"),
            "L4_effect": m("l4", "effect_consistency"),
            "L4_bind": m("l4", "bindability"),
            "steps_pred/ref": (m("l3", "pred_steps"),
                               m("l3", "ref_steps")),
            "steps_ref_raw": m("l3", None) if False else round(
                sum(x.get("ref_steps_raw", 0) for x in rs) / n, 2)}


by_cond = defaultdict(list)
for r in rows:
    by_cond[r["condition"]].append(r)
summary = {c: agg(by_cond[c]) for c in CONDS if by_cond[c]}

# by-domain breakdown for A_G0 / C_G0
bydom = {}
for cond in ("A_G0", "A_G1", "A_G2", "C_G0", "D_refplan"):
    bydom[cond] = {}
    for d in DOMS:
        rs = [r for r in by_cond[cond] if r["domain"] == d]
        if rs:
            bydom[cond][d] = agg(rs)

# attribution (rescored)
by = {}
for r in rows:
    by[(r["condition"], r["case_id"])] = r


def sem_ok(r):
    return bool(r and r["l0"]["valid"] and r["l2"]["exact_f1"] >= 0.99)


attr = Counter()
task_flags = defaultdict(list)
cids = sorted({r["case_id"] for r in rows})
for cid in cids:
    a = by.get(("A_G0", cid))
    b = by.get(("B_G0", cid))
    c = by.get(("C_G0", cid))
    dd = by.get(("D_refplan", cid))
    if a is None:
        continue
    if a["l0"]["valid"] and not sem_ok(a):
        task_flags[cid].append("validator_only_error")
        attr["validator_only_error"] += 1
    if not sem_ok(a) and sem_ok(b):
        task_flags[cid].append("canonicalizer_error")
        attr["canonicalizer_error"] += 1
    elif not sem_ok(b) and sem_ok(c):
        task_flags[cid].append("resolver_error")
        attr["resolver_error"] += 1
    elif not sem_ok(c):
        if c["l2"]["exact_f1"] < 0.99:
            task_flags[cid].append("composer_planning_error")
            attr["composer_planning_error"] += 1
        if c["l1"]["f1"] < 0.99 and c["l2"]["exact_f1"] >= 0.99:
            task_flags[cid].append("tool_binding_error")
            attr["tool_binding_error"] += 1
    if c is not None and sem_ok(c) and c["l1"]["f1"] < 0.99:
        task_flags[cid].append("tool_binding_error")
        attr["tool_binding_error"] += 1
    if dd is not None and not sem_ok(dd):
        task_flags[cid].append("binding_upper_bound_gap")
        attr["binding_upper_bound_gap"] += 1

n_multi = sum(1 for v in task_flags.values() if len(v) > 1)

# G1/G2 semantic transition vs G0 (Task 5)
trans = {}
for arm in ("A_G1", "A_G2"):
    sem_up = sem_down = 0
    for cid in cids:
        ra = by.get((arm, cid))
        r0 = by.get(("A_G0", cid))
        if not (ra and r0):
            continue
        if sem_ok(ra) and not sem_ok(r0):
            sem_up += 1
        elif sem_ok(r0) and not sem_ok(ra):
            sem_down += 1
    trans[arm] = {"semantic_fixed": sem_up, "semantic_broken": sem_down}

out = {"summary": summary, "by_domain": bydom,
       "attribution": {"counts": dict(attr),
                       "n_tasks": len(cids),
                       "n_flagged": len(task_flags),
                       "n_multi_cause": n_multi},
       "g_semantic_transitions": trans}
with open("results/phase8a/semantic_layers.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
with open("results/phase8a/error_attribution.json", "w") as f:
    json.dump({"counts": dict(attr), "n_tasks": len(cids),
               "n_flagged": len(task_flags), "n_multi_cause": n_multi,
               "task_flags": dict(task_flags)}, f, indent=2,
              ensure_ascii=False)
with open("results/phase8a/semantic_details_rescored.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

print(f"\n{'cond':<10}{'valid%':>8}{'L1f1':>7}{'L2ex':>7}{'seq%':>6}"
      f"{'ordR':>7}{'argA':>7}{'eff':>7}{'bind':>7}{'steps':>10}")
for c in CONDS:
    s = summary.get(c)
    if not s:
        continue
    print(f"{c:<10}{s['valid%']:>7.1f}%{s['L1_f1']:>7}{s['L2_exact_f1']:>7}"
          f"{s['L2_seq%']:>6.1f}{s['L3_ordRec']:>7}{str(s['L3_argAcc']):>7}"
          f"{s['L4_effect']:>7}{s['L4_bind']:>7}"
          f"{str(s['steps_pred/ref']):>10}")
print("\nattribution:", dict(attr), "multi:", n_multi)
print("G transitions:", trans)
print("RESCORE-DONE")
