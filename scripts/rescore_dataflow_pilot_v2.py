"""Phase 8D-1B.1 Task 5: rescore pilot with canonical matcher.

Reads original predictions from m0_m1_dataflow_pilot.json,
applies canonical graph matching (node-reorder invariant),
reports four metrics split by topology.
"""
import json, os, sys, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir("/ccfa2026/compile-anything"):
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
else:
    sys.path.insert(0, "/ccfa2026/compile-anything")
    os.chdir("/ccfa2026/compile-anything")

from src.eval.canonical_dataflow_matcher import (
    canonical_from_skeleton, canonical_from_taskir,
    canonical_call_binding_match, canonical_dag_match)

PILOT = "results/phase8d/m0_m1_dataflow_pilot.json"
TEST = "data/runtime_dataflow_v2/test.jsonl"
OUT = "results/phase8d/m0_m1_dataflow_rescored.json"


def main():
    pilot = json.load(open(PILOT))
    details = pilot["details"]
    test_recs = {json.loads(l)["task_id"]: json.loads(l)
                 for l in open(TEST) if l.strip()}

    # Rescore each detail
    rescored = []
    old_new_flips = {"old_fail_new_pass": 0, "old_pass_new_fail": 0}

    for d in details:
        rec = test_recs.get(d["task_id"])
        if not rec:
            continue
        r = dict(d)  # keep original fields

        # Parse the prediction's TaskIR (from the raw output)
        taskir = d.get("raw", "")
        if d["arm"] == "M1":
            # For M1, the raw is the skeleton JSON; we need the TaskIR
            # that was produced (stored in evaluate_one's taskir param,
            # but we stored 'raw' not 'taskir'). We need to re-derive.
            # The raw is the skeleton output text.
            # Reconstruct: parse skeleton → lower → TaskIR
            import re as _re
            from src.compiler.plan_skeleton import skeleton_from_dict, \
                validate_skeleton, hard_errors
            from src.compiler.skeleton_lowering import lower_skeleton
            try:
                m = _re.search(r"\{.*\}", d["raw"], _re.S)
                if m:
                    sk = skeleton_from_dict(json.loads(m.group(0)))
                    schemas = {}
                    for tid in rec["capability_registry"]:
                        meta = rec["capability_registry"][tid]
                        schemas[tid] = {
                            "params": [{"name": k, "optional": False,
                                         "type": v}
                                        for k, v in meta.get("input_schema", {}).items()],
                            "output_fields": meta.get("output_schema", {}),
                            "effect_class": meta.get("effect_class",
                                                      "read_only")}
                    errs = hard_errors(validate_skeleton(sk, None, schemas))
                    if not errs:
                        low = lower_skeleton(sk, None, schemas, strict=True)
                        taskir = low.get("taskir") or ""
                    else:
                        taskir = ""
                else:
                    taskir = ""
            except Exception:
                taskir = ""
        elif d["arm"] == "M0":
            taskir = d["raw"]  # M0's raw IS the TaskIR

        # Canonical evaluation
        ref = canonical_from_skeleton(rec["target_skeleton"]["nodes"])
        pred = canonical_from_taskir(taskir)

        if pred is None:
            r["A_call_binding_exact"] = False
            r["A_bindings_f1"] = 0.0
            r["B_data_exact"] = False
            r["B_extra_data"] = 0
            r["B_missing_data"] = 0
            r["B_data_f1"] = 0.0
            r["canonical_parse"] = False
        else:
            a = canonical_call_binding_match(ref, pred)
            b = canonical_dag_match(ref, pred)
            r["A_call_binding_exact"] = a["bindings_match"] and a["calls_match"]
            r["A_bindings_f1"] = round(
                2 * a["n_correct_bindings"] /
                max(1, a["n_ref_bindings"] + a["n_pred_bindings"]), 4) \
                if (a["n_ref_bindings"] + a["n_pred_bindings"]) else 1.0
            r["B_data_exact"] = b["data_exact"]
            r["B_extra_data"] = b["extra_data"]
            r["B_missing_data"] = b["missing_data"]
            r["B_data_f1"] = b["data_f1"]
            r["canonical_parse"] = True

        # Track flips
        old_pass = d.get("binding_exact", False)
        new_pass = r["A_call_binding_exact"]
        if not old_pass and new_pass:
            old_new_flips["old_fail_new_pass"] += 1
        elif old_pass and not new_pass:
            old_new_flips["old_pass_new_fail"] += 1

        rescored.append(r)

    # Aggregate by arm/protocol/topology
    by = defaultdict(list)
    for r in rescored:
        topo = r.get("topology", "?")
        by[(r["arm"], r["protocol"], topo)].append(r)
        by[(r["arm"], r["protocol"], "ALL")].append(r)

    summary = {}
    for (arm, proto, topo), rows in sorted(by.items()):
        n = len(rows)
        def pct(k):
            return round(100 * sum(1 for x in rows if x.get(k)) / n, 1)
        def avg(k):
            v = [x.get(k, 0) for x in rows]
            return round(sum(v) / len(v), 3) if v else 0
        summary[f"{proto}|{topo}|{arm}"] = {
            "n": n,
            "A_binding_exact%": pct("A_call_binding_exact"),
            "A_bindings_f1": avg("A_bindings_f1"),
            "B_data_exact%": pct("B_data_exact"),
            "B_extra_data_mean": avg("B_extra_data"),
            "B_missing_data_mean": avg("B_missing_data"),
            "B_data_f1": avg("B_data_f1"),
            # original metrics for comparison
            "old_binding_exact%": pct("binding_exact"),
            "old_strict%": pct("strict_semantic"),
        }

    # Paired stats per topology
    for proto in ("oracle", "distractor"):
        for topo in ("ALL", "fan_out_in", "repeated_tool"):
            m0 = {x["task_id"]: x for x in rescored
                   if x["arm"] == "M0" and x["protocol"] == proto
                   and (topo == "ALL" or x.get("topology") == topo)}
            m1 = {x["task_id"]: x for x in rescored
                   if x["arm"] == "M1" and x["protocol"] == proto
                   and (topo == "ALL" or x.get("topology") == topo)}
            for key in ("A_call_binding_exact", "B_data_exact"):
                deltas = []
                fx = bk = 0
                for tid in m0:
                    if tid not in m1:
                        continue
                    a = 1 if m1[tid].get(key) else 0
                    b = 1 if m0[tid].get(key) else 0
                    deltas.append(a - b)
                    if a and not b:
                        fx += 1
                    elif b and not a:
                        bk += 1
                if deltas:
                    rng = random.Random(13)
                    boots = sorted(
                        sum(deltas[rng.randrange(len(deltas))]
                            for _ in deltas) / len(deltas)
                        for _ in range(5000))
                    summary[f"{proto}|{topo}|paired|{key}"] = {
                        "n": len(deltas),
                        "diff_pp": round(100 * sum(deltas) / len(deltas), 1),
                        "ci95": [round(100 * boots[125], 1),
                                  round(100 * boots[4875], 1)],
                        "fixed": fx, "broken": bk}

    result = {"summary": summary, "details_count": len(rescored),
              "flips": old_new_flips, "details": rescored}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Rescored {len(rescored)} records")
    print(f"Score flips: {old_new_flips}")
    for k in sorted(summary):
        v = summary[k]
        if "paired" in k:
            print(f"  {k}: diff={v['diff_pp']}pp CI={v['ci95']} "
                  f"f/b={v['fixed']}/{v['broken']}")
        elif "ALL" in k:
            print(f"  {k}: A_binding={v['A_binding_exact%']}% "
                  f"A_f1={v['A_bindings_f1']} "
                  f"B_data_exact={v['B_data_exact%']}% "
                  f"extra={v['B_extra_data_mean']} "
                  f"(old_binding={v['old_binding_exact%']}%)")
    print("RESCORE-V2-DONE")


if __name__ == "__main__":
    main()
