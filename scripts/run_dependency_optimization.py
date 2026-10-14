"""Phase 8D-2A Tasks 3-5: dependency optimization + HEFT scheduling experiments.

Runs on the saved Phase 8D-1B predictions: applies dependency refinement,
computes critical path and HEFT makespan before/after optimization.
"""
import json, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir("/ccfa2026/compile-anything"):
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
else:
    sys.path.insert(0, "/ccfa2026/compile-anything")
    os.chdir("/ccfa2026/compile-anything")

from src.optimizer.dependency_refinement import refine_taskir_text
from src.scheduler.heft import dag_from_taskir, heft_schedule, DataflowDAG

PILOT = "results/phase8d/m0_m1_dataflow_pilot.json"
TEST = "data/runtime_dataflow_v2/test.jsonl"
OUT = "results/phase8d/dependency_optimization.json"


def main():
    pilot = json.load(open(PILOT))
    test_recs = {json.loads(l)["task_id"]: json.loads(l)
                 for l in open(TEST) if l.strip()}

    results = []
    stats = defaultdict(lambda: defaultdict(float))

    for d in pilot["details"]:
        if d["arm"] != "M1" or d["protocol"] != "oracle":
            continue  # focus on M1 oracle (the best-performing arm)
        rec = test_recs.get(d["task_id"])
        if not rec:
            continue

        # Get the M1 TaskIR (from the skeleton, lowered)
        import re
        from src.compiler.plan_skeleton import skeleton_from_dict, \
            validate_skeleton, hard_errors
        from src.compiler.skeleton_lowering import lower_skeleton
        try:
            m = re.search(r"\{.*\}", d.get("raw", ""), re.S)
            if not m:
                continue
            sk = skeleton_from_dict(json.loads(m.group(0)))
            schemas = {}
            for tid in rec["capability_registry"]:
                meta = rec["capability_registry"][tid]
                schemas[tid] = {
                    "params": [{"name": k, "optional": False, "type": v}
                                for k, v in meta.get("input_schema", {}).items()],
                    "output_fields": meta.get("output_schema", {}),
                    "effect_class": meta.get("effect_class", "read_only")}
            errs = hard_errors(validate_skeleton(sk, None, schemas))
            if errs:
                continue
            low = lower_skeleton(sk, None, schemas, strict=True)
            original_taskir = low.get("taskir")
            if not original_taskir:
                continue
        except Exception:
            continue

        # Build cap registry for refinement
        cap_reg = {}
        for tid in rec["capability_registry"]:
            meta = rec["capability_registry"][tid]
            cap_reg[tid] = {
                "input_schema": meta.get("input_schema", {}),
                "output_schema": meta.get("output_schema", {}),
                "effect_class": meta.get("effect_class", "read_only")}

        # 1. Original DAG analysis
        orig_dag = dag_from_taskir(original_taskir)
        if orig_dag is None:
            continue
        orig_cp = orig_dag.critical_path_length()
        orig_par = orig_dag.max_parallelism()
        orig_sched = heft_schedule(orig_dag, num_processors=2)
        orig_makespan = orig_sched.makespan

        # 2. Refine dependencies
        refined = refine_taskir_text(original_taskir, cap_registry=cap_reg)

        # 3. Refined DAG analysis
        if refined.get("refined"):
            ref_dag = dag_from_taskir(refined["refined"])
            if ref_dag:
                ref_cp = ref_dag.critical_path_length()
                ref_par = ref_dag.max_parallelism()
                ref_sched = heft_schedule(ref_dag, num_processors=2)
                ref_makespan = ref_sched.makespan
            else:
                ref_cp, ref_par, ref_makespan = orig_cp, orig_par, orig_makespan
        else:
            ref_cp, ref_par, ref_makespan = orig_cp, orig_par, orig_makespan

        # 4. Oracle DAG (from reference skeleton)
        oracle_schemas = dict(schemas)
        oracle_sk = skeleton_from_dict(rec["target_skeleton"])
        oracle_low = lower_skeleton(oracle_sk, None, oracle_schemas, strict=True)
        oracle_taskir = oracle_low.get("taskir", "")
        oracle_dag = dag_from_taskir(oracle_taskir) if oracle_taskir else None
        if oracle_dag:
            oracle_cp = oracle_dag.critical_path_length()
            oracle_par = oracle_dag.max_parallelism()
            oracle_sched = heft_schedule(oracle_dag, num_processors=2)
            oracle_makespan = oracle_sched.makespan
        else:
            oracle_cp = oracle_par = oracle_makespan = 0

        entry = {
            "task_id": d["task_id"],
            "topology": d.get("topology", "?"),
            "edges_removed": refined.get("removed", 0),
            "certificates": refined.get("certificates", []),
            "validation_pass": refined.get("validation_pass", False),
            "original": {"cp": round(orig_cp, 3), "parallelism": orig_par,
                          "makespan_2p": round(orig_makespan, 3),
                          "n_nodes": len(orig_dag.nodes),
                          "n_edges": len(orig_dag.edges)},
            "refined": {"cp": round(ref_cp, 3), "parallelism": ref_par,
                         "makespan_2p": round(ref_makespan, 3)},
            "oracle": {"cp": round(oracle_cp, 3), "parallelism": oracle_par,
                        "makespan_2p": round(oracle_makespan, 3)},
        }
        results.append(entry)

        # aggregate
        topo = entry["topology"]
        stats[topo]["total"] += 1
        stats[topo]["edges_removed"] += entry["edges_removed"]
        stats[topo]["orig_cp"] += entry["original"]["cp"]
        stats[topo]["ref_cp"] += entry["refined"]["cp"]
        stats[topo]["orig_mk"] += entry["original"]["makespan_2p"]
        stats[topo]["ref_mk"] += entry["refined"]["makespan_2p"]
        stats[topo]["oracle_mk"] += entry["oracle"]["makespan_2p"]
        stats[topo]["orig_par"] += entry["original"]["parallelism"]
        stats[topo]["ref_par"] += entry["refined"]["parallelism"]

    # Summary
    summary = {}
    for topo, s in sorted(stats.items()):
        n = max(1, int(s["total"]))
        summary[topo] = {
            "n": int(s["total"]),
            "avg_edges_removed": round(s["edges_removed"] / n, 2),
            "avg_cp_orig": round(s["orig_cp"] / n, 3),
            "avg_cp_refined": round(s["ref_cp"] / n, 3),
            "cp_improvement%": round(
                100 * (s["orig_cp"] - s["ref_cp"]) / max(0.001, s["orig_cp"]) / n, 1),
            "avg_makespan_orig": round(s["orig_mk"] / n, 3),
            "avg_makespan_refined": round(s["ref_mk"] / n, 3),
            "avg_makespan_oracle": round(s["oracle_mk"] / n, 3),
            "makespan_improvement%": round(
                100 * (s["orig_mk"] - s["ref_mk"]) / max(0.001, s["orig_mk"]) / n, 1),
            "avg_parallelism_orig": round(s["orig_par"] / n, 1),
            "avg_parallelism_refined": round(s["ref_par"] / n, 1),
        }

    # Also aggregate overall
    all_n = len(results)
    if all_n:
        total_removed = sum(r["edges_removed"] for r in results)
        summary["OVERALL"] = {
            "n": all_n,
            "total_edges_removed": total_removed,
            "avg_edges_removed": round(total_removed / all_n, 2),
            "tasks_with_removals": sum(1 for r in results if r["edges_removed"] > 0),
            "validation_all_pass": all(r["validation_pass"] for r in results),
        }

    output = {"summary": summary, "details": results}
    os.makedirs("results/phase8d", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Processed {all_n} M1 oracle predictions")
    for topo, s in sorted(summary.items()):
        print(f"  {topo}: {s}")
    print("OPT-DONE")


if __name__ == "__main__":
    main()
