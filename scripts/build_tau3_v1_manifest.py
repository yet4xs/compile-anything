"""Phase 9B-2.3 Task 5: freeze the v1 binding-exact manifest.

Admission criteria (all must hold):
  1. no unobservable constants        -> strict_v1 (every param LITERAL or
                                        sourced runtime_ref[_multi])
  2. every param has a source         -> same
  3. real-output schema consistency   -> refs use REAL replay paths by
                                        construction (repair pipeline)
  4. reconstruction executes          -> strict_v1_exec_ok (live runtime-ref
                                        resolution on the official simulator,
                                        args + final DB state match)
  5. split hygiene                    -> tasks used for error analysis are
                                        permanently excluded from any frozen
                                        test split

Output: results/phase9/tau3_binding_exact_v1_manifest.json
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "results" / "phase9" / "tau3_gold_compiled_v1.json"
EXEC = REPO / "results" / "phase9" / "tau3_gold_execution_validation.json"
OUT = REPO / "results" / "phase9" / "tau3_binding_exact_v1_manifest.json"

# tasks inspected during Phase 9B-2/9B-2.2 error analysis (predictions
# read, failures categorized) — never eligible for the frozen test split
EXPLORED_TASK_IDS = [
    "tau3_airline_37", "tau3_airline_42", "tau3_airline_43",
    "tau3_retail_11", "tau3_retail_3", "tau3_retail_4", "tau3_retail_48",
    "tau3_retail_58", "tau3_retail_6", "tau3_retail_60", "tau3_retail_64",
    "tau3_retail_8", "tau3_retail_91", "tau3_retail_95",
]


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    ex = json.loads(EXEC.read_text(encoding="utf-8"))
    ex_by = {t["task_id"]: t for t in ex["tasks"]}

    tasks = []
    admitted = []
    for t in gold["tasks"]:
        e = ex_by[t["task_id"]]
        ok = (t["strict_v1"] and e["reconstructed_exec_ok"]
              and e["final_db_state_match"] and not e["used_fallback_literals"])
        rec = {
            "task_id": t["task_id"], "domain": t["domain"],
            "origin_split": t["split"],
            "source_task_id": t["source_task_id"],
            "admitted_v1": ok,
            "strict_v1": t["strict_v1"],
            "executed_with_live_refs": e["reconstructed_exec_ok"],
            "final_state_match": e["final_db_state_match"],
            "param_class_counts": t["param_class_counts"],
            "ir_integration": (
                "current_ir" if t["current_ir_representable"]
                else "requires_typed_access_and_list_refs"),
            "explored_in_analysis": t["task_id"] in EXPLORED_TASK_IDS,
            "eligible_for_future_test_split": (
                ok and t["task_id"] not in EXPLORED_TASK_IDS),
        }
        tasks.append(rec)
        if ok:
            admitted.append(t["task_id"])

    eligible_test = [r["task_id"] for r in tasks
                     if r["eligible_for_future_test_split"]]
    summary = {
        "n_tasks": len(tasks),
        "admitted_v1": admitted,
        "n_admitted": len(admitted),
        "eligible_for_future_test_split": eligible_test,
        "split_policy": (
            "All admitted v1 tasks stay in the TRAIN partition; a frozen "
            "test split is deferred until >=15 admitted tasks exist "
            "(current: %d). Tasks used in Phase 9B-2/9B-2.2 error "
            "analysis are permanently excluded from test." % len(admitted)),
        "leakage_checks": {
            "task_ids_unique": len({r["task_id"] for r in tasks}) == len(tasks),
            "source_task_ids_unique": len(
                {(r["domain"], str(r["source_task_id"]))
                 for r in tasks}) == len(tasks),
            "admitted_overlap_with_explored": [
                t for t in admitted if t in EXPLORED_TASK_IDS],
            "note": ("tau3 exposes no template/family ids; source_task_id "
                     "uniqueness is the strongest available leakage "
                     "guarantee and is verified above"),
        },
        "criteria": [
            "1 no unobservable constants (strict_v1)",
            "2 every param has a determined source",
            "3 refs use real replay output paths",
            "4 reconstruction executes on the official simulator with "
            "live runtime-ref resolution (args + final DB state match)",
            "5 split hygiene: explored tasks excluded from test",
        ],
        "blockers_for_other_tasks": {},
    }
    from collections import Counter
    for t in gold["tasks"]:
        if t["strict_v1"]:
            continue
        c = t["param_class_counts"]
        key = tuple(sorted(k for k in c
                           if k not in ("LITERAL", "runtime_ref",
                                        "runtime_ref_multi")))
        summary["blockers_for_other_tasks"].setdefault(
            str(key), []).append(t["task_id"])

    OUT.write_text(json.dumps({"summary": summary, "tasks": tasks},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "blockers_for_other_tasks"}, indent=1))
    print("blocker groups:", {k: len(v) for k, v in
                              summary["blockers_for_other_tasks"].items()})


if __name__ == "__main__":
    main()
