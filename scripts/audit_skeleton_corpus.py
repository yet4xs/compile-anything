"""Phase 8B-1A Task 5: pre-training dataset audit.

Every training target must pass the full chain:
  skeleton parse -> skeleton validator -> deterministic lowering ->
  TaskIR parser -> TaskIR validator.
Failures are kept in a saved list, never silently dropped. Also reports
depth/DAG-shape distributions, identity & effect coverage, dedup and
family-isolation checks.
"""
import json, os, sys, hashlib
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import (skeleton_from_dict,
                                        validate_skeleton, hard_errors)
from src.compiler.skeleton_lowering import lower_skeleton

OUT = "results/phase8b/skeleton_data_audit.json"


def dag_shape(rec):
    nodes = rec["target_skeleton"]["nodes"]
    n = len(nodes)
    if n == 1:
        return "single"
    has_fork = any(len(x["depends_on"]) + len(x["order_on"]) > 1
                   for x in nodes)
    indeg = Counter()
    for x in nodes:
        for d in x["depends_on"] + x["order_on"]:
            indeg[d] += 1
    has_join = any(c > 1 for c in indeg.values())
    parallel = any(not x["depends_on"] and not x["order_on"]
                   for x in nodes[1:])
    if has_fork and has_join:
        return "fork_join"
    if parallel:
        return "independent_branches"
    return "linear_chain"


def critical_path(nodes):
    depth = {}
    for i, n in enumerate(nodes):
        deps = [depth[d] for d in n["depends_on"] + n["order_on"]
                if d in depth]
        depth[n["id"]] = 1 + (max(deps) if deps else 0)
    return max(depth.values()) if depth else 0


def main():
    splits = {}
    for k in ("train", "dev", "test"):
        rows = []
        for line in open(f"data/semantic_skeleton/{k}.jsonl",
                         encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
        splits[k] = rows

    pseudo_schemas = {}                 # corpus tools: no τ³ oracle
    audit = {"per_split": {}}
    all_fails = []
    for k, rows in splits.items():
        depth = Counter()
        shapes = Counter()
        chain = Counter()
        fails = []
        evidence = Counter()
        ref_ops = []
        crit_paths = []
        for r in rows:
            pseudo = {t: {"params": []} for t in
                      r.get("available_capabilities", [])}
            pseudo.update({n["capability_id"]: {"params": []}
                           for n in r["target_skeleton"]["nodes"]})
            n_nodes = len(r["target_skeleton"]["nodes"])
            depth[min(n_nodes, 8)] += 1
            shapes[dag_shape(r)] += 1
            evidence[r["provenance"]["evidence"]] += 1
            # reference operation count (merged duplicates included)
            ref_ops.append(sum(1 for _ in r["target_skeleton"]["nodes"]))
            crit_paths.append(critical_path(r["target_skeleton"]["nodes"]))
            try:
                sk = skeleton_from_dict(r["target_skeleton"])
                errs = hard_errors(validate_skeleton(sk, None, pseudo))
                if errs:
                    chain["skeleton_validation"] += 1
                    fails.append({"task_id": r["task_id"], "stage":
                                  "skeleton_validation", "err": errs[:1]})
                    continue
                low = lower_skeleton(sk, None, pseudo)
                if low.get("unresolved"):
                    chain["missing_params"] += 1
                    fails.append({"task_id": r["task_id"],
                                  "stage": "missing_params"})
                    continue
                if low.get("taskir") is None:
                    stage = ("taskir_parser" if str(low["errors"][0])
                             .startswith("parser") else "taskir_validator")
                    chain[stage] += 1
                    fails.append({"task_id": r["task_id"], "stage": stage,
                                  "err": low["errors"][:1]})
                    continue
                chain["ok"] += 1
            except Exception as e:
                chain["exception"] += 1
                fails.append({"task_id": r["task_id"], "stage": "exception",
                              "err": str(e)[:100]})
        audit["per_split"][k] = {
            "n": len(rows),
            "depth_hist_capped8": dict(sorted(depth.items())),
            "dag_shapes": dict(shapes),
            "evidence": dict(evidence),
            "chain": dict(chain),
            "lowering_success_pct": round(100 * chain["ok"]
                                          / max(1, len(rows)), 2),
            "mean_ref_op_count": round(sum(ref_ops) / max(1, len(ref_ops)),
                                       2),
            "mean_critical_path": round(sum(crit_paths)
                                        / max(1, len(crit_paths)), 2),
        }

        ok_ids = {r["task_id"] for r in rows}
        # SFT-ready subset: only chain-passing records (failures kept
        # separately, never silently dropped)
        sft = [r for r in rows
               if r["task_id"] not in {f["task_id"] for f in fails}]
        with open(f"data/semantic_skeleton/{k}_sft.jsonl", "w",
                  encoding="utf-8") as f:
            for r in sft:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        audit["per_split"][k]["sft_ready"] = len(sft)
        all_fails.extend({"split": k, **f} for f in fails)

    # near-duplicate check across splits: 8-gram Jaccard on a sample of
    # test instructions vs train
    def grams(t, k=8):
        w = t.split()
        return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}
    train_grams = [grams(r["instruction"]) for r in
                   splits["train"][:4000]]
    near = 0
    checked = 0
    for r in splits["test"][:300]:
        g = grams(r["instruction"])
        best = 0.0
        for tg in train_grams:
            if not tg or not g:
                continue
            j = len(g & tg) / len(g | tg)
            best = max(best, j)
        checked += 1
        if best > 0.8:
            near += 1
    audit["near_duplicates"] = {
        "test_checked": checked, "jaccard>0.8_vs_train": near}

    # family isolation
    fams = {k: {r["provenance"]["split_group"] for r in rows}
            for k, rows in splits.items()}
    audit["family_isolation"] = {
        "overlap_train_dev": sorted(fams["train"] & fams["dev"]),
        "overlap_train_test": sorted(fams["train"] & fams["test"]),
        "overlap_dev_test": sorted(fams["dev"] & fams["test"]),
    }
    # τ³ firewall
    audit["tau3_ids_present"] = sum(
        1 for rows in splits.values() for r in rows
        if any(d in r["task_id"] for d in
               ("airline-", "retail-", "telecom-", "banking_knowledge-")))
    # failure list preserved (all splits)
    with open("results/phase8b/skeleton_data_failures.json", "w") as f:
        json.dump(all_fails, f, ensure_ascii=False)
    with open(OUT, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    for k in ("train", "dev", "test"):
        a = audit["per_split"][k]
        print(f"{k}: n={a['n']} lowering_ok={a['lowering_success_pct']}% "
              f"depth={a['depth_hist_capped8']} shapes={a['dag_shapes']}")
    print("near-dup:", audit["near_duplicates"])
    print("family overlaps:", audit["family_isolation"])
    print("tau3 ids:", audit["tau3_ids_present"])
    print("AUDIT-DONE")


if __name__ == "__main__":
    main()
