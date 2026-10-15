"""Phase 9B-2.1 Task 5: Freeze the verified tau3 task subset.

Combines three evidence sources:
  1. Real-simulator oracle replay (results/phase9/tau3_oracle_replay.json):
     executability + actual tool outputs + per-dep ground truth.
  2. Benchmark records (data/tau3_runtime_benchmark/*.jsonl): skeleton,
     instruction text, reference deps.
  3. Static provenance audit (results/phase9/tau3_gold_provenance_audit.json):
     cross-reference only; replay evidence supersedes it.

A task is VERIFIED (paper-eligible) iff:
  - gold actions replay without exception on the canonical simulator, AND
  - every runtime dep's value is derivable from a real tool output
    (exact producer+field, wrong producer, or earlier output all count as
    "a runtime dep exists"; only NOT_IN_ANY_OUTPUT is a mislabel), AND
  - every literal argument is answerable: appears in the user-visible
    instruction OR in a tool output before its step.

Output: results/phase9/tau3_verified_subset_manifest.json
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "results" / "phase9" / "tau3_oracle_replay.json"
OUT = REPO / "results" / "phase9" / "tau3_verified_subset_manifest.json"

# replay dep verdicts that mean "a genuine runtime dep exists"
DERIVABLE = {"VERIFIED_BY_REPLAY", "WRONG_PRODUCER", "VALUE_IN_EARLIER_ONLY"}


def load_benchmark_records():
    recs = {}
    for split in ("train", "dev", "test"):
        with open(REPO / "data" / "tau3_runtime_benchmark" / f"{split}.jsonl",
                  encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                t["_split"] = split
                recs[t["task_id"]] = t
    return recs


def instruction_text(rec):
    instr = rec.get("instruction", "")
    if isinstance(instr, dict):
        return " ".join(str(v) for v in instr.values()).lower()
    return str(instr).lower()


def check_literal_answerability(rec, trace):
    """Every non-runtime_ref literal must be in user text or an earlier output.

    List-valued args are checked element-wise: each element must be derivable
    (a stringified list never appears verbatim in text or tool output).
    """
    text = instruction_text(rec)
    problems = []
    nodes = rec["target_skeleton"]["nodes"]

    def derivable(val_str, step):
        if val_str.lower() in text:
            return True
        return any(
            val_str in (trace[j]["output"] or "")
            for j in range(0, min(step, len(trace)))
        )

    for i, node in enumerate(nodes):
        for param, value in node["arguments"].items():
            if isinstance(value, dict) and value.get("kind") == "runtime_ref":
                continue
            elems = value if isinstance(value, list) else [value]
            bad = [str(e).strip() for e in elems
                   if str(e).strip() and not derivable(str(e).strip(), i)]
            if bad:
                problems.append({
                    "node": node["id"], "step": i, "param": param,
                    "tool": node["capability_id"],
                    "values": [b[:40] for b in bad[:4]],
                    "status": "UNANSWERABLE_LITERAL",
                })
    return problems


def main():
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    rep_by_id = {t["task_id"]: t for t in replay["tasks"]}
    recs = load_benchmark_records()
    static = json.loads((REPO / "results" / "phase9" /
                         "tau3_gold_provenance_audit.json").read_text(encoding="utf-8"))
    static_by_id = {t["task_id"]: t for t in static["task_details"]}

    manifest = []
    for tid in sorted(recs):
        rec, rep = recs[tid], rep_by_id[tid]
        trace = rep["trace"]
        dep_checks = rep["dep_checks"]
        n_exact = sum(d["verdict"] == "VERIFIED_BY_REPLAY" for d in dep_checks)
        n_derivable = sum(d["verdict"] in DERIVABLE for d in dep_checks)
        n_mislabeled = sum(d["verdict"] == "NOT_IN_ANY_OUTPUT" for d in dep_checks)
        unanswerable = check_literal_answerability(rec, trace)

        executable = rep["oracle_executable"]
        if executable and n_mislabeled == 0 and not unanswerable:
            tier = "VERIFIED"
        elif executable:
            tier = "DEGRADED"
        else:
            tier = "EXCLUDED_NOT_EXECUTABLE"

        manifest.append({
            "task_id": tid,
            "split": rec["_split"],
            "domain": rep["domain"],
            "source_task_id": rep["source_task_id"],
            "n_nodes": len(rec["target_skeleton"]["nodes"]),
            "n_runtime_deps": rec["n_runtime_deps"],
            "tier": tier,
            "oracle_executable": executable,
            "n_action_errors": rep["n_action_errors"],
            "deps": {
                "exact": n_exact,
                "derivable": n_derivable,
                "mislabeled": n_mislabeled,
                "total": len(dep_checks),
            },
            "n_unanswerable_literals": len(unanswerable),
            "unanswerable_examples": unanswerable[:5],
            "static_audit_hidden_high": static_by_id[tid]["n_hidden_env_high"],
        })

    tiers = {}
    for m in manifest:
        tiers.setdefault(m["tier"], []).append(m["task_id"])

    # strict variant: every dep label exactly right (producer+field+value)
    strict_ids = [m["task_id"] for m in manifest
                  if m["tier"] == "VERIFIED" and m["deps"]["exact"] == m["deps"]["total"]]

    summary = {
        "total": len(manifest),
        "tiers": {k: len(v) for k, v in tiers.items()},
        "verified_task_ids": tiers.get("VERIFIED", []),
        "degraded_task_ids": tiers.get("DEGRADED", []),
        "excluded_task_ids": tiers.get("EXCLUDED_NOT_EXECUTABLE", []),
        "strict_exact_deps_task_ids": strict_ids,
        "dep_totals": {
            "total": sum(m["deps"]["total"] for m in manifest),
            "exact": sum(m["deps"]["exact"] for m in manifest),
            "derivable": sum(m["deps"]["derivable"] for m in manifest),
            "mislabeled": sum(m["deps"]["mislabeled"] for m in manifest),
        },
        "verified_by_domain": {},
        "verified_by_split": {},
        "verified_dep_total": sum(m["deps"]["total"] for m in manifest
                                  if m["tier"] == "VERIFIED"),
    }
    for m in manifest:
        if m["tier"] == "VERIFIED":
            summary["verified_by_domain"][m["domain"]] = \
                summary["verified_by_domain"].get(m["domain"], 0) + 1
            summary["verified_by_split"][m["_split" if "_split" in m else "split"]] = \
                summary["verified_by_split"].get(m["split"], 0) + 1

    OUT.write_text(json.dumps({"summary": summary, "tasks": manifest}, indent=1,
                              ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if not k.endswith("_task_ids") and k != "strict_exact_deps_task_ids"},
                     indent=1))
    print("VERIFIED:", tiers.get("VERIFIED", []))
    print("STRICT (all deps exact):", len(strict_ids), strict_ids[:10])


if __name__ == "__main__":
    main()
