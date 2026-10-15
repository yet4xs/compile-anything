"""Phase 9B-2.2 Task 1: rebuild the tau3 gold quality manifest with three
evidence-backed tiers (supersedes the single-tier VERIFIED of 9B-2.1).

Tiers (cumulative):

  REPLAY_EXECUTABLE  all gold actions run without exception on the official
                     simulator (evidence: repair replay step_meta)
  VALUE_OBSERVABLE   REPLAY_EXECUTABLE + every dep value is carried by some
                     real prior tool output (verdict LABEL_EXACT /
                     LABEL_EXACT_ROOT_SCALAR / REPAIRED_UNIQUE / AMBIGUOUS)
                     + every literal arg answerable (user text or a real
                     prior output, structured path values)
  BINDING_EXACT      VALUE_OBSERVABLE + every dep has unique evidence-backed
                     provenance (LABEL_EXACT / LABEL_EXACT_ROOT_SCALAR /
                     REPAIRED_UNIQUE — AMBIGUOUS excluded). This is the only
                     tier admissible to strict Binding Exact evaluation.

AMBIGUOUS deps keep the task out of BINDING_EXACT but stay in
VALUE_OBSERVABLE; USER_LATENT / UNEXECUTABLE_CONTEXT break VALUE_OBSERVABLE.

Output: results/phase9/tau3_gold_verified_manifest.json
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPAIRED = REPO / "results" / "phase9" / "tau3_gold_repaired.json"
OUT = REPO / "results" / "phase9" / "tau3_gold_verified_manifest.json"

VALUE_OK = {"LABEL_EXACT", "LABEL_EXACT_ROOT_SCALAR", "REPAIRED_UNIQUE",
            "AMBIGUOUS"}
BINDING_OK = {"LABEL_EXACT", "LABEL_EXACT_ROOT_SCALAR", "REPAIRED_UNIQUE"}


def instruction_text(rec):
    instr = rec.get("instruction", "")
    if isinstance(instr, dict):
        return " ".join(str(v) for v in instr.values()).lower()
    return str(instr).lower()


def unanswerable_literals(rec, outputs_flat):
    text = instruction_text(rec)
    problems = []
    nodes = rec["target_skeleton"]["nodes"]
    for i, node in enumerate(nodes):
        prior_vals = []
        for j in range(0, min(i, len(outputs_flat))):
            prior_vals.extend(str(v) for v in outputs_flat[j].values())
        for param, value in node["arguments"].items():
            if isinstance(value, dict) and value.get("kind") == "runtime_ref":
                continue
            elems = value if isinstance(value, list) else [value]
            bad = [str(e).strip() for e in elems
                   if str(e).strip()
                   and str(e).strip().lower() not in text
                   and not any(str(e).strip() in pv for pv in prior_vals)]
            if bad:
                problems.append({"node": node["id"], "step": i,
                                 "param": param,
                                 "tool": node["capability_id"],
                                 "values": [b[:40] for b in bad[:4]]})
    return problems


def main():
    rep = json.loads(REPAIRED.read_text(encoding="utf-8"))
    rep_by_id = {t["task_id"]: t for t in rep["tasks"]}

    recs = {}
    for split in ("train", "dev", "test"):
        with open(REPO / "data" / "tau3_runtime_benchmark" / f"{split}.jsonl",
                  encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                recs[t["task_id"]] = t

    tasks_out = []
    tiers = {"REPLAY_EXECUTABLE": [], "VALUE_OBSERVABLE": [],
             "BINDING_EXACT": [], "FAILED_REPLAY": []}
    for tid in sorted(recs):
        t = rep_by_id[tid]
        rec = recs[tid]
        verdicts = [d["verdict"] for d in t["deps"]]
        n_deps = len(verdicts)
        counts = t["dep_verdicts"]

        executable = t["gold_replay_ok"]
        unans = unanswerable_literals(rec, t["outputs_flat"])
        value_ok = (executable and n_deps > 0
                    and all(v in VALUE_OK for v in verdicts)
                    and not unans) or (executable and n_deps == 0 and not unans)
        binding_ok = (value_ok and n_deps > 0
                      and all(v in BINDING_OK for v in verdicts))

        if not executable:
            tier = "FAILED_REPLAY"
        elif binding_ok:
            tier = "BINDING_EXACT"
        elif value_ok:
            tier = "VALUE_OBSERVABLE"
        else:
            tier = "REPLAY_EXECUTABLE"

        tiers[tier].append(tid)
        tasks_out.append({
            "task_id": tid, "domain": t["domain"], "split": t["split"],
            "tier": tier,
            "n_runtime_deps": n_deps,
            "dep_verdict_counts": counts,
            "n_unanswerable_literals": len(unans),
            "unanswerable_examples": unans[:4],
        })

    # BINDING_EXACT ⊂ VALUE_OBSERVABLE ⊂ REPLAY_EXECUTABLE by construction
    summary = {
        "n_tasks": len(tasks_out),
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "BINDING_EXACT": tiers["BINDING_EXACT"],
        "VALUE_OBSERVABLE": tiers["VALUE_OBSERVABLE"],
        "REPLAY_EXECUTABLE": tiers["REPLAY_EXECUTABLE"],
        "FAILED_REPLAY": tiers["FAILED_REPLAY"],
        "by_domain": {},
        "by_split_BINDING_EXACT": {},
        "dep_stats": {
            "total": sum(len(t["deps"]) for t in rep["tasks"]),
            **rep["summary"]["verdict_counts"],
        },
        "note": ("BINDING_EXACT is the only tier admissible to strict "
                 "Binding Exact evaluation; VALUE_OBSERVABLE tasks carry "
                 "AMBIGUOUS deps (value real, provenance not unique); "
                 "previous 9B-2.1 'VERIFIED' tier corresponds to none of "
                 "these exactly and is renamed OBSERVABLE_CANDIDATE "
                 "(superseded by this manifest)."),
    }
    for t in tasks_out:
        order = {"BINDING_EXACT": 3, "VALUE_OBSERVABLE": 2,
                 "REPLAY_EXECUTABLE": 1, "FAILED_REPLAY": 0}
        d = summary["by_domain"].setdefault(t["domain"], {})
        d[t["tier"]] = d.get(t["tier"], 0) + 1
        if t["tier"] == "BINDING_EXACT":
            summary["by_split_BINDING_EXACT"][t["split"]] = \
                summary["by_split_BINDING_EXACT"].get(t["split"], 0) + 1

    OUT.write_text(json.dumps({"summary": summary, "tasks": tasks_out},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if not isinstance(v, list) and k != "dep_stats"},
                     indent=1))
    print("dep_stats:", json.dumps(summary["dep_stats"]))


if __name__ == "__main__":
    main()
