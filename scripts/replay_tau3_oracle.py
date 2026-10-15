"""Phase 9B-2.1 Task 4: Oracle replay of tau3 gold trajectories in the REAL simulator.

Replays the gold action sequences of 5 representative benchmark tasks through the
actual tau2-bench simulator (no LLM, no AgentBoard) and verifies:

  1. Oracle executability: every gold action runs without exception.
  2. Gold env self-consistency: env_assertions pass on the gold-replayed env.
  3. Runtime dep ground truth: whether each runtime_dep value in our converted
     benchmark actually appears in the REAL tool output of its producer step
     (upgrades provenance UNKNOWN -> VERIFIED_BY_REPLAY / NOT_IN_OUTPUT).

Run with the tau3 conda env from the tau2-bench checkout:
  D:/miniconda3/envs/tau3/python.exe scripts/replay_tau3_oracle.py
"""

import json
import re
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAU2_SRC = Path("D:/ccfa-2026/tau3-bench-src")
OUT = REPO / "results" / "phase9" / "tau3_oracle_replay.json"

# 5 representative tasks: both domains, depth 2-19, VERIFIED/UNKNOWN dep mixes,
# clean + HIGH-hidden provenance.
SELECTED = [
    "tau3_airline_44",  # deepest airline: 19 nodes, 28 deps (3 V / 25 U), 1 HIGH hidden
    "tau3_airline_30",  # small airline, all deps VERIFIED (2/2)
    "tau3_retail_23",   # deep retail: 12 deps (4 V / 8 U), 0 HIGH hidden
    "tau3_retail_99",   # small retail, 4 HIGH hidden env values (worst provenance)
    "tau3_retail_0",    # simple clean retail baseline
]


def load_benchmark_records():
    recs = {}
    for split in ("train", "dev", "test"):
        with open(REPO / "data" / "tau3_runtime_benchmark" / f"{split}.jsonl",
                  encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                recs[t["task_id"]] = t
    return recs


def main():
    run_all = "--all" in sys.argv
    sys.path.insert(0, str(TAU2_SRC / "src"))

    from tau2.data_model.tasks import Task
    from tau2.domains.airline.environment import get_environment as airline_env
    from tau2.domains.retail.environment import get_environment as retail_env

    envs = {"airline": airline_env, "retail": retail_env}
    recs = load_benchmark_records()
    task_ids = sorted(recs) if run_all else SELECTED

    results = []
    for tid in task_ids:
        rec = recs[tid]
        dom = "airline" if tid.startswith("tau3_airline") else "retail"
        sid = str(rec["provenance"]["source_task_id"])

        cur_tasks = json.loads(
            (TAU2_SRC / "data" / "tau2" / "domains" / dom / "tasks.json")
            .read_text(encoding="utf-8"))
        raw = next(t for t in cur_tasks if str(t["id"]) == sid)
        task = Task.model_validate(raw)

        env = envs[dom]()
        init = task.initial_state
        env.set_state(
            initialization_data=(init.initialization_data if init else None),
            initialization_actions=(init.initialization_actions if init else None),
            message_history=[],
            strict=True,
        )

        actions = (task.evaluation_criteria.actions or []) if task.evaluation_criteria else []
        trace = []
        n_err = 0
        for i, act in enumerate(actions):
            entry = {"step": i, "tool": act.name, "ok": True, "output": None,
                     "error": None}
            try:
                out = env.make_tool_call(
                    tool_name=act.name,
                    requestor=act.requestor or "assistant",
                    **(act.arguments or {}),
                )
                entry["output"] = out if isinstance(out, str) else json.dumps(out, default=str)
            except Exception:
                entry["ok"] = False
                entry["error"] = traceback.format_exc(limit=3)
                n_err += 1
            trace.append(entry)

        # env_assertions on the gold-replayed env (self-consistency)
        assertion_results = []
        for ea in ((task.evaluation_criteria.env_assertions or [])
                   if task.evaluation_criteria else []):
            try:
                met = env.run_env_assertion(ea, raise_assertion_error=False)
            except Exception:
                met = False
            assertion_results.append({"assertion": str(ea), "met": bool(met)})

        # runtime dep ground truth: tau3 tools return Python repr strings like
        # "order_id='#W2378156' items=[OrderItem(...)]". Verify the dep value
        # against the producer step's REAL output (field=... regex + substring),
        # and also against ALL earlier outputs to catch wrong-producer cases.
        field_re = re.compile(r"(\w+)=('[^']*'|\"[^\"]*\"|[\w.+-]+)")
        dep_checks = []
        for dep in rec.get("reference_deps", []):
            cstep, pstep = dep["step"], dep["producer_step"]
            ptool, pfield = dep["producer_tool"], dep["producer_field"]
            val = dep.get("value")

            def field_matches(text):
                if text is None or val is None:
                    return False
                sval = str(val)
                for fname, fval in field_re.findall(text):
                    if fname == pfield and fval.strip("'\"") == sval:
                        return True
                return False

            def contains(text):
                return text is not None and val is not None and str(val) in text

            src_out = trace[pstep]["output"] if 0 <= pstep < len(trace) else None
            if not (0 <= pstep < len(trace) and trace[pstep]["tool"] == ptool):
                # wrong index: search backwards from consumer step
                src_out = None
                for j in range(min(cstep, len(trace)) - 1, -1, -1):
                    if trace[j]["tool"] == ptool:
                        src_out = trace[j]["output"]
                        pstep = j
                        break

            earlier_hits = [
                j for j in range(0, min(cstep, len(trace)))
                if contains(trace[j]["output"])
            ]
            dep_checks.append({
                "step": cstep, "tool": dep["tool"], "param": dep["param"],
                "producer_step": pstep, "producer_tool": ptool,
                "producer_field": pfield, "value": val,
                "producer_field_match": field_matches(src_out),
                "producer_contains_value": contains(src_out),
                "any_earlier_contains_value": bool(earlier_hits),
                "earlier_hit_steps": earlier_hits[:6],
                "verdict": (
                    "VERIFIED_BY_REPLAY" if field_matches(src_out) else
                    "WRONG_PRODUCER" if contains(src_out) else
                    "VALUE_IN_EARLIER_ONLY" if earlier_hits else
                    "NOT_IN_ANY_OUTPUT"),
            })

        results.append({
            "task_id": tid, "domain": dom, "source_task_id": sid,
            "n_gold_actions": len(actions), "n_action_errors": n_err,
            "oracle_executable": n_err == 0,
            "db_hash_after_replay": env.get_db_hash(),
            "env_assertions": assertion_results,
            "env_assertions_all_met": all(a["met"] for a in assertion_results)
                                      if assertion_results else None,
            "dep_checks": dep_checks,
            "dep_summary": {
                "total": len(dep_checks),
                "verified_by_replay": sum(d["verdict"] == "VERIFIED_BY_REPLAY"
                                          for d in dep_checks),
                "wrong_producer": sum(d["verdict"] == "WRONG_PRODUCER"
                                      for d in dep_checks),
                "value_in_earlier_only": sum(d["verdict"] == "VALUE_IN_EARLIER_ONLY"
                                             for d in dep_checks),
                "not_in_any_output": sum(d["verdict"] == "NOT_IN_ANY_OUTPUT"
                                         for d in dep_checks),
            },
            "trace": trace,
        })
        print(f"{tid}: {len(actions)} actions, {n_err} errors, "
              f"deps {results[-1]['dep_summary']}")

    summary = {
        "n_tasks": len(results),
        "oracle_executable": sum(r["oracle_executable"] for r in results),
        "total_actions": sum(r["n_gold_actions"] for r in results),
        "total_action_errors": sum(r["n_action_errors"] for r in results),
        "total_deps": sum(r["dep_summary"]["total"] for r in results),
        "deps_verified_by_replay": sum(r["dep_summary"]["verified_by_replay"]
                                       for r in results),
        "deps_wrong_producer": sum(r["dep_summary"]["wrong_producer"]
                                   for r in results),
        "deps_value_in_earlier_only": sum(r["dep_summary"]["value_in_earlier_only"]
                                          for r in results),
        "deps_not_in_any_output": sum(r["dep_summary"]["not_in_any_output"]
                                      for r in results),
    }
    OUT.write_text(json.dumps({"summary": summary, "run_all": run_all,
                               "representative_sample": SELECTED,
                               "tasks": results}, indent=1,
                              ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
