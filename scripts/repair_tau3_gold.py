"""Phase 9B-2.2 Task 2: Gold dependency repair from REAL tau3 tool returns.

Replays every task's gold actions through the official tau2-bench simulator,
calling the underlying tool functions directly (env.tools.tools[name]) so we
get the domain objects — not their repr/json stringification. Each output is
walked structurally (dataclass fields, dict keys, list indices) and every
value is indexed by its exact access path, e.g. ``orders[0].order_id``.

Every one of the 233 heuristic runtime-dep labels is then re-verified:

  LABEL_EXACT            labeled (producer_step, field) really carries the
                         value at that path
  REPAIRED_UNIQUE        exactly one (step, path) in prior outputs carries the
                         value — label corrected to it
  AMBIGUOUS              value appears in multiple prior outputs; no
                         auto-selection (all candidates recorded)
  NOT_RUNTIME_DEP_USER_TEXT   value never returned by a tool but is visible in
                         the user instruction — should be a literal arg
  USER_LATENT            value not in any tool output nor in the instruction —
                         only obtainable through user interaction
  UNEXECUTABLE_CONTEXT   a prior gold action errored, so outputs are partial

No label is changed to favor any model; the original label is always kept
side-by-side with the evidence-based verdict (old/new diff + provenance).

Run with the tau3 conda env:
  D:/miniconda3/envs/tau3/python.exe scripts/repair_tau3_gold.py
"""

import dataclasses
import json
import sys
import traceback
from pathlib import Path

from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
TAU2_SRC = Path("D:/ccfa-2026/tau3-bench-src")
OUT = REPO / "results" / "phase9" / "tau3_gold_repaired.json"


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


def walk(obj, path="", max_depth=8):
    """Yield (path, value) for every node (containers and leaves) of obj.

    pydantic models and dataclasses -> named fields, dicts -> keys,
    lists -> [i] indices. A bare primitive root yields the empty path
    (the entire tool output is that value).
    """
    if max_depth < 0:
        yield path, obj
        return
    if isinstance(obj, BaseModel):
        for name in obj.model_fields:
            yield from walk(getattr(obj, name),
                            f"{path}.{name}" if path else name,
                            max_depth - 1)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from walk(getattr(obj, f.name),
                            f"{path}.{f.name}" if path else f.name,
                            max_depth - 1)
    elif isinstance(obj, dict):
        if not obj:
            yield path, obj
        for k, v in obj.items():
            yield from walk(v,
                            f"{path}.{k}" if path else str(k),
                            max_depth - 1)
    elif isinstance(obj, (list, tuple)):
        if not obj:
            yield path, obj
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]", max_depth - 1)
    else:
        yield path, obj


def main():
    sys.path.insert(0, str(TAU2_SRC / "src"))

    from tau2.data_model.tasks import Task
    from tau2.domains.airline.environment import get_environment as airline_env
    from tau2.domains.retail.environment import get_environment as retail_env

    envs = {"airline": airline_env, "retail": retail_env}
    recs = load_benchmark_records()

    repaired_tasks = []
    verdict_counts = {}
    real_schema_observations = {}  # (domain, tool) -> {field: type_sample_count}

    for tid in sorted(recs):
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

        actions = ((task.evaluation_criteria.actions or [])
                   if task.evaluation_criteria else [])
        outputs = []   # per step: {path: value} flattened
        step_meta = []  # per step: {"tool":..., "ok":..., "error":...}
        for act in actions:
            entry = {"tool": act.name, "ok": True, "error": None}
            flat = {}
            try:
                obj = env.tools.tools[act.name](**(act.arguments or {}))
                for p, v in walk(obj):
                    flat[p] = v
            except Exception:
                entry["ok"] = False
                entry["error"] = traceback.format_exc(limit=2).strip().splitlines()[-1]
            outputs.append(flat)
            step_meta.append(entry)
            # real top-level schema observation
            key = (dom, act.name)
            obs = real_schema_observations.setdefault(key, {})
            for p, v in flat.items():
                top = p.split("[")[0].split(".")[0] or "(root)"
                t = type(v).__name__
                obs.setdefault(top, {}).setdefault(t, 0)
                obs[top][t] += 1

        text = instruction_text(rec)
        any_earlier_error = any(not m["ok"] for m in step_meta)

        dep_results = []
        for dep in rec.get("reference_deps", []):
            cstep = dep["step"]
            val = dep.get("value")
            pstep, ptool, pfield = (dep["producer_step"],
                                    dep["producer_tool"],
                                    dep["producer_field"])

            candidates = []
            for j in range(0, min(cstep, len(outputs))):
                for p, v in outputs[j].items():
                    if str(v) == str(val):
                        candidates.append({"step": j, "path": p,
                                           "tool": step_meta[j]["tool"]})

            def is_labeled(c):
                if c["step"] != pstep:
                    return False
                if c["path"] == pfield:
                    return True
                # Tool whose entire output is a bare scalar (e.g.
                # find_user_id_by_name_zip -> "user_123"): the field name is
                # the output itself; recorded distinctly, not silently.
                return c["path"] == "" and len(outputs[pstep]) == 1

            root_scalar = any(is_labeled(c) and c["path"] == ""
                              for c in candidates)
            labeled_hit = any(is_labeled(c) for c in candidates)

            if labeled_hit:
                verdict = ("LABEL_EXACT_ROOT_SCALAR" if root_scalar
                           else "LABEL_EXACT")
            elif len(candidates) == 1:
                verdict = "REPAIRED_UNIQUE"
            elif len(candidates) > 1:
                verdict = "AMBIGUOUS"
            else:
                if any_earlier_error:
                    verdict = "UNEXECUTABLE_CONTEXT"
                elif val is not None and str(val).lower() in text:
                    verdict = "NOT_RUNTIME_DEP_USER_TEXT"
                else:
                    verdict = "USER_LATENT"

            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            dep_results.append({
                "consumer": {"step": cstep, "tool": dep["tool"],
                             "param": dep["param"]},
                "original_label": {"producer_step": pstep,
                                   "producer_tool": ptool,
                                   "producer_field": pfield,
                                   "value": val},
                "verdict": verdict,
                "candidates": candidates[:8],
                "n_candidates": len(candidates),
                "labeled_producer_among_candidates": any(
                    c["step"] == pstep for c in candidates),
                "repaired_to": (candidates[0] if verdict == "REPAIRED_UNIQUE"
                                else None),
            })

        repaired_tasks.append({
            "task_id": tid, "domain": dom, "split": rec["_split"],
            "source_task_id": sid,
            "gold_replay_ok": all(m["ok"] for m in step_meta),
            "n_actions": len(actions),
            "step_meta": step_meta,
            "outputs_flat": [
                {p: str(v)[:80] for p, v in flat.items()} for flat in outputs],
            "deps": dep_results,
            "dep_verdicts": {v: sum(1 for d in dep_results if d["verdict"] == v)
                             for v in {d["verdict"] for d in dep_results}},
        })
        print(f"{tid}: {len(dep_results)} deps "
              f"{repaired_tasks[-1]['dep_verdicts']}")

    summary = {
        "n_tasks": len(repaired_tasks),
        "n_deps": sum(len(t["deps"]) for t in repaired_tasks),
        "verdict_counts": verdict_counts,
        "label_exact_rate": round(verdict_counts.get("LABEL_EXACT", 0) /
                                  max(1, sum(verdict_counts.values())), 4),
        "tasks_gold_replay_ok": sum(t["gold_replay_ok"]
                                    for t in repaired_tasks),
    }
    OUT.write_text(json.dumps({
        "summary": summary,
        "real_schema_observations": {
            f"{d}.{t}": fields for (d, t), fields in
            real_schema_observations.items()},
        "tasks": repaired_tasks,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
