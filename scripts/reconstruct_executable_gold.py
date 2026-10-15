"""Phase 9B-2.3 Tasks 1+2: reconstruct a truly executable gold program and
independently validate it against the official simulator.

Reconstruction (Task 1) classifies EVERY parameter element of EVERY gold
action — not just the 233 heuristic runtime-dep labels:

  LITERAL        value visible in the user instruction text
  runtime_ref    value carried by exactly one prior real tool output ->
                 {producer_step, path}; whole-list matches bind the list
  AMBIGUOUS      multiple prior-output candidates (kept, never auto-picked)
  USER_LATENT    not in text/outputs; present in the acting user's own DB
                 subtree -> needs user interaction
  ENV_STATE      not in text/outputs; present elsewhere in the initial DB
  UNKNOWN        nowhere -> cannot enter strict gold

A task's reconstructed program is STRICT_V1 iff every param element is
LITERAL or a unique runtime_ref whose producer precedes its consumer.

Execution validation (Task 2) runs BOTH programs on fresh official envs:
the original reference actions and the reconstructed program (runtime_refs
resolved live via typed paths). Compared per step: resolved argument values,
tool instances, structured outputs, DB hash trajectory, final DB state.
Reference-trajectory equivalence is reported as such; τ³'s own reward
semantics is end-state (DB hash) — both are recorded separately.

Run with the tau3 conda env:
  D:/miniconda3/envs/tau3/python.exe scripts/reconstruct_executable_gold.py
"""

import dataclasses
import hashlib
import json
import sys
import traceback
from pathlib import Path

from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
TAU2_SRC = Path("D:/ccfa-2026/tau3-bench-src")
OUT_GOLD = REPO / "results" / "phase9" / "tau3_gold_compiled_v1.json"
OUT_EXEC = REPO / "results" / "phase9" / "tau3_gold_execution_validation.json"

sys.path.insert(0, str(REPO))
from src.compiler.field_path import parse_path, resolve_path, PathTypeError


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
    if max_depth < 0:
        yield path, obj
        return
    if isinstance(obj, BaseModel):
        for name in obj.model_fields:
            yield from walk(getattr(obj, name),
                            f"{path}.{name}" if path else name, max_depth - 1)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from walk(getattr(obj, f.name),
                            f"{path}.{f.name}" if path else f.name,
                            max_depth - 1)
    elif isinstance(obj, dict):
        if not obj:
            yield path, obj
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k), max_depth - 1)
    elif isinstance(obj, (list, tuple)):
        if not obj:
            yield path, obj
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]", max_depth - 1)
    else:
        yield path, obj


def db_value_paths(db, value):
    """Paths in the initial DB JSON carrying `value` (string compare)."""
    hits = []
    def rec(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                rec(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                rec(v, f"{path}[{i}]")
        elif str(node) == str(value):
            hits.append(path)
    rec(db, "")
    return hits


def _canonical_param_match(param, path):
    """Path terminal field matches the consumer's parameter semantics
    (order_id <- ...order_id; item_ids <- ...item_id / ...item_ids;
    reservation_id <- reservations[i], a list holding bare ids)."""
    last = path.split(".")[-1].split("[")[0]
    stem = param[:-1] if param.endswith("s") else param
    if stem.endswith("_ids"):
        stem = stem[:-4] + "_id"
    base = stem[:-3] if stem.endswith("_id") else stem
    return last in (param, stem, base, base + "s", stem + "s")


def classify_value(value, text, prior_outputs, db, acting_user, param=None):
    """Classify one scalar/list element. prior_outputs: list of flat dicts
    (live outputs of steps 0..i-1). Returns (cls, info).

    Ambiguity policy: candidates spanning multiple producer steps stay
    AMBIGUOUS (never auto-picked). Candidates within a single producer are
    reduced by canonical terminal-field match against the consumer param;
    0 or >1 canonical matches stay AMBIGUOUS with all candidates recorded.
    """
    sval = str(value).strip()
    if not sval:
        return "LITERAL", {}
    if sval.lower() in text:
        return "LITERAL", {}

    cands = []
    for j, flat in enumerate(prior_outputs):
        for p, v in flat.items():
            if str(v) == sval:
                cands.append({"producer_step": j, "path": p})
    # prefer non-echo candidates: a dict-key segment equal to the sought
    # value (variants.<item_id>.item_id) is an identity index. Echoes are
    # kept only when they are the sole provenance (the id is genuinely
    # learned by enumerating that dict).
    nonecho = [c for c in cands
               if not any(seg == sval for seg in c["path"].split(".")
                          if "[" not in seg)]
    cands = nonecho or cands
    if len(cands) == 1:
        return "runtime_ref", {**cands[0],
                               "selected_by": "unique_candidate"}
    if cands:
        producers = {c["producer_step"] for c in cands}
        if len(producers) == 1:
            canon = [c for c in cands
                     if param and _canonical_param_match(param, c["path"])]
            if len(canon) == 1:
                return "runtime_ref", {**canon[0],
                                       "selected_by": "canonical_field",
                                       "all_candidates": cands[:8]}
            if len(canon) > 1:
                depths = {c["path"].count(".") + c["path"].count("[")
                          for c in canon}
                if len(depths) == 1 and len(canon) > 1:
                    # same-depth canonical duplicates: still ambiguous
                    return "AMBIGUOUS", {"candidates": cands[:8]}
                min_d = min(depths)
                minimal = [c for c in canon
                           if c["path"].count(".") + c["path"].count("[")
                           == min_d]
                if len(minimal) == 1:
                    return "runtime_ref", {**minimal[0],
                                           "selected_by":
                                               "canonical_minimal_depth",
                                           "all_candidates": cands[:8]}
        else:
            # multi-producer: a root-scalar candidate is the ORIGIN of the
            # value (that tool's entire output IS the value; later field
            # occurrences are echoes of it). Unique origin -> select.
            roots = [c for c in cands if c["path"] == ""]
            if len(roots) == 1:
                return "runtime_ref", {**roots[0],
                                       "selected_by": "root_scalar_origin",
                                       "all_candidates": cands[:8]}
        return "AMBIGUOUS", {"candidates": cands[:8]}

    if db is not None:
        hits = db_value_paths(db, value)
        if hits:
            owned = [h for h in hits if _owner_path(h, db, acting_user)]
            if owned and len(owned) == len(hits):
                return "USER_LATENT", {"db_paths": hits[:4]}
            return "ENV_STATE", {"db_paths": hits[:4]}
    return "UNKNOWN", {}


def _owner_path(path, db, acting_user):
    """True when a DB path points into the acting user's own entities
    (users.<uid>..., or an order/reservation whose user_id is the acting
    user's)."""
    if acting_user and path.startswith(f"users.{acting_user}."):
        return True
    root = path.split(".")[0].split("[")[0]
    if root in ("orders", "reservations") and acting_user:
        key = path.split(".")[1].split("[")[0] if "." in path else None
        if key:
            entity = db.get(root, {}).get(key)
            if isinstance(entity, dict) and \
                    str(entity.get("user_id")) == str(acting_user):
                return True
    return False


def classify_arg(value, text, prior_outputs, db, acting_user, param=None):
    """Whole-argument classification. Lists: exact whole-list output match
    first, then element-wise (mixed multi-source lists are not strict)."""
    if isinstance(value, (list, tuple)):
        # whole-list match against prior output nodes
        whole = []
        for j, flat in enumerate(prior_outputs):
            for p, v in flat.items():
                if isinstance(v, (list, tuple)) and \
                        [str(x) for x in v] == [str(x) for x in value]:
                    whole.append({"producer_step": j, "path": p})
        if len(whole) == 1:
            return "runtime_ref", {**whole[0],
                                   "selected_by": "whole_list_match"}
        if len(whole) > 1:
            return "AMBIGUOUS", {"candidates": whole[:8]}
        # element-wise
        elems = [classify_value(e, text, prior_outputs, db, acting_user,
                                param) for e in value]
        classes = {c for c, _ in elems}
        if not classes:
            return "LITERAL", {}
        if classes == {"LITERAL"}:
            return "LITERAL", {}
        if classes == {"runtime_ref"}:
            return "runtime_ref_multi", {
                "elements": [info for _, info in elems]}
        worst = sorted(classes)[0]  # AMBIGUOUS < ENV_STATE < UNKNOWN < ...
        return worst, {"elements": [
            {"class": c, **info} for c, info in elems][:8]}
    return classify_value(value, text, prior_outputs, db, acting_user, param)


NON_STRICT = {"AMBIGUOUS", "USER_LATENT", "ENV_STATE", "UNKNOWN"}
SOURCED = {"LITERAL", "runtime_ref", "runtime_ref_multi"}


def main():
    sys.path.insert(0, str(TAU2_SRC / "src"))
    from tau2.data_model.tasks import Task
    from tau2.domains.airline.environment import get_environment as airline_env
    from tau2.domains.retail.environment import get_environment as retail_env

    envs = {"airline": airline_env, "retail": retail_env}
    recs = load_benchmark_records()
    dbs = {d: json.loads(
        (REPO / "data" / "external_benchmarks" / "tau3_bench" / "domains" /
         d / "db.json").read_text(encoding="utf-8"))
        for d in ("retail", "airline")}

    gold_out, exec_out = [], []
    counts = {}

    for tid in sorted(recs):
        rec = recs[tid]
        dom = "airline" if tid.startswith("tau3_airline") else "retail"
        sid = str(rec["provenance"]["source_task_id"])
        cur_tasks = json.loads(
            (TAU2_SRC / "data" / "tau2" / "domains" / dom / "tasks.json")
            .read_text(encoding="utf-8"))
        raw = next(t for t in cur_tasks if str(t["id"]) == sid)
        task = Task.model_validate(raw)
        actions = ((task.evaluation_criteria.actions or [])
                   if task.evaluation_criteria else [])
        text = instruction_text(rec)
        db = dbs[dom]

        # acting user: any arg value that is a key in db['users']
        acting_user = None
        for a in actions:
            for v in (a.arguments or {}).values():
                if isinstance(v, str) and v in db.get("users", {}):
                    acting_user = v
                    break

        # ---- pass 1: original reference replay (fresh env) ----
        env1 = envs[dom]()
        init = task.initial_state
        for e in (env1,):
            e.set_state(
                initialization_data=(init.initialization_data if init else None),
                initialization_actions=(init.initialization_actions if init else None),
                message_history=[], strict=True)
        orig_trace = []
        for act in actions:
            entry = {"tool": act.name, "args": dict(act.arguments or {}),
                     "ok": True, "error": None, "output_full": {},
                     "output_flat": {}, "db_hash_after": None}
            try:
                obj = env1.tools.tools[act.name](**(act.arguments or {}))
                entry["output_full"] = dict(walk(obj))
                entry["output_flat"] = {p: str(v)[:60]
                                        for p, v in walk(obj)}
            except Exception:
                entry["ok"] = False
                entry["error"] = traceback.format_exc(limit=2).strip().splitlines()[-1]
            entry["db_hash_after"] = env1.tools.get_db_hash()
            orig_trace.append(entry)

        # ---- pass 2: classify every param, build reconstructed program ----
        live_outputs = []   # flat dicts from successful reference steps
        recon_nodes = []
        param_counts = {}
        for i, act in enumerate(actions):
            args_out = {}
            for param, value in (act.arguments or {}).items():
                cls, info = classify_arg(value, text, live_outputs, db,
                                         acting_user, param)
                param_counts[cls] = param_counts.get(cls, 0) + 1
                args_out[param] = {"class": cls, "value": value, **info}
            recon_nodes.append({"step": i, "tool": act.name,
                                "arguments": args_out})
            if orig_trace[i]["ok"]:
                live_outputs.append(orig_trace[i]["output_full"])

        strict = (all(e["ok"] for e in orig_trace) and bool(actions)
                  and set(param_counts) <= SOURCED)
        # producer-precedes-consumer check (incl. per-element refs)
        for node in recon_nodes:
            for param, a in node["arguments"].items():
                steps = [a.get("producer_step")] + [
                    e.get("producer_step") for e in a.get("elements", [])]
                if a["class"] in ("runtime_ref", "runtime_ref_multi"):
                    if any(s is not None and s >= node["step"] for s in steps):
                        strict = False
        # representable in the CURRENT skeleton format (single runtime_ref,
        # top-level or root-scalar path only)?
        current_ir = strict
        for node in recon_nodes:
            for a in node["arguments"].values():
                if a["class"] == "runtime_ref_multi":
                    current_ir = False
                elif a["class"] == "runtime_ref":
                    p = a.get("path", "")
                    if p and ("[" in p or "." in p):
                        current_ir = False

        # ---- pass 3: execute reconstructed program on a SECOND fresh env ----
        env2 = envs[dom]()
        env2.set_state(
            initialization_data=(init.initialization_data if init else None),
            initialization_actions=(init.initialization_actions if init else None),
            message_history=[], strict=True)
        recon_trace, live_outputs = [], []
        live_objects = {}  # producer_step -> live structured output object
        exec_ok = True
        for node in recon_nodes:
            call_args = {}
            resolve_errors = []
            for param, a in node["arguments"].items():
                if a["class"] == "LITERAL":
                    call_args[param] = a["value"]
                elif a["class"] == "runtime_ref":
                    try:
                        steps = parse_path(a["path"]) if a["path"] else []
                        call_args[param] = resolve_path(
                            live_objects[a["producer_step"]], steps)
                    except Exception as e:
                        resolve_errors.append(f"{param}: {e}")
                        call_args[param] = a["value"]  # fallback literal
                elif a["class"] == "runtime_ref_multi":
                    try:
                        vals = []
                        for e in a["elements"]:
                            steps = parse_path(e["path"]) if e["path"] else []
                            vals.append(resolve_path(
                                live_objects[e["producer_step"]], steps))
                        call_args[param] = vals
                    except Exception as e:
                        resolve_errors.append(f"{param}: {e}")
                        call_args[param] = a["value"]
                else:
                    resolve_errors.append(f"{param}: class {a['class']}")
                    call_args[param] = a["value"]
            entry = {"tool": node["tool"], "resolved_args": call_args,
                     "resolve_errors": resolve_errors, "ok": True,
                     "error": None, "output_flat": {}, "db_hash_after": None}
            try:
                obj = env2.tools.tools[node["tool"]](**call_args)
                live_objects[node["step"]] = obj
                entry["output_flat"] = {p: str(v)[:60]
                                        for p, v in walk(obj)}
                live_outputs.append(dict(walk(obj)))
            except Exception:
                entry["ok"] = False
                entry["error"] = traceback.format_exc(limit=2).strip().splitlines()[-1]
                exec_ok = False
            entry["db_hash_after"] = env2.tools.get_db_hash()
            recon_trace.append(entry)

        # step-by-step comparison (only meaningful for non-strict classes
        # that fell back to literals: identical programs compare trivially)
        steps_match = len(orig_trace) == len(recon_trace)
        if steps_match:
            for o, r in zip(orig_trace, recon_trace):
                if o["tool"] != r["tool"] or not o["ok"] or not r["ok"]:
                    steps_match = False
                    break
                if {k: str(v) for k, v in o["args"].items()} != \
                        {k: str(v) for k, v in r["resolved_args"].items()}:
                    steps_match = False
                    break
        final_state_match = (orig_trace and recon_trace and
                             orig_trace[-1]["db_hash_after"] ==
                             recon_trace[-1]["db_hash_after"])

        gold_out.append({
            "task_id": tid, "domain": dom, "split": rec["_split"],
            "source_task_id": sid,
            "acting_user": acting_user,
            "gold_replay_ok": all(e["ok"] for e in orig_trace),
            "param_class_counts": param_counts,
            "strict_v1": strict,
            "current_ir_representable": current_ir,
            "program": recon_nodes,
        })
        exec_out.append({
            "task_id": tid,
            "reference_replay_ok": all(e["ok"] for e in orig_trace),
            "reconstructed_exec_ok": exec_ok and not any(
                e["resolve_errors"] for e in recon_trace),
            "used_fallback_literals": any(
                e["resolve_errors"] for e in recon_trace),
            "n_steps": len(recon_trace),
            "steps_arg_and_tool_match": steps_match,
            "final_db_state_match": final_state_match,
            "reference_final_db_hash": orig_trace[-1]["db_hash_after"] if orig_trace else None,
            "reconstructed_final_db_hash": recon_trace[-1]["db_hash_after"] if recon_trace else None,
            "reference_trace": orig_trace,
            "reconstructed_trace": recon_trace,
        })
        for k, v in param_counts.items():
            counts[k] = counts.get(k, 0) + v
        print(f"{tid}: strict={strict} params={param_counts} "
              f"exec_ok={exec_ok} final_match={final_state_match}")

    # honest summary: state comparison only counts tasks whose
    # reconstruction resolved EVERY sourced param live (no literal
    # fallback); partial-resolution runs execute a Frankenstein program
    # whose mismatch means nothing.
    meaningful = [t for t in exec_out if not t["used_fallback_literals"]]
    summary = {
        "n_tasks": len(gold_out),
        "gold_replay_ok": sum(t["gold_replay_ok"] for t in gold_out),
        "strict_v1_tasks": sum(t["strict_v1"] for t in gold_out),
        "current_ir_representable": sum(t["current_ir_representable"]
                                        for t in gold_out),
        "param_class_counts": counts,
        "strict_v1_exec_ok": sum(
            1 for g, e in zip(gold_out, exec_out)
            if g["strict_v1"] and e["reconstructed_exec_ok"]),
        "reconstructed_exec_ok": sum(t["reconstructed_exec_ok"]
                                     for t in exec_out),
        "final_state_match": sum(t["final_db_state_match"]
                                 for t in meaningful),
        "final_state_match_denominator": len(meaningful),
    }
    OUT_GOLD.write_text(json.dumps({"summary": summary, "tasks": gold_out},
                                   indent=1, ensure_ascii=False),
                        encoding="utf-8")
    OUT_EXEC.write_text(json.dumps({"summary": summary, "tasks": exec_out},
                                   indent=1, ensure_ascii=False),
                         encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
