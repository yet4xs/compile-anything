"""Phase 9B-2.5 T3: full-pool gold reconstruction v2 (117 candidates).

Implements the preregistration (docs/phase9/gold-v2-preregistration.md):
compile_input_v2 text, parameter classes v2 (DUAL_SOURCE,
SAME_ENTITY_ECHO, COINCIDENTAL_VALUE, LOW_ENTROPY_MATCH added),
task flags (EXPECTED_ERROR_STEP / POLICY_DEPENDENT / REPLAY_ERROR),
equivalence-class acceptable_sources, and strict v2 admission.
No execution here (T6 runs the production E2E).

Run: D:/miniconda3/envs/tau3/python.exe scripts/reconstruct_executable_gold_v2.py
"""

import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
TAU2_SRC = Path("D:/ccfa-2026/tau3-bench-src")
OUT = REPO / "results" / "phase9" / "tau3_gold_compiled_v2.json"

sys.path.insert(0, str(REPO))
from src.eval.compile_input_v2 import render_compile_input

STRICT_CLASSES = {"LITERAL", "DUAL_SOURCE", "runtime_ref",
                  "runtime_ref_multi", "SAME_ENTITY_ECHO"}
LOW_ENTROPY_ENUM = {"economy", "basic", "regular", "false", "true",
                    "yes", "no", "none", " Str"}
POLICY_WORDS = ("policy", "lying", "lie ", "compensation", "compli",
                "refus", "ineligib", "not allowed", "violat", "cheat",
                "trick", "against the rules")
FALLBACK_FAMILY = {
    frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"}),
    frozenset({"exchange_delivered_order_items",
               "modify_pending_order_items"}),
}


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


def entity_prefix(path):
    """Path up to the LAST list index, e.g. orders[2].items[1] ->
    orders[2].items[1]; user_id -> ''."""
    m = re.match(r"^(.*\[\d+\])", path)
    return m.group(1) if m else ""


def is_low_entropy(v):
    s = str(v).strip()
    if s.lower() in ("true", "false"):
        return True
    if s.isdigit() and len(s) <= 2:
        return True
    if s.isalpha() and len(s) <= 3:
        return True
    return s.lower() in LOW_ENTROPY_ENUM


def db_paths_with_value(db, value):
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


def owner_path(path, db, acting_user):
    if acting_user and path.startswith(f"users.{acting_user}."):
        return True
    root = path.split(".")[0].split("[")[0]
    if root in ("orders", "reservations") and acting_user and "." in path:
        key = path.split(".")[1].split("[")[0]
        entity = db.get(root, {}).get(key)
        if isinstance(entity, dict) and \
                str(entity.get("user_id")) == str(acting_user):
            return True
    return False


def classify_value_v2(value, text, prior_outputs, db, acting_user, param):
    sval = str(value).strip()
    in_text = bool(sval) and sval.lower() in text
    cands = []
    if sval:
        for j, flat in enumerate(prior_outputs):
            for p, v in flat.items():
                if str(v) == sval:
                    cands.append({"producer_step": j, "path": p})
        # prefer non-echo candidates: a dict-key segment equal to the
        # sought value (variants.<item_id>.item_id) is an identity index;
        # echoes remain as provenance of last resort
        nonecho = [c for c in cands
                   if not any(seg == sval for seg in c["path"].split(".")
                              if "[" not in seg)]
        cands = nonecho or cands

    if in_text and not cands:
        return "LITERAL", {}, [None]
    if in_text and cands:
        return "DUAL_SOURCE", {"candidates": cands[:6]}, \
            [None] + cands[:6]

    if len(cands) == 1:
        return "runtime_ref", {**cands[0],
                               "selected_by": "unique_candidate"}, [cands[0]]
    if cands:
        # root-scalar origin applies GLOBALLY (later field occurrences
        # are echoes of the tool whose whole output IS the value)
        roots = [c for c in cands if c["path"] == ""]
        if len(roots) == 1 and len(cands) > 1:
            return "runtime_ref", {**roots[0],
                                   "selected_by": "root_scalar_origin",
                                   "all_candidates": cands[:8]}, [roots[0]]
        # entity key = entity prefix ONLY: the same entity echoed by a
        # later step is the same entity regardless of producer step
        keys = {entity_prefix(c["path"]) for c in cands}
        if len(keys) == 1 and next(iter(keys)):
            return "SAME_ENTITY_ECHO", {"candidates": cands[:8]}, cands[:8]
        if len({c["producer_step"] for c in cands}) == 1:
            canon = [c for c in cands if param and _canon(param, c["path"])]
            if len(canon) == 1:
                return "runtime_ref", {**canon[0],
                                       "selected_by": "canonical_field",
                                       "all_candidates": cands[:8]}, [canon[0]]
            if len(canon) > 1:
                ekeys = {entity_prefix(c["path"]) for c in canon}
                if len(ekeys) == 1 and next(iter(ekeys)):
                    return "SAME_ENTITY_ECHO", {
                        "candidates": canon[:8]}, canon[:8]
                depths = {_depth(c["path"]) for c in canon}
                if len(depths) == 1:
                    return "COINCIDENTAL_VALUE", {
                        "candidates": cands[:8],
                        "reason": "same-depth canonical duplicates"}, []
                min_d = min(depths)
                minimal = [c for c in canon if _depth(c["path"]) == min_d]
                if len(minimal) == 1:
                    return "runtime_ref", {
                        **minimal[0],
                        "selected_by": "canonical_minimal_depth",
                        "all_candidates": cands[:8]}, [minimal[0]]
        if is_low_entropy(value):
            return "LOW_ENTROPY_MATCH", {"candidates": cands[:8]}, []
        return "COINCIDENTAL_VALUE", {"candidates": cands[:8]}, []

    hits = db_paths_with_value(db, value) if db is not None else []
    if hits:
        if all(owner_path(h, db, acting_user) for h in hits):
            return "USER_LATENT", {"db_paths": hits[:4]}, []
        return "ENV_STATE", {"db_paths": hits[:4]}, []
    return "UNKNOWN", {}, []


def _canon(param, path):
    last = path.split(".")[-1].split("[")[0]
    stem = param[:-1] if param.endswith("s") else param
    if stem.endswith("_ids"):
        stem = stem[:-4] + "_id"
    base = stem[:-3] if stem.endswith("_id") else stem
    return last in (param, stem, base, base + "s", stem + "s")


def _depth(p):
    return p.count(".") + p.count("[")


def classify_arg_v2(value, text, prior_outputs, db, acting_user, param):
    if isinstance(value, (list, tuple)):
        whole = []
        for j, flat in enumerate(prior_outputs):
            for p, v in flat.items():
                if isinstance(v, (list, tuple)) and \
                        [str(x) for x in v] == [str(x) for x in value]:
                    whole.append({"producer_step": j, "path": p})
        if len(whole) == 1:
            return "runtime_ref", {**whole[0],
                                   "selected_by": "whole_list_match"}, \
                [whole[0]]
        if len(whole) > 1:
            keys = {(c["producer_step"], entity_prefix(c["path"]))
                    for c in whole}
            if len(keys) == 1:
                return "SAME_ENTITY_ECHO", {"candidates": whole[:8]}, whole[:8]
            return "COINCIDENTAL_VALUE", {"candidates": whole[:8]}, []
        elems = [classify_value_v2(e, text, prior_outputs, db, acting_user,
                                   param) for e in value]
        classes = {c for c, _, _ in elems}
        if not classes:
            return "LITERAL", {}, [None]
        if classes == {"LITERAL"}:
            return "LITERAL", {}, [None]
        if classes == {"runtime_ref"}:
            return "runtime_ref_multi", {
                "elements": [i for _, i, _ in elems]}, \
                [i for _, _, s in elems for i in s]
        if classes <= {"runtime_ref", "SAME_ENTITY_ECHO"}:
            return "SAME_ENTITY_ECHO", {
                "elements": [
                    {"class": c, **info} for c, info, _ in elems][:8]}, \
                [i for _, _, s in elems for i in s]
        worst = sorted(classes)[0]
        return worst, {"elements": [
            {"class": c, **info} for c, info, _ in elems][:8]}, []
    c, info, acc = classify_value_v2(value, text, prior_outputs, db,
                                     acting_user, param)
    return c, info, acc


def detect_task_flags(step_meta, actions, purpose):
    errs = [i for i, m in enumerate(step_meta) if not m["ok"]]
    flags = []
    if errs:
        expected = False
        for i in errs:
            fam = {step_meta[i]["tool"]}
            for l in range(i + 1, len(step_meta)):
                if step_meta[l]["ok"]:
                    lt = step_meta[l]["tool"]
                    pair = fam | {lt}
                    if lt in fam or any(pair == f or fam <= f
                                        for f in FALLBACK_FAMILY):
                        expected = True
                        break
            if expected:
                break
        flags.append("EXPECTED_ERROR_STEP" if expected else "REPLAY_ERROR")
    if purpose and any(w in purpose.lower() for w in POLICY_WORDS):
        flags.append("POLICY_DEPENDENT")
    return flags, errs


def main():
    sys.path.insert(0, str(TAU2_SRC / "src"))
    from tau2.data_model.tasks import Task
    from tau2.domains.airline.environment import get_environment as airline_env
    from tau2.domains.retail.environment import get_environment as retail_env
    envs = {"airline": airline_env, "retail": retail_env}

    dbs = {}
    splits = {}
    tasks_by_dom = {}
    for dom in ("retail", "airline"):
        d = TAU2_SRC / "data" / "tau2" / "domains" / dom
        tasks_by_dom[dom] = json.loads((d / "tasks.json").read_text(encoding="utf-8"))
        dbs[dom] = json.loads((d / "db.json").read_text(encoding="utf-8"))
        try:
            splits[dom] = set(json.loads(
                (d / "split_tasks.json").read_text(encoding="utf-8")).get("test", []))
        except Exception:
            splits[dom] = set()

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
        capture_output=True, text=True).stdout.strip()
    sha_in = {f"{dom}/{f}": hashlib.md5(
        (TAU2_SRC / "data" / "tau2" / "domains" / dom / f)
        .read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for dom in ("retail", "airline") for f in ("db.json", "tasks.json")}

    out_tasks = []
    counts = {}
    v1_by_id = {}
    try:
        v1 = json.loads((REPO / "results" / "phase9" /
                         "tau3_gold_compiled_v1.json").read_text(encoding="utf-8"))
        v1_by_id = {t["task_id"]: t for t in v1["tasks"]}
    except Exception:
        pass

    for dom in ("retail", "airline"):
        for raw in tasks_by_dom[dom]:
            actions = (raw.get("evaluation_criteria") or {}).get("actions") or []
            if len(actions) < 2:
                continue
            tid = f"tau3_{dom}_{raw['id']}"
            task = Task.model_validate(raw)
            text = render_compile_input(raw.get("user_scenario")).lower()
            purpose = (raw.get("description") or {}).get("purpose") or ""
            db = dbs[dom]
            acting_user = None
            for a in actions:
                for v in ((a.get("arguments") or {})
                          if isinstance(a, dict)
                          else (a.arguments or {})).values():
                    if isinstance(v, str) and v in db.get("users", {}):
                        acting_user = v
                        break

            env = envs[dom]()
            init = task.initial_state
            env.set_state(
                initialization_data=(init.initialization_data if init else None),
                initialization_actions=(init.initialization_actions if init else None),
                message_history=[], strict=True)
            step_meta, outputs = [], []
            for act in actions:
                name = act["name"] if isinstance(act, dict) else act.name
                args_d = (act.get("arguments") or {}) if isinstance(act, dict) \
                    else (act.arguments or {})
                m = {"tool": name, "ok": True, "error": None}
                flat = {}
                try:
                    obj = env.tools.tools[name](**args_d)
                    flat = dict(walk(obj))
                except Exception as e:
                    m["ok"] = False
                    m["error"] = str(e)[:120]
                step_meta.append(m)
                outputs.append(flat)

            flags, err_idx = detect_task_flags(step_meta, actions, purpose)

            prog, live = [], []
            pc = {}
            for i, act in enumerate(actions):
                name = act["name"] if isinstance(act, dict) else act.name
                args_d = (act.get("arguments") or {}) if isinstance(act, dict) \
                    else (act.arguments or {})
                args = {}
                for param, value in args_d.items():
                    cls, info, acc = classify_arg_v2(
                        value, text, live, db, acting_user, param)
                    pc[cls] = pc.get(cls, 0) + 1
                    args[param] = {"class": cls, "value": value,
                                   "acceptable": [
                                       {"producer_step": a.get("producer_step"),
                                        "path": a.get("path")}
                                       for a in (acc or []) if a],
                                   **info}
                # the step's OWN output must not be among its candidates
                if step_meta[i]["ok"]:
                    live.append(outputs[i])
                prog.append({"step": i, "tool": name, "arguments": args})

            strict = (not err_idx and bool(actions)
                      and "POLICY_DEPENDENT" not in flags
                      and set(pc) <= STRICT_CLASSES)
            for node in prog:
                for a in node["arguments"].values():
                    steps = [a.get("producer_step")] + [
                        e.get("producer_step")
                        for e in a.get("elements", [])]
                    if a["class"] in ("runtime_ref", "runtime_ref_multi") \
                            and any(s is not None and s >= node["step"]
                                    for s in steps):
                        strict = False

            out_tasks.append({
                "task_id": tid, "domain": dom,
                "official_test_split": str(raw["id"]) in splits.get(dom, set()),
                "acting_user": acting_user,
                "replay_ok": not err_idx,
                "error_steps": err_idx,
                "flags": flags,
                "purpose": purpose[:160],
                "param_class_counts": pc,
                "strict_v2": strict,
                "program": prog,
            })
            for k, v in pc.items():
                counts[k] = counts.get(k, 0) + v

    # v1 -> v2 migration matrix over the old 53
    migration = {}
    for t in out_tasks:
        old = v1_by_id.get(t["task_id"])
        if not old:
            continue
        for node, onode in zip(t["program"], old["program"]):
            if onode.get("step") != node["step"]:
                continue
            for p, a in node["arguments"].items():
                oa = (onode.get("arguments") or {}).get(p)
                if not oa:
                    continue
                key = (oa.get("class"), a["class"])
                migration[key] = migration.get(key, 0) + 1

    summary = {
        "meta": {"generated_commit": commit, "input_sha256": sha_in,
                 "tau2_bench_commit": "5ade54e",
                 "preregistration": "3d78f7a"},
        "n_candidates": len(out_tasks),
        "replay_ok": sum(t["replay_ok"] for t in out_tasks),
        "strict_v2": sum(t["strict_v2"] for t in out_tasks),
        "flags": {f: sum(f in t["flags"] for t in out_tasks)
                  for f in ("EXPECTED_ERROR_STEP", "REPLAY_ERROR",
                            "POLICY_DEPENDENT")},
        "param_class_counts": counts,
        "strict_by_domain": {d: sum(t["strict_v2"] for t in out_tasks
                                    if t["domain"] == d)
                             for d in ("retail", "airline")},
        "strict_in_official_test": sum(t["strict_v2"] and
                                       t["official_test_split"]
                                       for t in out_tasks),
        "v1_v2_migration": {f"{k[0]}->{k[1]}": v
                            for k, v in migration.items()},
    }
    OUT.write_text(json.dumps({"summary": summary, "tasks": out_tasks},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "v1_v2_migration"}, indent=1, ensure_ascii=False))
    print("migration:", json.dumps(summary["v1_v2_migration"], indent=1))


if __name__ == "__main__":
    main()
