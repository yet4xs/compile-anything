"""Phase 9B-2.2 Task 0.6 / Task 5: deterministic CPU scoring entry point.

Re-scores saved model predictions (raw generations) against the repaired,
evidence-backed gold. Stages are scored SEPARATELY (audit requirement):

  json_syntax      a JSON object was extractable (M0-J / M1)
  skeleton_valid   M1 skeleton passed structural validation
  lowering_ok      M1 lowering produced TaskIR
  taskir_valid     TaskIR parsed AND passed typed validation

Nested-field classification is based on the FIELD PATH SYNTAX and the
producer's REAL observed output fields — never on brackets appearing
elsewhere in the error message (the 9B-2.1 bug):

  GENUINE_NESTED_ACCESS   field path contains '[' or '.' AND its root
                          segment exists in the producer's real output
  NONEXISTENT_FIELD       field (or its root) absent from real output

Usage:
  python scripts/score_tau3_offline.py [--predictions results/phase9/tau3_real_evaluation.json]

Outputs results/phase9/tau3_m1_failure_reclassified.json (M1 deep metrics)
and prints stage-separated summaries for all arms.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ir.parser import parse_text
from src.ir.taskir import to_text, module_from_dict
from src.validator.validator import validate
from src.compiler.plan_skeleton import skeleton_from_dict, validate_skeleton, hard_errors
from src.compiler.skeleton_lowering import lower_skeleton
from src.eval.canonical_dataflow_matcher import (
    canonical_from_skeleton, canonical_from_taskir, canonical_call_binding_match)

FIELD_IN_ERR = re.compile(r"EXTRACT_FIELD '([^']+)'")


def schemas_for(rec):
    return {tid: {
        "params": [{"name": k, "optional": False, "type": v}
                   for k, v in meta.get("input_schema", {}).items()],
        "output_fields": meta.get("output_schema", {}),
        "effect_class": meta.get("effect_class", "read_only")}
        for tid, meta in rec["capability_registry"].items()}


def v2_registry(rec):
    return {t: {"input_schema": rec["capability_registry"][t].get("input_schema", {}),
                "output_schema": rec["capability_registry"][t].get("output_schema", {})}
            for t in rec["capability_registry"]}


def field_is_nested(field: str) -> bool:
    """Nested access is a property of the FIELD PATH syntax only."""
    return "[" in field or "." in field


def root_of(field: str) -> str:
    return field.split("[")[0].split(".")[0]


def classify_field_errors(low_errors, real_fields_by_tool, tool_of_node):
    """Split FIELD_NOT_FOUND lowering errors into genuine nested access vs
    nonexistent field using the field path itself + real output fields."""
    nested, plain = [], []
    for e in low_errors:
        m = FIELD_IN_ERR.search(e)
        if not m or "FIELD_NOT_FOUND" not in e:
            continue
        field = m.group(1)
        # find the producer tool named in the error to get its real fields
        tool = None
        for t, fields in real_fields_by_tool.items():
            if t in e:
                tool = t
                break
        real_roots = real_fields_by_tool.get(tool, set())
        if field_is_nested(field) and root_of(field) in real_roots:
            nested.append(field)
        else:
            plain.append(field)
    return nested, plain


def score_detail(detail, rec, repaired_task, real_fields_by_tool):
    """Stage-separated scoring of one prediction. Returns dict of stages +
    failure class using first-failing-stage semantics."""
    raw, arm = detail["raw"], detail["arm"]
    out = {"json_syntax": None, "skeleton_valid": None, "lowering_ok": None,
           "taskir_valid": None, "failure": None, "info": {}}

    taskir = ""
    if arm == "M0-T":
        taskir = raw
        try:
            parse_text(taskir)
            out["taskir_valid_parse"] = True
        except Exception:
            pass
    elif arm == "M0-J":
        try:
            d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            out["json_syntax"] = True
            taskir = to_text(module_from_dict(d))
        except Exception as e:
            out["json_syntax"] = False
            out["failure"] = "JSON_SYNTAX"
            out["info"]["error"] = str(e)[:150]
            return out
    else:  # M1
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            out["json_syntax"] = False
            out["failure"] = "JSON_SYNTAX"
            return out
        out["json_syntax"] = True
        try:
            sk = skeleton_from_dict(json.loads(m.group(0)))
        except Exception as e:
            out["skeleton_valid"] = False
            out["failure"] = "SKELETON_SCHEMA"
            out["info"]["error"] = str(e)[:150]
            return out
        errs = hard_errors(validate_skeleton(sk, None, schemas_for(rec)))
        if errs:
            out["skeleton_valid"] = False
            out["failure"] = "SKELETON_SCHEMA"
            out["info"]["errors"] = errs[:5]
            return out
        out["skeleton_valid"] = True
        low = lower_skeleton(sk, None, schemas_for(rec), strict=True)
        taskir = low.get("taskir") or ""
        if not taskir:
            out["lowering_ok"] = False
            low_errs = low.get("errors") or []
            out["info"]["lowering_errors"] = low_errs[:5]
            if any("not yet defined" in e for e in low_errs):
                out["failure"] = "FORWARD_REF"
                return out
            nested, plain = classify_field_errors(
                low_errs, real_fields_by_tool, None)
            if nested and not plain:
                out["failure"] = "GENUINE_NESTED_ACCESS"
                out["info"]["nested_fields"] = nested
                return out
            if nested or plain:
                out["failure"] = "NONEXISTENT_FIELD"
                out["info"]["nested_fields"] = nested
                out["info"]["nonexistent_fields"] = plain
                return out
            out["failure"] = "LOWERING_OTHER"
            return out
        out["lowering_ok"] = True

    if not taskir:
        out["failure"] = "SYNTAX_EMPTY"
        return out
    try:
        mod = parse_text(taskir)
    except Exception as e:
        out["failure"] = "TASKIR_SYNTAX"
        out["info"]["error"] = f"parse: {str(e)[:120]}"
        return out
    rep = validate(mod, capability_registry=v2_registry(rec))
    out["taskir_valid"] = rep.valid
    if not rep.valid:
        codes = [i.code for i in rep.errors]
        out["info"]["validator_codes"] = codes[:5]
        primary = codes[0] if codes else "UNKNOWN"
        out["failure"] = {"USE_BEFORE_DEF": "FORWARD_REF",
                          "FIELD_NOT_FOUND": "NONEXISTENT_FIELD",
                          "PARAM_NOT_IN_SCHEMA": "PARAM_SCHEMA",
                          "PARAM_CONFLICT": "PARAM_SCHEMA"}.get(
                              primary, "OTHER_INVALID")
        return out

    gold = sorted(rec.get("gold_capability_ids", []))
    pred_tools = sorted({n.params.get("action", "") for n in mod.program.nodes
                         if n.op == "EXEC_ACTION" and n.params.get("action")})
    if pred_tools != gold:
        out["failure"] = "WRONG_TOOL"
        out["info"]["gold_tools"], out["pred_tools"] = gold, pred_tools
        return out

    ref = canonical_from_skeleton(rec["target_skeleton"]["nodes"])
    pred = canonical_from_taskir(taskir)
    if pred:
        a = canonical_call_binding_match(ref, pred)
        out["info"]["binding"] = {
            "n_ref": a["n_ref_bindings"], "n_pred": a["n_pred_bindings"],
            "n_correct": a["n_correct_bindings"],
            "extra": a["extra_bindings"]}
        if a["bindings_match"] and a["calls_match"]:
            out["failure"] = "PASS"
        elif a["n_ref_bindings"] > 0 and a["n_pred_bindings"] == 0:
            out["failure"] = "NO_RUNTIME_BINDING"
        else:
            out["failure"] = "PARTIAL_BINDING"
    else:
        out["failure"] = "TASKIR_SYNTAX"
    return out


def m1_deep_metrics(detail, rec, repaired_task, real_fields_by_tool):
    """Advisor-required M1 metrics: count runtime refs and their quality
    against the repaired gold."""
    raw = detail["raw"]
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"json_syntax_pass": False}
    try:
        sk = skeleton_from_dict(json.loads(m.group(0)))
    except Exception:
        return {"json_syntax_pass": False}

    nodes = sk.nodes if hasattr(sk, "nodes") else sk["nodes"]
    node_list = [n.__dict__ if hasattr(n, "__dict__") else n for n in nodes] \
        if not isinstance(nodes, list) or hasattr(nodes[0], "__dict__") else nodes

    deps = repaired_task["deps"] if repaired_task else []
    by_consumer = {(d["consumer"]["step"], d["consumer"]["tool"],
                    d["consumer"]["param"]): d for d in deps}

    n_rt, n_valid_producer, n_correct_field = 0, 0, 0
    n_forward, n_nonexistent, n_nested = 0, 0, 0
    producer_tools = {n.get("capability_id"): n.get("id")
                      for n in node_list}
    for i, node in enumerate(node_list):
        args = node.get("arguments", {})
        for param, v in args.items():
            if not (isinstance(v, dict) and v.get("kind") == "runtime_ref"):
                continue
            n_rt += 1
            producer, field = v.get("producer", ""), v.get("field", "")
            # producer exists and is defined before this node?
            pid = producer
            ids_before = [n2.get("id") for j, n2 in enumerate(node_list)
                          if j < i]
            if pid not in producer_tools.values():
                n_nonexistent += 1
                continue
            if pid not in ids_before:
                n_forward += 1
                continue
            n_valid_producer += 1
            # field quality vs repaired gold for this consumer
            d = by_consumer.get((i, node.get("capability_id"), param))
            if d and d["verdict"] in ("LABEL_EXACT", "LABEL_EXACT_ROOT_SCALAR",
                                      "REPAIRED_UNIQUE"):
                if d["repaired_to"]:
                    expected = d["repaired_to"]["path"]
                else:
                    # label verified as-is: expected field is the frozen
                    # label (root-scalar tools carry a nominal field name)
                    expected = d["original_label"]["producer_field"]
                if expected == "" and not field_is_nested(field):
                    n_correct_field += 1
                elif field == expected:
                    n_correct_field += 1
            # genuine nested access: path syntax + root in real output
            ptool = None
            for n2 in node_list:
                if n2.get("id") == pid:
                    ptool = n2.get("capability_id")
            if field_is_nested(field) and \
                    root_of(field) in real_fields_by_tool.get(ptool, set()):
                n_nested += 1

    return {
        "json_syntax_pass": True,
        "n_valid_runtime_refs": n_rt,
        "n_valid_producer": n_valid_producer,
        "n_correct_output_field": n_correct_field,
        "n_forward_reference": n_forward,
        "n_nonexistent_producer": n_nonexistent,
        "n_genuine_nested_access": n_nested,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",
                    default="results/phase9/tau3_real_evaluation.json")
    ap.add_argument("--out",
                    default="results/phase9/tau3_m1_failure_reclassified.json")
    args = ap.parse_args()

    ev = json.loads((REPO / args.predictions).read_text(encoding="utf-8"))
    repaired = json.loads((REPO / "results/phase9/tau3_gold_repaired.json")
                          .read_text(encoding="utf-8"))
    rep_by_id = {t["task_id"]: t for t in repaired["tasks"]}
    manifest = json.loads(
        (REPO / "results/phase9/tau3_gold_verified_manifest.json")
        .read_text(encoding="utf-8"))
    tier_by_id = {t["task_id"]: t["tier"] for t in manifest["tasks"]}

    # real top-level output fields per tool from replay observations
    real_fields = {}
    for key, fields in repaired["real_schema_observations"].items():
        dom, tool = key.rsplit(".", 1)
        real_fields[f"{dom}.{tool}"] = set(fields) | {tool}
    # map bare tool name -> union of real fields across domains (errors name
    # the bare tool)
    real_fields_by_tool = {}
    for key, fields in repaired["real_schema_observations"].items():
        tool = key.rsplit(".", 1)[1]
        real_fields_by_tool.setdefault(tool, set()).update(fields)

    recs = {}
    for split in ("train", "dev", "test"):
        with open(REPO / "data" / "tau3_runtime_benchmark" / f"{split}.jsonl",
                  encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                recs[t["task_id"]] = t

    rows = []
    for d in ev["details"]:
        rec = recs[d["task_id"]]
        rt = rep_by_id[d["task_id"]]
        s = score_detail(d, rec, rt, real_fields_by_tool)
        row = {"task_id": d["task_id"], "arm": d["arm"],
               "tier": tier_by_id[d["task_id"]],
               "hit_generation_cap": d.get("gen_tokens", 0) >= 698,
               "raw_log_truncated": len(d.get("raw", "")) >= 2000,
               **s}
        if d["arm"] == "M1":
            row["m1_metrics"] = m1_deep_metrics(d, rec, rt, real_fields_by_tool)
        rows.append(row)

    def agg(sel):
        out = {}
        for arm in ("M0-T", "M0-J", "M1"):
            sub = [r for r in rows if r["arm"] == arm and sel(r)]
            out[arm] = {"n": len(sub),
                        "failures": dict(Counter(
                            r["failure"] for r in sub))}
        return out

    m1 = [r for r in rows if r["arm"] == "M1"]
    m1_summary = {
        "n": len(m1),
        "json_syntax_pass": sum(bool(r.get("json_syntax")) for r in m1),
        "skeleton_valid": sum(bool(r.get("skeleton_valid")) for r in m1),
        "failures": dict(Counter(r["failure"] for r in m1)),
        "totals": {
            k: sum(r["m1_metrics"].get(k, 0) for r in m1)
            for k in ("n_valid_runtime_refs", "n_valid_producer",
                      "n_correct_output_field", "n_forward_reference",
                      "n_nonexistent_producer", "n_genuine_nested_access")
        },
        "by_tier": {tier: dict(Counter(r["failure"] for r in m1
                                       if r["tier"] == tier))
                    for tier in {r["tier"] for r in m1}},
    }

    summary = {
        "all_tasks": agg(lambda r: True),
        "binding_exact_only": agg(lambda r: r["tier"] == "BINDING_EXACT"),
        "m1": m1_summary,
        "tier_distribution": dict(Counter(r["tier"] for r in rows)),
    }
    (REPO / args.out).write_text(json.dumps(
        {"summary": summary, "details": rows}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
