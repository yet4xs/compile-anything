"""Phase 9B-2.1 Task 6: Rescore existing tau3 predictions with stage-level
failure attribution, split by gold-integrity tier.

For every prediction detail in results/phase9/tau3_real_evaluation.json we
re-run the offline pipeline stages (parse -> skeleton -> lowering -> typed
validation -> canonical binding match) and attribute the FIRST failing stage:

  SYNTAX            raw generation unparseable
  SKELETON_SCHEMA   M1 skeleton JSON structurally invalid
  LOWERING          skeleton valid but lowering failed
  FORWARD_REF       validator USE_BEFORE_DEF
  NONEXISTENT_FIELD validator FIELD_NOT_FOUND
  PARAM_SCHEMA      validator PARAM_NOT_IN_SCHEMA / PARAM_CONFLICT
  OTHER_INVALID     other validator error
  WRONG_TOOL        tool multiset differs from gold
  NO_RUNTIME_BINDING prediction has 0 runtime bindings where gold has >0
                      (model emitted a literal / nothing)
  PARTIAL_BINDING   some bindings present, producer/field wrong
  ALL_STAGES_PASS

Gold-integrity context (from the verified-subset manifest) is attached per
task so the report separates model failures from gold/schema failures:
  tier = VERIFIED | DEGRADED | EXCLUDED_NOT_EXECUTABLE

Output: results/phase9/tau3_evaluation_rescored.json
"""

import json
import re
import sys
from collections import Counter, defaultdict
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

EVAL = REPO / "results" / "phase9" / "tau3_real_evaluation.json"
MANIFEST = REPO / "results" / "phase9" / "tau3_verified_subset_manifest.json"
OUT = REPO / "results" / "phase9" / "tau3_evaluation_rescored.json"


def schemas_for(rec):
    schemas = {}
    for tid, meta in rec["capability_registry"].items():
        schemas[tid] = {
            "params": [{"name": k, "optional": False, "type": v}
                       for k, v in meta.get("input_schema", {}).items()],
            "output_fields": meta.get("output_schema", {}),
            "effect_class": meta.get("effect_class", "read_only")}
    return schemas


def v2_registry(rec):
    return {t: {"input_schema": rec["capability_registry"][t].get("input_schema", {}),
                "output_schema": rec["capability_registry"][t].get("output_schema", {})}
            for t in rec["capability_registry"]}


def attribute(detail, rec):
    """Re-run stages offline; return (stage_failure, info)."""
    raw, arm = detail["raw"], detail["arm"]
    info = {}

    taskir = ""
    if arm == "M0-T":
        taskir = raw
    elif arm == "M0-J":
        try:
            d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            taskir = to_text(module_from_dict(d))
        except Exception as e:
            return "SYNTAX", {"error": f"JSON AST parse: {e}"}
    else:  # M1
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                return "SYNTAX", {"error": "no JSON object in output"}
            sk = skeleton_from_dict(json.loads(m.group(0)))
            errs = hard_errors(validate_skeleton(sk, None, schemas_for(rec)))
            if errs:
                return "SKELETON_SCHEMA", {"errors": errs[:5]}
            low = lower_skeleton(sk, None, schemas_for(rec), strict=True)
            taskir = low.get("taskir") or ""
            if not taskir:
                low_errs = low.get("errors") or []
                # Distinguish nested-path fields the IR cannot represent
                # (root segment exists in producer schema) from genuinely
                # nonexistent fields.
                nested = [e for e in low_errs
                          if "FIELD_NOT_FOUND" in e and
                          ("[" in e or "." in e)]
                plain = [e for e in low_errs
                         if "FIELD_NOT_FOUND" in e and e not in nested]
                info["lowering_errors"] = low_errs[:5]
                if any("not yet defined" in e for e in low_errs):
                    return "FORWARD_REF", info
                if nested and not plain:
                    return "UNREPRESENTABLE_NESTED_FIELD", info
                if "FIELD_NOT_FOUND" in " ".join(low_errs):
                    return "NONEXISTENT_FIELD", info
                return "LOWERING", info
        except Exception as e:
            return "SKELETON_SCHEMA", {"error": str(e)[:200]}

    if not taskir:
        return "SYNTAX", {"error": "empty TaskIR"}
    try:
        mod = parse_text(taskir)
    except Exception as e:
        return "SYNTAX", {"error": f"TaskIR parse: {str(e)[:150]}"}

    rep = validate(mod, capability_registry=v2_registry(rec))
    if not rep.valid:
        codes = [i.code for i in rep.errors]
        primary = codes[0] if codes else "UNKNOWN"
        info["validator_codes"] = codes[:5]
        if primary == "USE_BEFORE_DEF":
            return "FORWARD_REF", info
        if primary == "FIELD_NOT_FOUND":
            return "NONEXISTENT_FIELD", info
        if primary in ("PARAM_NOT_IN_SCHEMA", "PARAM_CONFLICT"):
            return "PARAM_SCHEMA", info
        return "OTHER_INVALID", info

    gold = sorted(rec.get("gold_capability_ids", []))
    pred_tools = sorted({n.params.get("action", "") for n in mod.program.nodes
                         if n.op == "EXEC_ACTION" and n.params.get("action")})
    if pred_tools != gold:
        info["gold_tools"], info["pred_tools"] = gold, pred_tools
        return "WRONG_TOOL", info

    ref = canonical_from_skeleton(rec["target_skeleton"]["nodes"])
    pred = canonical_from_taskir(taskir)
    if not pred:
        return "SYNTAX", {"error": "canonical extraction from TaskIR failed"}
    a = canonical_call_binding_match(ref, pred)
    info["n_ref_bindings"] = a["n_ref_bindings"]
    info["n_pred_bindings"] = a["n_pred_bindings"]
    info["n_correct"] = a["n_correct_bindings"]
    if a["n_ref_bindings"] > 0 and a["n_pred_bindings"] == 0:
        return "NO_RUNTIME_BINDING", info
    if a["bindings_match"] and a["calls_match"]:
        return "ALL_STAGES_PASS", info
    return "PARTIAL_BINDING", info


def main():
    ev = json.loads(EVAL.read_text(encoding="utf-8"))
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tier_by_id = {t["task_id"]: t["tier"] for t in man["tasks"]}
    unans_by_id = {t["task_id"]: t["n_unanswerable_literals"] for t in man["tasks"]}

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
        stage, info = attribute(d, rec)
        rows.append({
            "task_id": d["task_id"], "arm": d["arm"],
            "tier": tier_by_id[d["task_id"]],
            "task_gold_unanswerable_literals": unans_by_id[d["task_id"]],
            "truncated_at_token_cap": d.get("gen_tokens", 0) >= 698,
            "gen_tokens": d.get("gen_tokens", 0),
            "reported": {k: d.get(k) for k in
                         ("parse", "validator", "binding_exact", "tool_match",
                          "execution")},
            "stage_failure": stage,
            "info": info,
        })

    def agg(sel):
        out = {}
        for arm in ("M0-T", "M0-J", "M1"):
            sub = [r for r in rows if r["arm"] == arm and sel(r)]
            out[arm] = {
                "n": len(sub),
                "stages": dict(Counter(r["stage_failure"] for r in sub)),
            }
        return out

    summary = {
        "n_details": len(rows),
        "all_tasks": agg(lambda r: True),
        "verified_only": agg(lambda r: r["tier"] == "VERIFIED"),
        "tier_distribution": dict(Counter(r["tier"] for r in rows)),
        "verified_task_ids": sorted({r["task_id"] for r in rows
                                     if r["tier"] == "VERIFIED"}),
    }
    OUT.write_text(json.dumps({"summary": summary, "details": rows}, indent=1,
                              ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary["tier_distribution"]))
    print("== ALL ==", json.dumps(summary["all_tasks"], indent=1))
    print("== VERIFIED ONLY ==")
    print(json.dumps(summary["verified_only"], indent=1))


if __name__ == "__main__":
    main()
