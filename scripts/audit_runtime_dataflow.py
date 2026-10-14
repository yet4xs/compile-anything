"""Phase 9A-0: Real Runtime Dataflow Feasibility Audit.

Surveys real tool-use datasets for runtime value dependencies (Level 3):
cases where a downstream tool's parameter value comes from an upstream
tool's OUTPUT, not from the user's input text.

Level classification:
  0: single tool call
  1: multi-tool, no data dependency
  2: order dependency only (B must run after A but no value flows)
  3: runtime VALUE dependency (B consumes a value produced by A)

Prohibited: counting values visible in user text as runtime deps.
"""
import json, os, sys, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "results/phase9"


def classify_value_dependency(tool_calls, user_text):
    """Classify a trajectory's dependency level.

    tool_calls: list of {name, arguments} in execution order
    user_text: the natural language instruction

    Returns (level, detail) where level is 0-3.
    """
    if len(tool_calls) <= 1:
        return 0, "single_call"

    # Track all output values we've seen (simulated)
    # Since we don't have actual outputs, we check if any tool's INPUT
    # parameter value is NOT present in the user text but COULD come from
    # another tool's typical output
    user_lower = user_text.lower() if user_text else ""

    has_data_dep = False
    has_order_dep = False
    runtime_deps = []

    for i, call in enumerate(tool_calls):
        args = call.get("arguments") or {}
        for param_name, param_val in args.items():
            val_str = str(param_val).strip()
            if not val_str or val_str in ("null", "None", "{}", "[]"):
                continue
            # Check if value is NOT in user text
            val_lower = val_str.lower()
            in_user = val_lower in user_lower

            if not in_user and i > 0:
                # This value is not from user input — could be from upstream tool
                # Check if any earlier tool could have produced it
                for j in range(i):
                    prev = tool_calls[j]
                    prev_name = prev.get("name", "")
                    # Heuristic: if the value looks like an ID/hash/UUID that
                    # a tool would produce
                    is_id_like = bool(re.match(
                        r"^[A-Z]{1,3}[-_]?[A-Za-z0-9_-]{4,}$", val_str))
                    is_uuid_like = bool(re.match(
                        r"^[a-f0-9]{8}-[a-f0-9]{4}", val_str))
                    is_numeric_id = bool(re.match(r"^\d{4,}$", val_str))
                    is_email = "@" in val_str
                    is_url = val_str.startswith("http")

                    if is_id_like or is_uuid_like or is_numeric_id or is_email or is_url:
                        has_data_dep = True
                        runtime_deps.append({
                            "call_idx": i, "param": param_name,
                            "value": val_str[:30],
                            "possible_producer": prev_name,
                        })

    if has_data_dep:
        return 3, runtime_deps

    # Check for order dependency (same tools called in sequence)
    if len(tool_calls) > 1:
        has_order_dep = True  # any multi-call has at least ordering
        return 2, "multi_call_ordered"

    return 1, "multi_call_independent"


def audit_tau3():
    """Audit τ³-bench for runtime dependencies."""
    path = "runs/phase5b1/tau3_preds_2048.jsonl"
    if not os.path.exists(path):
        return {"error": "not found"}

    stats = Counter()
    level3_details = []
    total = 0

    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        total += 1
        ref_actions = r.get("ref_action_names") or []
        instruction = r.get("instruction", "")

        if len(ref_actions) <= 1:
            stats["L0_single"] += 1
            continue

        # For τ³, we don't have per-call arguments, only names
        # We check if the task description implies data flow
        # (e.g., "find the account, then check its transactions")
        # This is an approximation
        if len(ref_actions) > 1:
            # Check for implied data flow patterns
            has_flow_pattern = bool(re.search(
                r"(?:its|their|the|this|that)\s+(?:account|transaction|"
                r"reservation|booking|order|user|customer|invoice|payment)",
                instruction, re.I))
            if has_flow_pattern:
                stats["L3_likely_runtime_dep"] += 1
                level3_details.append({
                    "case_id": r.get("case_id"),
                    "n_actions": len(ref_actions),
                    "actions": ref_actions[:5],
                    "instruction_snippet": instruction[:120],
                    "note": "implied data flow from NL patterns; "
                             "arguments not available for verification",
                })
            else:
                stats["L2_ordered"] += 1
        else:
            stats["L1_multi"] += 1

    return {
        "source": "tau3_bench",
        "total_tasks": total,
        "level_distribution": dict(stats),
        "level3_count": stats.get("L3_likely_runtime_dep", 0),
        "level3_details_sample": level3_details[:10],
        "note": "τ³ stores reference action NAMES only, not arguments. "
                "Runtime dependency classification is heuristic based on "
                "NL patterns. Precise Level 3 verification requires "
                "per-call arguments.",
    }


def audit_bfcl():
    """Audit BFCL V4 for runtime dependencies."""
    path = "data/bfcl_training/bfcl_train.jsonl"
    if not os.path.exists(path):
        # Try alternate paths
        for alt in ["data/bfcl_training", "data/raw/bfcl"]:
            if os.path.exists(alt):
                path = os.path.join(alt, "bfcl_train.jsonl")
                break
    if not os.path.exists(path):
        return {"error": "not found"}

    stats = Counter()
    level3_details = []
    total = 0

    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        total += 1
        messages = r.get("messages", [])
        user_text = ""
        tool_calls = []

        for msg in messages:
            if msg.get("role") == "user":
                user_text += msg.get("content", "") + " "
            elif msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    tool_calls.append({"name": fn.get("name", ""),
                                       "arguments": args})

        if not tool_calls:
            stats["no_tool_calls"] += 1
            continue

        level, detail = classify_value_dependency(tool_calls, user_text)
        stats[f"L{level}"] += 1
        if level == 3:
            level3_details.append({
                "n_calls": len(tool_calls),
                "tools": [c["name"][:30] for c in tool_calls[:5]],
                "deps": detail[:3] if isinstance(detail, list) else detail,
            })

    return {
        "source": "bfcl_v4",
        "total": total,
        "level_distribution": dict(stats),
        "level3_count": stats.get("L3", 0),
        "level3_details_sample": level3_details[:10],
    }


def audit_toolbench():
    """Audit ToolBench for runtime dependencies."""
    results = {}
    for split in ("in_domain", "out_of_domain"):
        path = f"data/raw/toolbench_static/{split}.jsonl"
        if not os.path.exists(path):
            results[split] = {"error": "not found"}
            continue

        stats = Counter()
        level3_details = []
        total = 0

        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            messages = r.get("messages", [])
            user_text = ""
            tool_calls = []

            for msg in messages:
                if msg.get("role") == "user":
                    user_text += msg.get("content", "") + " "
                elif msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls", []):
                        fn = tc.get("function", {})
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                        except Exception:
                            args = {}
                        tool_calls.append({"name": fn.get("name", ""),
                                           "arguments": args})

            if not tool_calls:
                stats["no_tool_calls"] += 1
                continue

            level, detail = classify_value_dependency(tool_calls, user_text)
            stats[f"L{level}"] += 1
            if level == 3:
                level3_details.append({
                    "n_calls": len(tool_calls),
                    "tools": [c["name"][:30] for c in tool_calls[:5]],
                    "deps": detail[:3] if isinstance(detail, list) else detail,
                })

        results[split] = {
            "source": f"toolbench_{split}",
            "total": total,
            "level_distribution": dict(stats),
            "level3_count": stats.get("L3", 0),
            "level3_details_sample": level3_details[:5],
        }
    return results


def audit_xlam():
    """Audit xLAM function calling for runtime dependencies."""
    # xLAM data is in compiler_corpus but let's check raw
    path = "data/raw/xlam"
    if not os.path.exists(path):
        return {"error": "not found"}

    # Look for the actual data files
    data_files = [f for f in os.listdir(path) if f.endswith(".jsonl")]
    if not data_files:
        return {"error": "no jsonl files"}

    results = {}
    for fname in data_files[:3]:  # sample up to 3 files
        fpath = os.path.join(path, fname)
        stats = Counter()
        level3_details = []
        total = 0

        for line in open(fpath, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            # xLAM format varies; try common fields
            user_text = r.get("question", r.get("query", r.get("task", "")))
            tool_calls = r.get("tool_calls", r.get("answers", []))

            if not tool_calls:
                stats["no_tool_calls"] += 1
                continue

            # Normalize to {name, arguments}
            norm_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    norm_calls.append({
                        "name": tc.get("name", tc.get("tool", "")),
                        "arguments": tc.get("arguments",
                                             tc.get("parameters", {}))})
                elif isinstance(tc, str):
                    norm_calls.append({"name": tc, "arguments": {}})

            if not norm_calls:
                stats["no_tool_calls"] += 1
                continue

            level, detail = classify_value_dependency(norm_calls, user_text)
            stats[f"L{level}"] += 1
            if level == 3:
                level3_details.append({
                    "n_calls": len(norm_calls),
                    "tools": [c["name"][:30] for c in norm_calls[:5]],
                })

        results[fname] = {
            "source": f"xlam_{fname}",
            "total": total,
            "level_distribution": dict(stats),
            "level3_count": stats.get("L3", 0),
            "level3_details_sample": level3_details[:5],
        }
    return results


def main():
    os.makedirs(OUT, exist_ok=True)

    audits = {}
    print("=== Phase 9A-0: Runtime Dataflow Feasibility Audit ===\n")

    print("1. τ³-bench...")
    audits["tau3"] = audit_tau3()
    print(f"   total={audits['tau3'].get('total_tasks', '?')} "
          f"L3≈{audits['tau3'].get('level3_count', 0)}")

    print("2. BFCL V4...")
    audits["bfcl"] = audit_bfcl()
    if "error" not in audits["bfcl"]:
        print(f"   total={audits['bfcl'].get('total', '?')} "
              f"L3={audits['bfcl'].get('level3_count', 0)}")
    else:
        print(f"   {audits['bfcl']}")

    print("3. ToolBench...")
    audits["toolbench"] = audit_toolbench()
    for split, r in audits["toolbench"].items():
        if "error" not in r:
            print(f"   {split}: total={r.get('total', '?')} "
                  f"L3={r.get('level3_count', 0)}")
        else:
            print(f"   {split}: {r}")

    print("4. xLAM...")
    audits["xlam"] = audit_xlam()
    for fname, r in audits["xlam"].items():
        if "error" not in r:
            print(f"   {fname}: total={r.get('total', '?')} "
                  f"L3={r.get('level3_count', 0)}")
        else:
            print(f"   {fname}: {r}")

    # Summary
    print("\n=== SUMMARY ===")
    total_l3 = sum(a.get("level3_count", 0) for a in
                   [audits.get("tau3", {}), audits.get("bfcl", {})]
                   + list(audits.get("toolbench", {}).values())
                   + list(audits.get("xlam", {}).values())
                   if isinstance(a, dict) and "error" not in a)

    output = {
        "audits": audits,
        "summary": {
            "total_level3_found": total_l3,
            "answer_Q1": "See individual audits above",
            "answer_Q2": "To be determined based on L3 volume",
            "answer_Q3": "To be determined",
        }
    }

    with open(f"{OUT}/runtime_dataflow_audit.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nsaved {OUT}/runtime_dataflow_audit.json")
    print("AUDIT-DONE")


if __name__ == "__main__":
    main()
