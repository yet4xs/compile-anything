"""Phase 9B-2.1 Tasks 2-3: Gold provenance audit + runtime dep verification.

Audits every literal argument and runtime_ref in tau3_runtime_benchmark
to determine if the Gold labels are actually derivable from user-visible
information and prior tool outputs, or require hidden environment state.

Also verifies runtime_dep labels by checking value consistency between
producer inputs and consumer arguments.
"""
import json, os, sys, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

TAU3 = "third_party/tau3-bench/data/tau2/domains"
DATA = "data/tau3_runtime_benchmark"
OUT = "results/phase9"


def load_all_tasks():
    tasks = []
    for split in ("train", "dev", "test"):
        path = os.path.join(DATA, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def load_tau3_original(domain):
    path = os.path.join(TAU3, domain, "tasks.json")
    return json.load(open(path, encoding="utf-8"))


def audit_task_literals(task):
    """For every argument in every node, classify its source."""
    nodes = task["target_skeleton"]["nodes"]
    instr = task.get("instruction", "")
    if isinstance(instr, dict):
        instruction = " ".join(str(v) for v in instr.values()).lower()
    else:
        instruction = str(instr).lower()
    issues = []
    sources = []

    for i, node in enumerate(nodes):
        for param, value in node["arguments"].items():
            if isinstance(value, dict) and value.get("kind") == "runtime_ref":
                sources.append({
                    "node": node["id"], "param": param,
                    "source": "runtime_ref",
                    "producer": value.get("producer"),
                    "field": value.get("field"),
                })
                continue

            val_str = str(value).strip()
            if not val_str:
                continue

            # Check if value appears in user instruction
            if val_str.lower() in instruction:
                sources.append({
                    "node": node["id"], "param": param,
                    "source": "USER_TEXT",
                    "value": val_str[:30],
                })
            else:
                # Not in user text — could be env state or prior tool output
                issues.append({
                    "node": node["id"], "param": param,
                    "source": "NOT_IN_USER_TEXT",
                    "value": val_str[:40],
                    "tool": node["capability_id"],
                    "step": i,
                })
                sources.append({
                    "node": node["id"], "param": param,
                    "source": "ENV_STATE_OR_UNKNOWN",
                    "value": val_str[:30],
                })

    return sources, issues


def audit_runtime_deps(task):
    """Verify runtime_ref labels by checking value consistency."""
    nodes = task["target_skeleton"]["nodes"]
    instr = task.get("instruction", "")
    if isinstance(instr, dict):
        instruction = " ".join(str(v) for v in instr.values()).lower()
    else:
        instruction = str(instr).lower()
    verified = []
    refuted = []
    unknown = []

    node_by_id = {n["id"]: n for n in nodes}

    for i, node in enumerate(nodes):
        for param, value in node["arguments"].items():
            if not (isinstance(value, dict) and value.get("kind") == "runtime_ref"):
                continue

            producer_id = value.get("producer", "")
            field = value.get("field", "")
            producer = node_by_id.get(producer_id)

            if producer is None:
                refuted.append({
                    "node": node["id"], "param": param,
                    "reason": "producer_not_found",
                    "producer_id": producer_id,
                })
                continue

            producer_tool = producer["capability_id"]

            # Check if producer's output_schema actually has this field
            cap_registry = task.get("capability_registry", {})
            producer_schema = cap_registry.get(producer_tool, {})
            output_fields = producer_schema.get("output_schema", {})

            if field not in output_fields:
                # Field name not in declared output schema
                # Check heuristic matches (param name = field name)
                if param.rstrip('_id') in field.lower() or field.rstrip('_id') in param.lower():
                    unknown.append({
                        "node": node["id"], "param": param,
                        "reason": f"field '{field}' not in {producer_tool} declared output_schema "
                                   f"(has: {list(output_fields.keys())}); heuristic match only",
                        "producer": producer_tool,
                    })
                else:
                    refuted.append({
                        "node": node["id"], "param": param,
                        "reason": f"field '{field}' not in {producer_tool} output_schema "
                                   f"(has: {list(output_fields.keys())})",
                        "producer": producer_tool,
                    })
                continue

            # Check value consistency: producer's OWN arguments vs consumer's expected value
            # If producer was called with X and consumer expects Y≠X, and the field
            # is an echo field (like id), that's suspicious
            producer_args = producer.get("arguments", {})
            for p_param, p_value in producer_args.items():
                if isinstance(p_value, dict):
                    continue
                p_str = str(p_value).strip()
                # If producer's input param name matches the output field name
                # and it's an ID-like field, the output should echo the input
                if p_param.rstrip('_id') == field.rstrip('_id'):
                    # We can't verify without actual execution — mark as UNKNOWN
                    # unless the value is clearly echoed
                    unknown.append({
                        "node": node["id"], "param": param,
                        "reason": f"potential echo field '{field}' from {producer_tool}'s "
                                   f"input '{p_param}'='{p_str[:20]}'; needs runtime verification",
                        "producer": producer_tool,
                    })
                    break
            else:
                # No matching input param — the field is computed by the tool
                verified.append({
                    "node": node["id"], "param": param,
                    "producer": producer_tool,
                    "field": field,
                    "confidence": "declared_in_schema",
                })

    return verified, refuted, unknown


def check_hidden_env_values(task):
    """Identify literal values that the model cannot know from instruction
    or tool outputs — these require hidden environment state."""
    nodes = task["target_skeleton"]["nodes"]
    instr = task.get("instruction", "")
    if isinstance(instr, dict):
        instruction = " ".join(str(v) for v in instr.values()).lower()
    else:
        instruction = str(instr).lower()
    hidden = []

    for i, node in enumerate(nodes):
        for param, value in node["arguments"].items():
            if isinstance(value, dict):
                continue
            val_str = str(value).strip()
            if not val_str:
                continue
            if val_str.lower() not in instruction:
                # Check if any prior tool could produce this
                prior_tools = [n["capability_id"] for n in nodes[:i]]
                cap_registry = task.get("capability_registry", {})
                could_produce = False
                for pt in prior_tools:
                    outs = cap_registry.get(pt, {}).get("output_schema", {})
                    if param.rstrip('_id') in ' '.join(outs.keys()).lower():
                        could_produce = True
                        break

                hidden.append({
                    "node": node["id"], "step": i,
                    "param": param, "value": val_str[:40],
                    "tool": node["capability_id"],
                    "could_be_tool_output": could_produce,
                    "severity": "HIGH" if i == 0 else
                               "MEDIUM" if not could_produce else "LOW",
                })

    return hidden


def main():
    tasks = load_all_tasks()
    print(f"Auditing {len(tasks)} tau3 benchmark tasks\n")

    results = {
        "total_tasks": len(tasks),
        "literal_observability": {},
        "runtime_dep_verification": {},
        "hidden_env_values": {},
        "task_details": [],
    }

    all_sources = Counter()
    all_dep_status = Counter()
    all_hidden = Counter()
    tasks_with_hidden = 0
    tasks_with_refuted = 0

    for task in tasks:
        tid = task["task_id"]

        # Task 2: literal observability
        sources, issues = audit_task_literals(task)
        for s in sources:
            all_sources[s["source"]] += 1

        # Task 3: runtime dep verification
        verified, refuted, unknown = audit_runtime_deps(task)
        for v in verified:
            all_dep_status["VERIFIED"] += 1
        for r in refuted:
            all_dep_status["REFUTED"] += 1
        for u in unknown:
            all_dep_status["UNKNOWN"] += 1

        # Hidden env values
        hidden = check_hidden_env_values(task)
        high_hidden = [h for h in hidden if h["severity"] == "HIGH"]
        if hidden:
            tasks_with_hidden += 1
        all_hidden["HIGH"] += len(high_hidden)
        all_hidden["MEDIUM"] += len([h for h in hidden if h["severity"] == "MEDIUM"])
        all_hidden["LOW"] += len([h for h in hidden if h["severity"] == "LOW"])

        if refuted:
            tasks_with_refuted += 1

        results["task_details"].append({
            "task_id": tid,
            "n_nodes": len(task["target_skeleton"]["nodes"]),
            "n_runtime_deps": task["n_runtime_deps"],
            "n_verified_deps": len(verified),
            "n_refuted_deps": len(refuted),
            "n_unknown_deps": len(unknown),
            "n_hidden_env_high": len(high_hidden),
            "n_not_in_user_text": len(issues),
            "issues": issues[:3],
            "refuted": refuted[:2],
        })

    # Summary
    print("=== Literal Source Distribution ===")
    for k, v in all_sources.most_common():
        print(f"  {k}: {v}")

    print(f"\n=== Runtime Dependency Verification ===")
    for k, v in all_dep_status.most_common():
        print(f"  {k}: {v}")

    print(f"\n=== Hidden Environment Values ===")
    print(f"  Tasks with hidden env values: {tasks_with_hidden}/{len(tasks)}")
    for k, v in all_hidden.m_common() if hasattr(all_hidden, 'm_common') else all_hidden.items():
        print(f"  {k}: {v}")

    print(f"\n=== Task Quality Summary ===")
    print(f"  Tasks with refuted deps: {tasks_with_refuted}/{len(tasks)}")
    clean = sum(1 for d in results["task_details"]
                if d["n_refuted_deps"] == 0 and d["n_hidden_env_high"] == 0)
    print(f"  Clean tasks (no refuted deps, no HIGH hidden): {clean}/{len(tasks)}")

    results["literal_observability"] = dict(all_sources)
    results["runtime_dep_verification"] = dict(all_dep_status)
    results["hidden_env_values"] = {
        "tasks_with_hidden": tasks_with_hidden,
        "severity_distribution": dict(all_hidden),
    }
    results["summary"] = {
        "clean_tasks": clean,
        "tasks_with_refuted": tasks_with_refuted,
        "recommendation": "Only clean tasks (no refuted deps, no HIGH hidden env) "
                          "should be used for paper-level evaluation.",
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tau3_gold_provenance_audit.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved {OUT}/tau3_gold_provenance_audit.json")


if __name__ == "__main__":
    main()
