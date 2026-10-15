"""Phase 9B-2: Convert τ³ L3 tasks to Semantic Skeleton + TaskIR v0.2.

Reads τ³ reference action sequences, identifies runtime deps, generates
Skeleton JSON with runtime_ref + TaskIR v0.2 with EXTRACT_FIELD/bind().
Outputs benchmark-ready JSONL + validation report.
"""
import json, os, sys, re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

TAU3 = os.path.join("third_party", "tau3-bench", "data", "tau2", "domains")
OUT = "data/tau3_runtime_benchmark"

# ── Tool schemas (from τ³ source code analysis) ──

AIRLINE_TOOLS = {
    "get_user_details": {
        "input_schema": {"user_id": "Str"},
        "output_schema": {"user_id": "Str", "reservations": "List[Any]", "membership_level": "Str"},
        "effect_class": "read_only",
    },
    "get_reservation_details": {
        "input_schema": {"reservation_id": "Str"},
        "output_schema": {"reservation_id": "Str", "flights": "List[Any]", "cabin": "Str", "status": "Str"},
        "effect_class": "read_only",
    },
    "get_flight_status": {
        "input_schema": {"flight_number": "Str"},
        "output_schema": {"status": "Str", "departure": "Str"},
        "effect_class": "read_only",
    },
    "search_direct_flight": {
        "input_schema": {"date": "Str", "origin": "Str", "destination": "Str"},
        "output_schema": {"flight_number": "Str", "origin": "Str", "destination": "Str", "departure_time": "Str"},
        "effect_class": "read_only",
    },
    "update_reservation_flights": {
        "input_schema": {"reservation_id": "Str", "cabin": "Str", "flights": "List[Any]", "payment_id": "Str"},
        "output_schema": {"reservation_id": "Str"},
        "effect_class": "state_update_reversible",
    },
    "cancel_reservation": {
        "input_schema": {"reservation_id": "Str"},
        "output_schema": {"reservation_id": "Str", "status": "Str"},
        "effect_class": "state_transition_irreversible",
    },
    "send_certificate": {
        "input_schema": {"user_id": "Str", "amount": "Int"},
        "output_schema": {"certificate_id": "Str"},
        "effect_class": "state_creation",
    },
}

RETAIL_TOOLS = {
    "find_user_id_by_name_zip": {
        "input_schema": {"first_name": "Str", "last_name": "Str", "zip": "Str"},
        "output_schema": {"user_id": "Str"},
        "effect_class": "read_only",
    },
    "find_user_id_by_email": {
        "input_schema": {"email": "Str"},
        "output_schema": {"user_id": "Str"},
        "effect_class": "read_only",
    },
    "get_user_details": {
        "input_schema": {"user_id": "Str"},
        "output_schema": {"user_id": "Str", "orders": "List[Any]", "default_address": "Str"},
        "effect_class": "read_only",
    },
    "get_order_details": {
        "input_schema": {"order_id": "Str"},
        "output_schema": {"order_id": "Str", "items": "List[Any]", "status": "Str", "payment_method_id": "Str"},
        "effect_class": "read_only",
    },
    "get_product_details": {
        "input_schema": {"product_id": "Str"},
        "output_schema": {"product_id": "Str", "price": "Float", "sku": "Str", "in_stock": "Bool"},
        "effect_class": "read_only",
    },
    "get_item_details": {
        "input_schema": {"item_id": "Str"},
        "output_schema": {"item_id": "Str", "product_id": "Str", "price": "Float"},
        "effect_class": "read_only",
    },
    "list_all_product_types": {
        "input_schema": {},
        "output_schema": {"product_types": "List[Any]"},
        "effect_class": "read_only",
    },
    "exchange_delivered_order_items": {
        "input_schema": {"order_id": "Str", "item_ids": "List[Any]", "new_item_ids": "List[Any]", "payment_method_id": "Str"},
        "output_schema": {"order_id": "Str", "success": "Bool"},
        "effect_class": "state_transition_irreversible",
    },
    "return_delivered_order_items": {
        "input_schema": {"order_id": "Str", "item_ids": "List[Any]", "payment_method_id": "Str"},
        "output_schema": {"order_id": "Str", "success": "Bool"},
        "effect_class": "state_transition_irreversible",
    },
    "modify_pending_order_items": {
        "input_schema": {"order_id": "Str", "item_ids": "List[Any]", "new_item_ids": "List[Any]", "payment_method_id": "Str"},
        "output_schema": {"order_id": "Str", "success": "Bool"},
        "effect_class": "state_update_reversible",
    },
    "modify_user_address": {
        "input_schema": {"user_id": "Str", "address1": "Str", "address2": "Str", "city": "Str", "state": "Str", "country": "Str", "zip": "Str"},
        "output_schema": {"user_id": "Str", "success": "Bool"},
        "effect_class": "state_update_reversible",
    },
}

DOMAIN_TOOLS = {"airline": AIRLINE_TOOLS, "retail": RETAIL_TOOLS}


def get_task_user_text(task):
    parts = []
    desc = task.get("description") or {}
    if isinstance(desc, dict):
        parts.append(desc.get("purpose") or "")
    us = task.get("user_scenario") or {}
    if isinstance(us, dict):
        parts.append(us.get("instructions") or "")
    return " ".join(str(p) for p in parts if p)


def build_dependency_chain(actions, user_text, tools):
    """Identify runtime deps by checking which argument values could come
    from prior tool outputs (matching field names)."""
    user_lower = user_text.lower()
    deps = []
    for i, action in enumerate(actions):
        if i == 0:
            continue
        tool_name = action.get("name", "")
        args = action.get("arguments") or {}
        prior_names = [a.get("name", "") for a in actions[:i]]

        for param, value in args.items():
            val_str = str(value).strip()
            if not val_str or val_str.lower() in user_lower:
                continue  # from user text

            # Check if any prior tool's output_schema has this param as a field
            is_runtime = False
            producer = None
            producer_field = None
            for pn in prior_names:
                if pn in tools and param in tools[pn].get("output_schema", {}):
                    is_runtime = True
                    producer = pn
                    producer_field = param
                    break
            # Heuristic: order_id from get_user_details.orders, etc.
            if not is_runtime:
                for pn in prior_names:
                    if pn in tools:
                        outs = tools[pn].get("output_schema", {})
                        for field in outs:
                            if field.rstrip('_id') == param.rstrip('_id') or param in field.lower():
                                is_runtime = True
                                producer = pn
                                producer_field = field
                                break
                    if is_runtime:
                        break

            if is_runtime:
                deps.append({
                    "step": i, "tool": tool_name, "param": param,
                    "producer_step": None, "producer_tool": producer,
                    "producer_field": producer_field,
                    "value": val_str[:40],
                })
    return deps


def find_producer_step(dep, actions):
    """Find which prior step actually calls the producer tool."""
    for j, a in enumerate(actions):
        if a.get("name") == dep["producer_tool"] and j < dep["step"]:
            dep["producer_step"] = j
            return j
    return None


def convert_task(task, domain, tools):
    """Convert one τ³ task to our benchmark format."""
    actions = task.get("evaluation_criteria", {}).get("actions") or []
    if len(actions) <= 1:
        return None

    user_text = get_task_user_text(task)
    deps = build_dependency_chain(actions, user_text, tools)
    if not deps:
        return None  # Not an L3 task

    # Find producer steps
    for dep in deps:
        find_producer_step(dep, actions)

    # Build capability registry (only tools used in this task)
    used_tools = sorted({a["name"] for a in actions})
    cap_registry = {t: tools[t] for t in used_tools if t in tools}
    if len(cap_registry) < len(used_tools):
        return None  # Some tools unknown

    # Build skeleton nodes
    skeleton_nodes = []
    for i, action in enumerate(actions):
        tool_name = action["name"]
        args = action.get("arguments") or {}

        node_args = {}
        for param, value in args.items():
            val_str = str(value).strip()
            # Check if this param is a runtime dep
            matching_dep = next(
                (d for d in deps
                 if d["step"] == i and d["param"] == param), None)
            if matching_dep and matching_dep.get("producer_step") is not None:
                node_args[param] = {
                    "kind": "runtime_ref",
                    "producer": f"p{matching_dep['producer_step']}",
                    "field": matching_dep["producer_field"],
                }
            else:
                node_args[param] = value

        # depends_on: link to all producers of our runtime deps
        depends_on = sorted(set(
            f"p{d['producer_step']}" for d in deps
            if d["step"] == i and d.get("producer_step") is not None))

        skeleton_nodes.append({
            "id": f"p{i}",
            "capability_id": tool_name,
            "skill": "EXEC_ACTION",
            "depends_on": depends_on,
            "order_on": [],
            "arguments": node_args,
        })

    # Build instruction (use the task description)
    desc = task.get("description") or {}
    purpose = desc.get("purpose") or ""
    if not purpose:
        us = task.get("user_scenario") or {}
        purpose = us.get("instructions") or ""
    if not purpose:
        purpose = f"Execute a {domain} workflow with {len(actions)} steps"

    return {
        "task_id": f"tau3_{domain}_{task['id']}",
        "instruction": purpose,
        "topology": f"tau3_{domain}",
        "gold_capability_ids": used_tools,
        "offered_capability_ids": used_tools,  # oracle pool for now
        "distractor_capability_ids": [],
        "capability_registry": cap_registry,
        "target_skeleton": {
            "task_id": f"tau3_{domain}_{task['id']}",
            "nodes": skeleton_nodes,
        },
        "reference_deps": deps,
        "n_runtime_deps": len(deps),
        "depth": len(actions),
        "provenance": {
            "dataset": f"tau3_{domain}",
            "evidence": "converted_from_reference",
            "source_task_id": task["id"],
        },
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    all_tasks = []
    stats = Counter()

    for domain in ("airline", "retail"):
        tools = DOMAIN_TOOLS[domain]
        tasks = json.load(open(os.path.join(TAU3, domain, "tasks.json")))

        converted = 0
        skipped = 0
        for task in tasks:
            result = convert_task(task, domain, tools)
            if result is None:
                skipped += 1
                continue
            all_tasks.append(result)
            converted += 1
            stats[f"{domain}_converted"] += 1
            stats[f"{domain}_deps"] += result["n_runtime_deps"]

        print(f"{domain}: converted {converted}, skipped {skipped}")

    # Split: 70% train, 15% dev, 15% test (by domain balance)
    import random
    rng = random.Random(20261015)
    rng.shuffle(all_tasks)

    n = len(all_tasks)
    n_test = max(5, n // 7)
    n_dev = max(3, n // 7)
    test = all_tasks[:n_test]
    dev = all_tasks[n_test:n_test + n_dev]
    train = all_tasks[n_test + n_dev:]

    for split, tasks in (("train", train), ("dev", dev), ("test", test)):
        path = os.path.join(OUT, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(tasks)} tasks")

    # Report
    dep_counts = [t["n_runtime_deps"] for t in all_tasks]
    depth_counts = [t["depth"] for t in all_tasks]
    print(f"\nTotal: {len(all_tasks)} L3 tasks")
    print(f"Runtime deps: total={sum(dep_counts)}, mean={sum(dep_counts)/len(all_tasks):.1f}")
    print(f"Depth: mean={sum(depth_counts)/len(all_tasks):.1f}, "
          f"min={min(depth_counts)}, max={max(depth_counts)}")

    # By domain
    for domain in ("airline", "retail"):
        dt = [t for t in all_tasks if domain in t["topology"]]
        dd = sum(t["n_runtime_deps"] for t in dt)
        print(f"  {domain}: {len(dt)} tasks, {dd} deps")

    manifest = {
        "total": len(all_tasks),
        "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
        "stats": dict(stats),
        "depth_range": [min(depth_counts), max(depth_counts)],
        "total_runtime_deps": sum(dep_counts),
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nsaved {OUT}/")


if __name__ == "__main__":
    main()
