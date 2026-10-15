"""Phase 9B-1: τ³ Runtime Dependency Extraction (static analysis).

Analyzes reference action sequences + DB state to classify each argument
value's source: user_text, environment, or runtime_dep (from prior tool).

Method:
1. For each multi-step task, check each argument value at step N>0
2. If value appears in user instructions → user_text
3. If value appears in DB but NOT reachable from prior tool outputs → environment
4. If value matches a field that a prior tool's return type contains → runtime_dep
"""
import json, os, sys, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU3 = os.path.join(ROOT, "third_party", "tau3-bench", "data", "tau2", "domains")

# Tool output field hints (from reading τ³ tool source code)
TOOL_OUTPUTS = {
    "airline": {
        "get_user_details": {"reservations": "list_of_reservations", "user_id": "Str"},
        "get_reservation_details": {"reservation_id": "Str", "flights": "list"},
        "search_direct_flight": {"flight_number": "Str", "origin": "Str", "destination": "Str"},
        "search_onestop_flight": {"flight_number": "Str"},
    },
    "retail": {
        "find_user_id_by_email": {"user_id": "Str"},
        "find_user_id_by_name_zip": {"user_id": "Str"},
        "get_order_details": {"order_id": "Str", "items": "list"},
        "get_product_details": {"product_id": "Str", "price": "Float"},
        "get_item_details": {"item_id": "Str"},
        "list_all_product_types": {"product_id": "Str"},
    },
    "banking_knowledge": {
        "get_user_information_by_id": {"id": "Str"},
        "get_user_information_by_email": {"id": "Str"},
        "get_user_information_by_name": {"id": "Str"},
        "get_credit_card_accounts_by_user": {"account_id": "Str"},
    },
    "telecom": {
        "get_customer_by_id": {"customer_id": "Str", "lines": "list"},
        "get_customer_by_name": {"customer_id": "Str"},
        "get_customer_by_phone": {"customer_id": "Str"},
        "get_details_by_id": {"customer_id": "Str", "line_id": "Str"},
    },
}


def load_domain(domain):
    tasks = json.load(open(os.path.join(TAU3, domain, "tasks.json")))
    db_path = os.path.join(TAU3, domain, "db.json")
    db = json.load(open(db_path)) if os.path.exists(db_path) else {}
    return tasks, db


def get_task_user_text(task):
    parts = []
    desc = task.get("description", {})
    if isinstance(desc, dict):
        parts.append(desc.get("purpose", ""))
    us = task.get("user_scenario", {})
    if isinstance(us, dict):
        parts.append(us.get("instructions", ""))
    return " ".join(str(p) for p in parts if p)


def db_contains_value(db, value):
    """Check if a value exists anywhere in the DB (recursive search)."""
    val_str = str(value).strip()
    if not val_str:
        return False

    def search(obj):
        if isinstance(obj, dict):
            return any(search(v) for v in obj.values())
        if isinstance(obj, list):
            return any(search(v) for v in obj)
        return str(obj).strip() == val_str
    return search(db)


def can_prior_tool_produce(prior_tools, param_name, value, domain):
    """Check if any prior tool's return signature includes this field."""
    outputs = TOOL_OUTPUTS.get(domain, {})
    for tool_name in prior_tools:
        if tool_name in outputs:
            if param_name in outputs[tool_name]:
                return True
            # Check if any output field name matches the param name
            # (heuristic: get_X returns X_id)
            for field in outputs[tool_name]:
                if field == param_name or param_name.rstrip('_id') in field:
                    return True
    return False


def classify_task(task, db, domain):
    """Classify all dependencies in a task's reference action sequence."""
    actions = task.get("evaluation_criteria", {}).get("actions", [])
    if len(actions) <= 1:
        return {"level": 0, "deps": []}

    user_text = get_task_user_text(task).lower()
    deps = []
    has_runtime_dep = False
    has_order_dep = False

    for i, action in enumerate(actions):
        if i == 0:
            continue
        tool_name = action.get("name", "")
        args = action.get("arguments", {})
        prior_tools = [a.get("name", "") for a in actions[:i]]

        for param, value in args.items():
            val_str = str(value).strip()
            if not val_str:
                continue

            if val_str.lower() in user_text:
                deps.append({"step": i, "param": param, "value": val_str[:30],
                            "source": "user_text", "tool": tool_name})
            elif can_prior_tool_produce(prior_tools, param, val_str, domain):
                has_runtime_dep = True
                deps.append({"step": i, "param": param, "value": val_str[:30],
                            "source": "runtime_dep", "tool": tool_name,
                            "prior_tools": [t for t in prior_tools
                                             if t in TOOL_OUTPUTS.get(domain, {})]})
            elif db_contains_value(db, value):
                deps.append({"step": i, "param": param, "value": val_str[:30],
                            "source": "environment", "tool": tool_name})
            else:
                has_order_dep = True
                deps.append({"step": i, "param": param, "value": val_str[:30],
                            "source": "unknown", "tool": tool_name})

    if has_runtime_dep:
        level = 3
    elif has_order_dep or len(actions) > 1:
        level = 2
    else:
        level = 1

    return {"level": level, "deps": deps, "n_actions": len(actions)}


def main():
    all_results = {}
    grand_total = Counter()

    for domain in ("airline", "banking_knowledge", "retail", "telecom"):
        tasks, db = load_domain(domain)
        stats = Counter()
        level3_tasks = []
        level3_dep_count = 0

        for task in tasks:
            result = classify_task(task, db, domain)
            stats[f"L{result['level']}"] += 1
            stats["total"] += 1

            runtime_deps = [d for d in result["deps"]
                            if d["source"] == "runtime_dep"]
            level3_dep_count += len(runtime_deps)

            if result["level"] == 3:
                level3_tasks.append({
                    "task_id": task.get("id"),
                    "n_actions": result["n_actions"],
                    "runtime_deps": runtime_deps,
                    "user_text_snippet": get_task_user_text(task)[:100],
                })

        print(f"\n{domain}:")
        print(f"  total={stats['total']} "
              f"L0={stats.get('L0',0)} L1={stats.get('L1',0)} "
              f"L2={stats.get('L2',0)} **L3={stats.get('L3',0)}**")
        print(f"  runtime_dep_count={level3_dep_count}")
        if level3_tasks:
            t = level3_tasks[0]
            print(f"  sample: {t['task_id']} ({t['n_actions']} steps)")
            for d in t["runtime_deps"][:3]:
                print(f"    [{d['step']}] {d['tool'][:25]}({d['param']}={d['value']})"
                      f" ← {d.get('prior_tools', [])}")

        all_results[domain] = {
            "total": stats["total"],
            "L0": stats.get("L0", 0),
            "L1": stats.get("L1", 0),
            "L2": stats.get("L2", 0),
            "L3": stats.get("L3", 0),
            "runtime_dep_count": level3_dep_count,
            "level3_tasks_sample": level3_tasks[:5],
        }
        for k, v in stats.items():
            grand_total[f"{domain}_{k}"] = v

    # Summary
    total_l3 = sum(all_results[d]["L3"] for d in all_results)
    total_multi = sum(all_results[d]["L1"] + all_results[d]["L2"] + all_results[d]["L3"]
                      for d in all_results)
    total_tasks = sum(all_results[d]["total"] for d in all_results)

    print(f"\n{'='*50}")
    print(f"GRAND TOTAL: {total_tasks} tasks, {total_multi} multi-step, {total_l3} L3")
    print(f"runtime deps total: {sum(all_results[d]['runtime_dep_count'] for d in all_results)}")

    output = {
        "domains": all_results,
        "summary": {
            "total_tasks": total_tasks,
            "multi_step": total_multi,
            "level3_tasks": total_l3,
            "runtime_dep_edges": sum(all_results[d]["runtime_dep_count"] for d in all_results),
        }
    }
    os.makedirs("results/phase9", exist_ok=True)
    with open("results/phase9/tau3_runtime_audit.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nsaved results/phase9/tau3_runtime_audit.json")


if __name__ == "__main__":
    main()
