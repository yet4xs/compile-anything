"""Phase 8D-1A: build executable dynamic dataflow corpus (100-300 tasks).

Tasks feature REAL runtime value passing: downstream parameters come from
upstream tool outputs (NOT from instruction text). Each task:
- Has typed Capability schemas (input + output)
- Instruction describes WHAT to do, not the intermediate IDs
- Skeleton uses runtime_ref for dynamic params
- Mock Runtime executes the full chain and verifies value propagation
- Both M0 (TaskIR) and M1 (Skeleton) labels generated from same plan

Topologies: linear 3-8 step, fork, join, fan-out/fan-in, read→mutate.
"""
import json, os, sys, random, hashlib, time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import SkeletonNode, PlanSkeleton
from src.compiler.skeleton_lowering import lower_skeleton
from src.compiler.capability_schema import normalize_registry
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.runtime.mock_capability_registry import (
    MockCapabilityRegistry, MockCapability, DataflowRuntime)

OUT = "data/runtime_dataflow_v1"
SEED = 20261020

# ── Tool domain definitions ──

TOOLS = {
    "lookup_user": {
        "input_schema": {"name": "Str"},
        "output_schema": {"user_id": "Str", "email": "Str", "age": "Int"},
        "exec": lambda p: {"user_id": f"U_{p['name']}",
                           "email": f"{p['name']}@test.com",
                           "age": hash(p['name']) % 50 + 18},
    },
    "get_account": {
        "input_schema": {"user_id": "Str"},
        "output_schema": {"account_id": "Str", "balance": "Float"},
        "exec": lambda p: {"account_id": f"AC_{p['user_id']}",
                           "balance": float(hash(p['user_id']) % 10000)},
    },
    "get_transactions": {
        "input_schema": {"account_id": "Str"},
        "output_schema": {"latest_tx_id": "Str", "tx_count": "Int"},
        "exec": lambda p: {"latest_tx_id": f"TX_{p['account_id']}",
                           "tx_count": hash(p['account_id']) % 20 + 1},
    },
    "get_tx_status": {
        "input_schema": {"tx_id": "Str"},
        "output_schema": {"status": "Str", "timestamp": "Str"},
        "exec": lambda p: {"status": "completed",
                           "timestamp": "2024-01-15T10:00:00Z"},
    },
    "get_credit_score": {
        "input_schema": {"user_id": "Str"},
        "output_schema": {"score": "Int", "rating": "Str"},
        "exec": lambda p: {"score": hash(p['user_id']) % 300 + 500,
                           "rating": "good"},
    },
    "update_balance": {
        "input_schema": {"account_id": "Str", "new_balance": "Float"},
        "output_schema": {"success": "Bool"},
        "effect_class": "state_update_reversible",
        "exec": lambda p: {"success": True},
    },
    "send_alert": {
        "input_schema": {"user_id": "Str", "message": "Str"},
        "output_schema": {"sent": "Bool"},
        "effect_class": "communication",
        "exec": lambda p: {"sent": True},
    },
    "approve_loan": {
        "input_schema": {"user_id": "Str", "score": "Int",
                          "amount": "Float"},
        "output_schema": {"approved": "Bool", "loan_id": "Str"},
        "effect_class": "state_creation",
        "exec": lambda p: {"approved": p["score"] >= 650,
                           "loan_id": f"LN_{p['user_id']}"},
    },
    "search_products": {
        "input_schema": {"query": "Str"},
        "output_schema": {"top_product_id": "Str", "result_count": "Int"},
        "exec": lambda p: {"top_product_id": f"PR_{hash(p['query']) % 999}",
                           "result_count": hash(p['query']) % 50 + 1},
    },
    "get_product_details": {
        "input_schema": {"product_id": "Str"},
        "output_schema": {"price": "Float", "in_stock": "Bool",
                           "sku": "Str"},
        "exec": lambda p: {"price": float(hash(p['product_id']) % 500),
                           "in_stock": hash(p['product_id']) % 2 == 0,
                           "sku": f"SKU_{p['product_id']}"},
    },
    "place_order": {
        "input_schema": {"product_id": "Str", "sku": "Str",
                          "price": "Float"},
        "output_schema": {"order_id": "Str"},
        "effect_class": "state_creation",
        "exec": lambda p: {"order_id": f"ORD_{p['product_id']}"},
    },
}

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace",
         "Heidi", "Ivan", "Judy", "Mallory", "Niaj", "Olive", "Peggy"]
QUERIES = ["wireless earbuds", "gaming laptop", "coffee maker",
           "running shoes", "smart watch", "desk lamp", "keyboard",
           "monitor stand", "usb hub", "webcam"]


def build_registry():
    reg = MockCapabilityRegistry()
    for tid, spec in TOOLS.items():
        reg.register(MockCapability(
            capability_id=tid,
            input_schema=spec["input_schema"],
            output_schema=spec["output_schema"],
            effect_class=spec.get("effect_class", "read_only"),
            mock_executor=spec["exec"]))
    return reg


def cap_schema_for_lowering():
    """Schemas in the format expected by lower_skeleton."""
    out = {}
    for tid, spec in TOOLS.items():
        params = [{"name": k, "optional": False}
                  for k in spec["input_schema"]]
        out[tid] = {"params": params,
                    "output_fields": spec["output_schema"],
                    "effect_class": spec.get("effect_class",
                                              "read_only")}
    return out


# ── Task generators by topology ──

def gen_linear(rng, depth):
    """Linear chain: lookup → account → [tx → status] with depth steps."""
    name = rng.choice(NAMES)
    nodes = [
        SkeletonNode(id="p0", capability_id="lookup_user",
                     skill="EXEC_ACTION", depends_on=[],
                     arguments={"name": name}),
        SkeletonNode(id="p1", capability_id="get_account",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={"user_id": {"kind": "runtime_ref",
                                            "producer": "p0",
                                            "field": "user_id"}}),
    ]
    instruction = (f"Look up user {name}, then retrieve their account "
                   f"details")
    if depth >= 3:
        nodes.append(SkeletonNode(
            id="p2", capability_id="get_transactions",
            skill="EXEC_ACTION", depends_on=["p1"],
            arguments={"account_id": {"kind": "runtime_ref",
                                       "producer": "p1",
                                       "field": "account_id"}}))
        instruction += ", find their recent transactions"
    if depth >= 4:
        nodes.append(SkeletonNode(
            id="p3", capability_id="get_tx_status",
            skill="EXEC_ACTION", depends_on=["p2"],
            arguments={"tx_id": {"kind": "runtime_ref",
                                  "producer": "p2",
                                  "field": "latest_tx_id"}}))
        instruction += ", and check the status of the latest one"
    if depth >= 5:
        nodes.append(SkeletonNode(
            id="p4", capability_id="get_credit_score",
            skill="EXEC_ACTION", depends_on=["p0"],
            arguments={"user_id": {"kind": "runtime_ref",
                                    "producer": "p0",
                                    "field": "user_id"}}))
        instruction += f", and get their credit score"
    if depth >= 6:
        nodes.append(SkeletonNode(
            id="p5", capability_id="approve_loan", skill="EXEC_ACTION",
            depends_on=["p0", "p4"],
            arguments={
                "user_id": {"kind": "runtime_ref", "producer": "p0",
                             "field": "user_id"},
                "score": {"kind": "runtime_ref", "producer": "p4",
                           "field": "score"},
                "amount": float(rng.randint(1000, 50000))}))
        instruction += f", then evaluate a loan application"
    instruction += "."
    return nodes, instruction


def gen_fork(rng, _depth):
    """Fork: one producer feeds two independent consumers."""
    name = rng.choice(NAMES)
    nodes = [
        SkeletonNode(id="p0", capability_id="lookup_user",
                     skill="EXEC_ACTION", depends_on=[],
                     arguments={"name": name}),
        SkeletonNode(id="p1", capability_id="get_account",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={"user_id": {"kind": "runtime_ref",
                                            "producer": "p0",
                                            "field": "user_id"}}),
        SkeletonNode(id="p2", capability_id="get_credit_score",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={"user_id": {"kind": "runtime_ref",
                                            "producer": "p0",
                                            "field": "user_id"}}),
    ]
    instruction = (f"Look up {name}, get their account info and "
                   f"credit score in parallel.")
    return nodes, instruction


def gen_join(rng, _depth):
    """Join: two producers feed one consumer."""
    name = rng.choice(NAMES)
    nodes = [
        SkeletonNode(id="p0", capability_id="lookup_user",
                     skill="EXEC_ACTION", depends_on=[],
                     arguments={"name": name}),
        SkeletonNode(id="p1", capability_id="get_credit_score",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={"user_id": {"kind": "runtime_ref",
                                            "producer": "p0",
                                            "field": "user_id"}}),
        SkeletonNode(id="p2", capability_id="approve_loan",
                     skill="EXEC_ACTION", depends_on=["p0", "p1"],
                     arguments={
                         "user_id": {"kind": "runtime_ref",
                                      "producer": "p0",
                                      "field": "user_id"},
                         "score": {"kind": "runtime_ref",
                                    "producer": "p1",
                                    "field": "score"},
                         "amount": float(rng.randint(5000, 80000))}),
    ]
    instruction = (f"Look up {name}, get their credit score, and "
                   f"evaluate a loan application.")
    return nodes, instruction


def gen_shop_chain(rng, depth):
    """Read→mutate: search → details → order."""
    query = rng.choice(QUERIES)
    nodes = [
        SkeletonNode(id="p0", capability_id="search_products",
                     skill="EXEC_ACTION", depends_on=[],
                     arguments={"query": query}),
        SkeletonNode(id="p1", capability_id="get_product_details",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={"product_id": {"kind": "runtime_ref",
                                                "producer": "p0",
                                                "field": "top_product_id"}}),
        SkeletonNode(id="p2", capability_id="place_order",
                     skill="EXEC_ACTION", depends_on=["p0", "p1"],
                     arguments={
                         "product_id": {"kind": "runtime_ref",
                                         "producer": "p0",
                                         "field": "top_product_id"},
                         "sku": {"kind": "runtime_ref",
                                  "producer": "p1",
                                  "field": "sku"},
                         "price": {"kind": "runtime_ref",
                                    "producer": "p1",
                                    "field": "price"}}),
    ]
    instruction = (f"Search for {query}, get the details of the top "
                   f"result, and place an order for it.")
    return nodes, instruction


def gen_read_mutate(rng, _depth):
    """Read chain then mutation."""
    name = rng.choice(NAMES)
    nodes = [
        SkeletonNode(id="p0", capability_id="lookup_user",
                     skill="EXEC_ACTION", depends_on=[],
                     arguments={"name": name}),
        SkeletonNode(id="p1", capability_id="get_account",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={"user_id": {"kind": "runtime_ref",
                                            "producer": "p0",
                                            "field": "user_id"}}),
        SkeletonNode(id="p2", capability_id="update_balance",
                     skill="EXEC_ACTION", depends_on=["p1"],
                     arguments={
                         "account_id": {"kind": "runtime_ref",
                                         "producer": "p1",
                                         "field": "account_id"},
                         "new_balance": float(rng.randint(0, 5000))}),
        SkeletonNode(id="p3", capability_id="send_alert",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={
                         "user_id": {"kind": "runtime_ref",
                                      "producer": "p0",
                                      "field": "user_id"},
                         "message": "Your balance has been updated."}),
    ]
    instruction = (f"Look up {name}, get their account, update the "
                   f"balance, and send them a notification.")
    return nodes, instruction


def gen_fan_out_in(rng, _depth):
    """Fan-out/fan-in: one lookup fans to two reads, both join at mutation."""
    name = rng.choice(NAMES)
    nodes = [
        SkeletonNode(id="p0", capability_id="lookup_user",
                     skill="EXEC_ACTION", depends_on=[],
                     arguments={"name": name}),
        SkeletonNode(id="p1", capability_id="get_account",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={"user_id": {"kind": "runtime_ref",
                                            "producer": "p0",
                                            "field": "user_id"}}),
        SkeletonNode(id="p2", capability_id="get_credit_score",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={"user_id": {"kind": "runtime_ref",
                                            "producer": "p0",
                                            "field": "user_id"}}),
        SkeletonNode(id="p3", capability_id="approve_loan",
                     skill="EXEC_ACTION", depends_on=["p0", "p2"],
                     arguments={
                         "user_id": {"kind": "runtime_ref",
                                      "producer": "p0",
                                      "field": "user_id"},
                         "score": {"kind": "runtime_ref",
                                    "producer": "p2",
                                    "field": "score"},
                         "amount": float(rng.randint(10000, 90000))}),
        SkeletonNode(id="p4", capability_id="send_alert",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={
                         "user_id": {"kind": "runtime_ref",
                                      "producer": "p0",
                                      "field": "user_id"},
                         "message": "Loan application processed."}),
    ]
    instruction = (f"Look up {name}, check their account and credit "
                   f"score, evaluate a loan, and notify them.")
    return nodes, instruction


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    rt = DataflowRuntime(build_registry())
    schemas = cap_schema_for_lowering()

    generators = [
        ("linear_3", lambda r: gen_linear(r, 3)),
        ("linear_4", lambda r: gen_linear(r, 4)),
        ("linear_5", lambda r: gen_linear(r, 5)),
        ("linear_6", lambda r: gen_linear(r, 6)),
        ("fork", lambda r: gen_fork(r, 0)),
        ("join", lambda r: gen_join(r, 0)),
        ("shop_chain", lambda r: gen_shop_chain(r, 0)),
        ("read_mutate", lambda r: gen_read_mutate(r, 0)),
        ("fan_out_in", lambda r: gen_fan_out_in(r, 0)),
    ]
    quotas = [30, 25, 20, 15, 20, 20, 25, 20, 20]  # total ~195

    all_tasks = []
    stats = Counter()
    fails = []

    for (gen_name, gen_fn), quota in zip(generators, quotas):
        for i in range(quota):
            nodes, instruction = gen_fn(rng)
            task_id = f"df_{gen_name}_{i:03d}"

            sk = PlanSkeleton(task_id=task_id)
            sk.nodes = nodes

            # strict lowering
            low = lower_skeleton(sk, None, schemas, strict=True)
            if low.get("taskir") is None:
                fails.append({"id": task_id, "stage": "lowering",
                              "err": str(low.get("errors", []))[:100]})
                stats[f"{gen_name}_lower_fail"] += 1
                continue

            # strict validation
            mod = parse_text(low["taskir"])
            v2_reg = {t: {"input_schema": TOOLS[t]["input_schema"],
                          "output_schema": TOOLS[t]["output_schema"]}
                      for t in TOOLS}
            rep = validate(mod, capability_registry=v2_reg)
            if not rep.valid:
                fails.append({"id": task_id, "stage": "validation",
                              "err": str(rep.errors[:2])[:100]})
                stats[f"{gen_name}_valid_fail"] += 1
                continue

            # runtime execution
            tr = rt.execute(mod, capability_registry=v2_reg)
            if not tr.success:
                fails.append({"id": task_id, "stage": "runtime",
                              "err": tr.error[:100]})
                stats[f"{gen_name}_exec_fail"] += 1
                continue

            # verify dataflow: at least one param_ref resolved
            has_dataflow = any(
                s.op == "EXTRACT_FIELD" for s in tr.steps)
            if not has_dataflow:
                fails.append({"id": task_id, "stage": "no_dataflow"})
                stats[f"{gen_name}_no_flow"] += 1
                continue

            # count runtime_refs
            n_refs = sum(
                1 for n in nodes
                for v in n.arguments.values()
                if isinstance(v, dict) and v.get("kind") == "runtime_ref")

            task = {
                "task_id": task_id,
                "instruction": instruction,
                "topology": gen_name,
                "available_capabilities": sorted(
                    {n.capability_id for n in nodes}),
                "capability_registry": {
                    t: {"input_schema": TOOLS[t]["input_schema"],
                        "output_schema": TOOLS[t]["output_schema"],
                        "effect_class": TOOLS[t].get("effect_class",
                                                     "read_only")}
                    for t in {n.capability_id for n in nodes}},
                "target_skeleton": {
                    "task_id": task_id,
                    "nodes": [{"id": n.id,
                               "capability_id": n.capability_id,
                               "skill": n.skill,
                               "depends_on": n.depends_on,
                               "order_on": [],
                               "arguments": n.arguments}
                              for n in nodes]},
                "m0_taskir": low["taskir"],
                "provenance": {
                    "dataset": "runtime_dataflow_v1",
                    "evidence": "verified",
                    "topology": gen_name,
                    "depth": len(nodes),
                    "runtime_refs": n_refs,
                    "execution_steps": len(tr.steps),
                    "final_result_type": type(tr.final_result).__name__,
                },
            }
            all_tasks.append(task)
            stats[gen_name] += 1

    # dedup by instruction
    seen = set()
    unique = []
    for t in all_tasks:
        h = hashlib.sha1(t["instruction"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(t)

    # split: train 70%, dev 15%, test 15% (by topology balance)
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(unique)
    n = len(unique)
    n_test = max(10, n // 7)
    n_dev = max(10, n // 7)
    test = unique[:n_test]
    dev = unique[n_test:n_test + n_dev]
    train = unique[n_test + n_dev:]

    for k, rows in (("train", train), ("dev", dev), ("test", test)):
        with open(f"{OUT}/{k}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "total_generated": len(all_tasks),
        "total_unique": len(unique),
        "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
        "topology_stats": {k: v for k, v in stats.items()
                           if not k.endswith("_fail")
                           and not k.endswith("_no_flow")},
        "failures": dict((k, v) for k, v in stats.items()
                          if k.endswith("_fail") or k.endswith("_no_flow")),
        "failure_details": fails[:20],
        "tools_used": sorted(TOOLS.keys()),
        "seed": SEED,
    }
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"generated: {len(all_tasks)}, unique: {len(unique)}")
    print(f"splits: {manifest['splits']}")
    print(f"topologies: {manifest['topology_stats']}")
    print(f"failures: {manifest['failures']}")
    depth_dist = Counter(t["provenance"]["depth"] for t in unique)
    print(f"depth: {dict(sorted(depth_dist.items()))}")
    ref_dist = Counter(t["provenance"]["runtime_refs"] for t in unique)
    print(f"runtime_refs per task: {dict(sorted(ref_dist.items()))}")


if __name__ == "__main__":
    main()
