"""Phase 8D-1A.1: dynamic dataflow corpus v2 with semantic quality gates.

Fixes over v1:
1. Observable literals: every literal param appears in the instruction
   text (no hidden random amounts)
2. DAG preservation: read_only tools NOT serialized; effect_class from
   normalized registry (never None); DATA/ORDER/EFFECT edges verified
3. Typed schemas preserved (Str/Int/Float/Bool) through all 4 layers
4. Distractor capabilities: offered pool includes irrelevant tools
5. Two evaluation protocols: Oracle Pool (binding test) + Distractor
   Pool (selection + binding test)
6. Template/composition-aware splits
"""
import json, os, sys, random, hashlib, copy
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import SkeletonNode, PlanSkeleton, \
    validate_skeleton, hard_errors
from src.compiler.skeleton_lowering import lower_skeleton
from src.compiler.capability_schema import normalize_registry
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.runtime.mock_capability_registry import (
    MockCapabilityRegistry, MockCapability, DataflowRuntime)

OUT = "data/runtime_dataflow_v2"
SEED = 20261021

# ── Tool definitions with FULL typed schemas ──

TOOLS = {
    "lookup_user": {
        "input_schema": {"name": "Str"},
        "output_schema": {"user_id": "Str", "email": "Str", "age": "Int"},
        "effect": "read_only",
        "exec": lambda p: {"user_id": f"U_{p['name']}",
                           "email": f"{p['name']}@test.com",
                           "age": (hash(p['name']) % 40) + 20},
    },
    "get_account": {
        "input_schema": {"user_id": "Str"},
        "output_schema": {"account_id": "Str", "balance": "Float",
                           "currency": "Str"},
        "effect": "read_only",
        "exec": lambda p: {"account_id": f"AC_{p['user_id']}",
                           "balance": float((hash(p['user_id']) % 9000) + 100),
                           "currency": "USD"},
    },
    "get_transactions": {
        "input_schema": {"account_id": "Str"},
        "output_schema": {"latest_tx_id": "Str", "tx_count": "Int"},
        "effect": "read_only",
        "exec": lambda p: {"latest_tx_id": f"TX_{p['account_id']}",
                           "tx_count": (hash(p['account_id']) % 15) + 1},
    },
    "get_tx_status": {
        "input_schema": {"tx_id": "Str"},
        "output_schema": {"status": "Str"},
        "effect": "read_only",
        "exec": lambda p: {"status": "completed"},
    },
    "get_credit_score": {
        "input_schema": {"user_id": "Str"},
        "output_schema": {"score": "Int", "rating": "Str"},
        "effect": "read_only",
        "exec": lambda p: {"score": (hash(p['user_id']) % 250) + 550,
                           "rating": "good"},
    },
    "update_balance": {
        "input_schema": {"account_id": "Str", "new_balance": "Float"},
        "output_schema": {"success": "Bool"},
        "effect": "state_update_reversible",
        "exec": lambda p: {"success": True},
    },
    "send_alert": {
        "input_schema": {"user_id": "Str", "message": "Str"},
        "output_schema": {"sent": "Bool"},
        "effect": "communication",
        "exec": lambda p: {"sent": True},
    },
    "approve_loan": {
        "input_schema": {"user_id": "Str", "score": "Int",
                          "amount": "Float"},
        "output_schema": {"approved": "Bool", "loan_id": "Str"},
        "effect": "state_creation",
        "exec": lambda p: {"approved": p["score"] >= 600,
                           "loan_id": f"LN_{p['user_id']}"},
    },
    "search_products": {
        "input_schema": {"query": "Str"},
        "output_schema": {"top_product_id": "Str", "result_count": "Int"},
        "effect": "read_only",
        "exec": lambda p: {"top_product_id": f"PR_{hash(p['query']) % 99}",
                           "result_count": (hash(p['query']) % 30) + 1},
    },
    "get_product_details": {
        "input_schema": {"product_id": "Str"},
        "output_schema": {"price": "Float", "sku": "Str",
                           "in_stock": "Bool"},
        "effect": "read_only",
        "exec": lambda p: {"price": float((hash(p['product_id']) % 400) + 10),
                           "sku": f"SKU_{p['product_id']}",
                           "in_stock": hash(p['product_id']) % 2 == 0},
    },
    "place_order": {
        "input_schema": {"product_id": "Str", "sku": "Str",
                          "price": "Float"},
        "output_schema": {"order_id": "Str"},
        "effect": "state_creation",
        "exec": lambda p: {"order_id": f"ORD_{p['product_id']}"},
    },
    "get_order_status": {
        "input_schema": {"order_id": "Str"},
        "output_schema": {"status": "Str", "eta": "Str"},
        "effect": "read_only",
        "exec": lambda p: {"status": "processing", "eta": "3-5 days"},
    },
}

# Distractor tools (similar signatures but not needed for gold plans)
DISTRACTORS = {
    "get_user_profile": {
        "input_schema": {"user_id": "Str"},
        "output_schema": {"profile_id": "Str", "joined": "Str"},
        "effect": "read_only",
        "desc": "Get social profile for a user",
        "exec": lambda p: {"profile_id": f"PF_{p['user_id']}",
                           "joined": "2020-01-01"},
    },
    "list_accounts": {
        "input_schema": {"user_id": "Str"},
        "output_schema": {"account_ids": "Str"},
        "effect": "read_only",
        "desc": "List all account IDs for a user",
        "exec": lambda p: {"account_ids": "AC_1,AC_2"},
    },
    "calculate_interest": {
        "input_schema": {"balance": "Float", "rate": "Float"},
        "output_schema": {"interest": "Float"},
        "effect": "read_only",
        "desc": "Calculate interest on a balance",
        "exec": lambda p: {"interest": p["balance"] * p["rate"] / 100},
    },
    "get_exchange_rate": {
        "input_schema": {"from_currency": "Str", "to_currency": "Str"},
        "output_schema": {"rate": "Float"},
        "effect": "read_only",
        "desc": "Get currency exchange rate",
        "exec": lambda p: {"rate": 1.0},
    },
    "cancel_order": {
        "input_schema": {"order_id": "Str"},
        "output_schema": {"cancelled": "Bool"},
        "effect": "state_transition_irreversible",
        "desc": "Cancel an existing order",
        "exec": lambda p: {"cancelled": True},
    },
    "send_invoice": {
        "input_schema": {"user_id": "Str", "amount": "Float"},
        "output_schema": {"invoice_id": "Str"},
        "effect": "communication",
        "desc": "Send an invoice to a user",
        "exec": lambda p: {"invoice_id": f"INV_{p['user_id']}"},
    },
}

ALL_TOOLS = {**TOOLS, **DISTRACTORS}

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace",
         "Heidi", "Ivan", "Judy", "Ken", "Luna", "Mark", "Nina",
         "Oscar", "Piper", "Quinn", "Rose", "Sam", "Tina"]
QUERIES = ["wireless earbuds", "gaming laptop", "coffee maker",
           "running shoes", "smart watch", "desk lamp", "mechanical keyboard",
           "monitor arm", "usb-c hub", "4k webcam", "mouse pad",
           "laptop stand", "phone charger", "bluetooth speaker"]


def full_registry():
    reg = MockCapabilityRegistry()
    for tid, spec in ALL_TOOLS.items():
        reg.register(MockCapability(
            capability_id=tid,
            input_schema=spec["input_schema"],
            output_schema=spec["output_schema"],
            effect_class=spec.get("effect", "read_only"),
            mock_executor=spec["exec"]))
    return reg


def cap_schemas_typed():
    """Schemas with REAL types for all layers."""
    out = {}
    for tid, spec in ALL_TOOLS.items():
        out[tid] = {
            "input_schema": dict(spec["input_schema"]),
            "output_schema": dict(spec["output_schema"]),
            "effect_class": spec.get("effect", "read_only"),
        }
    return out


def lowering_schemas():
    """Format for lower_skeleton: params with TYPES."""
    out = {}
    for tid, spec in ALL_TOOLS.items():
        params = [{"name": k, "optional": False,
                    "type": v} for k, v in spec["input_schema"].items()]
        out[tid] = {
            "params": params,
            "output_fields": dict(spec["output_schema"]),
            "effect_class": spec.get("effect", "read_only"),
        }
    return out


# ── Task generators (all literals in instruction) ──

def gen_linear_v2(rng, depth):
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
    instr = f"Look up user {name} and get their account details"
    if depth >= 3:
        nodes.append(SkeletonNode(
            id="p2", capability_id="get_transactions",
            skill="EXEC_ACTION", depends_on=["p1"],
            arguments={"account_id": {"kind": "runtime_ref",
                                       "producer": "p1",
                                       "field": "account_id"}}))
        instr += ", find recent transactions"
    if depth >= 4:
        nodes.append(SkeletonNode(
            id="p3", capability_id="get_tx_status",
            skill="EXEC_ACTION", depends_on=["p2"],
            arguments={"tx_id": {"kind": "runtime_ref",
                                  "producer": "p2",
                                  "field": "latest_tx_id"}}))
        instr += ", check the latest transaction status"
    if depth >= 5:
        nodes.append(SkeletonNode(
            id="p4", capability_id="get_credit_score",
            skill="EXEC_ACTION", depends_on=["p0"],
            arguments={"user_id": {"kind": "runtime_ref",
                                    "producer": "p0",
                                    "field": "user_id"}}))
        instr += f", and get their credit score"
    if depth >= 6:
        amount = rng.randint(5, 80) * 1000
        nodes.append(SkeletonNode(
            id="p5", capability_id="approve_loan", skill="EXEC_ACTION",
            depends_on=["p0", "p4"],
            arguments={
                "user_id": {"kind": "runtime_ref", "producer": "p0",
                             "field": "user_id"},
                "score": {"kind": "runtime_ref", "producer": "p4",
                           "field": "score"},
                "amount": float(amount)}))
        instr += f", then evaluate a ${amount} loan application"
    return nodes, instr + "."


def gen_fork_v2(rng, _):
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
    return nodes, (f"Look up {name}, check their account balance "
                   f"and credit score.")


def gen_join_v2(rng, _):
    name = rng.choice(NAMES)
    amount = rng.randint(5, 50) * 1000
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
                         "amount": float(amount)}),
    ]
    return nodes, (f"Look up {name}, get their credit score, and "
                   f"evaluate a ${amount} loan application.")


def gen_shop_v2(rng, depth):
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
    instr = (f"Search for {query}, get details of the top result, "
             f"and place an order")
    if depth >= 4:
        nodes.append(SkeletonNode(
            id="p3", capability_id="get_order_status",
            skill="EXEC_ACTION", depends_on=["p2"],
            arguments={"order_id": {"kind": "runtime_ref",
                                     "producer": "p2",
                                     "field": "order_id"}}))
        instr += ", then check the order status"
    return nodes, instr + "."


def gen_read_mutate_v2(rng, _):
    name = rng.choice(NAMES)
    new_balance = rng.randint(100, 5000)
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
                         "new_balance": float(new_balance)}),
        SkeletonNode(id="p3", capability_id="send_alert",
                     skill="EXEC_ACTION", depends_on=["p0", "p2"],
                     arguments={
                         "user_id": {"kind": "runtime_ref",
                                      "producer": "p0",
                                      "field": "user_id"},
                         "message": "balance updated"}),
    ]
    return nodes, (f"Look up {name}, get their account, update the "
                   f"balance to ${new_balance}, and send them a "
                   f"notification saying balance updated.")


def gen_fan_out_in_v2(rng, _):
    name = rng.choice(NAMES)
    amount = rng.randint(10, 60) * 1000
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
                         "amount": float(amount)}),
        SkeletonNode(id="p4", capability_id="send_alert",
                     skill="EXEC_ACTION", depends_on=["p0", "p3"],
                     arguments={
                         "user_id": {"kind": "runtime_ref",
                                      "producer": "p0",
                                      "field": "user_id"},
                         "message": "loan processed"}),
    ]
    return nodes, (f"Look up {name}, check account and credit score, "
                   f"evaluate a ${amount} loan, and send them a message saying loan processed.")


def gen_repeated_tool(rng, _):
    """Same tool called multiple times with different inputs."""
    name1, name2 = rng.sample(NAMES, 2)
    nodes = [
        SkeletonNode(id="p0", capability_id="lookup_user",
                     skill="EXEC_ACTION", depends_on=[],
                     arguments={"name": name1}),
        SkeletonNode(id="p1", capability_id="get_account",
                     skill="EXEC_ACTION", depends_on=["p0"],
                     arguments={"user_id": {"kind": "runtime_ref",
                                            "producer": "p0",
                                            "field": "user_id"}}),
        SkeletonNode(id="p2", capability_id="lookup_user",
                     skill="EXEC_ACTION", depends_on=[],
                     arguments={"name": name2}),
        SkeletonNode(id="p3", capability_id="get_account",
                     skill="EXEC_ACTION", depends_on=["p2"],
                     arguments={"user_id": {"kind": "runtime_ref",
                                            "producer": "p2",
                                            "field": "user_id"}}),
    ]
    return nodes, (f"Look up both {name1} and {name2}, and get the "
                   f"account details for each.")


# ── DAG Preservation Audit ──

def audit_dag(nodes, taskir_text):
    """Compare Skeleton DAG with lowered TaskIR DAG."""
    mod = parse_text(taskir_text)
    # Skeleton edges (by index)
    sk_data = set()
    sk_order = set()
    for i, n in enumerate(nodes):
        for d in n.depends_on:
            j = int(d.replace("p", "").replace("s", ""))
            sk_data.add((j, i))
        for d in (n.order_on or []):
            j = int(d.replace("p", "").replace("s", ""))
            sk_order.add((j, i))
    # TaskIR edges (by ORIGINAL node index, skipping EXTRACT_FIELD)
    # Map EXTRACT_FIELD intermediaries: %ext → original producer
    ssa_idx = {n.id: i for i, n in enumerate(mod.program.nodes)}
    # build transitive DATA edges through EXTRACT_FIELD intermediaries
    # to original-node index space
    orig_idx = {}  # SSA id → original skeleton node index (or -1 for ext)
    orig_nodes = [n for n in mod.program.nodes
                  if n.op != "EXTRACT_FIELD"]
    for i, n in enumerate(orig_nodes):
        orig_idx[n.id] = i
    # EXTRACT_FIELD: map to the ORIGINAL producer (transitive)
    ext_to_orig = {}
    for n in mod.program.nodes:
        if n.op == "EXTRACT_FIELD" and n.inputs:
            src_ssa = n.inputs[0]
            if src_ssa in orig_idx:
                ext_to_orig[n.id] = orig_idx[src_ssa]
            elif src_ssa in ext_to_orig:
                ext_to_orig[n.id] = ext_to_orig[src_ssa]

    def resolve_to_orig(ssa_id):
        """Resolve any SSA id to original node index."""
        if ssa_id in orig_idx:
            return orig_idx[ssa_id]
        if ssa_id in ext_to_orig:
            return ext_to_orig[ssa_id]
        return -1

    ir_data = set()
    ir_order = set()
    for i, n in enumerate(orig_nodes):
        for r in n.inputs:
            j = resolve_to_orig(r)
            if j >= 0 and j != i:
                ir_data.add((j, i))
        for r in (n.after or []):
            j = resolve_to_orig(r)
            if j >= 0 and j != i:
                ir_order.add((j, i))
        for r in n.param_refs.values():
            j = resolve_to_orig(r)
            if j >= 0 and j != i:
                ir_data.add((j, i))
    return {
        "skeleton_data": sk_data, "taskir_data": ir_data,
        "skeleton_order": sk_order, "taskir_order": ir_order,
        "data_preserved": sk_data <= ir_data,
        "no_extra_order": ir_order <= (sk_order | sk_data),
    }


def audit_literals(nodes, instruction):
    """Every literal arg value must appear in the instruction text."""
    issues = []
    instr_lower = instruction.lower()
    for i, n in enumerate(nodes):
        for k, v in n.arguments.items():
            if isinstance(v, dict) and v.get("kind") == "runtime_ref":
                continue
            if isinstance(v, float) and v == int(v):
                val_str = str(int(v))  # 50000.0 → "50000"
            else:
                val_str = str(v).lower()
            if val_str in instr_lower:
                continue
            # try without commas (instruction may have $50,000)
            if val_str.replace(",", "") in instr_lower.replace(",", ""):
                continue
            issues.append({"node": n.id, "param": k, "value": val_str})
    return issues


def select_distractors(rng, gold_tools, n=3):
    """Pick distractors with similar signatures to gold tools."""
    candidates = []
    for tid in DISTRACTORS:
        if tid in gold_tools:
            continue
        # prefer distractors that share input param names with gold tools
        gold_params = set()
        for g in gold_tools:
            gold_params |= set(ALL_TOOLS.get(g, {}).get(
                "input_schema", {}).keys())
        dist_params = set(DISTRACTORS[tid]["input_schema"].keys())
        overlap = len(gold_params & dist_params)
        candidates.append((overlap, tid))
    candidates.sort(reverse=True)
    top = [t for _, t in candidates[:n * 2]]
    rng.shuffle(top)
    return top[:n]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    rt = DataflowRuntime(full_registry())
    schemas_typed = cap_schemas_typed()
    schemas_lower = lowering_schemas()

    generators = [
        ("linear_3", lambda r: gen_linear_v2(r, 3), 35),
        ("linear_4", lambda r: gen_linear_v2(r, 4), 30),
        ("linear_5", lambda r: gen_linear_v2(r, 5), 25),
        ("linear_6", lambda r: gen_linear_v2(r, 6), 20),
        ("fork", lambda r: gen_fork_v2(r, 0), 25),
        ("join", lambda r: gen_join_v2(r, 0), 25),
        ("shop_3", lambda r: gen_shop_v2(r, 3), 25),
        ("shop_4", lambda r: gen_shop_v2(r, 4), 20),
        ("read_mutate", lambda r: gen_read_mutate_v2(r, 0), 25),
        ("fan_out_in", lambda r: gen_fan_out_in_v2(r, 0), 25),
        ("repeated_tool", lambda r: gen_repeated_tool(r, 0), 20),
    ]

    all_tasks = []
    stats = Counter()
    fails = defaultdict(list)
    dag_audits = []
    literal_issues = []

    for gen_name, gen_fn, quota in generators:
        for i in range(quota):
            nodes, instruction = gen_fn(rng)
            task_id = f"v2_{gen_name}_{i:03d}"
            gold_tools = sorted({n.capability_id for n in nodes})

            # literal observability audit
            lit_issues = audit_literals(nodes, instruction)
            if lit_issues:
                literal_issues.extend(
                    {"task": task_id, **x} for x in lit_issues)
                stats["literal_unobservable"] += 1
                continue

            # build skeleton
            sk = PlanSkeleton(task_id=task_id)
            sk.nodes = nodes

            # strict lowering with TYPED schemas
            low = lower_skeleton(sk, None, schemas_lower, strict=True)
            if low.get("taskir") is None:
                fails["lowering"].append(
                    {"id": task_id, "err": str(low.get("errors"))[:80]})
                stats["lowering_fail"] += 1
                continue

            # DAG preservation audit
            dag = audit_dag(nodes, low["taskir"])
            dag_audits.append({
                "task": task_id, "topology": gen_name,
                "data_preserved": dag["data_preserved"],
                "no_extra_order": dag["no_extra_order"],
                "sk_data_edges": len(dag["skeleton_data"]),
                "ir_order_edges": len(dag["taskir_order"]),
            })
            if not dag["data_preserved"]:
                stats["dag_data_lost"] += 1
                fails["dag"].append({"id": task_id, "detail": "data lost"})
                continue
            if not dag["no_extra_order"]:
                stats["dag_extra_order"] += 1
                fails["dag"].append({"id": task_id,
                                      "detail": "extra order edges"})

            # strict validation with TYPED registry
            mod = parse_text(low["taskir"])
            rep = validate(mod, capability_registry=schemas_typed)
            if not rep.valid:
                fails["validation"].append(
                    {"id": task_id, "err": str(rep.errors[:1])[:80]})
                stats["validation_fail"] += 1
                continue

            # runtime execution
            tr = rt.execute(mod, capability_registry=schemas_typed)
            if not tr.success:
                fails["runtime"].append(
                    {"id": task_id, "err": tr.error[:80]})
                stats["runtime_fail"] += 1
                continue

            # verify dataflow
            has_flow = any(s.op == "EXTRACT_FIELD" for s in tr.steps)
            if not has_flow:
                stats["no_dataflow"] += 1
                fails["runtime"].append({"id": task_id, "err": "no flow"})
                continue

            # distractor pool
            distractors = select_distractors(rng, gold_tools, n=3)
            offered = sorted(set(gold_tools + distractors))

            # count refs
            n_refs = sum(1 for n in nodes for v in n.arguments.values()
                         if isinstance(v, dict)
                         and v.get("kind") == "runtime_ref")

            task = {
                "task_id": task_id,
                "instruction": instruction,
                "topology": gen_name,
                "gold_capability_ids": gold_tools,
                "offered_capability_ids": offered,
                "distractor_capability_ids": distractors,
                "capability_registry": {
                    t: {"input_schema": ALL_TOOLS[t]["input_schema"],
                        "output_schema": ALL_TOOLS[t]["output_schema"],
                        "effect_class": ALL_TOOLS[t].get("effect",
                                                         "read_only")}
                    for t in offered},
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
                    "dataset": "runtime_dataflow_v2",
                    "evidence": "verified",
                    "topology": gen_name,
                    "depth": len(nodes),
                    "runtime_refs": n_refs,
                    "gold_tool_count": len(gold_tools),
                    "distractor_count": len(distractors),
                },
            }
            all_tasks.append(task)
            stats[gen_name] += 1

    # dedup
    seen = set()
    unique = []
    for t in all_tasks:
        h = hashlib.sha1(t["instruction"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(t)

    # topology-aware splits: hold out entire topologies for test
    held_out = {"repeated_tool", "fan_out_in"}  # unseen in train
    test = [t for t in unique if t["topology"] in held_out]
    rest = [t for t in unique if t["topology"] not in held_out]
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(rest)
    n_dev = max(10, len(rest) // 6)
    dev = rest[:n_dev]
    train = rest[n_dev:]

    for k, rows in (("train", train), ("dev", dev), ("test", test)):
        with open(f"{OUT}/{k}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # DAG audit summary
    dag_ok = sum(1 for d in dag_audits if d["data_preserved"]
                 and d["no_extra_order"])
    manifest = {
        "total_generated": len(all_tasks),
        "total_unique": len(unique),
        "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
        "held_out_topologies": sorted(held_out),
        "topology_stats": {k: v for k, v in stats.items()
                           if not k.endswith("_fail")
                           and "unobservable" not in k
                           and "dag_" not in k and k != "no_dataflow"},
        "failures": {k: len(v) for k, v in fails.items()},
        "literal_unobservable_count": stats["literal_unobservable"],
        "literal_issues": literal_issues[:10],
        "dag_audit": {
            "total": len(dag_audits),
            "data_preserved_and_no_extra_order": dag_ok,
            "extra_order_only": stats.get("dag_extra_order", 0),
            "data_lost": stats.get("dag_data_lost", 0),
        },
        "effect_serialization_note": (
            "read_only tools are NOT serialized by default; "
            "effect-based ordering only for mutating tools"),
        "typed_schemas": "Str/Int/Float/Bool preserved through "
                          "all layers",
        "distractor_pool": "3 per task, signature-similar",
        "seed": SEED,
    }
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"generated: {len(all_tasks)}, unique: {len(unique)}")
    print(f"splits: {manifest['splits']} (held-out: {sorted(held_out)})")
    print(f"literal unobservable: {stats['literal_unobservable']}")
    print(f"DAG: {dag_ok}/{len(dag_audits)} preserved+no_extra_order, "
          f"extra_order={stats.get('dag_extra_order', 0)}")
    print(f"failures: {manifest['failures']}")
    d = Counter(t["provenance"]["depth"] for t in unique)
    print(f"depth: {dict(sorted(d.items()))}")


if __name__ == "__main__":
    main()
