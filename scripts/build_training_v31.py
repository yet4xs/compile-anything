"""Phase 9B-3: training data v3.1 — template-diversified (>=8 shapes).

Round-1 diagnosis: v3 had only 2 templates; the model learned entity
extraction and binding syntax but under-planned lookup depth. v3.1
varies: number of order lookups (1-4), product-query prefixes, address
family (profile-copy nested paths), cancel family, multi-write plans,
and DUAL-SOURCE instructions. Same proven pipeline: every task must
pass strict lowering + typed validation; entities from the real retail
DB; A-test acting users excluded; retail only (condition-B isolation);
no task_id in targets.

Run: D:/miniconda3/envs/tau3/python.exe scripts/build_training_v31.py
"""
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAU2 = Path("D:/ccfa-2026/tau3-bench-src")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TAU2 / "src"))
from src.compiler.plan_skeleton import skeleton_from_dict
from src.compiler.skeleton_lowering import lower_skeleton
from src.ir.parser import parse_text
from src.validator.validator import validate

OUT = REPO / "data" / "runtime_dataflow_v31"
random.seed(45)


def flat(v):
    if isinstance(v, str):
        return v
    k = v.get("kind")
    if k == "scalar":
        return v.get("type", "Any")
    return {"list": "List[Any]", "struct": "Struct", "dict": "Dict",
            "scalar_root": "Any"}.get(k, "Any")


def R(producer, field):
    return {"kind": "runtime_ref", "producer": producer, "field": field}


def M(*elems):
    return {"kind": "runtime_ref_multi",
            "elements": [{"producer": p, "field": f} for p, f in elems]}


def main():
    from tau2.domains.retail.environment import get_environment
    reg = json.loads((REPO / "results/phase9/verified_registry_manifest.json")
                     .read_text(encoding="utf-8"))
    schemas = {k.split(".", 1)[1]: {
        "input_schema": {k2: flat(v2) for k2, v2 in v["input_schema"].items()},
        "output_schema": v["output_schema"],
        "effect_class": v["effect_class"],
        "params": [{"name": k2, "optional": False, "type": flat(v2)}
                   for k2, v2 in v["input_schema"].items()],
        "output_fields": {}} for k, v in reg["tools"].items()
        if k.startswith("retail.")}
    v2reg = {t: {"input_schema": s["input_schema"],
                 "output_schema": s["output_schema"]}
             for t, s in schemas.items()}
    db = json.loads((TAU2 / "data/tau2/domains/retail/db.json")
                    .read_text(encoding="utf-8"))
    fm = json.loads((REPO / "results/phase9/tau3_gold_v2_2_freeze_manifest.json")
                    .read_text(encoding="utf-8"))
    test_users = set()
    for g, ids in fm["groups"].items():
        m = re.match(r"\('retail', '([^']+)'", g)
        if m and any(i in fm["summary"]["condition_A_test"] for i in ids):
            test_users.add(m.group(1))

    users = [u for u in db["users"].values()
             if u["user_id"] not in test_users
             and sum(1 for o in u.get("orders", [])
                     if o in db["orders"]
                     and db["orders"][o].get("status") == "delivered") >= 1]
    random.shuffle(users)
    env = get_environment()
    DISTRACTORS = ["get_item_details", "list_all_product_types",
                   "modify_user_address", "calculate", "find_user_id_by_email",
                   "get_product_details", "get_order_details"]

    def make_task(u, tmpl_id, n_target):
        delivered = [o for o in u["orders"] if o in db["orders"]
                     and db["orders"][o].get("status") == "delivered"]
        name, zipc = u["name"], u["address"]["zip"]
        k_orders = random.randint(1, min(3, len(delivered)))
        chosen = random.sample(delivered, k_orders)
        target_order = chosen[0]
        order = db["orders"][target_order]
        items = order.get("items", [])
        pm = (order.get("payment_history") or [{}])[0].get(
            "payment_method_id")
        if not pm or len(items) < 1:
            return None
        oidx = {o: u["orders"].index(o) for o in chosen}
        dual = random.random() < 0.3
        instr = (f"reason_for_call: 处理订单"
                 f"{(' ' + target_order) if dual else '（订单号你自己查）'}。"
                 f"known_info: 你是 {name['first_name']} {name['last_name']}，"
                 f"邮编 {zipc}。task_instructions: 一次处理完。")
        nodes = [
            {"id": "p0", "capability_id": "find_user_id_by_name_zip",
             "skill": "EXEC_ACTION", "depends_on": [], "order_on": [],
             "arguments": {"first_name": name["first_name"],
                           "last_name": name["last_name"], "zip": zipc}},
            {"id": "p1", "capability_id": "get_user_details",
             "skill": "EXEC_ACTION", "depends_on": ["p0"], "order_on": [],
             "arguments": {"user_id": R("p0", "user_id")}},
        ]
        op_ids = []
        for j, o in enumerate(chosen):
            pid = f"p{2 + j}"
            nodes.append({"id": pid, "capability_id": "get_order_details",
                          "skill": "EXEC_ACTION", "depends_on": ["p1"],
                          "order_on": [], "arguments": {
                              "order_id": R("p1", f"orders[{oidx[o]}]")}})
            op_ids.append(pid)
        top = op_ids[0]          # the write targets the FIRST looked-up order
        k1 = random.randrange(len(items))

        def add(cap, args, deps):
            pid = f"p{len(nodes)}"
            nodes.append({"id": pid, "capability_id": cap,
                          "skill": "EXEC_ACTION", "depends_on": deps,
                          "order_on": [], "arguments": args})
            return pid

        tmpl = tmpl_id % 8
        if tmpl == 0:      # return
            k2 = random.randrange(len(items))
            add("return_delivered_order_items",
                {"order_id": R(top, "order_id"),
                 "item_ids": M((top, f"items[{k1}].item_id"),
                               (top, f"items[{k2}].item_id"))
                 if len(items) > 1 else M((top, f"items[{k1}].item_id")),
                 "payment_method_id": R(top, "payment_history[0]."
                                        "payment_method_id")}, [top])
        elif tmpl == 1:    # exchange with product prefix
            pr = items[k1].get("product_id")
            pp = add("get_product_details",
                     {"product_id": R(top, f"items[{k1}].product_id")}, [top])
            add("exchange_delivered_order_items",
                {"order_id": R(top, "order_id"),
                 "item_ids": M((top, f"items[{k1}].item_id")),
                 "new_item_ids": M((pp, f"variants.{items[k1]['item_id']}"
                                    ".item_id")),
                 "payment_method_id": R(top, "payment_history[0]."
                                        "payment_method_id")}, [top, pp])
        elif tmpl == 2:    # modify items
            add("modify_pending_order_items",
                {"order_id": R(top, "order_id"),
                 "item_ids": M((top, f"items[{k1}].item_id")),
                 "new_item_ids": M((top, f"items[{k1}].item_id")),
                 "payment_method_id": R(top, "payment_history[0]."
                                        "payment_method_id")}, [top])
        elif tmpl == 3:    # address (profile-copy nested paths)
            add("modify_pending_order_address",
                {"order_id": R(top, "order_id"),
                 "address1": R("p1", "address.address1"),
                 "address2": R("p1", "address.address2"),
                 "city": R("p1", "address.city"),
                 "state": R("p1", "address.state"),
                 "country": R("p1", "address.country"),
                 "zip": R("p1", "address.zip")}, [top, "p1"])
        elif tmpl == 4:    # cancel
            add("cancel_pending_order",
                {"order_id": R(top, "order_id"),
                 "reason": "no longer needed"}, [top])
        elif tmpl == 5 and len(op_ids) > 1:   # two writes, two orders
            add("return_delivered_order_items",
                {"order_id": R(top, "order_id"),
                 "item_ids": M((top, f"items[{k1}].item_id")),
                 "payment_method_id": R(top, "payment_history[0]."
                                        "payment_method_id")}, [top])
            o2 = db["orders"][chosen[1]]
            it2 = o2.get("items", [])
            if it2:
                add("exchange_delivered_order_items",
                    {"order_id": R(op_ids[1], "order_id"),
                     "item_ids": M((op_ids[1], "items[0].item_id")),
                     "new_item_ids": M((op_ids[1], "items[0].item_id")),
                     "payment_method_id": R(op_ids[1], "payment_history[0]."
                                            "payment_method_id")},
                    [op_ids[1]])
        elif tmpl in (6, 7):   # return + extra product query / plain return
            if tmpl == 6:
                add("get_product_details",
                    {"product_id": R(top, f"items[{k1}].product_id")}, [top])
            add("return_delivered_order_items",
                {"order_id": R(top, "order_id"),
                 "item_ids": M((top, f"items[{k1}].item_id")),
                 "payment_method_id": R(top, "payment_history[0]."
                                        "payment_method_id")}, [top])
        else:
            return None

        # production validation
        sk = skeleton_from_dict({"task_id": "t", "nodes": nodes})
        low = lower_skeleton(sk, None, schemas, strict=True)
        if not low.get("taskir"):
            return None
        if not validate(parse_text(low["taskir"]),
                        capability_registry=v2reg).valid:
            return None
        offered = {t: schemas[t] for t in
                   ["find_user_id_by_name_zip", "get_user_details",
                    "get_order_details"]
                   + random.sample(DISTRACTORS, 3)}
        for n in nodes:
            offered.setdefault(n["capability_id"], schemas[n["capability_id"]])
        return {"task_id": f"v31_{n_target:03d}", "instruction": instr,
                "domain": "retail",
                "capability_registry": {
                    t: {"input_schema": s["input_schema"],
                        "output_schema": s["output_schema"],
                        "effect_class": s["effect_class"]}
                    for t, s in offered.items()},
                "target_skeleton": {"nodes": nodes},
                "template": tmpl}

    tasks, seen_users = [], {}
    n_target = 0
    tmpl_seen = set()
    for u in users * 3:
        if n_target >= 330:
            break
        tid = random.randrange(8)
        t = make_task(u, tid, n_target)
        if t is None:
            continue
        # cap per-user to force entity diversity
        if seen_users.get(u["user_id"], 0) >= 3:
            continue
        seen_users[u["user_id"]] = seen_users.get(u["user_id"], 0) + 1
        tasks.append(t)
        tmpl_seen.add(t["template"])
        n_target += 1
    from collections import Counter
    _tc = Counter(x["template"] for x in tasks)
    print("per-template:", dict(sorted(_tc.items())))

    OUT.mkdir(parents=True, exist_ok=True)
    cut = int(len(tasks) * 0.85)
    (OUT / "train.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n"
                for t in tasks[:cut]), encoding="utf-8")
    (OUT / "dev.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n"
                for t in tasks[cut:]), encoding="utf-8")
    man = {"n_generated": len(tasks), "n_train": cut,
           "n_dev": len(tasks) - cut,
           "templates": sorted(tmpl_seen), "n_templates": len(tmpl_seen),
           "distinct_users": len(seen_users),
           "isolation": {"domains": ["retail"],
                         "test_acting_users_excluded": len(test_users),
                         "no_task_id_in_targets": True},
           "provenance": {"registry": "verified_registry_manifest",
                          "entities": "official retail db (5ade54e)"},
           "valid": "all tasks pass strict lowering + typed validation"}
    (OUT / "manifest.json").write_text(
        json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
