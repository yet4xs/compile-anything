"""Phase 9B-2.5 T8: training data v3 — spec + generator (NO training).

Synthesizes executable programs over the VERIFIED retail registry with
real DB entities: typed nested paths (orders[i], payment_history[i].x,
items[i].item_id), runtime_ref_multi + MAKE_LIST, scalar-root producers
(find_user), DUAL_SOURCE, distractor tools, effect classes. Targets
carry NO non-derivable identifiers (no task_id). Condition B isolation:
retail tools only (no airline). Value isolation: acting users of the
frozen A-test are excluded. Validates every task through the production
pipeline and sample-executes on the official simulator.

Run: D:/miniconda3/envs/tau3/python.exe scripts/build_training_v3.py
"""
import dataclasses, hashlib, json, random, re, sys
from pathlib import Path
from pydantic import BaseModel
REPO = Path(__file__).resolve().parents[1]
TAU2 = Path("D:/ccfa-2026/tau3-bench-src")
sys.path.insert(0, str(REPO))
from src.compiler.plan_skeleton import skeleton_from_dict
from src.compiler.skeleton_lowering import lower_skeleton
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.eval.schema_render_v2 import render_registry

OUT = REPO / "data" / "runtime_dataflow_v3"
random.seed(44)

def walk(o, p="", d=0):
    if d > 8: yield p, o; return
    if isinstance(o, BaseModel):
        for n in o.model_fields: yield from walk(getattr(o, n), f"{p}.{n}" if p else n, d+1)
    elif isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, f"{p}.{k}" if p else str(k), d+1)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): yield from walk(v, f"{p}[{i}]", d+1)
    else: yield p, o

def main():
    sys.path.insert(0, str(TAU2 / "src"))
    from tau2.domains.retail.environment import get_environment
    _FLAT = {"list": "List[Any]", "struct": "Struct", "dict": "Dict", "scalar_root": "Any"}
    def _flat(v):
        if isinstance(v, str):
            return v
        k = v.get("kind")
        if k == "scalar":
            return v.get("type", "Any")
        return _FLAT.get(k, "Any")
    reg = json.loads((REPO/"results/phase9/verified_registry_manifest.json").read_text(encoding="utf-8"))
    schemas = {k.split(".",1)[1]: {"input_schema": {k2: _flat(v2) for k2, v2 in v["input_schema"].items()},
        "output_schema": v["output_schema"], "effect_class": v["effect_class"],
        "params": [{"name": k2, "optional": False, "type": _flat(v2)}
                    for k2, v2 in v["input_schema"].items()],
        "output_fields": {}} for k, v in reg["tools"].items() if k.startswith("retail.")}
    v2reg = {t: {"input_schema": {k2: _flat(v2) for k2, v2 in s["input_schema"].items()},
                 "output_schema": s["output_schema"]} for t, s in schemas.items()}
    db = json.loads((TAU2/"data/tau2/domains/retail/db.json").read_text(encoding="utf-8"))
    fm = json.loads((REPO/"results/phase9/tau3_gold_v2_1_freeze_manifest.json").read_text(encoding="utf-8"))
    test_users = set()
    for g, ids in fm["groups"].items():
        m = re.match(r"\('retail', '([^']+)'", g)
        if m and any(i in fm["summary"]["condition_A_test"] for i in ids):
            test_users.add(m.group(1))

    users = [u for u in db["users"].values() if u["user_id"] not in test_users
             and any(o in db["orders"] and db["orders"][o].get("status") == "delivered"
                     for o in u.get("orders", []))]
    random.shuffle(users)
    env = get_environment()
    WRITES = ["return_delivered_order_items", "exchange_delivered_order_items"]
    DISTRACTORS = ["get_item_details", "list_all_product_types", "modify_user_address", "calculate"]
    tasks, stats = [], {"nested": 0, "multi": 0, "scalar_root": 0, "dual": 0,
                        "templates": set(), "valid": 0, "exec_ok": 0, "exec_tried": 0}
    n_target = 160
    for u in users:
        if len(tasks) >= n_target: break
        name = u["name"]; zipc = u["address"]["zip"]
        delivered = [o for o in u["orders"] if o in db["orders"]
                     and db["orders"][o].get("status") == "delivered"]
        if not delivered: continue
        oid = random.choice(delivered)
        order = db["orders"][oid]
        items = order.get("items", [])
        if len(items) < 2: continue
        pm = (order.get("payment_history") or [{}])[0].get("payment_method_id")
        if not pm: continue
        k1, k2 = random.sample(range(len(items)), 2)
        write = random.choice(WRITES)
        dual = random.random() < 0.35
        instr = (f"reason_for_call: 退货/换货。订单号 {oid if dual else '你自己查'}。"
                 f"known_info: 你是 {name['first_name']} {name['last_name']},"
                 f"邮编 {zipc}。task_instructions: 一次性处理完。")
        nodes = [
            {"id":"p0","capability_id":"find_user_id_by_name_zip","skill":"EXEC_ACTION",
             "depends_on":[],"order_on":[],"arguments":{
                 "first_name": name["first_name"], "last_name": name["last_name"], "zip": zipc}},
            {"id":"p1","capability_id":"get_user_details","skill":"EXEC_ACTION",
             "depends_on":["p0"],"order_on":[],"arguments":{
                 "user_id":{"kind":"runtime_ref","producer":"p0","field":"user_id"}}},
            {"id":"p2","capability_id":"get_order_details","skill":"EXEC_ACTION",
             "depends_on":["p1"],"order_on":[],"arguments":{
                 "order_id": {"kind":"runtime_ref","producer":"p1",
                          "field": f"orders[{u['orders'].index(oid)}]"}}},
        ]
        wargs = {"order_id":{"kind":"runtime_ref","producer":"p2","field":"order_id"},
                 "item_ids":{"kind":"runtime_ref_multi","elements":[
                     {"producer":"p2","field":f"items[{k1}].item_id"},
                     {"producer":"p2","field":f"items[{k2}].item_id"}]},
                 "payment_method_id":{"kind":"runtime_ref","producer":"p2",
                                       "field":"payment_history[0].payment_method_id"}}
        if write == "exchange_delivered_order_items":
            pr = items[k1].get("product_id")
            wargs["new_item_ids"] = {"kind":"runtime_ref_multi","elements":[
                {"producer":"p2","field":f"items[{k1}].item_id"}]}
            wargs["payment_method_id"] = {"kind":"runtime_ref","producer":"p2",
                                           "field":"payment_history[0].payment_method_id"}
        nodes.append({"id":"p3","capability_id":write,"skill":"EXEC_ACTION",
                      "depends_on":["p2"],"order_on":[],"arguments":wargs})
        stats["nested"] += 2; stats["multi"] += 1; stats["scalar_root"] += 1
        dual_params = ["order_id"] if dual else []
        tmpl = tuple(n["capability_id"] for n in nodes)
        stats["templates"].add(tmpl)
        # production validation
        sk = skeleton_from_dict({"task_id":"t","nodes":nodes})
        low = lower_skeleton(sk, None, schemas, strict=True)
        ok = bool(low.get("taskir"))
        if ok:
            rep = validate(parse_text(low["taskir"]), capability_registry=v2reg)
            ok = rep.valid
        stats["valid"] += ok
        # official executability sample (first 30 valid)
        exec_ok = None
        if ok and stats["exec_tried"] < 30:
            stats["exec_tried"] += 1
            e2 = get_environment()
            try:
                live = {}
                for n in nodes:
                    args = {}
                    for p, a in n["arguments"].items():
                        if isinstance(a, dict) and a.get("kind") == "runtime_ref":
                            c = (a.get("candidates") or [{"producer": a["producer"],
                                    "path": a["field"]}])[0]
                            from src.compiler.field_path import parse_path, resolve_path
                            src_val = live[c["producer"]]
                            if isinstance(src_val, (str, int, float, bool)):
                                args[p] = src_val  # scalar-root producer
                            elif c["path"]:
                                args[p] = resolve_path(src_val, parse_path(c["path"]))
                            else:
                                args[p] = src_val
                        elif isinstance(a, dict) and a.get("kind") == "runtime_ref_multi":
                            from src.compiler.field_path import parse_path, resolve_path
                            args[p] = [resolve_path(live[e["producer"]], parse_path(e["field"]))
                                       for e in a["elements"]]
                        else:
                            args[p] = a
                    live[n["id"]] = e2.tools.tools[n["capability_id"]](**args)
                exec_ok = True
            except Exception:
                exec_ok = False
            stats["exec_ok"] += bool(exec_ok)
        offered = {t: schemas[t] for t in
                   ["find_user_id_by_name_zip","get_user_details","get_order_details",write]
                   + random.sample(DISTRACTORS, 2)}
        tasks.append({"task_id": f"v3_{len(tasks):03d}",
                      "instruction": instr, "domain": "retail",
                      "capability_registry": {t: {"input_schema": {k2: _flat(v2) for k2, v2 in s2["input_schema"].items()},
                          "output_schema": s2["output_schema"],
                          "effect_class": s2["effect_class"]} for t, s2 in offered.items()},
                      "target_skeleton": {"nodes": nodes},
                      "valid": ok, "exec_sample": exec_ok, "dual_params": dual_params})
    OUT.mkdir(parents=True, exist_ok=True)
    valid_tasks = [t for t in tasks if t["valid"]]
    (OUT/"train.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + chr(10) for t in valid_tasks[:130]), encoding="utf-8")
    (OUT/"dev.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + chr(10) for t in valid_tasks[130:]), encoding="utf-8")
    man = {"n_generated": len(tasks), "n_valid": len(valid_tasks),
           "n_train": min(130, len(valid_tasks)), "n_dev": max(0, len(valid_tasks)-130),
           "stats": {k: (len(v) if isinstance(v, set) else v) for k, v in stats.items()},
           "isolation": {"domains": ["retail"], "airline_tools": 0,
                          "test_acting_users_excluded": len(test_users),
                          "no_task_id_in_targets": True},
           "provenance": {"registry": "verified_registry_manifest (957e1ba)",
                           "entities": "official retail db.json (5ade54e)"}}
    (OUT/"manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(man, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
