"""Phase 9B-2.5 (owner option B): expand the strict pool past the 15-task
threshold by reconstructing LOOKUP PREFIXES for compressed-gold tasks.

Compressed gold = reference actions contain only write ops; their params
landed ENV_STATE because no prior output existed. We prepend the natural
lookup chain (find_user -> get_user_details -> get_order_details per
order -> get_product_details per exchange product); the v2 classifier
then sources every write param through the prefix outputs. A second
pass discovers additional source ORDERS from blocked params' db_paths
(e.g. modify-address tasks copying the user's other order's address).
Machine adjudication (R1-R4) resolves residual COINCIDENTAL/LOW_ENTROPY.
The read-only prefix cannot change the final DB state; E2E re-verifies.

Run: D:/miniconda3/envs/tau3/python.exe scripts/expand_gold_pool.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAU2 = Path("D:/ccfa-2026/tau3-bench-src")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TAU2 / "src"))
from src.eval.compile_input_v2 import render_compile_input
from scripts.reconstruct_executable_gold_v2 import walk, classify_arg_v2
from scripts.machine_adjudicate import adjudicate

OUT = REPO / "results" / "phase9" / "tau3_gold_compiled_v2_2.json"
OK = {"LITERAL", "DUAL_SOURCE", "runtime_ref", "runtime_ref_multi",
      "SAME_ENTITY_ECHO"}


def process(env_ctor, aug, text, db, u, task_idx):
    """Replay + classify + adjudicate one augmented action list."""
    e2 = env_ctor()
    outputs, step_meta = [], []
    for a in aug:
        m = {"tool": a["name"], "ok": True, "error": None}
        flat = {}
        try:
            obj = e2.tools.tools[a["name"]](**a["arguments"])
            flat = dict(walk(obj))
        except Exception as ex:
            m["ok"] = False
            m["error"] = str(ex)[:100]
        step_meta.append(m)
        outputs.append(flat)

    prog, live, pc = [], [], {}
    for i, a in enumerate(aug):
        args = {}
        for p, v in a["arguments"].items():
            cls, info, acc = classify_arg_v2(v, text, live, db, u, p)
            if cls in ("COINCIDENTAL_VALUE", "LOW_ENTROPY_MATCH"):
                cands = info.get("candidates") or []
                if info.get("elements") and not cands:
                    cands = [x for el in info["elements"]
                             for x in (el.get("candidates") or [])][:8]
                item = {"item_id": f"XP-{task_idx:02d}-{i}-{p}",
                        "class": cls, "consumer": f"{a['name']}.{p}",
                        "value": v,
                        "candidates": [
                            {"producer_step": cc.get("producer_step"),
                             "producer_tool": aug[cc["producer_step"]]["name"]
                             if isinstance(cc.get("producer_step"), int)
                             and 0 <= cc["producer_step"] < len(aug)
                             else "?",
                             "path": cc.get("path")}
                            for cc in cands]}
                j, why = adjudicate(item)
                if j.startswith("ACCEPT_") and cands:
                    cc = cands[int(j.split("_")[1]) - 1]
                    cls, info = "runtime_ref", {
                        "producer_step": cc["producer_step"],
                        "path": cc["path"],
                        "selected_by": "machine_adjudication_"
                                       + why.split(":")[0]}
            pc[cls] = pc.get(cls, 0) + 1
            args[p] = {"class": cls, "value": v, **info}
        if step_meta[i]["ok"]:
            live.append(outputs[i])
        prog.append({"step": i, "tool": a["name"], "arguments": args})

    strict = all(m["ok"] for m in step_meta) and set(pc) <= OK
    for node in prog:
        for a in node["arguments"].values():
            st = [a.get("producer_step")] + [
                e.get("producer_step") for e in a.get("elements", [])]
            if a["class"] in ("runtime_ref", "runtime_ref_multi") \
                    and any(s is not None and s >= node["step"] for s in st):
                strict = False
    return prog, pc, strict


def main():
    from tau2.domains.retail.environment import get_environment
    g = json.loads((REPO / "results" / "phase9" / "tau3_gold_compiled_v2_1.json")
                   .read_text(encoding="utf-8"))
    db = json.loads((TAU2 / "data" / "tau2" / "domains" / "retail" / "db.json")
                    .read_text(encoding="utf-8"))
    raw_by_id = {str(t["id"]): t for t in json.loads(
        (TAU2 / "data" / "tau2" / "domains" / "retail" / "tasks.json")
        .read_text(encoding="utf-8"))}

    expanded, failed = [], []
    for t in g["tasks"]:
        c = t["param_class_counts"]
        if (t.get("strict_v2_1") or not t["replay_ok"]
                or "EXPECTED_ERROR_STEP" in t["flags"]
                or "POLICY_DEPENDENT" in t["flags"]
                or set(c) - OK - {"ENV_STATE"}):
            continue
        raw = raw_by_id[str(t.get("source_task_id")
                            or t["task_id"].rsplit("_", 1)[1])]
        acts = (raw.get("evaluation_criteria") or {}).get("actions") or []
        u = t.get("acting_user")
        if not u:
            for a in acts:
                oid = (a.get("arguments") or {}).get("order_id")
                if isinstance(oid, str) and oid in db.get("orders", {}):
                    u = db["orders"][oid].get("user_id")
                    break
        urec = db["users"].get(u)
        if not urec:
            failed.append((t["task_id"], "no acting user"))
            continue
        text = render_compile_input(raw.get("user_scenario")).lower()
        name, zipc = urec["name"], urec["address"]["zip"]

        order_ids, product_ids = [], []
        for a in acts:
            args = a.get("arguments") or {}
            oid = args.get("order_id")
            if isinstance(oid, str) and oid not in order_ids:
                order_ids.append(oid)
            for nid in (args.get("new_item_ids") or []):
                for pid, prec in db["products"].items():
                    if nid in (prec.get("variants") or {}) \
                            and pid not in product_ids:
                        product_ids.append(pid)

        def build_aug(extra):
            return (
                [{"name": "find_user_id_by_name_zip",
                  "arguments": {"first_name": name["first_name"],
                                "last_name": name["last_name"],
                                "zip": zipc}},
                 {"name": "get_user_details", "arguments": {"user_id": u}}]
                + [{"name": "get_order_details", "arguments": {"order_id": o}}
                   for o in order_ids + [x for x in extra
                                         if x not in order_ids]]
                + [{"name": "get_product_details",
                    "arguments": {"product_id": p}} for p in product_ids]
                + [{"name": a["name"],
                    "arguments": dict(a.get("arguments") or {})}
                   for a in acts])

        extra_orders = []
        aug = build_aug(extra_orders)
        prog = pc = None
        strict = False
        for attempt in range(4):
            prog, pc, strict = process(get_environment, aug, text, db, u,
                                       len(expanded))
            if strict or attempt == 3:
                break
            new_extra, user_votes = [], {}
            for node in prog:
                for a in node["arguments"].values():
                    if a["class"] in ("ENV_STATE", "USER_LATENT"):
                        for dp in (a.get("db_paths") or []):
                            m2 = re.match(r"orders\.([^\.\[]+)\.", dp)
                            if m2 and m2.group(1) not in order_ids + extra_orders:
                                new_extra.append(m2.group(1))
                            mu = re.match(r"users\.([^\.\[]+)\.", dp)
                            if mu:
                                user_votes[mu.group(1)] = \
                                    user_votes.get(mu.group(1), 0) + 1
            new_users = sorted(user_votes, key=user_votes.get,
                               reverse=True)
            if not new_extra and not new_users:
                break
            extra_orders += new_extra
            # majority-vote acting user from blocked params' db_paths —
            # the first write's owner can differ from the scenario user
            if new_users and new_users[0] != u:
                u = new_users[0]
                urec = db["users"].get(u) or urec
                name, zipc = urec["name"], urec["address"]["zip"]
            aug = build_aug(extra_orders)

        t2 = dict(t)
        t2["program"] = prog
        t2["param_class_counts"] = pc
        t2["strict_v2_2"] = strict
        t2["expanded_prefix"] = True
        t2["n_prefix_steps"] = len(aug) - len(acts)
        t2["extra_source_orders"] = extra_orders
        expanded.append(t2)
        print(t["task_id"], "strict_v2_2 =", strict, pc)

    by_id = {t["task_id"]: t for t in expanded}
    for t in g["tasks"]:
        if t["task_id"] not in by_id:
            t["strict_v2_2"] = t.get("strict_v2_1", False)
    g["tasks"] = sorted(
        [t for t in g["tasks"] if t["task_id"] not in by_id] + expanded,
        key=lambda x: x["task_id"])
    n_new = sum(1 for t in expanded if t["strict_v2_2"])
    g["summary"]["strict_v2_2"] = sum(
        t.get("strict_v2_2", False) for t in g["tasks"])
    g["summary"]["meta"]["expansion"] = {
        "method": "lookup-prefix reconstruction, 2-pass order discovery, "
                  "classifier re-sourcing, machine adjudication R1-R4",
        "candidates": len(expanded),
        "flipped_strict": n_new,
        "failed": failed}
    OUT.write_text(json.dumps(g, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print("flipped:", n_new, "/", len(expanded),
          "| total strict_v2_2:", g["summary"]["strict_v2_2"])


if __name__ == "__main__":
    main()
