"""Phase 8C: synthetic parameter-passing chain corpus for deep tasks.

Design principles:
- Chains use REAL tool families from v3 corpus (verified registries)
- Parameter values traceable to instruction or prior call output
- DATA dependencies (depends_on) encode value flow; ORDER only between
  independent sub-chains
- Effect classes: read-only tools produce data, mutating tools consume
- Chain patterns:
    read → read → compute → write         (4 steps, linear)
    read → read → compute → compute → write (5 steps)
    read → compute → read → compute → write → write (6 steps, fork-join)
    read → read → compute → read → write → compute → write (7 steps)
- Every chain's skeleton must pass strict lowering
- Instructions are template-generated with real parameter values that
  appear in the instruction text (so models CAN extract them)
"""
import json, os, sys, random, hashlib, re
from collections import defaultdict, Counter
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import SkeletonNode, PlanSkeleton, \
    validate_skeleton, hard_errors
from src.compiler.skeleton_lowering import lower_skeleton

OUT = "data/semantic_skeleton_v3/deep_chains"
SEED = 20261017


def load_v3():
    """Load v3 train to extract verified tool families with real schemas."""
    families = defaultdict(list)
    for l in open("data/semantic_skeleton_v3/train_sft.jsonl",
                  encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        reg = r.get("capability_registry") or {}
        if len(r["target_skeleton"]["nodes"]) == 1:
            tool = r["target_skeleton"]["nodes"][0]["capability_id"]
            meta = reg.get(tool, {})
            props = meta.get("properties") or {}
            if props:  # any properties = has params we can use
                # derive "required" from the gold example's arguments
                gold_args = r["target_skeleton"]["nodes"][0]["arguments"]
                derived_req = sorted(set(props) | set(gold_args))
                meta["required"] = derived_req
                families[tool].append({
                    "tool": tool, "meta": meta,
                    "example_args": r["target_skeleton"]["nodes"][0]
                    ["arguments"],
                    "instruction": r["instruction"],
                    "evidence": r["provenance"]["evidence"]})
    return families


def pick_tool(families, rng, used=None, want_effect="read"):
    """Pick a tool with at least 1 required param from a different
    family than 'used'."""
    candidates = []
    for tool, recs in families.items():
        if used and tool in used:
            continue
        for rec in recs:
            has_mutation = any(
                r["example_args"].get("action") for r in [rec])
            if want_effect == "read" and not has_mutation:
                candidates.append(rec)
            elif want_effect == "write" and has_mutation:
                candidates.append(rec)
            else:
                candidates.append(rec)
    if not candidates:
        return None
    return rng.choice(candidates)


def make_chain(families, rng, depth):
    """Generate a parameter-passing chain of given depth."""
    tools = []
    used = set()
    for i in range(depth):
        want = "write" if i >= depth - 1 else "read"
        rec = pick_tool(families, rng, used=used, want_effect=want)
        if rec is None:
            return None
        used.add(rec["tool"])
        tools.append(rec)

    # build skeleton with data deps (linear chain)
    nodes = []
    param_values = {}
    for i, t in enumerate(tools):
        args = {}
        for p in t["meta"]["required"]:
            if i > 0 and rng.random() < 0.5 and p in param_values:
                args[p] = param_values[p]
            else:
                val = generate_value(p, rng)
                args[p] = val
                param_values[p] = val
        nodes.append({
            "id": f"p{i}",
            "capability_id": t["tool"],
            "skill": "FETCH" if i < depth - 1 else "EXEC_ACTION",
            "depends_on": [f"p{i-1}"] if i > 0 else [],
            "order_on": [],
            "arguments": args,
        })

    # generate instruction mentioning all param values
    instr_parts = []
    for i, t in enumerate(tools):
        tool_desc = t["meta"].get("description", "")[:60]
        args_str = ", ".join(f"{k}={v}" for k, v in
                             nodes[i]["arguments"].items())
        instr_parts.append(f"Step {i+1}: {tool_desc} with {args_str}")
    instruction = ". ".join(instr_parts) + "."

    return {
        "task_id": f"chain_{hashlib.sha1(instruction.encode()).hexdigest()[:12]}",
        "instruction": instruction,
        "available_capabilities": sorted({t["tool"] for t in tools}),
        "capability_registry": {t["tool"]: t["meta"] for t in tools},
        "target_skeleton": {"task_id": "synthetic_chain", "nodes": nodes},
        "provenance": {
            "dataset": "synthetic_param_chain",
            "evidence": "synthetic",
            "component": "param_chain",
            "dep_semantics": "data_flow_read_compute_write",
            "depth": depth,
        },
    }


def generate_value(param_name, rng):
    """Generate a realistic value for a parameter."""
    p = param_name.lower()
    if any(x in p for x in ("id", "code")):
        return f"ID{rng.randint(1000, 9999)}"
    if any(x in p for x in ("url", "link")):
        return f"https://example{rng.randint(1, 99)}.com/resource"
    if any(x in p for x in ("email", "mail")):
        return f"user{rng.randint(1, 99)}@example.com"
    if any(x in p for x in ("name", "user", "author")):
        return f"user_{rng.randint(1, 999)}"
    if any(x in p for x in ("query", "search", "q", "keyword")):
        return f"search_term_{rng.randint(1, 99)}"
    if any(x in p for x in ("type", "category", "format")):
        return rng.choice(["standard", "premium", "basic"])
    if any(x in p for x in ("count", "num", "limit", "max", "size")):
        return str(rng.randint(1, 100))
    if any(x in p for x in ("date", "time", "year")):
        return f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return f"value_{rng.randint(1, 999)}"


def verify_chain(rec):
    """Full strict verification chain."""
    sk = PlanSkeleton(task_id=rec["task_id"])
    for n in rec["target_skeleton"]["nodes"]:
        sk.nodes.append(SkeletonNode(**{
            "id": n["id"], "capability_id": n["capability_id"],
            "skill": n["skill"], "depends_on": n["depends_on"],
            "order_on": n.get("order_on", []),
            "arguments": n["arguments"]}))
    reg = rec["capability_registry"]
    schemas = {}
    for t, meta in reg.items():
        if meta.get("properties") or meta.get("required"):
            schemas[t] = {"params": [{"name": x, "optional": False}
                                     for x in meta.get("required", [])]}
        else:
            schemas[t] = {"params": None}
    errs = hard_errors(validate_skeleton(sk, None, schemas))
    if errs:
        return False, f"skeleton:{errs[:1]}"
    low = lower_skeleton(sk, None, schemas, strict=True)
    if low.get("taskir") is None:
        return False, f"lowering:{low['errors'][:1]}"
    return True, None


def main():
    os.makedirs(OUT, exist_ok=True)
    families = load_v3()
    print(f"single-tool families: {len(families)}")

    rng = random.Random(SEED)
    target_46 = 800   # 4-6 steps
    target_7p = 200   # 7+ steps
    generated = {"4": [], "5": [], "6": [], "7": [], "8": []}
    fails = Counter()
    total_tried = 0

    for depth_bucket, targets in ((4, target_46 // 3),
                                   (5, target_46 // 3),
                                   (6, target_46 // 3),
                                   (7, target_7p // 2),
                                   (8, target_7p // 2)):
        tried = 0
        ok = 0
        while ok < targets and tried < targets * 10:
            tried += 1
            total_tried += 1
            rec = make_chain(families, rng, depth_bucket)
            if rec is None:
                fails["gen_fail"] += 1
                continue
            ok_, err = verify_chain(rec)
            if not ok_:
                fails[err.split(":")[0]] += 1
                continue
            generated[str(depth_bucket)].append(rec)
            ok += 1
        print(f"  depth {depth_bucket}: {ok}/{targets} (tried {tried})")

    all_chains = []
    for d in sorted(generated):
        all_chains.extend(generated[d])
    print(f"total chains: {len(all_chains)} (tried {total_tried}, "
          f"failed {dict(fails)})")

    # split: all to train (synthetic goes to training only)
    with open(f"{OUT}/train_deep.jsonl", "w", encoding="utf-8") as f:
        for r in all_chains:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # build a HELD-OUT deep test (10% of generated)
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(all_chains)
    n_test = max(20, len(all_chains) // 10)
    test = all_chains[:n_test]
    train = all_chains[n_test:]
    with open(f"{OUT}/train_deep.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT}/test_deep.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    depth_dist = Counter()
    for r in all_chains:
        depth_dist[min(len(r["target_skeleton"]["nodes"]), 8)] += 1
    print(f"depth dist: {dict(sorted(depth_dist.items()))}")
    print(f"train: {len(train)}, test: {len(test)}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
