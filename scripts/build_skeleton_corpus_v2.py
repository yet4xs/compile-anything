"""Phase 8B-1A.1: strict semantic supervision corpus v2.

Fixes over v1 (advisor review):
  A. synthetic compositions KEEP subgraph data dependencies; order_on is
     added only BETWEEN consecutive subtasks; no data->order conversion.
  B. per-record tool registry built ONLY from available_capabilities
     (with description + parameter schema); schema_missing tools are
     marked schema_unknown, never treated as 'no required params'.
  C. splits are CONNECTED COMPONENTS of the tool co-occurrence graph
     (all tools, not the first gold tool); synthetic lineage inherits
     source components and stays in the source split.
  D. strict lowering: type-incompatible data deps FAIL compilation
     (semantic_loss tier); runtime-input needs are unresolved; the
     diagnostic coercion mode remains available but its outputs never
     enter strict SFT.
  E. original SSA id -> skeleton id mapping (no positional indexing
     across the POLICY_OPS filter); deps through policy nodes or
     dangling refs are marked unsupported.
Output: data/semantic_skeleton_v2/{train,dev,test}.jsonl + manifest.
v1 files are left untouched.
"""
import json, os, re, sys, random, hashlib
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import SkeletonNode, PlanSkeleton
from src.compiler.skeleton_lowering import lower_skeleton
from src.ir.parser import parse_text

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
OUT_DIR = "data/semantic_skeleton_v2"
SPLIT_SEED = 20261015


def cap_registry(caps_raw):
    """available_capabilities -> {tool: {...schema...}, ...}."""
    reg = {}
    for c in caps_raw:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name:
            continue
        p = c.get("parameters") or {}
        if isinstance(p, str):
            try:
                import ast
                p = ast.literal_eval(p)
            except Exception:
                p = {}
        req = []
        props = {}
        if isinstance(p, dict):
            req = [x for x in (p.get("required") or [])
                   if isinstance(x, str)]
            props = {k: v for k, v in (p.get("properties") or {}).items()
                     if isinstance(k, str)}
        reg[name] = {
            "description": c.get("description", ""),
            "required": req,
            "properties": props,
            "schema_known": bool(req or props),
        }
    return reg


def extract_skeleton(rec, reg):
    """Task 1+E: original-SSA-id-keyed extraction with dependency
    cross-check. Returns (skeleton_dict, tier, diagnostics)."""
    text = rec.get("plan_target") or rec.get("plan_text") or ""
    instruction = rec.get("instruction", "")
    if not text or not instruction:
        return None, "ambiguous", ["empty"]
    try:
        mod = parse_text(text)
    except Exception as e:
        return None, "ambiguous", [f"parse:{type(e).__name__}"]
    lowering = rec.get("lowering") or []
    if isinstance(lowering, str):
        try:
            import ast
            lowering = ast.literal_eval(lowering)
        except Exception:
            lowering = []
    lowering_seq = [e for e in lowering if isinstance(e, dict)]

    nodes = []                       # skeleton nodes in plan order
    id_map = {}                      # original SSA id -> skeleton node idx
    diags = []
    li = 0
    for n in mod.program.nodes:
        if n.op in POLICY_OPS:
            diags.append({"kind": "policy_node_dropped", "ssa": n.id})
            continue
        tool = None
        if li < len(lowering_seq) and lowering_seq[li].get("skill") == n.op:
            tool = lowering_seq[li].get("tool")
            li += 1
        if tool is None and n.op == "EXEC_ACTION" \
                and n.params.get("action") in {e.get("tool")
                                               for e in lowering_seq}:
            tool = str(n.params["action"])
        id_map[n.id] = len(nodes)
        nodes.append({"ssa": n.id, "op": n.op, "tool": tool,
                      "inputs": [r for r in n.inputs if r.startswith("%")],
                      "params": {k: v for k, v in n.params.items()
                                 if k != "action"}})
    if not nodes:
        return None, "ambiguous", ["policy_only"]
    if any(not n["tool"] for n in nodes):
        return None, "ambiguous", ["identity_unresolved"]

    sk_nodes = []
    unsupported = []
    for i, n in enumerate(nodes):
        deps, order = [], []
        for ref in n["inputs"]:
            if ref not in id_map:
                unsupported.append({"node": f"p{i}", "dep": ref,
                                    "reason": "dangling_or_policy"})
                continue
            j = id_map[ref]
            if j == i:
                continue
            deps.append(f"p{j}")
        sk_nodes.append({"id": f"p{i}", "capability_id": n["tool"],
                         "skill": n["op"],
                         "depends_on": list(dict.fromkeys(deps)),
                         "order_on": order,
                         "arguments": n["params"]})
    # registry membership: every called tool must be in the record's OWN
    # available capabilities
    missing_reg = sorted({n["capability_id"] for n in sk_nodes}
                         - set(reg))
    if missing_reg:
        return {"task_id": rec.get("id", ""), "nodes": sk_nodes}, \
            "semantic_loss", [f"registry_missing:{missing_reg[:3]}"]
    if unsupported:
        return {"task_id": rec.get("id", ""), "nodes": sk_nodes}, \
            "semantic_loss", [f"unsupported_dep:{len(unsupported)}"]

    sk = PlanSkeleton(task_id=rec.get("id", ""))
    for n in sk_nodes:
        sk.nodes.append(SkeletonNode(**{
            "id": n["id"], "capability_id": n["capability_id"],
            "skill": n["skill"], "depends_on": n["depends_on"],
            "order_on": n["order_on"], "arguments": n["arguments"]}))

    # strict lowering (registry = record capabilities; required params
    # from real schemas when known)
    schemas = {t: {"params": ([{"name": r, "optional": False}
                               for r in reg[t]["required"]]
                              if reg[t]["schema_known"] else None),
                   "schema_known": reg[t]["schema_known"]}
               for t in reg}
    low = lower_skeleton(sk, None, schemas, strict=True)
    if low.get("taskir") is None:
        errs = low.get("errors") or [f"unresolved:{low.get('unresolved')}"]
        e0 = str(errs[0])
        tier = ("semantic_loss" if "semantic_loss" in e0
                or "incompatible" in e0 else
                "semantic_loss" if e0.startswith("unresolved")
                else "semantic_loss")
        return {"task_id": rec.get("id", ""), "nodes": sk_nodes}, \
            tier, [e0[:120]]
    schema_known = all(reg[n["capability_id"]]["schema_known"]
                       for n in sk_nodes)
    tier = "strict_verified" if schema_known else "schema_unknown"
    # derived tier when identity came from lowering (not action param)
    return {"task_id": rec.get("id", ""), "nodes": sk_nodes,
            "lowered_taskir": low["taskir"]}, tier, diags


def synth_compose_v2(records_by_component, target_n):
    """Task 2+A: preserve subgraph DATA deps; remap ids; one order_on
    edge between consecutive subtasks; components never cross splits."""
    out = []
    rng = random.Random(SPLIT_SEED + 1)
    comps = {c: [r for r in rs
                 if 1 <= len(r["sk"]["nodes"]) <= 3]
             for c, rs in records_by_component.items()}
    for comp, rs in sorted(comps.items()):
        if len(rs) < 8:
            continue
        rng.shuffle(rs)
        i = 0
        while i + 2 <= len(rs) and len(out) < target_n:
            k = rng.choice([2, 2, 3])
            parts = rs[i:i + k]
            i += k
            nodes = []
            instr = []
            for j, p in enumerate(parts):
                base = len(nodes)
                id_remap = {}
                for nd in p["sk"]["nodes"]:
                    id_remap[nd["id"]] = f"s{len(nodes)}"
                    nodes.append({
                        "id": id_remap[nd["id"]],
                        "capability_id": nd["capability_id"],
                        "skill": nd["skill"],
                        # DATA dependencies preserved via the remap
                        "depends_on": [id_remap[d]
                                       for d in nd["depends_on"]
                                       if d in id_remap],
                        "order_on": [],
                        "arguments": dict(nd["arguments"]),
                    })
                if j > 0 and base < len(nodes):
                    # single inter-subtask ORDER edge (no data claim)
                    nodes[base]["order_on"] = [nodes[base - 1]["id"]]
                instr.append(f"({j+1}) " + p["instruction"].strip())
            out.append({
                "task_id": "synv2_" + hashlib.sha1(
                    "|".join(p["task_id"] for p in parts)
                    .encode()).hexdigest()[:12],
                "instruction": " ".join(instr),
                "available_capabilities": sorted(
                    {n["capability_id"] for n in nodes}),
                "target_skeleton": {"task_id": "synthetic_v2",
                                    "nodes": nodes},
                "provenance": {
                    "dataset": "synthetic_composition_v2",
                    "evidence": "synthetic",
                    "component": comp,
                    "components": [p["task_id"] for p in parts],
                    "dep_semantics": "subgraph_data_preserved+inter_order",
                },
                "sk": {"nodes": nodes},
            })
    return out[:target_n]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = []
    for split in ("train", "val"):
        for line in open(f"data/compiler_corpus_v4/{split}.jsonl",
                         encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["_orig_split"] = split
                raw.append(r)
    print(f"raw records: {len(raw)}")

    converted = []
    tiers = Counter()
    for r in raw:
        caps = r.get("available_capabilities") or []
        if isinstance(caps, str):
            try:
                import ast
                caps = ast.literal_eval(caps)
            except Exception:
                caps = []
        reg = cap_registry(caps)
        sk, tier, diags = extract_skeleton(r, reg)
        tiers[tier] += 1
        if sk is None:
            continue
        tools = sorted({n["capability_id"] for n in sk["nodes"]})
        converted.append({
            "task_id": r.get("id", ""),
            "instruction": r.get("instruction", ""),
            "available_capabilities": [c.get("name") for c in caps
                                       if isinstance(c, dict)],
            "capability_registry": reg,
            "target_skeleton": sk,
            "provenance": {
                "dataset": "compiler_corpus_v4",
                "evidence": tier,
                "orig_split": r["_orig_split"],
                "gold_tools": tools,
                "diagnostics": [str(d) for d in diags][:4],
            },
            "sk": sk,
            "instruction_text": r.get("instruction", ""),
        })
    print("tiers:", dict(tiers))

    # ---- Task 5: connected components over tool co-occurrence ----
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for r in converted:
        ts = r["provenance"]["gold_tools"]
        for t in ts[1:]:
            union(ts[0], t)
    comp_of = {r["task_id"]: find(r["provenance"]["gold_tools"][0])
               for r in converted if r["provenance"]["gold_tools"]}
    comps = defaultdict(list)
    for r in converted:
        c = comp_of.get(r["task_id"])
        if c is not None:
            comps[c].append(r)
    comp_sizes = sorted((len(v) for v in comps.values()), reverse=True)
    print(f"components: {len(comps)}; top sizes: {comp_sizes[:5]}")

    # split by component (largest component forces non-strict isolation;
    # reported honestly instead of hidden)
    comp_ids = sorted(comps, key=lambda c: -len(comps[c]))
    rng = random.Random(SPLIT_SEED)
    small = [c for c in comp_ids if len(comps[c]) < 400]
    big = [c for c in comp_ids if len(comps[c]) >= 400]
    rng.shuffle(small)
    n_small = len(small)
    test_c = set(small[:max(1, n_small // 10)])
    dev_c = set(small[max(1, n_small // 10):max(2, 2 * n_small // 10)])
    assign = {}
    for c in small:
        assign[c] = ("test" if c in test_c else
                     "dev" if c in dev_c else "train")
    # big components all go to train; reported as non-isolatable
    for c in big:
        assign[c] = "train"
    splits = defaultdict(list)
    for c, rs in comps.items():
        for r in rs:
            splits[assign[c]].append(r)
    # synthetic: only from TRAIN components, inherit the component
    train_by_comp = {c: rs for c, rs in comps.items() if assign[c] == "train"}
    synth = synth_compose_v2(train_by_comp, target_n=1000)
    for s in synth:
        splits["train"].append(s)
    non_isolatable = [str(c) for c in big]

    # same-family test slice: sampled from the giant components with
    # exact-instruction exclusion vs train — explicitly NOT unseen-family
    samefam = []
    rng2 = random.Random(SPLIT_SEED + 2)
    for c in big:
        cands = [r for r in comps[c]
                 if r["provenance"]["evidence"] == "strict_verified"]
        rng2.shuffle(cands)
        take = cands[:250]
        samefam.extend(take)
        drop = {r["task_id"] for r in take}
        splits["train"] = [r for r in splits["train"]
                           if r["task_id"] not in drop]

    # exact-instruction dedup within splits
    for k in list(splits):
        seen, uniq = set(), []
        for r in splits[k]:
            h = hashlib.sha1(r["instruction"].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                uniq.append(r)
        splits[k] = uniq
    # cross-set exact-dup removal (samefam vs train)
    if samefam:
        train_hashes = {hashlib.sha1(r["instruction"].encode()).hexdigest()
                        for r in splits["train"]}
        samefam = [r for r in samefam
                   if hashlib.sha1(r["instruction"].encode()).hexdigest()
                   not in train_hashes]
        splits["test_samefamily"] = samefam

    manifest = {
        "v": 2,
        "tiers": dict(tiers),
        "splits": {k: len(v) for k, v in splits.items()},
        "components_total": len(comps),
        "non_isolatable_components_forced_train": non_isolatable,
        "isolation_claim": ("strict unseen-family for dev/test ONLY IF "
                            "non_isolatable list is empty; otherwise "
                            "dev/test families unseen, train contains "
                            "large shared components (reported)")
            if non_isolatable else "strict unseen-family",
        "synthetic_count": len(synth),
        "test_samefamily": {
            "n": len(splits.get("test_samefamily", [])),
            "claim": "SAME-FAMILY held-out tasks from the giant tool "
                     "component (exact-dup removed vs train); NOT an "
                     "unseen-family claim. Primary unseen-family test = "
                     "strict dev/test splits + tau3-unused external pool."},
        "tau3_used": False,
    }
    for k in ("train", "dev", "test", "test_samefamily"):
        if k not in splits:
            continue
        with open(f"{OUT_DIR}/{k}.jsonl", "w", encoding="utf-8") as f:
            for r in splits[k]:
                out = {x: r[x] for x in ("task_id", "instruction",
                                         "available_capabilities",
                                         "target_skeleton", "provenance")
                       if x in r}
                if "capability_registry" in r:
                    out["capability_registry"] = r["capability_registry"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
    with open(f"{OUT_DIR}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("splits:", {k: len(v) for k, v in splits.items()})
    print("saved", OUT_DIR)


if __name__ == "__main__":
    main()
