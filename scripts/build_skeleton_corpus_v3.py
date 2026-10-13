"""Phase 8B-1A.2: freeze v3 SFT data (final semantic gate).

Over v2:
- synthetic records MERGE the real capability registries of their source
  subtasks (description + schema); same tool id with conflicting schema
  rejects the composition; every synthetic re-runs FULL strict validation
- schema tiers split into strict_verified / schema_explicit_empty
  (properties known, zero required - checkable) / schema_unknown (no
  properties - excluded from formal SFT)
- EXEC_ACTION data dependencies preserved as operands (8B-1A.2 Task 2)
- files frozen with SHA256 in the manifest
"""
import json, os, sys, random, hashlib
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import SkeletonNode, PlanSkeleton
from src.compiler.skeleton_lowering import lower_skeleton
from src.compiler.plan_skeleton import validate_skeleton, hard_errors
from src.ir.parser import parse_text

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
OUT = "data/semantic_skeleton_v3"
SEED = 20261016


def cap_registry(caps_raw):
    reg = {}
    for c in caps_raw:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        p = c.get("parameters") or {}
        if isinstance(p, str):
            try:
                import ast
                p = ast.literal_eval(p)
            except Exception:
                p = {}
        req = [x for x in (p.get("required") or []) if isinstance(x, str)] \
            if isinstance(p, dict) else []
        props = {k: v for k, v in
                 (p.get("properties") or {}).items()} \
            if isinstance(p, dict) else {}
        reg[c["name"]] = {
            "description": c.get("description", ""),
            "required": req, "properties": props,
            "schema": ("props" if props else
                       "explicit_empty" if isinstance(p, dict) and p
                       else "unknown"),
        }
    return reg


def strict_check(sk_nodes, reg, task_id):
    """Full strict chain: registry membership -> schema -> strict
    lowering. Returns (tier, error)."""
    sk = PlanSkeleton(task_id=task_id)
    for n in sk_nodes:
        sk.nodes.append(SkeletonNode(
            id=n["id"], capability_id=n["capability_id"],
            skill=n["skill"], depends_on=n["depends_on"],
            order_on=n.get("order_on", []), arguments=n["arguments"]))
    schemas = {}
    for t, meta in reg.items():
        if meta["schema"] == "props":
            schemas[t] = {"params": [{"name": x, "optional": False}
                                     for x in meta["required"]]}
        else:
            schemas[t] = {"params": None}
    errs = hard_errors(validate_skeleton(sk, None, schemas))
    called = {n.capability_id for n in sk.nodes}
    missing = called - set(reg)
    if errs or missing:
        return "semantic_loss", f"registry:{sorted(missing)[:2] or errs[:1]}"
    for n in sk.nodes:
        meta = reg[n.capability_id]
        if meta["schema"] == "props":
            for r in meta["required"]:
                if r not in n.arguments:
                    return "semantic_loss", f"missing_param:{n.id}.{r}"
    low = lower_skeleton(sk, None, schemas, strict=True)
    if low.get("taskir") is None:
        return "semantic_loss", str(low["errors"][:1])[:120]
    tier = ("schema_explicit_empty"
            if all(reg[n.capability_id]["schema"] != "props"
                   for n in sk.nodes) else "strict_verified")
    return tier, None


def extract(rec):
    text = rec.get("plan_target") or ""
    if not text or not rec.get("instruction"):
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
    lseq = [e for e in lowering if isinstance(e, dict)]
    reg = cap_registry(rec.get("available_capabilities") or [])
    nodes, id_map = [], {}
    li = 0
    for n in mod.program.nodes:
        if n.op in POLICY_OPS:
            continue
        tool = None
        if li < len(lseq) and lseq[li].get("skill") == n.op:
            tool = lseq[li].get("tool")
            li += 1
        if tool is None and n.op == "EXEC_ACTION" \
                and n.params.get("action") in {e.get("tool")
                                               for e in lseq}:
            tool = str(n.params["action"])
        id_map[n.id] = len(nodes)
        nodes.append({"ssa": n.id, "op": n.op, "tool": tool,
                      "inputs": [r for r in n.inputs
                                 if r.startswith("%")],
                      "params": {k: v for k, v in n.params.items()
                                 if k != "action"}})
    if not nodes:
        return None, "ambiguous", ["policy_only"]
    if any(not n["tool"] for n in nodes):
        return None, "ambiguous", ["identity_unresolved"]
    sk_nodes = []
    lost = 0
    for i, n in enumerate(nodes):
        deps = []
        for ref in n["inputs"]:
            if ref in id_map and id_map[ref] != i:
                deps.append(f"p{id_map[ref]}")
            elif ref.startswith("%"):
                lost += 1
        sk_nodes.append({"id": f"p{i}", "capability_id": n["tool"],
                         "skill": n["op"], "depends_on":
                             list(dict.fromkeys(deps)),
                         "order_on": [], "arguments": n["params"]})
    if lost:
        return {"task_id": rec.get("id", ""), "nodes": sk_nodes}, \
            "semantic_loss", [f"unsupported_dep:{lost}"]
    unknown_schema = any(reg.get(n["capability_id"], {})
                         .get("schema") == "unknown"
                         for n in sk_nodes)
    tier, err = strict_check(sk_nodes, reg, rec.get("id", ""))
    if tier == "semantic_loss":
        return {"task_id": rec.get("id", ""), "nodes": sk_nodes}, tier, [err]
    if unknown_schema:
        tier = "schema_unknown"
    return {"task_id": rec.get("id", ""), "nodes": sk_nodes}, tier, []


def synth(parts):
    """Compose with REAL merged registries; reject schema conflicts."""
    reg = {}
    for p in parts:
        for t, meta in p["registry"].items():
            if t in reg and reg[t] != meta:
                return None, f"schema_conflict:{t}"
            reg[t] = meta
    nodes, instr = [], []
    for j, p in enumerate(parts):
        base = len(nodes)
        remap = {n["id"]: f"s{base + k}"
                 for k, n in enumerate(p["sk"]["nodes"])}
        for n in p["sk"]["nodes"]:
            nodes.append({"id": remap[n["id"]],
                          "capability_id": n["capability_id"],
                          "skill": n["skill"],
                          "depends_on": [remap[d] for d in n["depends_on"]
                                         if d in remap],
                          "order_on": [], "arguments": dict(n["arguments"])})
        if j > 0:
            nodes[base]["order_on"] = [nodes[base - 1]["id"]]
        instr.append(f"({j+1}) " + p["instruction"].strip())
    # FULL strict validation of the composition
    tier, err = strict_check(nodes, reg, "synthetic")
    if tier != "strict_verified" and tier != "schema_explicit_empty":
        return None, err
    return {
        "task_id": "synv3_" + hashlib.sha1(
            "|".join(p["task_id"] for p in parts).encode()
        ).hexdigest()[:12],
        "instruction": " ".join(instr),
        "available_capabilities": sorted(reg),
        "capability_registry": reg,
        "target_skeleton": {"task_id": "synthetic_v3", "nodes": nodes},
        "provenance": {"dataset": "synthetic_composition_v3",
                       "evidence": "synthetic",
                       "component": parts[0]["component"],
                       "components": [p["task_id"] for p in parts],
                       "dep_semantics": "subgraph_data_preserved+inter_order"},
        "sk": {"nodes": nodes},
        "instruction_text": " ".join(instr),
    }, None


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = []
    for split in ("train", "val"):
        for line in open("data/compiler_corpus_v4/"
                         f"{split}.jsonl", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["_orig_split"] = split
                raw.append(r)
    conv = []
    tiers = Counter()
    for r in raw:
        caps = r.get("available_capabilities") or []
        if isinstance(caps, str):
            try:
                import ast
                caps = ast.literal_eval(caps)
            except Exception:
                caps = []
        sk, tier, diag = extract(r)
        tiers[tier] += 1
        if sk is None:
            continue
        conv.append({
            "task_id": r.get("id", ""),
            "instruction": r.get("instruction", ""),
            "available_capabilities": [c.get("name") for c in caps
                                       if isinstance(c, dict)],
            "capability_registry": cap_registry(caps),
            "target_skeleton": sk,
            "provenance": {"dataset": "compiler_corpus_v4",
                           "evidence": tier,
                           "orig_split": r["_orig_split"],
                           "gold_tools": sorted({n["capability_id"]
                                                 for n in sk["nodes"]}),
                           "diagnostics": [str(d) for d in diag][:3]},
            "sk": sk, "instruction_text": r.get("instruction", ""),
        })
    print("real tiers:", dict(tiers))

    # components over all tools
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for r in conv:
        ts = r["provenance"]["gold_tools"]
        for t in ts[1:]:
            union(ts[0], t)
    for r in conv:
        r["component"] = find(r["provenance"]["gold_tools"][0])
    comps = defaultdict(list)
    for r in conv:
        comps[r["component"]].append(r)

    # split by component (same policy as v2: giant components -> train)
    rng = random.Random(SEED)
    sizes = {c: len(v) for c, v in comps.items()}
    small = sorted([c for c, n in sizes.items() if n < 400])
    big = [c for c in sizes if sizes[c] >= 400]
    rng.shuffle(small)
    n_s = len(small)
    assign = {}
    for i, c in enumerate(small):
        assign[c] = ("test" if i < max(1, n_s // 10) else
                     "dev" if i < max(2, 2 * n_s // 10) else "train")
    for c in big:
        assign[c] = "train"
    splits = defaultdict(list)
    for c, rs in comps.items():
        splits[assign[c]].extend(rs)

    # synthetic from TRAIN components only, with registry merge + recheck
    train_comps = {c: rs for c, rs in comps.items() if assign[c] == "train"}
    syn_ok, syn_rej = 0, Counter()
    cand = []
    for c, rs in sorted(train_comps.items()):
        pool_ = [r for r in rs if 1 <= len(r["sk"]["nodes"]) <= 3]
        if len(pool_) < 8:
            continue
        rng.shuffle(pool_)
        for i in range(0, len(pool_) - 1, 3):
            k = rng.choice([2, 2, 3])
            parts = pool_[i:i + k]
            if len(parts) < 2:
                continue
            regs = [{t: m for t, m in p["capability_registry"].items()}
                    for p in parts]
            cand.append(([{**p, "registry": reg, "component": c}
                          for p, reg in zip(parts, regs)], c))
            if len(cand) >= 1400:
                break
        if len(cand) >= 1400:
            break
    for parts, c in cand:
        s, err = synth(parts)
        if s is None:
            syn_rej[err.split(":")[0]] += 1
            continue
        splits["train"].append(s)
        syn_ok += 1
    print(f"synthetic ok={syn_ok} rejected={dict(syn_rej)}")

    # same-family test slice from giant components
    rng2 = random.Random(SEED + 2)
    samefam = []
    for c in big:
        cands = [r for r in comps[c]
                 if r["provenance"]["evidence"] == "strict_verified"]
        rng2.shuffle(cands)
        take = cands[:250]
        samefam.extend(take)
        drop = {r["task_id"] for r in take}
        splits["train"] = [r for r in splits["train"]
                           if r["task_id"] not in drop]
    splits["test_samefamily"] = samefam

    # exact-dedup within + across (keep dev/test)
    prio = {"dev": 0, "test": 0, "train": 1, "test_samefamily": 2}
    seen_global = defaultdict(set)     # hash -> set of splits
    keep = defaultdict(list)
    order = sorted(splits.keys(),
                   key=lambda k: (prio[k], 0 if k in ("dev", "test") else 1))
    docs = {k: splits[k] for k in splits}
    drop_count = 0
    for k in order:
        out = []
        for r in docs[k]:
            h = hashlib.sha1(r["instruction"].encode()).hexdigest()
            if any(h in seen_global[k2] for k2 in seen_global
                   if prio.get(k2, 9) < prio[k]):
                drop_count += 1
                continue
            seen_global[k].add(h)
            out.append(r)
        splits[k] = out
    print("exact-dup drops:", drop_count)

    # final stats + SHA256 freeze
    manifest = {"v": 3, "tiers_real": dict(tiers),
                "synthetic_ok": syn_ok, "synthetic_rejected": dict(syn_rej),
                "tau3_used": False, "splits": {}, "sha256": {},
                "non_isolatable_components_forced_train":
                    [str(c) for c in big]}
    for k in ("train", "dev", "test", "test_samefamily"):
        rows = splits[k]
        path = f"{OUT}/{k}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    x: r[x] for x in ("task_id", "instruction",
                                      "available_capabilities",
                                      "capability_registry",
                                      "target_skeleton", "provenance")
                    if x in r}, ensure_ascii=False) + "\n")
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        manifest["sha256"][k] = h
        manifest["splits"][k] = len(rows)
        depth = Counter()
        syn = sum(1 for r in rows
                  if r["provenance"].get("evidence") == "synthetic")
        for r in rows:
            n = len(r["target_skeleton"]["nodes"])
            depth["1" if n == 1 else "2-3" if n <= 3
                  else "4-6" if n <= 6 else "7+"] += 1
        manifest.setdefault("depth", {})[k] = dict(depth)
        manifest.setdefault("synthetic", {})[k] = syn
    with open(f"{OUT}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("splits:", manifest["splits"])
    print("depth:", manifest["depth"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
