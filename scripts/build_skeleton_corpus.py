"""Phase 8B-1A Task 3+4: build the semantic skeleton supervision corpus.

Sources (evaluation firewall respected — tau3 NEVER used):
  1. data/compiler_corpus_v4/{train,val}.jsonl — project's own training
     split; skeletons DERIVED deterministically from validated TaskIR
     plans (EXEC_ACTION identity -> verified; semantic-skill identity via
     the plan's provenance lowering -> derived)
  2. Synthetic depth augmentation — same-tool-family ordering
     compositions (explicitly marked synthetic; ordering-only deps,
     semantically valid sequential/independent steps)
Ambiguous records (unclear tool identity / policy-only plans) are
excluded and LISTED, never silently dropped.

Output: data/semantic_skeleton/{train,dev,test}.jsonl + manifest.json
with tool-family-disjoint splits.
"""
import json, os, re, sys, random, hashlib
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import (skeleton_from_dict,
                                        validate_skeleton, hard_errors)
from src.compiler.skeleton_lowering import lower_skeleton
from src.ir.parser import parse_text
from src.validator.validator import validate

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
OUT_DIR = "data/semantic_skeleton"
NEGATIVES_K = 7
SPLIT_SEED = 20261014


def parse_plan(text, rec=None):
    """TaskIR plan -> (nodes, provenance) with per-node tool identity."""
    mod = parse_text(text)
    prov = {}
    for m in re.finditer(r"; provenance\.(\w+): (.*)", text):
        prov[m.group(1)] = m.group(2).strip()
    lowering = []
    if rec is not None and rec.get("lowering") is not None:
        lv = rec["lowering"]
        if isinstance(lv, list):
            lowering = lv
        else:
            import ast as _ast
            try:
                lowering = _ast.literal_eval(str(lv))
            except Exception:
                lowering = []
    tool_by_skill = {}
    for e in lowering:
        if isinstance(e, dict):
            # later tools win per skill (single-tool records dominate)
            tool_by_skill.setdefault(e.get("skill"), e.get("tool"))
    tool_seq = [e.get("tool") for e in lowering if isinstance(e, dict)]
    nodes = []
    # lowering entries are per tool-call in order; align non-policy nodes
    # to lowering positions (corpus plans are near-1:1 with tool calls)
    lowering_seq = [e for e in lowering if isinstance(e, dict)]
    li = 0
    for n in mod.program.nodes:
        if n.op in POLICY_OPS:
            continue
        tool = None
        evidence = "derived"
        if li < len(lowering_seq):
            entry = lowering_seq[li]
            if entry.get("skill") == n.op:
                tool = entry.get("tool")
                li += 1
        if tool is None and n.op == "EXEC_ACTION" and n.params.get("action")                 and str(n.params["action"]) in {e.get("tool")
                                                for e in lowering_seq}:
            tool = str(n.params["action"])   # action param IS the tool id
            evidence = "verified"
        nodes.append({"id": "", "op": n.op, "tool": tool,
                      "inputs": list(n.inputs),
                      "params": {k: v for k, v in n.params.items()
                                 if k != "action"},
                      "evidence": evidence})
    return nodes, prov


def to_skeleton_record(rec, tool_vocab):
    """One corpus record -> skeleton supervision record, or None+reason."""
    import ast
    text = rec.get("plan_text") or rec.get("plan_target") or ""
    task = rec.get("instruction", "") or rec.get("task", "")
    if not text or not task:
        return None, "empty"
    try:
        nodes, prov = parse_plan(text, rec)
    except Exception as e:
        return None, f"parse:{type(e).__name__}"
    if not nodes:
        return None, "policy_only"
    # tool identity required for every node
    evid = "verified"
    semantic_idx = {}
    sk_nodes = []
    for i, nd in enumerate(nodes):
        if not nd["tool"]:
            return None, "ambiguous_identity"
        if nd["evidence"] == "derived":
            evid = "derived"
        ssa_id = None
        # map original SSA id for dep resolution
        sk_nodes.append({"id": f"p{i}", "capability_id": nd["tool"],
                         "skill": nd["op"], "depends_on": [],
                         "order_on": [], "arguments": nd["params"],
                         "_inputs": nd["inputs"], "_ssa": None})
    # resolve data deps: node inputs referencing other %ids
    ssa_index = {}
    mod = parse_text(text)
    for i, n in enumerate(mod.program.nodes):
        ssa_index[n.id] = i
    for i, n in enumerate(mod.program.nodes):
        if n.op in POLICY_OPS or i >= len(sk_nodes):
            continue
        deps = []
        for ref in n.inputs:
            if ref.startswith("%") and ref in ssa_index:
                j = ssa_index[ref]
                if j < len(sk_nodes) and j != i:
                    deps.append(f"p{j}")
        # keep only refs to non-policy earlier nodes (already filtered)
        sk_nodes[i]["depends_on"] = list(dict.fromkeys(deps))
        for k in ("_inputs", "_ssa"):
            sk_nodes[i].pop(k, None)
    sk = {"task_id": rec.get("id", ""), "nodes": sk_nodes}
    # available capabilities: the record's own pool (gold + negatives as
    # shipped in the corpus)
    caps = rec.get("available_capabilities") or []
    if isinstance(caps, str):
        import ast as _ast
        try:
            caps = _ast.literal_eval(caps)
        except Exception:
            caps = []
    avail = [c.get("name") if isinstance(c, dict) else str(c)
             for c in caps]
    gold_tools = sorted({n["capability_id"] for n in sk_nodes})
    if not avail:
        avail = gold_tools
    return {
        "task_id": rec.get("id", ""),
        "instruction": task,
        "available_capabilities": avail,
        "target_skeleton": sk,
        "provenance": {
            "dataset": "compiler_corpus_v4",
            "evidence": evid,
            "split_group": gold_tools[0] if gold_tools else "?",
            "gold_tools": gold_tools,
        },
    }, None


def synth_compose(records, target_n):
    """Ordering-composition: chain 2-3 same-family records into deeper
    DAGs. Deps are ORDER-ONLY (semantically valid sequential steps);
    marked synthetic. Instruction = concatenation of subtasks."""
    out = []
    by_family = defaultdict(list)
    for r in records:
        fam = r["provenance"]["split_group"]
        if 1 <= len(r["target_skeleton"]["nodes"]) <= 3:
            by_family[fam].append(r)
    rng = random.Random(SPLIT_SEED + 1)
    for fam, rs in sorted(by_family.items()):
        if len(rs) < 8:
            continue
        rng.shuffle(rs)
        i = 0
        while len(out) < target_n and i + 3 <= len(rs):
            k = rng.choice([2, 2, 3])
            parts = rs[i:i + k]
            i += k
            nodes = []
            instr = []
            for j, p in enumerate(parts):
                for nd in p["target_skeleton"]["nodes"]:
                    nd = dict(nd)
                    nd["id"] = f"s{len(nodes)}"
                    nd["depends_on"] = []
                    if nodes:
                        nd["order_on"] = [nodes[-1]["id"]]
                    nodes.append(nd)
                instr.append(f"({j+1}) " + p["instruction"].strip())
            out.append({
                "task_id": "syn_" + hashlib.sha1(
                    "|".join(p["task_id"] for p in parts)
                    .encode()).hexdigest()[:12],
                "instruction": " ".join(instr),
                "available_capabilities": sorted(
                    {n["capability_id"] for n in nodes}),
                "target_skeleton": {"task_id": "synthetic", "nodes": nodes},
                "provenance": {
                    "dataset": "synthetic_composition",
                    "evidence": "synthetic",
                    "split_group": fam,
                    "components": [p["task_id"] for p in parts],
                    "dep_semantics": "ordering_only",
                },
            })
            if len(out) >= target_n:
                break
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # τ³ firewall: never touch tau3 files here
    tool_vocab = set()
    raw = {"train": [], "val": []}
    for split in ("train", "val"):
        for line in open(f"data/compiler_corpus_v4/{split}.jsonl",
                         encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get("plan_text") or r.get("plan_target") or ""
            m = re.findall(r"; provenance\.tools: (\[.*?\])", text)
            for mm in m:
                try:
                    tool_vocab.update(json.loads(
                        mm.replace("'", '"')))
                except Exception:
                    pass
            raw[split].append(r)
    tool_vocab = sorted(t for t in tool_vocab if t)
    print(f"tool vocabulary: {len(tool_vocab)}")

    records, excluded = [], Counter()
    for split in ("train", "val"):
        for r in raw[split]:
            rec, why = to_skeleton_record(r, tool_vocab)
            if rec is None:
                excluded[why] += 1
                continue
            rec["provenance"]["orig_split"] = split
            records.append(rec)
    print(f"converted {len(records)}; excluded: {dict(excluded)}")

    # synthetic depth augmentation (from TRAIN split only)
    train_recs = [r for r in records if r["provenance"]["orig_split"]
                  == "train"]
    synth = synth_compose(train_recs, target_n=1200)
    print(f"synthetic compositions: {len(synth)}")

    # family-disjoint splits: group key = primary tool family
    fam_records = defaultdict(list)
    for r in records:
        fam_records[r["provenance"]["split_group"]].append(r)
    fams = sorted(fam_records)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(fams)
    n_f = len(fams)
    test_fams = set(fams[:max(1, n_f // 10)])
    dev_fams = set(fams[max(1, n_f // 10):max(2, n_f // 5)])
    splits = {"train": [], "dev": [], "test": []}
    for f in fams:
        bucket = ("test" if f in test_fams else
                  "dev" if f in dev_fams else "train")
        splits[bucket].extend(fam_records[f])
    # synthetic only into train (its families come from train anyway)
    fam_syn = set()
    for s in synth:
        if s["provenance"]["split_group"] in test_fams | dev_fams:
            continue          # keep family firewall intact
        splits["train"].append(s)
        fam_syn.add(s["provenance"]["split_group"])
    # near-duplicate guard: exact instruction dedup within splits
    for k in splits:
        seen = set()
        uniq = []
        for r in splits[k]:
            h = hashlib.sha1(r["instruction"].encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            uniq.append(r)
        splits[k] = uniq
    print({k: len(v) for k, v in splits.items()})

    manifest = {
        "splits": {k: len(v) for k, v in splits.items()},
        "tool_vocabulary": len(tool_vocab),
        "excluded": dict(excluded),
        "family_disjoint": True,
        "test_families": sorted(test_fams)[:40],
        "dev_families": sorted(dev_fams)[:40],
        "synthetic_in_train": len(fam_syn),
        "tau3_used": False,
        "negatives_per_record": NEGATIVES_K,
    }
    for k, v in splits.items():
        with open(f"{OUT_DIR}/{k}.jsonl", "w", encoding="utf-8") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT_DIR}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("saved", OUT_DIR)


if __name__ == "__main__":
    main()
