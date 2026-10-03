"""Phase 6A Tasks 2-4: build the skill-grounding probe dataset.

Unit: one original tool call -> one probe sample (schema + skill label only;
no TaskIR, no oracle nodes, no downstream plans).

Tiers (frozen in phase6a_manifest.json):
  G1 exact + EXEC_ACTION contamination filter   -> primary
  G2 heuristic                                   -> secondary ablation
  G3 fallback | contradictory_exact              -> audit only (never trained)

Splits (group-aware, frozen):
  train/dev/test_ood by TOOL FAMILY (first-2-tokens group) 80/10/10
  test_id: calls held out at random from train families (in-distribution control)
  action_boundary_test: boundary-skill slice of test_ood with known/unseen verb tags
"""
import json, os, re, sys, random, hashlib
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.audit_phase6a_labels import (load_xlam_schemas, load_tbs_schemas,
                                          family_key, RETRIEVAL_PREFIX_RE,
                                          MUTATION_START_RE)

CORPUS = os.path.join(ROOT, "data/compiler_corpus_v3_1/train.jsonl")
OUT = os.path.join(ROOT, "data/phase6a_grounding")
SEED = 42

BOUNDARY = {"RETRIEVAL": {"SEARCH", "FETCH", "QUERY_DB"},
            "ACTION": {"EXEC_ACTION", "SEND", "SAVE"}}


def skill_family(skill):
    for fam, skills in BOUNDARY.items():
        if skill in skills:
            return fam
    return "OTHER"


def main():
    random.seed(SEED)
    xlam, tbs = load_xlam_schemas(), load_tbs_schemas()
    os.makedirs(OUT, exist_ok=True)

    samples, seen_pairs = [], set()
    tier_counts = Counter()
    skipped = Counter()
    for line in open(CORPUS, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        instruction = (r.get("instruction") or "").strip()
        src = r.get("source", "")
        for low in r.get("lowering") or []:
            tool = low.get("tool", "")
            skill = low.get("skill", "")
            mk = low.get("mapping_kind", "")
            sch = xlam.get(tool) or tbs.get(tool)
            if not sch:
                skipped["no_schema"] += 1
                continue
            # tier assignment (frozen)
            if mk == "exact":
                tier = "G1"
                if skill == "EXEC_ACTION" and RETRIEVAL_PREFIX_RE.match(tool) \
                        and not MUTATION_START_RE.match(sch.get("description", "") or ""):
                    tier = "G3"  # contradictory_exact
                    tier_counts["G3_contradictory_exact"] += 1
            elif mk == "heuristic":
                tier = "G2"
                tier_counts[tier] += 1
            else:
                tier = "G3"
                tier_counts["G3_fallback"] += 1

            key = (tool, hashlib.md5(instruction.encode()).hexdigest()[:8])
            if key in seen_pairs:
                skipped["dup_call"] += 1
                continue
            seen_pairs.add(key)

            fam = family_key(tool)
            samples.append({
                "sample_id": f"p6a-{len(samples):06d}",
                "instruction": instruction,
                "capability": {
                    "name": tool,
                    "description": (sch.get("description", "") or "")[:400],
                    "parameters": _slim_params(sch.get("parameters")),
                },
                "target_skill": skill,
                "skill_family": skill_family(skill),
                "mapping_kind": mk,
                "tier": tier,
                "label_source": f"toolmap:{mk}",
                "source": src,
                "tool_name": tool,
                "tool_family": fam,
                "group_id": fam,
            })

    print(f"samples: {len(samples)}  tiers: {dict(tier_counts)}  skipped: {dict(skipped)}")

    # ── Group-aware split on G1+G2 pool (G3 excluded entirely) ──
    pool = [s for s in samples if s["tier"] in ("G1", "G2")]
    groups = defaultdict(list)
    for s in pool:
        groups[s["group_id"]].append(s)

    fam_names = sorted(groups.keys())
    random.shuffle(fam_names)
    n = len(pool)
    target_train, target_dev = int(n * 0.8), int(n * 0.1)

    train_groups, dev_groups, test_groups = [], [], []
    cnt = 0
    for g in fam_names:
        if cnt < target_train:
            train_groups.append(g)
        elif cnt < target_train + target_dev:
            dev_groups.append(g)
        else:
            test_groups.append(g)
        cnt += len(groups[g])

    def by_group(gs):
        out = []
        for g in gs:
            out.extend(groups[g])
        return out

    train_all = by_group(train_groups)
    dev = by_group(dev_groups)
    test_ood = by_group(test_groups)

    # in-distribution control: hold out 10% of train calls at random (same families)
    random.shuffle(train_all)
    n_id = min(int(len(train_all) * 0.1), 2000)
    test_id = train_all[:n_id]
    train = train_all[n_id:]

    # ── overlap checks (frozen constraints) ──
    def names_of(ss):
        return set(s["tool_name"] for s in ss)
    g_overlap = len(set(train_groups) & set(dev_groups)) + \
        len(set(train_groups) & set(test_groups)) + len(set(dev_groups) & set(test_groups))
    name_overlap = len(names_of(train) & names_of(dev)) + \
        len(names_of(train) & names_of(test_ood)) + len(names_of(dev) & names_of(test_ood))

    # instruction near-duplicate (document-level): an instruction whose shingle
    # set overlaps train's by jaccard >= 0.8 counts as leakage (frozen rule)
    def doc_shingles(instruction):
        toks = re.findall(r"\w+", instruction.lower())
        if len(toks) < 5:
            return {hash(" ".join(toks))}
        return {" ".join(toks[i:i + 5]) for i in range(len(toks) - 4)}

    train_docs = [doc_shingles(s["instruction"]) for s in train]
    train_sh_union = set().union(*train_docs) if train_docs else set()
    train_sh_counts = Counter()
    for d in train_docs:
        train_sh_counts.update(d)

    def near_dups(rows):
        n = 0
        for s in rows:
            d = doc_shingles(s["instruction"])
            inter = sum(1 for x in d if x in train_sh_counts)
            union = len(d) + len(train_sh_union) - inter
            if union and inter / union >= 0.8:
                n += 1
        return n

    near_overlap = near_dups(dev) + near_dups(test_ood)

    # ── action_boundary_test: boundary slice of test_ood with verb/family tags ──
    train_verbs = set(s["tool_name"].split("_")[0].lower() for s in train)
    abt = []
    for s in test_ood:
        if s["skill_family"] in ("RETRIEVAL", "ACTION"):
            verb = s["tool_name"].split("_")[0].lower()
            s2 = dict(s)
            s2["verb"] = verb
            s2["verb_seen"] = verb in train_verbs
            s2["family_seen"] = False  # group split => all test families unseen
            abt.append(s2)

    # ── write ──
    def dump(name, rows):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for s in rows:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")

    dump("train.jsonl", train)
    dump("dev.jsonl", dev)
    dump("test_ood.jsonl", test_ood)
    dump("test_id.jsonl", test_id)
    dump("action_boundary_test.jsonl", abt)
    dump("audit_g3.jsonl", [s for s in samples if s["tier"] == "G3"])

    def dist(rows):
        c = Counter((s["target_skill"], s["tier"]) for s in rows)
        return {f"{k[0]}/{k[1]}": v for k, v in sorted(c.items(), key=lambda x: -x[1])[:12]}

    manifest = {
        "seed": SEED,
        "total_calls": len(samples),
        "tiers": dict(tier_counts),
        "groups": {"train": len(train_groups), "dev": len(dev_groups),
                   "test_ood": len(test_groups)},
        "sizes": {"train": len(train), "dev": len(dev),
                  "test_ood": len(test_ood), "test_id": len(test_id),
                  "action_boundary_test": len(abt)},
        "overlap_checks": {"family_group_overlap": g_overlap,
                           "tool_name_overlap": name_overlap,
                           "instruction_shingle_overlap": near_overlap,
                           "all_must_be_zero": True},
        "skill_dist_train": dist(train),
        "skill_dist_test_ood": dist(test_ood),
        "boundary_dist_test_ood": dict(Counter(s["skill_family"] for s in test_ood
                                               if s["skill_family"] != "OTHER")),
    }
    with open(os.path.join(OUT, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=1, ensure_ascii=False)[:1800])


def _slim_params(p):
    """Keep parameter schema small; handles both shapes:
    {"properties": {...}, "required": [...]} (toolbench) and
    {param: {"type": ...}} flat (xLAM)."""
    if not isinstance(p, dict):
        return {}
    req = p.get("required") if isinstance(p.get("required"), list) else []
    props = p.get("properties")
    if not isinstance(props, dict):
        props = {k: v for k, v in p.items()
                 if k not in ("required", "type")}
    return {"required": req[:8],
            "properties": {k: str(v.get("type", v) if isinstance(v, dict) else v)[:12]
                           for k, v in list(props.items())[:10]}}


if __name__ == "__main__":
    main()
