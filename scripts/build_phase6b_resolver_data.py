"""Phase 6B-2 B2-B8: build Resolver training/eval data from corpus v4.

Outputs (data/phase6b_resolver/):
  r1_train.jsonl / r1_test.jsonl   — record-level generative format
  r2_pairs_train.jsonl             — (task, capability) -> relevant|irrelevant,
                                      1:3 pos:neg, hard negatives typed
  r2_tasks_test.jsonl              — per-task available table + gold ids (for scoring eval)
  none_tasks.jsonl                 — synthetic NO_CALL tasks (test side)
Leakage: capability blocks never contain selected/reference markers.
"""
import json, os, sys, random, re
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
random.seed(42)

V4 = os.path.join(ROOT, "data/compiler_corpus_v4")
OUT = os.path.join(ROOT, "data/phase6b_resolver")
os.makedirs(OUT, exist_ok=True)

ACTION_FAMS = {"EXEC_ACTION", "SEND", "SAVE"}
RETR_FAMS = {"SEARCH", "FETCH", "QUERY_DB"}


def obj_of(name):
    toks = [t for t in re.split(r"[_\s]", name.lower()) if t]
    verbs = {"get", "list", "search", "find", "fetch", "query", "lookup", "retrieve",
             "show", "view", "cancel", "create", "add", "update", "delete", "remove",
             "book", "reserve", "order", "purchase", "send", "post", "toggle", "grant",
             "revoke", "reset", "set", "make", "apply", "check"}
    keep = [t for t in toks if t not in verbs]
    return "_".join(keep[:2]) if keep else name.lower()


def fam_class(skill):
    if skill in RETR_FAMS:
        return "RETRIEVAL"
    if skill in ACTION_FAMS:
        return "ACTION"
    return "OTHER"


def cap_block(c, canon_by_id):
    cn = canon_by_id.get(c["capability_id"], {})
    lines = [f"[{c['capability_id']}] name: {c['name']}"]
    if c.get("description"):
        lines.append(f"description: {c['description'][:200]}")
    lines.append(f"canonical_skill: {cn.get('canonical_skill', 'UNKNOWN')}")
    return "\n".join(lines)


def classify_negative(cap, sel_caps, canon):
    """Type the difficulty of a negative w.r.t. the selected positive(s)."""
    cn = canon.get(cap["capability_id"], {})
    neg_fam = fam_class(cn.get("canonical_skill", ""))
    neg_obj = obj_of(cap["name"])
    for s in sel_caps:
        scn = canon.get(s["capability_id"], {})
        pos_fam = fam_class(scn.get("canonical_skill", ""))
        pos_obj = obj_of(s["name"])
        if neg_obj and pos_obj and neg_obj == pos_obj and neg_fam != pos_fam:
            return "same_object_opposite_intent"
        if pos_fam == neg_fam and neg_fam != "OTHER":
            return "same_family_wrong_object"
        if " ".join(cap["name"].lower().split("_"))[:6] == \
           " ".join(s["name"].lower().split("_"))[:6]:
            return "same_domain_similar_name"
    return "random_cross_domain"


def load(split):
    recs = []
    for line in open(os.path.join(V4, f"{split}.jsonl"), encoding="utf-8"):
        if line.strip():
            recs.append(json.loads(line))
    return recs


def usable(r):
    canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
    sel = [cid for cid in r["selected_capabilities"]
           if cid in canon and canon[cid]["label_tier"] != "G3"]
    return sel


def main():
    train = load("train")
    test = load("test")

    tooluse_train = [r for r in train if usable(r)]
    print(f"tool-use train records: {len(tooluse_train)}", flush=True)

    # ── R1 record-level ──
    def r1_rows(recs):
        rows = []
        for r in recs:
            sel = usable(r)
            if not sel:
                continue
            caps = {c["capability_id"]: c for c in r["available_capabilities"]}
            canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
            blocks = [cap_block(caps[cid], canon) for cid in caps if cid in canon][:15]
            if not blocks:
                continue
            rows.append({"id": r["id"],
                         "task": r["instruction"],
                         "available_blocks": blocks,
                         "gold_ids": sorted(sel)})
        return rows

    r1_train = r1_rows(tooluse_train)
    r1_test = r1_rows(test)
    with open(f"{OUT}/r1_train.jsonl", "w", encoding="utf-8") as f:
        for x in r1_train:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(f"{OUT}/r1_test.jsonl", "w", encoding="utf-8") as f:
        for x in r1_test:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"R1: train {len(r1_train)} / test {len(r1_test)}", flush=True)

    # ── R2 pairs (1 pos : 3 neg, hard negatives preferred, typed) ──
    pairs = []
    neg_type_dist = Counter()
    NONE_SHARE = 0.10
    all_recs = tooluse_train
    for i, r in enumerate(tooluse_train):
        sel = usable(r)
        caps = {c["capability_id"]: c for c in r["available_capabilities"]}
        canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
        sel_caps = [caps[cid] for cid in sel if cid in caps]
        # positives
        for cid in sel:
            if cid in caps:
                pairs.append({"task": r["instruction"], "block": cap_block(caps[cid], canon),
                              "label": "relevant", "negative_type": None})
        # negatives: rank by hardness
        cands = [c for c in caps.values() if c["capability_id"] not in sel]
        typed = [(classify_negative(c, sel_caps, canon), c) for c in cands]
        typed.sort(key=lambda x: {"same_object_opposite_intent": 0,
                                  "same_domain_similar_name": 1,
                                  "same_family_wrong_object": 2,
                                  "random_cross_domain": 3}[x[0]])
        k = 0
        for t, c in typed:
            if k >= 3:
                break
            pairs.append({"task": r["instruction"], "block": cap_block(c, canon),
                          "label": "irrelevant", "negative_type": t})
            neg_type_dist[t] += 1
            k += 1
    # NONE synthetic tasks (~10% of record count): task + foreign-domain table
    n_none = int(len(tooluse_train) * NONE_SHARE)
    rng = random.Random(43)
    idxs = rng.sample(range(len(tooluse_train)), min(n_none, len(tooluse_train)))
    for j, i in enumerate(idxs):
        r = tooluse_train[i]
        other = tooluse_train[(i + 137) % len(tooluse_train)]
        caps = {c["capability_id"]: c for c in other["available_capabilities"]}
        canon = {c["capability_id"]: c for c in other["canonical_capabilities"]}
        blocks = [cap_block(c, canon) for c in caps.values() if c["capability_id"] in canon][:8]
        if not blocks:
            continue
        pairs.append({"task": r["instruction"], "available_blocks": blocks,
                      "label": "NONE", "negative_type": "synthetic_negative",
                      "synthetic_negative": True})
        neg_type_dist["synthetic_negative(none)"] += 1
    rng.shuffle(pairs)
    # cap at 50k
    if len(pairs) > 50000:
        pairs = pairs[:50000]
    with open(f"{OUT}/r2_pairs_train.jsonl", "w", encoding="utf-8") as f:
        for x in pairs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"R2 pairs: {len(pairs)} | labels: {Counter(p['label'] for p in pairs)}", flush=True)
    print(f"negative types: {dict(neg_type_dist)}", flush=True)

    # ── test-side: per-task tables for scoring eval + synthetic NONE tasks ──
    with open(f"{OUT}/r2_tasks_test.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            sel = usable(r)
            if not sel:
                continue
            caps = {c["capability_id"]: c for c in r["available_capabilities"]}
            canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
            f.write(json.dumps({
                "id": r["id"], "task": r["instruction"],
                "blocks": {cid: cap_block(c, canon) for cid, c in caps.items() if cid in canon},
                "gold_ids": sorted(sel),
                "gold_skills": sorted({canon[cid]["canonical_skill"] for cid in sel if cid in canon}),
            }, ensure_ascii=False) + "\n")
    # NONE test tasks
    ttest = [r for r in test if usable(r)]
    with open(f"{OUT}/none_tasks_test.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(ttest[:150]):
            other = ttest[(i + 71) % len(ttest)]
            caps = {c["capability_id"]: c for c in other["available_capabilities"]}
            canon = {c["capability_id"]: c for c in other["canonical_capabilities"]}
            blocks = {cid: cap_block(c, canon) for cid, c in caps.items() if cid in canon}
            if blocks:
                f.write(json.dumps({"id": r["id"] + "-none", "task": r["instruction"],
                                    "blocks": blocks, "gold_ids": [],
                                    "synthetic_negative": True},
                                   ensure_ascii=False) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
