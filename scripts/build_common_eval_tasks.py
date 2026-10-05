"""Fixes 3/6/7: build the matched common candidate set + typed test metadata
+ synthetic-NONE collision audit. Deterministic. Train data untouched.

common_eval_tasks.jsonl: per task exactly
  all gold capabilities + hardest typed negatives (same ranking as training)
  + random fill to 15, each candidate tagged {capability_id, is_gold,
  negative_type}.
none_tasks_common.jsonl: synthetic NONE tasks on the SAME candidate-set
  protocol, with collision audit (skill/object overlap with the task's gold).
"""
import json, os, re, random
from collections import Counter

ROOT = "/ccfa2026/compile-anything"
sys_path = os.path.join(ROOT)
import sys
sys.path.insert(0, sys_path)
from scripts.build_phase6b_resolver_data import (cap_block, classify_negative,
                                                 usable, obj_of)

D = os.path.join(ROOT, "data/phase6b_resolver")
V4 = os.path.join(ROOT, "data/compiler_corpus_v4")
random.seed(4242)


def canon_of(r):
    return {c["capability_id"]: c for c in r["canonical_capabilities"]}


def build_task_entry(r, force_ids=None):
    sel = usable(r)
    if not sel:
        return None
    caps = {c["capability_id"]: c for c in r["available_capabilities"]}
    canon = canon_of(r)
    sel_caps = [caps[cid] for cid in sel if cid in caps]
    # candidates: gold + typed-hardest negatives + fill
    cands = [(True, "gold", cid) for cid in sel if cid in caps]
    others = [c for c in caps.values() if c["capability_id"] not in sel]
    typed = sorted(((classify_negative(c, sel_caps, canon), c["capability_id"])
                    for c in others),
                   key=lambda x: {"same_object_opposite_intent": 0,
                                  "same_domain_similar_name": 1,
                                  "same_family_wrong_object": 2,
                                  "random_cross_domain": 3}[x[0]])
    for t, cid in typed:
        if len(cands) >= 15:
            break
        cands.append((False, t, cid))
    return {
        "id": r["id"], "task": r["instruction"],
        "candidates": [{"capability_id": cid, "is_gold": is_gold,
                         "negative_type": nt,
                         "block": cap_block(caps[cid], canon)}
                        for is_gold, nt, cid in cands],
        "gold_ids": sorted(sel),
        "gold_skills": sorted({canon[cid]["canonical_skill"]
                               for cid in sel if cid in canon}),
    }


def main():
    test = [json.loads(l) for l in open(os.path.join(V4, "test.jsonl"),
                                        encoding="utf-8") if l.strip()]
    entries = []
    for r in test:
        e = build_task_entry(r)
        if e:
            entries.append(e)
    with open(os.path.join(D, "common_eval_tasks.jsonl"), "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    ntype_dist = Counter(c["negative_type"] for e in entries for c in e["candidates"])
    print(f"common_eval_tasks: {len(entries)} | candidate types: {dict(ntype_dist)}")

    # Fix 7: synthetic NONE with collision audit
    tooltest = [r for r in test if usable(r)]
    none_entries = []
    collisions = 0
    for i, r in enumerate(tooltest[:150]):
        other = tooltest[(i + 71) % len(tooltest)]
        e = build_task_entry(other)
        if not e:
            continue
        # collision: any candidate's skill/object matches THIS task's gold
        my_canon = canon_of(r)
        my_gold_skills = {my_canon[cid]["canonical_skill"] for cid in usable(r)
                          if cid in my_canon}
        my_objs = {obj_of(my_canon[cid]["name"]) for cid in usable(r)
                   if cid in my_canon}
        coll = any((c["is_gold"] and
                    (re.search(r"canonical_skill: (\S+)", c["block"]).group(1)
                     in my_gold_skills))
                   for c in e["candidates"])
        coll = coll or any(c["is_gold"] and
                           obj_of(re.search(r"name: (\S+)", c["block"]).group(1))
                           in my_objs for c in e["candidates"])
        if coll:
            collisions += 1
        none_entries.append({**e, "id": r["id"] + "-none", "orig_gold_ids": e["gold_ids"],
                             "gold_ids": [], "synthetic_negative": True,
                             "potential_collision": bool(coll)})
    clean = [e for e in none_entries if not e["potential_collision"]]
    with open(os.path.join(D, "none_tasks_common.jsonl"), "w", encoding="utf-8") as f:
        for e in none_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"synthetic NONE: n={len(none_entries)} collisions={collisions} "
          f"clean={len(clean)}")
    print("DONE")


if __name__ == "__main__":
    main()
