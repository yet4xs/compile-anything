"""Conditional (Fix 2 branch): rebuild R1 data with gold blocks ALWAYS visible.

Only invoked by the eval-fix chain when audit finds gold visibility < 100%.
Ordering: gold blocks first, then hardest typed negatives, fill to 15.
Overwrites r1_train.jsonl / r1_test.jsonl ONLY (R2 pair data untouched).
"""
import json, os, sys
ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
from scripts.build_phase6b_resolver_data import (cap_block, classify_negative,
                                                 usable)
from scripts.build_common_eval_tasks import canon_of

D = os.path.join(ROOT, "data/phase6b_resolver")
V4 = os.path.join(ROOT, "data/compiler_corpus_v4")

for split, out_name in (("train", "r1_train.jsonl"), ("test", "r1_test.jsonl")):
    rows = []
    for line in open(os.path.join(V4, f"{split}.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        sel = usable(r)
        if not sel:
            continue
        caps = {c["capability_id"]: c for c in r["available_capabilities"]}
        canon = canon_of(r)
        sel_caps = [caps[cid] for cid in sel if cid in caps]
        blocks = [cap_block(caps[cid], canon) for cid in sel if cid in caps]
        others = [c for c in caps.values() if c["capability_id"] not in sel]
        typed = sorted(((classify_negative(c, sel_caps, canon), c["capability_id"])
                        for c in others),
                       key=lambda x: {"same_object_opposite_intent": 0,
                                      "same_domain_similar_name": 1,
                                      "same_family_wrong_object": 2,
                                      "random_cross_domain": 3}[x[0]])
        for t, cid in typed:
            if len(blocks) >= 15:
                break
            blocks.append(cap_block(caps[cid], canon))
        rows.append({"id": r["id"], "task": r["instruction"],
                     "available_blocks": blocks, "gold_ids": sorted(sel)})
    with open(os.path.join(D, out_name), "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"rebuilt {out_name}: {len(rows)} rows (gold-first)")
print("R1-REBUILD-DONE")
