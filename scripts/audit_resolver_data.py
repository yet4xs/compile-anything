"""Phase 6B-2 data audits (reviewer Fixes 2/8/9) — CPU, run before eval.

Fix 2: R1 gold visibility (gold_ids present in the <=15 visible blocks?)
Fix 8: effective training distribution of the ACTUAL 50k r2_pairs file
Fix 9 (part): seen/unseen family counts on the test tasks
Outputs results/phase6b/resolver_data_audit.json
"""
import json, os, re
from collections import Counter

ROOT = "/ccfa2026/compile-anything"
D = os.path.join(ROOT, "data/phase6b_resolver")


def visible_block_ids(blocks):
    ids = set()
    for b in blocks:
        m = re.match(r"\[(c\d+)\]", b)
        if m:
            ids.add(m.group(1))
    return ids


# Fix 2: R1 gold visibility
for split in ("r1_train", "r1_test"):
    n = hidden = 0
    for line in open(os.path.join(D, f"{split}.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        vis = visible_block_ids(r["available_blocks"])
        if any(g not in vis for g in r["gold_ids"]):
            hidden += 1
    print(f"{split}: n={n} hidden_gold={hidden} "
          f"visible_rate={100*(n-hidden)/max(1,n):.2f}%")

# Fix 8: effective distribution of the truncated 50k
labels = Counter()
negtypes = Counter()
for line in open(os.path.join(D, "r2_pairs_train.jsonl"), encoding="utf-8"):
    if not line.strip():
        continue
    p = json.loads(line)
    labels[p["label"]] += 1
    if p["label"] == "irrelevant":
        negtypes[p.get("negative_type")] += 1
pos = labels["relevant"]
neg = labels["irrelevant"]
print(f"\neffective 50k distribution: {dict(labels)}")
print(f"pos:neg = 1:{neg/max(1,pos):.2f} | NONE share {100*labels['NONE']/sum(labels.values()):.2f}%")
print(f"negative types: {dict(negtypes)}")

# Fix 9 part: test family counts
train_fams = set()
for line in open(os.path.join(ROOT, "data/compiler_corpus_v4/train.jsonl"), encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        for c in r["canonical_capabilities"]:
            train_fams.add("_".join(c["name"].split("_")[:2]))
seen = unseen = 0
for line in open(os.path.join(D, "r2_tasks_test.jsonl"), encoding="utf-8"):
    if not line.strip():
        continue
    t = json.loads(line)
    fams = {"_".join(re.search(r"name: (\S+)", b.split("\n")[0]).group(1).split("_")[:2])
            for b in t["blocks"].values() if re.search(r"name: (\S+)", b.split("\n")[0])}
    if fams & train_fams:
        seen += 1
    else:
        unseen += 1
print(f"\ntest tasks: seen_family={seen} unseen_family={unseen}")

audit = {
    "fix2_r1_gold_visibility": {"train_visible_pct": None, "test_visible_pct": None},
    "fix8_effective_distribution": {"labels": dict(labels),
                                     "pos_neg_ratio": f"1:{neg/max(1,pos):.2f}",
                                     "none_share_pct": round(100*labels['NONE']/sum(labels.values()), 2),
                                     "negative_types": dict(negtypes)},
    "fix9_test_family_counts": {"seen": seen, "unseen": unseen},
}
# refill fix2 numbers
for i, split in enumerate(("r1_train", "r1_test")):
    n = hidden = 0
    for line in open(os.path.join(D, f"{split}.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        vis = visible_block_ids(r["available_blocks"])
        if any(g not in vis for g in r["gold_ids"]):
            hidden += 1
    key = "train_visible_pct" if i == 0 else "test_visible_pct"
    audit["fix2_r1_gold_visibility"][key] = round(100*(n-hidden)/max(1, n), 2)

os.makedirs(os.path.join(ROOT, "results/phase6b"), exist_ok=True)
with open(os.path.join(ROOT, "results/phase6b/resolver_data_audit.json"), "w",
          encoding="utf-8") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)
print("\nsaved results/phase6b/resolver_data_audit.json")
