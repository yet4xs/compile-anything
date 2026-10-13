"""Purge cross-split near-duplicates (J>0.8, 8-gram) from semantic_skeleton_v2.
Priority: keep dev/test; drop from test_samefamily first, then train."""
import json, hashlib
from collections import defaultdict

D = 'data/semantic_skeleton_v2'

def grams(text, k=8):
    w = text.split()
    return {" ".join(w[i:i+k]) for i in range(max(1, len(w)-k+1))}

splits = {}
for name in ('train', 'dev', 'test', 'test_samefamily'):
    try:
        splits[name] = [json.loads(l) for l in open(f'{D}/{name}.jsonl', encoding='utf-8') if l.strip()]
    except FileNotFoundError:
        pass
prio = {'dev': 0, 'test': 0, 'train': 1, 'test_samefamily': 2}
index = defaultdict(list)
docs = []
for k, rows in splits.items():
    for i, r in enumerate(rows):
        docs.append((k, i, grams(r['instruction'])))
        for g in docs[-1][2]:
            index[g].append(len(docs) - 1)
drop = set()
for g, lst in index.items():
    if 1 < len(lst) <= 4:
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                d1, d2 = docs[lst[a]], docs[lst[b]]
                if d1[0] == d2[0]:
                    continue
                j = len(d1[2] & d2[2]) / max(1, len(d1[2] | d2[2]))
                if j > 0.8:
                    drop.add(lst[a] if prio[d1[0]] > prio[d2[0]] else lst[b])
for gi in drop:
    k, i, _ = docs[gi]
    splits[k][i] = None
for k in splits:
    splits[k] = [x for x in splits[k] if x]
    with open(f'{D}/{k}.jsonl', 'w', encoding='utf-8') as f:
        for r in splits[k]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
man = json.load(open(f'{D}/manifest.json'))
man['splits'] = {k: len(v) for k, v in splits.items()}
man['near_dup_purged'] = len(drop)
json.dump(man, open(f'{D}/manifest.json', 'w'), indent=2, ensure_ascii=False)
print('purged', len(drop), {k: len(v) for k, v in splits.items()})
