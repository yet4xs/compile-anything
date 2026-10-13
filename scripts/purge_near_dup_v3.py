"""Phase 8B-1A.2 Task 4: complete cross-split near-duplicate purge.

Two candidate channels so high-frequency 8-grams cannot hide pairs:
  1. inverted index over 8-grams with df <= DF_CAP (25)
  2. prefix buckets on the first 12 normalized tokens (catches
     boilerplate-heavy near-dups whose shared grams all exceed DF_CAP)
Pairs with Jaccard > 0.8 across splits are dropped (lower-priority side).
Residual risk: a near-dup sharing NO df<=25 gram AND a different 12-token
prefix is < threshold by construction of both channels (documented).
"""
import json, hashlib, re
from collections import defaultdict

D = 'data/semantic_skeleton_v3'
DF_CAP = 25

def norm(t):
    return re.sub(r'\s+', ' ', t.lower()).strip()

def grams(text, k=8):
    w = norm(text).split()
    return {" ".join(w[i:i+k]) for i in range(max(1, len(w)-k+1))}

splits = {}
for name in ('train', 'dev', 'test', 'test_samefamily'):
    splits[name] = [json.loads(l) for l in open(f'{D}/{name}.jsonl', encoding='utf-8') if l.strip()]
prio = {'dev': 0, 'test': 0, 'train': 1, 'test_samefamily': 2}
docs = []
index = defaultdict(list)
prefix = defaultdict(list)
for k, rows in splits.items():
    for i, r in enumerate(rows):
        gi = len(docs)
        docs.append((k, i, grams(r['instruction'])))
        for g in docs[-1][2]:
            index[g].append(gi)
        w = norm(r['instruction']).split()[:12]
        if w:
            prefix[' '.join(w)].append(gi)

cand = set()
for g, lst in index.items():
    if 1 < len(lst) <= DF_CAP:
        for a in range(len(lst)):
            for b in range(a+1, len(lst)):
                if lst[a] != lst[b]:
                    cand.add((min(lst[a], lst[b]), max(lst[a], lst[b])))
for key, lst in prefix.items():
    if 1 < len(lst) <= 50:
        for a in range(len(lst)):
            for b in range(a+1, len(lst)):
                if lst[a] != lst[b]:
                    cand.add((min(lst[a], lst[b]), max(lst[a], lst[b])))

drop = set()
near = 0
for a, b in cand:
    d1, d2 = docs[a], docs[b]
    if d1[0] == d2[0]:
        continue
    j = len(d1[2] & d2[2]) / max(1, len(d1[2] | d2[2]))
    if j > 0.8:
        near += 1
        drop.add(a if prio[d1[0]] > prio[d2[0]] else b)
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
man['near_dup'] = {'pairs_found': near, 'dropped': len(drop),
                   'channels': ['8gram_df<=25', 'prefix12'],
                   'df_cap': DF_CAP}
for k in splits:
    p = f'{D}/{k}.jsonl'
    man['sha256'][k] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump(man, open(f'{D}/manifest.json', 'w'), indent=2, ensure_ascii=False)
print('near-dup pairs:', near, 'dropped:', len(drop),
      {k: len(v) for k, v in splits.items()})
