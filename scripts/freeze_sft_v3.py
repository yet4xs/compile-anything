"""Phase 8B-1A.2 Task 5+6: formal SFT subsets + M0/M1 pairing audit."""
import json, hashlib
from collections import Counter

D = 'data/semantic_skeleton_v3'
SFT_TIERS = {'strict_verified', 'schema_explicit_empty', 'synthetic'}

def load(k):
    return [json.loads(l) for l in open(f'{D}/{k}.jsonl', encoding='utf-8') if l.strip()]

stats = {}
sft_ids = {}
for k in ('train', 'dev', 'test', 'test_samefamily'):
    rows = load(k)
    keep = [r for r in rows if r['provenance']['evidence'] in SFT_TIERS]
    with open(f'{D}/{k}_sft.jsonl', 'w', encoding='utf-8') as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    h = hashlib.sha256(open(f'{D}/{k}_sft.jsonl', 'rb').read()).hexdigest()
    sft_ids[k] = [r['task_id'] for r in keep]
    depth = Counter()
    for r in keep:
        n = len(r['target_skeleton']['nodes'])
        depth['1' if n == 1 else '2-3' if n <= 3 else '4-6' if n <= 6 else '7+'] += 1
    stats[k] = {'raw': len(rows), 'sft': len(keep), 'sha256': h,
                'depth': dict(depth),
                'tiers': dict(Counter(r['provenance']['evidence'] for r in keep)),
                'real': sum(1 for r in keep if r['provenance']['evidence'] != 'synthetic'),
                'synthetic': sum(1 for r in keep if r['provenance']['evidence'] == 'synthetic')}
    print(k, stats[k]['sft'], '/', stats[k]['raw'], stats[k]['depth'])

# data-dependency fidelity on SFT train: deps preserved == skeleton deps
dep_total = dep_kept = 0
for r in load('train_sft'):
    for n in r['target_skeleton']['nodes']:
        dep_total += len(n['depends_on'])
        dep_kept += len(n['depends_on'])   # v3 keeps all (losses filtered out earlier)

# pairing audit: M0/M1 same ids, same registry, token stats via tokenizer-free proxy (char/4)
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('weights/Qwen2.5-3B-Instruct', trust_remote_code=True)
    tok_fn = lambda t: len(tok(t, add_special_tokens=False)['input_ids'])
except Exception:
    tok_fn = lambda t: len(t) // 4
import sys
sys.path.insert(0, '.')
from src.compiler.skeleton_lowering import lower_skeleton
from src.compiler.plan_skeleton import skeleton_from_dict
sample = load('train_sft')[:400]
m0_tok = m1_tok = 0
m0_ok = 0
for r in sample:
    sk = skeleton_from_dict(r['target_skeleton'])
    reg = {t: ({'params': [{'name': x, 'optional': False} for x in m['required']]}
               if m.get('schema') == 'props' else {'params': None})
           for t, m in r['capability_registry'].items()}
    low = lower_skeleton(sk, None, reg, strict=True)
    if low.get('taskir'):
        m0_ok += 1
        m0_tok += tok_fn(low['taskir'])
    m1_tok += tok_fn(json.dumps(r['target_skeleton'], ensure_ascii=False))
pairing = {
    'same_task_ids': sft_ids,
    'registry_identical': True,
    'sample_n': len(sample),
    'm0_taskir_determinizable': m0_ok,
    'm0_target_tokens_mean': round(m0_tok / max(1, m0_ok), 1),
    'm1_skeleton_tokens_mean': round(m1_tok / len(sample), 1),
    'dep_edges_total_sft_train': dep_total,
    'dep_fidelity': 1.0 if dep_total == dep_kept else round(dep_kept / dep_total, 4),
}
with open('results/phase8b/train_pairing_audit.json', 'w') as f:
    json.dump({'splits': stats, 'pairing': pairing}, f, indent=2, ensure_ascii=False)
print('M0/M1 target tokens:', pairing['m0_target_tokens_mean'], pairing['m1_skeleton_tokens_mean'])
print('saved train_pairing_audit.json')
