"""Phase 8C-1 Task 5: rescore with v3.1 evaluator."""
import json, sys, os, random
sys.path.insert(0, '.')
os.chdir('/ccfa2026/compile-anything')
from src.eval.semantic_eval_v31 import evaluate_v31
from collections import defaultdict

rows = json.load(open('results/phase8c/m0_m1_deep_comparison.json'))['details']
recs = {}
for sp in ('test_sft', 'test_samefamily', 'test_deep'):
    for l in open(f'data/semantic_skeleton_v4/{sp}.jsonl', encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            recs[r['task_id']] = r

out = []
for r in rows:
    rec = recs.get(r['task_id'])
    if not rec:
        continue
    v31 = evaluate_v31(r.get('taskir_text', ''), rec)
    v31.update({'arm': r['arm'], 'split': r['split'],
                'task_id': r['task_id'], 'depth': r['depth']})
    out.append(v31)

by = defaultdict(list)
for r in out:
    by[(r['arm'], r['split'])].append(r)

summary = {}
for (arm, sp), rows2 in sorted(by.items()):
    n = len(rows2)
    def m(k):
        v = [x.get(k) for x in rows2 if x.get(k) is not None]
        return round(sum(1 for x in v if x) / len(v) * 100, 1) if v else None
    def avg(k):
        v = [x.get(k) for x in rows2 if x.get(k) is not None]
        return round(sum(v) / len(v), 2) if v else None
    summary[f'{sp}|{arm}'] = {
        'n': n,
        'validator_pass': m('validator_pass'),
        'identity': m('identity_exact'),
        'args': m('args_exact'),
        'data_dag': m('data_dag_exact'),
        'order_dag': m('order_dag_exact'),
        'strict_joint': m('strict_joint'),
        'missing_params': avg('missing_params'),
        'wrong_values': avg('wrong_values'),
    }

for sp in ('test_sft', 'test_samefamily', 'test_deep'):
    m0 = {x['task_id']: x for x in out if x['arm'] == 'M0' and x['split'] == sp}
    m1 = {x['task_id']: x for x in out if x['arm'] == 'M1' and x['split'] == sp}
    for key in ('strict_joint', 'identity_exact', 'validator_pass'):
        deltas = []
        fx = bk = 0
        for tid in m0:
            if tid not in m1:
                continue
            a = 1 if m1[tid][key] else 0
            b = 1 if m0[tid][key] else 0
            deltas.append(a - b)
            if a and not b:
                fx += 1
            elif b and not a:
                bk += 1
        if deltas:
            rng = random.Random(13)
            boots = sorted(
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
                for _ in range(10000))
            summary[f'{sp}|paired|{key}'] = {
                'n': len(deltas),
                'diff': round(100 * sum(deltas) / len(deltas), 1),
                'ci95': [round(100 * boots[250], 1),
                         round(100 * boots[9750], 1)],
                'fixed': fx, 'broken': bk}

os.makedirs('results/phase8c', exist_ok=True)
with open('results/phase8c/semantic_eval_v31.json', 'w') as f:
    json.dump({'summary': summary, 'details': out}, f, indent=2,
              ensure_ascii=False)

for k in sorted(summary):
    v = summary[k]
    if 'paired' in k:
        print(k, v)
    else:
        print(k, 'vp={} id={} args={} joint={} dag={}'.format(
            v['validator_pass'], v['identity'], v['args'],
            v['strict_joint'], v['data_dag']))
print('RESCORE-V31-DONE')
