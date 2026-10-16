"""Phase 9B-2.5 T4: rescore existing predictions with scorer v3
(equivalence classes, machine-adjudicated gold v2.1). Diagnostic only."""
import json, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.ir.parser import parse_text
from src.ir.taskir import to_text, module_from_dict
from src.eval.gold_binding_scorer import (
    score_prediction_v3, calls_from_skeleton_nodes,
    bindings_from_skeleton_nodes)
from scripts.score_tau3_offline import extract_pred_from_taskir

ev = json.loads((REPO/'results/phase9/tau3_real_evaluation.json').read_text(encoding='utf-8'))
gold = json.loads((REPO/'results/phase9/tau3_gold_compiled_v2_1.json').read_text(encoding='utf-8'))
g_by = {t['task_id']: t for t in gold['tasks']}

rows = []
for d in ev['details']:
    t = g_by.get(d['task_id'])
    if not t:
        continue
    try:
        if d['arm'] == 'M1':
            m = re.search(r'\{.*\}', d['raw'], re.S)
            nodes = json.loads(m.group(0)).get('nodes', [])
            calls = calls_from_skeleton_nodes(nodes)
            bindings = bindings_from_skeleton_nodes(nodes)
        elif d['arm'] == 'M0-J':
            ast = json.loads(re.search(r'\{.*\}', d['raw'], re.S).group(0))
            calls, bindings = extract_pred_from_taskir(
                parse_text(to_text(module_from_dict(ast))))
        else:
            calls, bindings = extract_pred_from_taskir(parse_text(d['raw']))
    except Exception:
        continue
    r = score_prediction_v3(calls, bindings, t['program'])
    r.update({'task_id': d['task_id'], 'arm': d['arm'],
              'strict_v2_1_task': t.get('strict_v2_1', False)})
    rows.append(r)

from collections import Counter
arm_stats = {}
for arm in ('M0-T', 'M0-J', 'M1'):
    sub = [r for r in rows if r['arm'] == arm]
    arm_stats[arm] = {
        'n_scored': len(sub),
        'value_match': sum(r['value_match'] for r in sub),
        'equivalence_class_exact': sum(r['equivalence_class_exact'] for r in sub),
        'strict_origin_exact': sum(r['strict_origin_exact'] for r in sub),
        'slots_total': sum(r['n_binding_slots'] for r in sub),
        'bindings_in_class': sum(r['n_bindings_in_class'] for r in sub),
        'bindings_canonical': sum(r['n_bindings_canonical'] for r in sub),
    }
out = {'summary': {'n_details_scored': len(rows), 'by_arm': arm_stats,
                   'note': 'diagnostic only: legacy raw truncated, '
                           'machine-adjudicated gold v2.1, scorer v3'},
       'details': rows}
(REPO/'results/phase9/strict_binding_scorer_v3.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(arm_stats, indent=1))
