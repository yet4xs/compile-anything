"""Fix-up: (1) empty-pred programs score ordered_recall=0 (not excluded);
(2) D-condition plan uses RAW traces (over-long) -> report D binding-only
as upper bound; recompute summary + attribution from rescored details."""
import json, sys
from collections import Counter, defaultdict

rows = [json.loads(l) for l in open(
    'results/phase8a/semantic_details_rescored.jsonl', encoding='utf-8')
        if l.strip()]
for r in rows:
    if r['l0']['parse_ok'] and r['l3']['ordered_recall'] is None:
        r['l3']['ordered_recall'] = 0.0
    if r['l0']['parse_ok'] and not r['l2']['pred_seq']:
        r['l2']['exact_f1'] = 0.0

CONDS = ['A_G0', 'A_G1', 'A_G2', 'B_G0', 'C_G0', 'D_refplan']
by_cond = defaultdict(list)
for r in rows:
    by_cond[r['condition']].append(r)

def agg(rs):
    n = max(1, len(rs))
    def m(a, b):
        vals = [x[a][b] for x in rs if a in x and x[a].get(b) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {'n': len(rs),
            'valid%': round(100 * sum(x['l0']['valid'] for x in rs) / n, 1),
            'L1_f1': m('l1', 'f1'), 'L2_exact_f1': m('l2', 'exact_f1'),
            'L2_seq%': round(100 * sum(x['l2']['exact_sequence_match'] for x in rs) / n, 1),
            'L3_ordRec': m('l3', 'ordered_recall'),
            'L3_argAcc': m('l3', 'arg_value_acc'),
            'L4_effect': m('l4', 'effect_consistency'),
            'L4_bind': m('l4', 'bindability'),
            'steps_pred/ref': (m('l3', 'pred_steps'), m('l3', 'ref_steps'))}

summary = {c: agg(by_cond[c]) for c in CONDS if by_cond[c]}
by = {(r['condition'], r['case_id']): r for r in rows}
def sem_ok(r):
    return bool(r and r['l0']['valid'] and r['l2']['exact_f1'] >= 0.99)
attr = Counter()
flags = defaultdict(list)
cids = sorted({r['case_id'] for r in rows})
for cid in cids:
    a, b, c = (by.get((k, cid)) for k in ('A_G0', 'B_G0', 'C_G0'))
    dd = by.get(('D_refplan', cid))
    if a is None:
        continue
    if a['l0']['valid'] and not sem_ok(a):
        flags[cid].append('validator_only_error'); attr['validator_only_error'] += 1
    if not sem_ok(a) and sem_ok(b):
        flags[cid].append('canonicalizer_error'); attr['canonicalizer_error'] += 1
    elif not sem_ok(b) and sem_ok(c):
        flags[cid].append('resolver_error'); attr['resolver_error'] += 1
    elif not sem_ok(c):
        if c['l2']['exact_f1'] < 0.99:
            flags[cid].append('composer_planning_error'); attr['composer_planning_error'] += 1
        if c['l1']['f1'] < 0.99 and c['l2']['exact_f1'] >= 0.99:
            flags[cid].append('tool_binding_error'); attr['tool_binding_error'] += 1
    if c is not None and sem_ok(c) and c['l1']['f1'] < 0.99:
        flags[cid].append('tool_binding_error'); attr['tool_binding_error'] += 1
    if dd is not None and not sem_ok(dd):
        flags[cid].append('binding_upper_bound_gap'); attr['binding_upper_bound_gap'] += 1

trans = {}
for arm in ('A_G1', 'A_G2'):
    up = down = 0
    for cid in cids:
        ra, r0 = by.get((arm, cid)), by.get(('A_G0', cid))
        if ra and r0:
            if sem_ok(ra) and not sem_ok(r0): up += 1
            elif sem_ok(r0) and not sem_ok(ra): down += 1
    trans[arm] = {'semantic_fixed': up, 'semantic_broken': down}

out = json.load(open('results/phase8a/semantic_layers.json'))
out['summary'] = summary
out['attribution'] = {'counts': dict(attr), 'n_tasks': len(cids),
                      'n_flagged': len(flags),
                      'n_multi_cause': sum(1 for v in flags.values() if len(v) > 1)}
out['g_semantic_transitions'] = trans
out['caveats'] = [
    'D_refplan is built from RAW traces (one node per action) and is '
    'over-long vs the normalized reference: its L2/seq are LOWER bounds; '
    'its binding metrics (argAcc/bindability/effect) are the upper-bound '
    'signal',
    'ordered_recall scores 0 for parsed-but-empty programs (not excluded)',
    'attribution keeps multi-cause overlaps; B-condition valid-collapse '
    '(15%) reflects composer training-distribution coupling, so the '
    'canonicalizer branch rarely fires solo']
json.dump(out, open('results/phase8a/semantic_layers.json', 'w'), indent=2,
          ensure_ascii=False)
json.dump({'counts': dict(attr), 'n_tasks': len(cids), 'n_flagged': len(flags),
           'n_multi_cause': sum(1 for v in flags.values() if len(v) > 1),
           'task_flags': dict(flags)},
          open('results/phase8a/error_attribution.json', 'w'), indent=2,
          ensure_ascii=False)
print(f"{'cond':<10}{'valid%':>8}{'L1f1':>7}{'L2ex':>7}{'seq%':>6}{'ordR':>7}{'eff':>7}{'bind':>7}{'steps':>12}")
for c in CONDS:
    s = summary.get(c)
    if s:
        print(f"{c:<10}{s['valid%']:>7.1f}%{s['L1_f1']:>7}{s['L2_exact_f1']:>7}{s['L2_seq%']:>6.1f}{s['L3_ordRec']:>7}{s['L4_effect']:>7}{s['L4_bind']:>7}{str(s['steps_pred/ref']):>12}")
print('attribution:', dict(attr))
print('transitions:', trans)
