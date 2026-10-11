#!/bin/bash
# Phase 7C-2 driver: dev-30 completion + 3-seed confirmation run.
set -e
cd /ccfa2026/compile-anything
PY=/opt/conda/bin/python

# 1) retag smoke records (same frozen params, same ids) as confirmation
$PY - << 'EOF'
import json
path = 'results/phase7/online_large_runs.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]
for r in rows:
    if r['set'] == 'confirmation-smoke':
        r['set'] = 'confirmation'
        r['retag_note'] = 'smoke run under identical frozen params'
with open(path, 'w') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('retagged', sum(1 for r in rows if r.get('retag_note')))
EOF

# 2) import 7C-1 dev-30 G0/G1/G2 (s43) with recomputed metrics
$PY scripts/import_dev30_7c1.py

# 3) dev-30 s43: G2a + G3 only (G0/G1/G2 imported above)
$PY scripts/eval_online_large.py --seed s43 --dev30 --arms 2,4

# 4) confirmation 150 x 3 seeds x 5 arms
for SEED in s42 s43 s44; do
  echo "=== confirmation $SEED ==="
  $PY scripts/eval_online_large.py --seed $SEED
done

# 5) aggregate + stats
$PY scripts/analyze_online_large.py
echo "DRIVER-DONE"
