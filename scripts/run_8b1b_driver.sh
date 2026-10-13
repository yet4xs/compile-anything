#!/bin/bash
set -e
cd /ccfa2026/compile-anything
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/bin/python

echo "=== M0 full training ==="
$PY scripts/train_m0_m1.py --model m0

echo "=== patch budget manifest ==="
$PY - << 'PYEOF'
import json
log = json.load(open('runs/phase8b/m0_s42/train_log.json'))
mp = 'experiments/phase8b/m0_m1_training_manifest.json'
man = json.load(open(mp))
man['budget']['max_steps'] = log['max_steps']
man['budget']['m0_target_tokens'] = log['target_tokens']
man['budget']['note'] = ('both models run identical optimizer steps; '
                         'set by M0 2-epoch plan')
json.dump(man, open(mp, 'w'), indent=2)
print('budget max_steps =', log['max_steps'])
PYEOF

echo "=== M1 full training (same steps) ==="
$PY scripts/train_m0_m1.py --model m1

echo "=== paired evaluation ==="
$PY scripts/eval_m0_m1.py
echo "DRIVER-8B1B-DONE"
