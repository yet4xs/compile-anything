#!/bin/bash
# Phase 6B-1 chain: C2 x3 seeds -> C1 x2 seeds -> C3 x1 -> composer eval
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for seed in 42 43 44; do
  echo "=== composer C2 seed${seed} ==="
  /opt/conda/bin/python scripts/train_phase6b_composer.py --arm C2 --seed ${seed} || echo "C2-${seed} FAILED"
done
for seed in 42 43; do
  echo "=== composer C1 seed${seed} ==="
  /opt/conda/bin/python scripts/train_phase6b_composer.py --arm C1 --seed ${seed} || echo "C1-${seed} FAILED"
done
echo "=== composer C3 seed42 ==="
/opt/conda/bin/python scripts/train_phase6b_composer.py --arm C3 --seed 42 || echo "C3 FAILED"

echo "=== composer eval ==="
/opt/conda/bin/python scripts/eval_phase6b_composer.py || echo "EVAL FAILED"
echo "PHASE6B1-CHAIN-DONE"
