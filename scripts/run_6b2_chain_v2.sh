#!/bin/bash
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for SEED in 42 43 44; do
  echo "=== resolver R2 seed${SEED} ==="
  /opt/conda/bin/python scripts/train_phase6b_resolver.py --mode R2 --seed "${SEED}" || echo "R2-${SEED} FAILED"
done
echo "=== resolver eval ==="
/opt/conda/bin/python scripts/eval_phase6b_resolver.py || echo "EVAL FAILED"
echo "PHASE6B2V2-CHAIN-DONE"
