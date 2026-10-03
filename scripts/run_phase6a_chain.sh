#!/bin/bash
# Phase 6A chain: 4 training runs then full internal evaluation
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== train g1 seed42 ==="
/opt/conda/bin/python scripts/train_phase6a_grounder.py --config g1 --seed 42 || exit 1
echo "=== train g1 seed43 ==="
/opt/conda/bin/python scripts/train_phase6a_grounder.py --config g1 --seed 43 || exit 1
echo "=== train g1 seed44 ==="
/opt/conda/bin/python scripts/train_phase6a_grounder.py --config g1 --seed 44 || exit 1
echo "=== train g1g2 seed42 ==="
/opt/conda/bin/python scripts/train_phase6a_grounder.py --config g1g2 --seed 42 || exit 1
echo "=== eval ==="
/opt/conda/bin/python scripts/eval_phase6a.py || exit 1
echo "PHASE6A-CHAIN-DONE"
