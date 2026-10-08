#!/bin/bash
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== 7B MAX: 3 epochs, full 60k xLAM, seq 2048 ==="
/opt/conda/bin/python scripts/train_bfcl_max.py --seed 42 --epochs 3.0 || echo "MAX-TRAIN-FAILED"
echo "=== Dual evaluation (held-out + full) ==="
/opt/conda/bin/python scripts/eval_bfcl_max.py || echo "MAX-EVAL-FAILED"
echo "BFCL-MAX-ALL-DONE"
