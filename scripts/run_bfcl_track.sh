#!/bin/bash
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== training BFCL Track A (3B, 2 epochs) ==="
/opt/conda/bin/python scripts/train_bfcl_competitive.py --seed 42 --epochs 2.0 || echo TRAIN-FAILED
echo "=== evaluating ==="
/opt/conda/bin/python scripts/eval_bfcl_track.py || echo EVAL-FAILED
echo "BFCL-TRACK-DONE"
