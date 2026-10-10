#!/bin/bash
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/opt/conda/bin/python scripts/train_bfcl_competitive.py \
  --seed 42 --epochs 1.0 \
  --model weights/Qwen2.5-7B-Instruct \
  --out runs/bfcl_track/fc_7b_s42 \
  || echo "7B-TRAIN-FAILED"
/opt/conda/bin/python scripts/eval_bfcl_heldout.py || echo "EVAL-FAILED"
echo "SEVEN-B-DONE"
