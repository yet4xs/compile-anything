#!/bin/bash
# 7-Hour Plan v2 — reordered for maximum value within time budget
# Block 1 (2h):  6B-3 external tau3 (paper's core architecture test)
# Block 2 (1h):  BFCL held-out evaluation (3B + zero-shot baseline)
# Block 3 (3.5h): BFCL 7B training 1 epoch (leaderboard competitive)
# Block 4 (0.5h): BFCL 7B quick eval on held-out
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "BLOCK 1: 6B-3 External tau3 Diagnostic (~2h)"
echo "=============================================="
/opt/conda/bin/python scripts/eval_phase6b3_external_tau3.py || echo "TAU3-FAILED"

echo "=============================================="
echo "BLOCK 2: BFCL Held-out Eval 3B (~1h)"
echo "=============================================="
/opt/conda/bin/python scripts/eval_bfcl_heldout.py || echo "HELDOUT-FAILED"

echo "=============================================="
echo "BLOCK 3: BFCL 7B Training 1 epoch (~3.5h)"
echo "=============================================="
/opt/conda/bin/python scripts/train_bfcl_competitive.py \
  --seed 42 --epochs 1.0 \
  --model weights/Qwen2.5-7B-Instruct \
  --out runs/bfcl_track/fc_7b_s42 \
  || echo "7B-TRAIN-FAILED"

echo "=============================================="
echo "BLOCK 4: BFCL 7B Quick Eval (~30min)"
echo "=============================================="
/opt/conda/bin/python scripts/eval_bfcl_heldout.py || echo "7B-EVAL-FAILED"

echo "SEVEN-HOUR-V2-DONE"
