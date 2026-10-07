#!/bin/bash
# 7-Hour Experiment Plan — four sequential blocks
# Block 1 (2.5h): BFCL 7B training — leaderboard competitive
# Block 2 (1h):  BFCL 3B+7B held-out evaluation — honest generalization
# Block 3 (2h):  6B-3 external tau3 diagnostic — paper's core architecture test
# Block 4 (1h):  BFCL 3B multi_turn supplement + seq fix — quick improvement
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "BLOCK 1: BFCL 7B Training (~2.5h)"
echo "=============================================="
/opt/conda/bin/python scripts/train_bfcl_competitive.py \
  --seed 42 --epochs 2.0 \
  --model weights/Qwen2.5-7B-Instruct \
  --out runs/bfcl_track/fc_7b_s42 \
  || echo "7B-TRAIN-FAILED"

echo "=============================================="
echo "BLOCK 2: BFCL Held-out Evaluation (3B + 7B) (~1h)"
echo "=============================================="
/opt/conda/bin/python scripts/eval_bfcl_heldout.py || echo "HELDOUT-FAILED"

echo "=============================================="
echo "BLOCK 3: 6B-3 External tau3 Diagnostic (~2h)"
echo "=============================================="
/opt/conda/bin/python scripts/eval_phase6b3_external_tau3.py || echo "TAU3-FAILED"

echo "=============================================="
echo "BLOCK 4: BFCL 3B v2 (multi_turn + seq 2048) (~1h)"
echo "=============================================="
/opt/conda/bin/python scripts/train_bfcl_v2.py --seed 42 || echo "V2-TRAIN-FAILED"
/opt/conda/bin/python scripts/eval_bfcl_heldout.py --adapter runs/bfcl_track/fc_3b_v2_s42/final || echo "V2-EVAL-FAILED"

echo "ALL-BLOCKS-DONE"
