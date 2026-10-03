#!/bin/bash
# Recovery chain (2026-10-04 early AM): internal eval (trainings already done)
# -> external diagnostics -> g1g2 s43/s44 -> E3 7B -> E3 eval.
# Replaces both the dead tail of run_phase6a_chain.sh and the deadlocked night queue.
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [0] internal eval (all 4 trainings complete) ==="
/opt/conda/bin/python scripts/eval_phase6a.py && echo "PHASE6A-CHAIN-DONE" >> /tmp/phase6a_chain.log || { echo "INTERNAL-EVAL FAILED"; exit 1; }

echo "=== [1] external grounding diagnostics ==="
/opt/conda/bin/python scripts/eval_phase6a_external.py || echo "EXTERNAL FAILED (continuing)"

echo "=== [2] grounder g1g2 seed43 ==="
/opt/conda/bin/python scripts/train_phase6a_grounder.py --config g1g2 --seed 43 || echo "G1G2-S43 FAILED"

echo "=== [3] grounder g1g2 seed44 ==="
/opt/conda/bin/python scripts/train_phase6a_grounder.py --config g1g2 --seed 44 || echo "G1G2-S44 FAILED"

echo "=== [4] E3 7B QLoRA ==="
/opt/conda/bin/python -m src.compiler.train.train_lora \
  --config src/compiler/train/config/qwen7b_24gb.yaml \
  --model weights/Qwen2.5-7B-Instruct \
  --train data/compiler_corpus_v3_1/train.jsonl \
  --val data/compiler_corpus_v3_1/val.jsonl \
  --out runs/phase6/e3_7b_qlora || echo "E3 FAILED"

echo "=== [5] E3 quick eval ==="
/opt/conda/bin/python scripts/eval_e3_quick.py || echo "E3-EVAL SKIPPED"

echo "RECOVERY-CHAIN-DONE"
