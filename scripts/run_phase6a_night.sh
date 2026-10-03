#!/bin/bash
# Overnight queue (launched 2026-10-04): external diagnostics -> g1g2 extra seeds -> E3 7B
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/5] external grounding diagnostics (waits for internal chain) ==="
/opt/conda/bin/python scripts/eval_phase6a_external.py || echo "EXTERNAL FAILED (continuing)"

echo "=== [2/5] grounder g1g2 seed43 ==="
/opt/conda/bin/python scripts/train_phase6a_grounder.py --config g1g2 --seed 43 || echo "G1G2-S43 FAILED"

echo "=== [3/5] grounder g1g2 seed44 ==="
/opt/conda/bin/python scripts/train_phase6a_grounder.py --config g1g2 --seed 44 || echo "G1G2-S44 FAILED"

echo "=== [4/5] E3 7B QLoRA (supplementary, phase6 artifacts only) ==="
/opt/conda/bin/python -m src.compiler.train.train_lora \
  --config src/compiler/train/config/qwen7b_24gb.yaml \
  --model weights/Qwen2.5-7B-Instruct \
  --train data/compiler_corpus_v3_1/train.jsonl \
  --val data/compiler_corpus_v3_1/val.jsonl \
  --out runs/phase6/e3_7b_qlora || echo "E3 FAILED"

echo "=== [5/5] E3 internal quick eval (parse/valid on val split) ==="
/opt/conda/bin/python scripts/eval_e3_quick.py || echo "E3-EVAL SKIPPED"

echo "NIGHT-CHAIN-DONE"
