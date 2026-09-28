#!/usr/bin/env bash
# Phase 5B-1 one-command runner (server-side).
#
#   bash scripts/run_phase5b1.sh preflight     # full preflight (3B)
#   bash scripts/run_phase5b1.sh e0            # 3B zero-shot baseline
#   bash scripts/run_phase5b1.sh e1-sanity     # 3B QLoRA, 50 steps
#   bash scripts/run_phase5b1.sh e1            # 3B QLoRA, full
#   bash scripts/run_phase5b1.sh e2            # 7B zero-shot baseline
#   bash scripts/run_phase5b1.sh e3            # 7B LoRA/QLoRA, full
#   bash scripts/run_phase5b1.sh eval-all      # collect results table
#
# Frozen: Tier A+B, plan_target, capability context OFF, corpus v3 splits.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
RUNS=runs/phase5b1
CORPUS=data/compiler_corpus_v3
M3=${M3:-weights/Qwen2.5-3B-Instruct}
M7=${M7:-weights/Qwen2.5-7B-Instruct}

_eval_run () {  # $1 run dir, $2 tag
  $PY -m src.compiler.train.infer --model "$1/final" \
      --test "$CORPUS/test.jsonl" --out "$RUNS/$2/preds_test.jsonl"
  $PY benchmark/neural_compiler_eval.py \
      --predictions "$RUNS/$2/preds_test.jsonl" \
      --train "$CORPUS/train.jsonl" \
      --dump-unseen "$RUNS/$2/unseen_cases.json" \
      --out "$RUNS/$2/eval.json"
}

case "${1:-help}" in
  preflight)
    $PY scripts/training_preflight.py --model "$M3" --quantize 4bit ;;
  e0)
    mkdir -p "$RUNS/e0_3b_base"
    $PY -m src.compiler.train.infer --model "$M3" \
        --test "$CORPUS/test.jsonl" --out "$RUNS/e0_3b_base/preds_test.jsonl"
    $PY benchmark/neural_compiler_eval.py \
        --predictions "$RUNS/e0_3b_base/preds_test.jsonl" \
        --train "$CORPUS/train.jsonl" \
        --dump-unseen "$RUNS/e0_3b_base/unseen_cases.json" \
        --out "$RUNS/e0_3b_base/eval.json" ;;
  e1-sanity)
    $PY -m src.compiler.train.train_lora \
        --config src/compiler/train/config/qwen3b.yaml --model "$M3" \
        --train "$CORPUS/train.jsonl" --val "$CORPUS/val.jsonl" \
        --out "$RUNS/e1_3b_qlora" --max-steps 50
    _eval_run "$RUNS/e1_3b_qlora" e1_3b_qlora ;;
  e1)
    $PY -m src.compiler.train.train_lora \
        --config src/compiler/train/config/qwen3b.yaml --model "$M3" \
        --train "$CORPUS/train.jsonl" --val "$CORPUS/val.jsonl" \
        --out "$RUNS/e1_3b_qlora" --resume auto
    _eval_run "$RUNS/e1_3b_qlora" e1_3b_qlora ;;
  e2)
    mkdir -p "$RUNS/e2_7b_base"
    $PY -m src.compiler.train.infer --model "$M7" \
        --test "$CORPUS/test.jsonl" --out "$RUNS/e2_7b_base/preds_test.jsonl"
    $PY benchmark/neural_compiler_eval.py \
        --predictions "$RUNS/e2_7b_base/preds_test.jsonl" \
        --train "$CORPUS/train.jsonl" \
        --dump-unseen "$RUNS/e2_7b_base/unseen_cases.json" \
        --out "$RUNS/e2_7b_base/eval.json" ;;
  e3)
    $PY -m src.compiler.train.train_lora \
        --config src/compiler/train/config/qwen7b.yaml --model "$M7" \
        --train "$CORPUS/train.jsonl" --val "$CORPUS/val.jsonl" \
        --out "$RUNS/e3_7b_lora" --resume auto
    _eval_run "$RUNS/e3_7b_lora" e3_7b_lora ;;
  eval-all)
    $PY scripts/collect_phase5b1_results.py ;;
  *)
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -14 ;;
esac
