#!/bin/bash
# SOTA pipeline: EF-SC → V-DPO → CG-RLVR on BFCL
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Phase 1: EF-SC (Error-Feedback Self-Correction)"
echo "=============================================="
/opt/conda/bin/python scripts/train_ef_sc.py || echo "EF-SC-FAILED"

echo "=============================================="
echo "Phase 2: V-DPO (Validator-Guided DPO)"
echo "=============================================="
/opt/conda/bin/python scripts/train_vdpo.py || echo "V-DPO-FAILED"

echo "=============================================="
echo "Phase 3: Final Evaluation with AST"
echo "=============================================="
/opt/conda/bin/python scripts/eval_final_ast.py || echo "FINAL-EVAL-FAILED"

echo "SOTA-PIPELINE-DONE"
