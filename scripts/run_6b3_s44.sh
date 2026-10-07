#!/bin/bash
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/opt/conda/bin/python scripts/eval_phase6b3_integration.py --seed 44 || echo INT44-FAILED
echo PHASE6B3-S44-DONE
