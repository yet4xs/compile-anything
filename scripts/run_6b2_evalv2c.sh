#!/bin/bash
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/opt/conda/bin/python scripts/eval_phase6b_resolver_v2.py
echo EVALV2C-DONE
