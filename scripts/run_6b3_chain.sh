#!/bin/bash
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== integration seed42 (primary) ==="
/opt/conda/bin/python scripts/eval_phase6b3_integration.py --seed 42 || echo INT42-FAILED
echo "=== integration seed44 (replicate) ==="
/opt/conda/bin/python scripts/eval_phase6b3_integration.py --seed 44 || echo INT44-FAILED
echo "PHASE6B3-INTERNAL-DONE"
