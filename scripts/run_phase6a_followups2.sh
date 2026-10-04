#!/bin/bash
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q RECOVERY-CHAIN-DONE /tmp/phase6a_recovery.log 2>/dev/null; do sleep 120; done
echo re-running FIXED external diagnostics
/opt/conda/bin/python scripts/eval_phase6a_external.py
echo running followups
/opt/conda/bin/python scripts/eval_phase6a_followups.py
echo FOLLOWUPS2-DONE
