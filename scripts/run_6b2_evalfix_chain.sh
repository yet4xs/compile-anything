#!/bin/bash
# Waits for R2 trainings, then: audits -> common eval set -> eval v2
# Conditional: if R1 gold visibility < 100%, rebuild R1 data (gold-first) + retrain R1-s42
cd /ccfa2026/compile-anything
export PYTHONPATH=/ccfa2026/compile-anything:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -aq PHASE6B2V2-CHAIN-DONE /tmp/phase6b2v2.log 2>/dev/null; do sleep 180; done

echo "=== data audits (fix 2/8/9) ==="
/opt/conda/bin/python scripts/audit_resolver_data.py || echo AUDIT-FAILED

VIS=$(python3 -c "import json;d=json.load(open('results/phase6b/resolver_data_audit.json'));print(d['fix2_r1_gold_visibility']['train_visible_pct'])" 2>/dev/null || echo 0)
echo "R1 train gold visibility: ${VIS}%"
RETRAIN_NEEDED=$(python3 -c "print(1 if float('${VIS}') < 100.0 else 0)")
if [ "${RETRAIN_NEEDED}" = "1" ]; then
  echo "=== rebuilding R1 data gold-first + retraining R1-s42 ==="
  /opt/conda/bin/python scripts/rebuild_r1_gold_first.py || echo R1-REBUILD-FAILED
  /opt/conda/bin/python scripts/train_phase6b_resolver.py --mode R1 --seed 42 || echo R1-RETRAIN-FAILED
fi

echo "=== common eval tasks (fix 3/6/7) ==="
/opt/conda/bin/python scripts/build_common_eval_tasks.py || exit 1
echo "=== resolver eval v2 ==="
/opt/conda/bin/python scripts/eval_phase6b_resolver_v2.py || echo EVALV2-FAILED
echo "EVALFIX-CHAIN-DONE"
