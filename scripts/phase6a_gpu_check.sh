#!/bin/bash
echo "=== compute apps ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "=== phase6a procs ==="
pgrep -af phase6a | grep -v pgrep | head -5
echo "=== kill leftovers ==="
pkill -9 -f run_phase6a_chain 2>/dev/null
pkill -9 -f train_phase6a_grounder 2>/dev/null
sleep 2
echo "=== gpu mem after ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== bnb sanity ==="
cd /ccfa2026/compile-anything
/opt/conda/bin/python - <<'PYEOF'
import torch
import bitsandbytes as bnb
print('bnb', bnb.__version__)
import torch.nn as nn
l = nn.Linear(8, 8).cuda()
print('cuda ok', torch.cuda.get_device_name(0))
PYEOF
