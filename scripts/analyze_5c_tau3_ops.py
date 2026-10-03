"""τ³ op-distribution analysis: what do models predict vs what the reference lowers to?
CPU only. Informs Phase 6 direction (why is recall stuck at 0.6%?)."""
import json, sys, os
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")

from src.eval.tau3_skill_oracle import lower_reference_actions
from collections import Counter

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

FILES = {
    "E5C-SD": "runs/phase5c/e5c_sd_tau3_preds.jsonl",
    "E1-A":  "runs/phase5b1/tau3_preds_2048.jsonl",
}

for label, path in FILES.items():
    from src.ir.parser import parse_text
    pred_ops = Counter()
    ref_ops = Counter()
    conf = Counter()  # (ref_skill, pred_skill) for first ref action
    n = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            rl = lower_reference_actions([{"name": x} for x in rec.get("ref_action_names", [])])
            for r in rl:
                ref_ops[r["skill"]] += 1
            try:
                module = parse_text(rec["taskir_text"])
                nodes = [x for x in module.program.nodes if x.op not in POLICY_OPS]
            except Exception:
                nodes = []
            for x in nodes:
                pred_ops[x.op] += 1
            if rl and nodes:
                conf[(rl[0]["skill"], nodes[0].op)] += 1
    print(f"\n=== {label} (n={n}) ===")
    print("Reference lowered skills (top 12):")
    for k, v in ref_ops.most_common(12):
        print(f"  {k:<20s} {v:6d}")
    print("Predicted skills (top 12):")
    for k, v in pred_ops.most_common(12):
        print(f"  {k:<20s} {v:6d}")
    print("Top ref→pred confusions (first-action):")
    for (r, p), v in conf.most_common(12):
        print(f"  {r:<18s} -> {p:<18s} {v:5d}")
print("\nDONE")
