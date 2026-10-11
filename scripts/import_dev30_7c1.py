"""Convert 7C-1 s43 dev-30 records (G0/G1/G2) into the large-run schema.

Recomputes all Task-6 metrics from the saved texts so dev-30 and
confirmation sets share one metric definition. G2a/G3 for dev-30 are
generated separately by eval_online_large.py --dev30 --arms 2,4.
"""
import json, os, sys

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_online_large import load_tasks, judge, RUNS_PATH

src = json.load(open("results/phase7/online_constrained.json"))
tasks = load_tasks()

existing = set()
if os.path.exists(RUNS_PATH):
    for line in open(RUNS_PATH, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            existing.add((r["seed"], r["id"], r["arm"]))

n = 0
with open(RUNS_PATH, "a", encoding="utf-8") as f:
    for tid, arms in src["samples"].items():
        for arm, rec in arms.items():
            key = (src["seed"], tid, arm)
            if key in existing:
                continue
            j = judge(rec["text"], tasks[tid]["ref_text"])
            out = {"seed": src["seed"], "id": tid, "arm": arm,
                   "set": "dev30", "latency_s": rec.get("latency_s", 0),
                   "tokens": None, "text": rec["text"], "judge": j,
                   "imported_from": "phase7/online_constrained.json (7C-1)"}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
print(f"imported {n} dev-30 records (G0/G1/G2, s43)")
