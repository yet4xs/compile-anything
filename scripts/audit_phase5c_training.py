"""Task 1: Training depth audit — verify if v3.1 training corpus is biased
toward short plans, which would explain E1's under-planning on τ³."""
import json, sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from src.ir.parser import parse_text, TaskIRSyntaxError
from collections import Counter, defaultdict

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
DEPTHS = [(1,1,"1"), (2,3,"2-3"), (4,6,"4-6"), (7,999,"7+")]

def action_count(plan_json):
    nodes = plan_json.get("program", {}).get("nodes", [])
    return len([n for n in nodes if n.get("op","") not in POLICY_OPS])

print("=== Task 1: Training Depth Audit ===\n", flush=True)

for split in ("train", "val", "test"):
    path = f"data/compiler_corpus_v3_1/{split}.jsonl"
    stats = defaultdict(lambda: {"n":0, "count_sum":0, "sources":Counter(), "tiers":Counter()})
    all_counts = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cnt = action_count(r.get("plan_json", {}))
            all_counts.append(cnt)
            for lo, hi, label in DEPTHS:
                if lo <= cnt <= hi:
                    d = stats[label]
                    d["n"] += 1
                    d["count_sum"] += cnt
                    d["sources"][r.get("source","")] += 1
                    d["tiers"][r.get("quality_tier","")] += 1
                    break

    total = len(all_counts)
    mean = sum(all_counts)/max(1,total)
    print(f"\n{split.upper()} ({total} samples, mean={mean:.2f} actions/sample):", flush=True)
    print(f"{'Depth':<8s} {'Samples':>8s} {'%':>7s} {'Avg':>6s} {'Top Sources':>40s}", flush=True)
    print("-" * 75, flush=True)
    for _, _, label in DEPTHS:
        d = stats[label]
        if d["n"] == 0:
            print(f"{label:<8s} {0:>8d} {0:>6.1f}% {'—':>6s}", flush=True)
            continue
        pct = 100*d["n"]/total
        avg = d["count_sum"]/d["n"]
        top_src = ", ".join(f"{s}:{c}" for s,c in d["sources"].most_common(3))
        print(f"{label:<8s} {d['n']:>8d} {pct:>6.1f}% {avg:>6.2f} {top_src:>40s}", flush=True)

# Also check dependency edges
print(f"\n\nDependency edges per sample (train):", flush=True)
edge_counts = []
with open("data/compiler_corpus_v3_1/train.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        nodes = r.get("plan_json",{}).get("program",{}).get("nodes",[])
        edges = sum(len(n.get("inputs",[])) for n in nodes)
        edge_counts.append(edges)
mean_edges = sum(edge_counts)/max(1,len(edge_counts))
zero_edges = sum(1 for e in edge_counts if e == 0)
print(f"  Mean edges/sample: {mean_edges:.2f}", flush=True)
print(f"  Zero edges (single node or no deps): {zero_edges}/{len(edge_counts)} ({100*zero_edges/len(edge_counts):.1f}%)", flush=True)

print("\nDONE", flush=True)
