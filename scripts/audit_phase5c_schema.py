"""Task 2: Capability schema coverage audit for v3.1 training sources."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter

print("=== Task 2: Capability Schema Coverage Audit ===\n")

# For each source, check what capability metadata exists in the corpus
corpus_path = "data/compiler_corpus_v3_1/train.jsonl"

source_stats = {}

with open(corpus_path) as f:
    for line in f:
        r = json.loads(line)
        src = r.get("source", "unknown")
        caps = r.get("capabilities") or []
        lowering = r.get("lowering") or []

        if src not in source_stats:
            source_stats[src] = {
                "n": 0,
                "has_capabilities": 0,
                "cap_count_sum": 0,
                "has_lowering": 0,
                "lowering_kinds": Counter(),
                "cap_examples": [],
            }

        s = source_stats[src]
        s["n"] += 1

        if caps:
            s["has_capabilities"] += 1
            s["cap_count_sum"] += len(caps)
            if len(s["cap_examples"]) < 2:
                s["cap_examples"].append(str(caps[:2])[:200])

        if lowering:
            s["has_lowering"] += 1
            for l in lowering:
                s["lowering_kinds"][l.get("mapping_kind","unknown")] += 1

# Classify schema quality
CLASSIFICATIONS = {
    "xlam": {
        "schema_source": "xLAM dataset tools field",
        "concrete_schema": "function name + description + parameters",
        "quality": "FULL_SCHEMA",
        "note": "xLAM provides tool definitions with parameter schemas"
    },
    "toolbench_static": {
        "schema_source": "ToolBench tools field",
        "concrete_schema": "API description + params",
        "quality": "FULL_SCHEMA",
        "note": "ToolBench provides API schemas"
    },
    "spider": {
        "schema_source": "Spider DB schema (not in corpus v3.1)",
        "concrete_schema": "table/column names",
        "quality": "SEMANTIC_ONLY",
        "note": "Spider SQL includes table references but not full schema in training data"
    },
    "humaneval": {
        "schema_source": "none (code generation)",
        "concrete_schema": "none",
        "quality": "NONE",
        "note": "Code tasks don't have tool schemas"
    },
    "mbpp": {
        "schema_source": "none (code generation)",
        "concrete_schema": "none",
        "quality": "NONE",
        "note": "Code tasks don't have tool schemas"
    },
    "verilogeval": {
        "schema_source": "none (RTL generation)",
        "concrete_schema": "none",
        "quality": "NONE",
        "note": "RTL tasks don't have tool schemas"
    },
}

print(f"{'Source':<20s} {'n':>6s} {'HasCaps':>8s} {'%':>6s} {'AvgCaps':>8s} {'Schema':>15s}", flush=True)
print("-" * 70, flush=True)

for src in sorted(source_stats):
    s = source_stats[src]
    cls = CLASSIFICATIONS.get(src, {"quality": "UNKNOWN"})
    pct = 100*s["has_capabilities"]/max(1,s["n"])
    avg = s["cap_count_sum"]/max(1,s["has_capabilities"])
    print(f"{src:<20s} {s['n']:>6d} {s['has_capabilities']:>8d} {pct:>5.1f}% {avg:>8.1f} {cls['quality']:>15s}", flush=True)

print(f"\n{'='*60}", flush=True)
print("Schema Quality Classification:", flush=True)
print(f"{'='*60}", flush=True)
total = sum(s["n"] for s in source_stats.values())
for src in sorted(source_stats):
    s = source_stats[src]
    cls = CLASSIFICATIONS.get(src, {"quality":"UNKNOWN","note":""})
    print(f"\n{src} ({s['n']} samples, {100*s['n']/total:.1f}% of corpus):", flush=True)
    print(f"  Quality: {cls['quality']}", flush=True)
    print(f"  Schema: {cls.get('concrete_schema','')}", flush=True)
    print(f"  Note: {cls.get('note','')}", flush=True)
    print(f"  Has capabilities: {s['has_capabilities']}/{s['n']} ({100*s['has_capabilities']/max(1,s['n']):.1f}%)", flush=True)
    if s["cap_examples"]:
        print(f"  Example: {s['cap_examples'][0][:150]}", flush=True)
    print(f"  Lowering kinds: {dict(s['lowering_kinds'])}", flush=True)

# Summary
full = sum(s["n"] for src, s in source_stats.items() if CLASSIFICATIONS.get(src,{}).get("quality") == "FULL_SCHEMA")
sem = sum(s["n"] for src, s in source_stats.items() if CLASSIFICATIONS.get(src,{}).get("quality") == "SEMANTIC_ONLY")
none_c = sum(s["n"] for src, s in source_stats.items() if CLASSIFICATIONS.get(src,{}).get("quality") == "NONE")

print(f"\n{'='*60}", flush=True)
print(f"Summary:", flush=True)
print(f"  FULL_SCHEMA:    {full} ({100*full/total:.1f}%)", flush=True)
print(f"  SEMANTIC_ONLY:  {sem} ({100*sem/total:.1f}%)", flush=True)
print(f"  NONE:           {none_c} ({100*none_c/total:.1f}%)", flush=True)
print(f"  TOTAL:          {total}", flush=True)
print(f"\nDONE", flush=True)
