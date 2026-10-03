"""Phase 6A urgent audit: does Phase 5C 'capabilities' leak the target skill set?

For every corpus v3.1 record (train/val/test):
  C = set(record['capabilities'])            # what 5C conditioning put in the prompt
  T = set(non-policy ops in plan_target)     # what the model was supervised to emit
Report per source: |C| vs |T|, C==T, T⊆C, C⊆T, precision/recall/Jaccard.
Provenance: capabilities were built by semantic_capabilities(tools) — derive
the semantic skills of the record's AVAILABLE tools via the same toolmap,
so we can tell 'available-tool view' (Case A) from 'used-tool view' (Case B).
"""
import json, os, sys, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.lifter.toolmap import map_tool

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
SPLITS = ("train", "val", "test")


def main():
    # rebuild the available-tool semantic view: join lowering-tool is the USED
    # tool; the raw record's full tool list is not in the corpus, so we use the
    # xLAM/toolbench raw join from Phase 6A Task 1 for provenance of |available|.
    # Within the corpus we can still answer the core question: is C == T?
    results = {}
    for split in SPLITS:
        path = os.path.join(ROOT, f"data/compiler_corpus_v3_1/{split}.jsonl")
        per_src = defaultdict(lambda: Counter())
        stats = defaultdict(lambda: {"n": 0, "c_eq_t": 0, "t_sub_c": 0, "c_sub_t": 0,
                                     "prec_sum": 0.0, "rec_sum": 0.0, "jac_sum": 0.0,
                                     "c_sum": 0, "t_sum": 0, "no_caps": 0})
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            src = r.get("source", "")
            C = set(r.get("capabilities") or [])
            try:
                import ast
                tgt = r.get("plan_target", "")
                # parse ops from plan text: lines start with %id = OP(...)
                T = set(re.findall(r"=\s*([A-Z_]+)\s*\(", tgt)) - POLICY_OPS
            except Exception:
                T = set()
            st = stats[src]
            st["n"] += 1
            if not C:
                st["no_caps"] += 1
                continue
            st["c_sum"] += len(C)
            st["t_sum"] += len(T)
            if C == T:
                st["c_eq_t"] += 1
            if T <= C:
                st["t_sub_c"] += 1
            if C <= T:
                st["c_sub_t"] += 1
            inter = len(C & T)
            st["prec_sum"] += inter / len(C)
            st["rec_sum"] += inter / len(T) if T else 1.0
            st["jac_sum"] += inter / len(C | T) if (C | T) else 1.0

        results[split] = {}
        for src, st in stats.items():
            n_with = st["n"] - st["no_caps"]
            results[split][src] = {
                "n": st["n"], "with_capabilities": n_with,
                "avg_C_size": round(st["c_sum"] / max(1, n_with), 2),
                "avg_T_size": round(st["t_sum"] / max(1, n_with), 2),
                "C_eq_T_pct": round(100 * st["c_eq_t"] / max(1, n_with), 2),
                "T_subset_C_pct": round(100 * st["t_sub_c"] / max(1, n_with), 2),
                "C_subset_T_pct": round(100 * st["c_sub_t"] / max(1, n_with), 2),
                "precision_C_T": round(st["prec_sum"] / max(1, n_with), 4),
                "recall_T_C": round(st["rec_sum"] / max(1, n_with), 4),
                "jaccard": round(st["jac_sum"] / max(1, n_with), 4),
            }
        print(f"\n=== {split} ===")
        for src, st in results[split].items():
            print(f"  {src:<18s} n={st['n']:>6} |C|={st['avg_C_size']:5.2f} |T|={st['avg_T_size']:5.2f} "
                  f"C==T {st['C_eq_T_pct']:6.2f}%  T⊆C {st['T_subset_C_pct']:6.2f}%  "
                  f"P={st['precision_C_T']:.3f} R={st['recall_T_C']:.3f} J={st['jaccard']:.3f}")

    out = {
        "question": "Is Phase 5C 'capabilities' an available-tool view (Case A) or the target op set (Case B)?",
        "provenance": ("record['capabilities'] was built by semantic_capabilities(record tools) "
                       "in the corpus builder: for xLAM/ToolBench the raw record's tool list is the "
                       "candidate tool set presented to the model, mapped through the same toolmap "
                       "that produced the plan ops — so C is the semantic view of AVAILABLE tools, "
                       "deduplicated; T is the ops the trajectory actually used."),
        "by_split": results,
    }
    os.makedirs(os.path.join(ROOT, "results/phase6"), exist_ok=True)
    with open(os.path.join(ROOT, "results/phase6/phase5c_capability_audit.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved results/phase6/phase5c_capability_audit.json")


if __name__ == "__main__":
    main()
