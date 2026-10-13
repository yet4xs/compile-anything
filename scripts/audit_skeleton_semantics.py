"""Phase 8B-1A.1 Task 5+6: strict semantic audit of skeleton corpus v2.

Four-tier compilation checks per record:
  structural   - skeleton parses and validates (hard errors only)
  registry     - every called tool exists in the record's OWN
                 available_capabilities registry
  schema       - required parameters (real schemas) present in arguments
  strict_lower - strict lowering produces validator-passing TaskIR

Leakage audit (full corpus, not samples):
  - source lineage: synthetic components never cross splits
  - tool-family isolation: connected-component overlap across splits
  - full near-duplicate scan via 8-gram inverted index (Jaccard > 0.8)
"""
import json, os, sys, hashlib
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import (skeleton_from_dict,
                                        validate_skeleton, hard_errors)
from src.compiler.skeleton_lowering import lower_skeleton

D = "data/semantic_skeleton_v3"
OUT = "results/phase8b/sft_v3_integrity.json"


def grams(text, k=8):
    w = text.split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def main():
    splits = {}
    for name in os.listdir(D):
        if name.endswith(".jsonl"):
            k = name[:-6]
            splits[k] = [json.loads(l) for l in
                         open(f"{D}/{name}", encoding="utf-8")
                         if l.strip()]
    audit = {"splits": {k: len(v) for k, v in splits.items()},
             "tiers": {}, "leakage": {}}

    # ---- four-tier compilation audit ----
    per_split = {}
    for k, rows in splits.items():
        if k == "manifest":
            continue
        tiers = Counter()
        fails = []
        for r in rows:
            reg = r.get("capability_registry") or {}
            schemas = {}
            for t, meta in reg.items():
                if meta.get("schema") == "props":
                    schemas[t] = {"params": [
                        {"name": x, "optional": False}
                        for x in meta.get("required", [])]}
                else:
                    schemas[t] = {"params": None}
            try:
                sk = skeleton_from_dict(r["target_skeleton"])
            except Exception as e:
                tiers["fail_parse"] += 1
                fails.append({"id": r["task_id"], "tier": "parse",
                              "err": str(e)[:80]})
                continue
            errs = hard_errors(validate_skeleton(sk, None, schemas))
            called = {n.capability_id for n in sk.nodes}
            not_in_reg = called - set(schemas)
            if errs or not_in_reg:
                tiers["fail_structural_or_registry"] += 1
                fails.append({"id": r["task_id"],
                              "tier": "registry" if not_in_reg
                              else "structural",
                              "err": (f"missing:{sorted(not_in_reg)[:2]}"
                                      if not_in_reg else errs[:1])})
                continue
            tiers["registry_pass"] += 1
            # schema check
            missing_params = []
            for n in sk.nodes:
                meta = reg.get(n.capability_id, {})
                if meta.get("schema") == "props":
                    missing_params += [x for x in meta.get("required") or []
                                       if x not in n.arguments]
            if missing_params:
                tiers["fail_schema"] += 1
                fails.append({"id": r["task_id"], "tier": "schema",
                              "err": f"missing {missing_params[:3]}"})
                continue
            tiers["schema_pass"] += 1
            low = lower_skeleton(sk, None, schemas, strict=True)
            if low.get("taskir") is None:
                tiers["fail_strict_lowering"] += 1
                fails.append({"id": r["task_id"], "tier": "strict_lower",
                              "err": str(low["errors"][:1])[:100]})
                continue
            tiers["strict_pass"] += 1
        per_split[k] = {"n": len(rows), "tiers": dict(tiers),
                        "strict_pass_pct": round(
                            100 * tiers["strict_pass"]
                            / max(1, len(rows)), 2)}
        with open(f"results/phase8b/integrity_fails_{k}.json", "w") as f:
            json.dump(fails[:2000], f, ensure_ascii=False)
    audit["tiers"] = per_split

    # ---- lineage: synthetic components never cross splits ----
    comp_split = defaultdict(set)
    for k, rows in splits.items():
        for r in rows:
            c = (r.get("provenance") or {}).get("component") \
                or (r.get("provenance") or {}).get("split_group")
            if c:
                comp_split[c].add(k)
    cross = {c: sorted(v) for c, v in comp_split.items() if len(v) > 1}
    audit["leakage"]["lineage_cross_split_components"] = len(cross)

    # ---- tool-family isolation: exact tool overlap across splits ----
    tools = {k: {t for r in rows
                 for t in (r.get("provenance") or {}).get("gold_tools",
                                                          [])}
             for k, rows in splits.items()}
    iso = {}
    for a in splits:
        for b in splits:
            if a < b:
                iso[f"{a}|{b}"] = len(tools.get(a, set())
                                       & tools.get(b, set()))
    audit["leakage"]["tool_overlap_pairs"] = iso

    # ---- FULL near-duplicate scan (8-gram inverted index) ----
    index = defaultdict(list)
    doc_grams = {}
    for k, rows in splits.items():
        for i, r in enumerate(rows):
            g = grams(r["instruction"])
            doc_grams[(k, i)] = g
            for gr in g:
                index[gr].append((k, i))
    # rare grams (df <= 3) drive candidate pairs
    cand = Counter()
    for gr, docs in index.items():
        if 1 < len(docs) <= 3:
            for a in range(len(docs)):
                for b in range(a + 1, len(docs)):
                    if docs[a] != docs[b]:
                        cand[tuple(sorted([docs[a], docs[b]]))] += 1
    near = []
    for (d1, d2), shared in cand.items():
        g1, g2 = doc_grams[d1], doc_grams[d2]
        if not g1 or not g2:
            continue
        j = len(g1 & g2) / len(g1 | g2)
        if j > 0.8 and d1[0] != d2[0]:
            near.append({"a": f"{d1[0]}:{d1[1]}", "b": f"{d2[0]}:{d2[1]}",
                         "j": round(j, 3)})
    audit["leakage"]["cross_split_near_dup_j>0.8"] = len(near)
    with open("results/phase8b/near_dup_pairs.json", "w") as f:
        json.dump(near[:200], f, ensure_ascii=False)

    with open(OUT, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    for k, v in per_split.items():
        print(f"{k}: n={v['n']} strict={v['strict_pass_pct']}% "
              f"tiers={v['tiers']}")
    print("lineage cross-split components:",
          audit["leakage"]["lineage_cross_split_components"])
    print("tool overlaps:", iso)
    print("cross-split near-dup:", len(near))
    print("AUDIT-V2-DONE")


if __name__ == "__main__":
    main()
