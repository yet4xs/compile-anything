"""Training-contamination audit (Task 9): corpus v3 ↔ external benchmarks.

Report-only — overlapping samples are NOT removed (per spec).

    python scripts/audit_external_overlap.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.dedup import normalize, minhash, tokens, _jaccard  # noqa
from src.eval.adapters import (load_bfcl, load_tau3, load_agentboard,   # noqa
                               load_rtl_repo, load_bird)
from src.lifter.toolmap import map_tool

V3_TRAIN = ROOT / "data" / "compiler_corpus_v3" / "train.jsonl"

BANDS, N_PERM = 8, 32


def load_v3():
    rows = []
    for line in open(V3_TRAIN, encoding="utf-8"):
        r = json.loads(line)
        rows.append({"instruction": r["instruction"], "source": r["source"],
                     "lowering": r.get("lowering")})
    return rows


def v3_skill_families(rows):
    fams = Counter()
    for r in rows:
        for c in (r["lowering"] or []):
            fams[c.get("skill")] += 1
    return set(fams)


def ext_tool_families(samples, get_tools):
    """Map external tool/capability names to semantic families (toolmap is
    read-only here; it is an analyzer, not a trainer)."""
    fams = set()
    for s in samples:
        for t in get_tools(s) or []:
            fams.add(map_tool(t, {}).get("skill"))
    return fams


def near_overlap(v3_texts, ext_texts, threshold=0.8):
    """LSH over combined set; count cross pairs with true Jaccard."""
    all_texts = v3_texts + ext_texts
    sigs = [minhash(tokens(t)) for t in all_texts]
    rows = N_PERM // BANDS
    pairs = set()
    for band in range(BANDS):
        buckets = defaultdict(list)
        lo = band * rows
        for i, sig in enumerate(sigs):
            buckets[sig[lo:lo + rows]].append(i)
        for idxs in buckets.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pairs.add((min(idxs[a], idxs[b]), max(idxs[a], idxs[b])))
    n_v3 = len(v3_texts)
    near = exact = 0
    v3_norm = {normalize(t) for t in v3_texts}
    ext_norm = [normalize(t) for t in ext_texts]
    exact = sum(1 for t in ext_norm if t in v3_norm)
    for i, j in pairs:
        if i < n_v3 <= j:
            if _jaccard(set(tokens(all_texts[i])),
                        set(tokens(all_texts[j]))) >= threshold:
                near += 1
    return exact, near


def main() -> int:
    v3 = load_v3()
    v3_texts = [r["instruction"] for r in v3]
    v3_norm = {normalize(t) for t in v3_texts}
    v3_fams = v3_skill_families(v3)
    v3_opseqs = set()
    for line in open(V3_TRAIN, encoding="utf-8"):
        r = json.loads(line)
        v3_opseqs.add("|".join(n["op"] for n in
                               r["plan_json"]["program"]["nodes"]))

    benchmarks = {
        "bfcl_v4": (load_bfcl(),
                    lambda s: [f.get("name") for f in (s.capabilities or [])
                               if isinstance(f, dict)]),
        "tau3_bench": (load_tau3(), lambda s: []),
        "agentboard_tools": (load_agentboard(tasks=["tool-query",
                                                     "tool-operation"]),
                             lambda s: []),
        "agentboard_webshop": (load_agentboard(tasks=["webshop"]),
                               lambda s: []),
        "rtl_repo": (load_rtl_repo("test"), lambda s: []),
        "bird_mini_dev": (load_bird(), lambda s: []),
    }

    report = {}
    for name, (samples, get_tools) in benchmarks.items():
        texts = [s.instruction for s in samples]
        exact = sum(1 for t in texts if normalize(t) in v3_norm)
        # near-dup on a capped sample for tractability (LSH over 124k+28k
        # is fine, but toolbench excluded here anyway)
        cap = 20000
        _, near = near_overlap(v3_texts, texts[:cap])
        fams = ext_tool_families(samples, get_tools)
        fam_overlap = len(fams & v3_fams) / len(fams) if fams else None
        # composition overlap: BFCL oracle op-seqs vs v3 plan op-seqs
        comp = None
        if name == "bfcl_v4":
            from src.eval.oracle_bfcl import oracle
            ops_in_v3 = ops_total = 0
            for s in samples:
                r = oracle(s)
                if r.get("module") is not None:
                    seq = "|".join(n.op for n in r["module"].program.nodes)
                    ops_total += 1
                    if seq in v3_opseqs:
                        ops_in_v3 += 1
            comp = round(100 * ops_in_v3 / ops_total, 2) if ops_total else 0
        n = len(samples) or 1
        entry = {
            "external_samples": len(samples),
            "exact_overlap": exact,
            "exact_overlap_pct": round(100 * exact / n, 3),
            "near_overlap_pairs(vs_20k_cap)": near,
            "tool_family_overlap_pct": round(100 * fam_overlap, 1)
            if fam_overlap is not None else None,
            "composition_overlap_pct": comp,
        }
        # OOD classification
        if entry["exact_overlap_pct"] > 1 or near > max(50, 0.01 * n):
            ood = "IID-like (needs manual review)"
        elif entry["tool_family_overlap_pct"] and \
                entry["tool_family_overlap_pct"] >= 60:
            ood = "cross-dataset (same tool families)"
        elif entry["tool_family_overlap_pct"] and \
                entry["tool_family_overlap_pct"] >= 20:
            ood = "tool-family partial OOD"
        else:
            ood = ("composition OOD" if comp is not None and comp < 20
                   else "strong OOD")
        entry["classification"] = ood
        report[name] = entry
        print(f"{name:20s} exact={entry['exact_overlap_pct']}% "
              f"near={near} fam={entry['tool_family_overlap_pct']}% "
              f"comp={comp} -> {ood}")

    out = ROOT / "data" / "reports" / "external_overlap.json"
    out.write_text(json.dumps({"v3_train_size": len(v3), "benchmarks":
                               report}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    md = ["# External Overlap Audit (training contamination, report-only)", "",
          f"v3 train = {len(v3)} instructions. Overlapping samples are "
          "NOT removed.", "",
          "| benchmark | n | exact % | near pairs | tool-family % | "
          "composition % | classification |", "|---|---:|---:|---:|---:|"
          "---:|---|"]
    for name, e in report.items():
        md.append(f"| {name} | {e['external_samples']} "
                  f"| {e['exact_overlap_pct']} | "
                  f"{e['near_overlap_pairs(vs_20k_cap)']} | "
                  f"{e['tool_family_overlap_pct']} | "
                  f"{e['composition_overlap_pct']} | "
                  f"{e['classification']} |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
