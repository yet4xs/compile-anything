"""Training-corpus quality audit (Phase 5B-0 Task 2 + Task 5).

Reports — over REAL benchmark data, plan view (pure lowering):
  - lowering provenance: exact / heuristic / fallback %
  - EXEC_ACTION split: intentional action vs unknown fallback
  - per-dataset fallback rate; per-skill mapping-kind distribution
  - semantic_mapping_coverage: % of samples whose trajectory has ZERO
    fallback calls (distinct from lift/validator/execution coverage)
  - policy-tail effect: op distribution with vs without GENERATE/VERIFY tail
  - Task 5: instruction-only ambiguity — same (near-duplicate) instruction
    mapping to different semantic plans; and whether differing capability
    context could disambiguate (A: instruction-only vs B: +capabilities)

Output: data/reports/training_corpus_audit.{json,md}
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.pipeline import run_pipeline            # noqa: E402
from src.dataset.dedup import normalize, near_duplicate_families  # noqa: E402

TOOLUSE_SOURCES = {"toolbench", "toolbench_static", "xlam", "apibank",
                   "agentbench"}
TAIL_SOURCES = TOOLUSE_SOURCES | {"spider"}      # sources with policy tail


def lowering_of(rec):
    prov = rec["module"].meta.get("provenance", {})
    low = prov.get("lowering")
    if low:
        return low
    # non-tool sources have no lowering; synthetic tagging not applicable
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "reports"
                                         / "training_corpus_audit.json"))
    args = ap.parse_args()

    print("lifting (plan view, simulate on)...")
    _, lifted = run_pipeline(pathlib.Path(args.raw_root), view="plan")

    kinds = Counter()
    exec_action = {"intentional": 0, "unknown_fallback": 0}
    skill_kind = defaultdict(Counter)
    per_ds = {}
    op_dist_plan = Counter()
    op_dist_exec = Counter()
    ambiguity = {"exact_groups": 0, "exact_ambiguous": 0,
                 "near_groups": 0, "near_ambiguous": 0,
                 "ambig_ctx_differ": 0, "ambig_ctx_same": 0}
    xlam_stats = {"total": 0, "zero_fallback": 0, "some_fallback": 0,
                  "all_fallback": 0}

    # collect per-sample info for ambiguity analysis
    by_source_samples = defaultdict(list)

    for rec in lifted:
        src = rec["sample"]["source"]
        if rec["module"] is None or not rec.get("valid"):
            continue
        ops = [n.op for n in rec["module"].program.nodes]
        op_dist_plan.update(ops)
        if src in TAIL_SOURCES:
            op_dist_exec.update(ops)
            op_dist_exec.update(["GENERATE", "VERIFY"])
        else:
            op_dist_exec.update(ops)

        low = lowering_of(rec)
        ds = per_ds.setdefault(src, {"samples": 0, "calls": 0,
                                     "exact": 0, "heuristic": 0,
                                     "fallback": 0, "zero_fallback": 0,
                                     "all_fallback": 0})
        ds["samples"] += 1
        by_source_samples[src].append({
            "text": rec["sample"].get("input_text") or "",
            "ops": tuple(ops),
            "caps": tuple(sorted((rec["sample"].get("metadata")
                                  or {}).get("capabilities") or [])),
        })
        if low is None:
            # non-tool sources lower directly (no toolmap): semantic
            # mapping is exact by construction
            ds["zero_fallback"] += 1
            continue
        n_fb = 0
        for c in low:
            kinds[c["mapping_kind"]] += 1
            skill_kind[c["skill"]][c["mapping_kind"]] += 1
            if c["skill"] == "EXEC_ACTION":
                if c["mapping_kind"] == "fallback":
                    exec_action["unknown_fallback"] += 1
                    n_fb += 1
                else:
                    exec_action["intentional"] += 1
            elif c["mapping_kind"] == "fallback":
                n_fb += 1
            ds["calls"] += 1
            ds[c["mapping_kind"]] += 1
        if n_fb == 0:
            ds["zero_fallback"] += 1
        if low and n_fb == len(low):
            ds["all_fallback"] += 1
        if src == "xlam":
            xlam_stats["total"] += 1
            if n_fb == 0:
                xlam_stats["zero_fallback"] += 1
            elif n_fb == len(low):
                xlam_stats["all_fallback"] += 1
            else:
                xlam_stats["some_fallback"] += 1

    # ---- Task 5: ambiguity (tool-use sources only) ----
    for src in TOOLUSE_SOURCES:
        samples = by_source_samples.get(src, [])
        if not samples:
            continue
        # exact-instruction groups
        groups = defaultdict(set)
        caps_by_group = defaultdict(set)
        for s in samples:
            key = normalize(s["text"])
            groups[key].add(s["ops"])
            caps_by_group[key].add(s["caps"])
        for key, opsets in groups.items():
            ambiguity["exact_groups"] += 1
            if len(opsets) > 1:
                ambiguity["exact_ambiguous"] += 1
                if len(caps_by_group[key]) > 1:
                    ambiguity["ambig_ctx_differ"] += 1
                else:
                    ambiguity["ambig_ctx_same"] += 1
        # near-duplicate families
        fams = near_duplicate_families([s["text"] for s in samples],
                                       op_seqs=[ "|".join(s["ops"])
                                                 for s in samples])
        fam_ops = defaultdict(set)
        fam_caps = defaultdict(set)
        for s, f in zip(samples, fams):
            fam_ops[f].add(s["ops"])
            fam_caps[f].add(s["caps"])
        for f, opsets in fam_ops.items():
            ambiguity["near_groups"] += 1
            if len(opsets) > 1:
                ambiguity["near_ambiguous"] += 1

    total_calls = sum(kinds.values()) or 1
    report = {
        "note": "plan view (no policy tail); lift/validator/execution "
                "coverage lives in dataset_coverage.json — fallbacks are "
                "NOT counted as semantic coverage here",
        "lowering_provenance_pct": {
            k: round(100 * v / total_calls, 2) for k, v in kinds.items()},
        "exec_action_split": exec_action,
        "per_dataset": {
            src: {**d,
                  "fallback_rate_pct": round(100 * d["fallback"]
                                             / d["calls"], 2)
                  if d["calls"] else 0.0,
                  "semantic_mapping_coverage_pct": round(
                      100 * d["zero_fallback"] / d["samples"], 2)
                  if d["samples"] else 0.0}
            for src, d in per_ds.items()},
        "xlam_detail": xlam_stats,
        "skill_mapping_kind_distribution": {
            k: dict(v) for k, v in skill_kind.items()},
        "policy_tail_effect": {
            "op_distribution_plan_view": dict(op_dist_plan.most_common()),
            "op_distribution_with_policy_tail": dict(op_dist_exec.most_common()),
        },
        "instruction_only_ambiguity": {
            **ambiguity,
            "exact_ambiguity_rate_pct": round(
                100 * ambiguity["exact_ambiguous"]
                / ambiguity["exact_groups"], 2)
            if ambiguity["exact_groups"] else 0.0,
            "near_ambiguity_rate_pct": round(
                100 * ambiguity["near_ambiguous"]
                / ambiguity["near_groups"], 2)
            if ambiguity["near_groups"] else 0.0,
            "interpretation": "ambig_ctx_differ > 0 means a capability "
                              "context (view B) could disambiguate those "
                              "groups; ambig_ctx_same groups are ambiguous "
                              "even with full context",
        },
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    md = out.with_suffix(".md")
    L = ["# Training Corpus Quality Audit", ""]
    L += ["## Lowering provenance", "",
          f"- exact: **{report['lowering_provenance_pct'].get('exact', 0)}%**"
          f"  heuristic: {report['lowering_provenance_pct'].get('heuristic', 0)}%"
          f"  fallback: **{report['lowering_provenance_pct'].get('fallback', 0)}%**",
          f"- EXEC_ACTION: intentional {exec_action['intentional']}"
          f" vs unknown-fallback {exec_action['unknown_fallback']}", ""]
    L += ["## Per dataset", "",
          "| source | samples | calls | exact | heuristic | fallback % | "
          "semantic_mapping_coverage % |", "|---|---|---|---|---|---|---|"]
    for src, d in report["per_dataset"].items():
        L.append(f"| {src} | {d['samples']} | {d['calls']} | {d['exact']} "
                 f"| {d['heuristic']} | {d['fallback_rate_pct']} "
                 f"| {d['semantic_mapping_coverage_pct']} |")
    L += ["", f"xLAM detail: {json.dumps(xlam_stats)}", "",
          "## Policy tail effect (GENERATE/VERIFY)", "",
          f"- plan view GENERATE count: {op_dist_plan.get('GENERATE', 0)}, "
          f"VERIFY: {op_dist_plan.get('VERIFY', 0)}",
          f"- with tail GENERATE: {op_dist_exec.get('GENERATE', 0)}, "
          f"VERIFY: {op_dist_exec.get('VERIFY', 0)}", "",
          "## Instruction-only ambiguity (Task 5)", "",
          f"- exact groups: {ambiguity['exact_groups']}, ambiguous: "
          f"{ambiguity['exact_ambiguous']} "
          f"({report['instruction_only_ambiguity']['exact_ambiguity_rate_pct']}%)",
          f"- near-dup families: {ambiguity['near_groups']}, ambiguous: "
          f"{ambiguity['near_ambiguous']} "
          f"({report['instruction_only_ambiguity']['near_ambiguity_rate_pct']}%)",
          f"- ambiguous w/ differing capability context: "
          f"{ambiguity['ambig_ctx_differ']}; same context: "
          f"{ambiguity['ambig_ctx_same']}", ""]
    md.write_text("\n".join(L), encoding="utf-8")

    print(json.dumps({k: report[k] for k in
                      ("lowering_provenance_pct", "exec_action_split",
                       "instruction_only_ambiguity")}, indent=2)[:1500])
    print(f"report -> {out} / {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
