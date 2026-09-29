"""Full-deterministic semantic label audit over corpus v3 (Task 8+).

    python scripts/audit_semantic_labels.py --corpus data/compiler_corpus_v3

Re-links every v3 record to its RAW ground truth (via the training-side
dataset adapters in data/raw/) and runs the source-specific checkers.
Nothing is modified; suspect samples are reported only. Also emits the
stratified judge sample (Task 9), the 256-sample sanity-overfit set
(Task 14), and the audit manifest (Task 15).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import subprocess
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audit.checkers import (audit_trajectory, audit_spider,   # noqa
                                audit_code, audit_rtl)
from src.dataset.adapters import (XlamAdapter, ToolBenchStaticAdapter,  # noqa
                                  ToolBenchAdapter, SpiderAdapter,
                                  HumanEvalAdapter, MBPPAdapter,
                                  VerilogEvalAdapter)


def build_raw_index():
    """sample_id -> raw ground-truth payload (from TRAINING raw data)."""
    raw_root = ROOT / "data" / "raw"
    idx = {}
    for adapter in (XlamAdapter(), ToolBenchStaticAdapter(),
                    ToolBenchAdapter(), SpiderAdapter(), HumanEvalAdapter(),
                    MBPPAdapter(), VerilogEvalAdapter()):
        try:
            for s in adapter.load(raw_root):
                idx[s["id"]] = s
        except Exception as e:                           # noqa: BLE001
            print(f"  [WARN] adapter {adapter.sources}: {e}")
    return idx


def raw_calls_of(raw) -> list:
    return (raw.get("trajectory") or []) if raw else []


def audit_record(record, raw) -> object:
    src = record["source"]
    if raw is None:
        from src.audit.semantic_label import SemanticAuditResult
        r = SemanticAuditResult(sample_id=record.get("id", ""), source=src,
                                tier=record.get("quality_tier", ""))
        r.status = "unverifiable"
        r.reasons.append("UNVERIFIABLE:raw ground truth not found")
        return r
    if src in ("xlam", "toolbench_static", "toolbench"):
        return audit_trajectory(record, raw_calls_of(raw))
    if src == "spider":
        return audit_spider(record, raw.get("sql") or "")
    if src in ("humaneval", "mbpp"):
        return audit_code(record, raw.get("code") or "",
                          strict_ops=(src == "humaneval"))
    if src == "verilogeval":
        return audit_rtl(record, raw.get("input_text") or "")
    from src.audit.semantic_label import SemanticAuditResult
    r = SemanticAuditResult(sample_id=record.get("id", ""), source=src,
                            tier=record.get("quality_tier", ""))
    r.status = "unverifiable"
    r.reasons.append(f"UNVERIFIABLE:no checker for source {src}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data"
                                            / "compiler_corpus_v3"))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    corpus = pathlib.Path(args.corpus)

    print("building raw ground-truth index ...")
    raw_idx = build_raw_index()
    print(f"  raw samples indexed: {len(raw_idx)}")

    records = []
    for split in ("train", "val", "test"):
        p = corpus / f"{split}.jsonl"
        for line in open(p, encoding="utf-8"):
            records.append(json.loads(line))
    print(f"corpus records: {len(records)}")

    results = []
    for rec in records:
        results.append(audit_record(rec, raw_idx.get(rec.get("id"))))

    # ---- aggregate ----------------------------------------------------
    def agg(sel):
        n = len(sel) or 1
        c = Counter(r.status for r in sel)
        return {"n": len(sel),
                "verified_pct": round(100 * c["verified"] / n, 3),
                "suspect_pct": round(100 * c["suspect"] / n, 3),
                "unverifiable_pct": round(100 * c["unverifiable"] / n, 3)}

    report = {"total": len(results),
              "overall": agg(results),
              "by_tier": {t: agg([r for r in results if r.tier == t])
                          for t in sorted({r.tier for r in results})},
              "by_source": {s: agg([r for r in results if r.source == s])
                            for s in sorted({r.source for r in results})}}

    # error taxonomy breakdown (Task 12)
    err = Counter()
    err_by_source = defaultdict(Counter)
    err_by_skill = defaultdict(Counter)
    for r in results:
        if r.status != "suspect":
            continue
        for reason in r.reasons:
            tax = reason.split(":")[0]
            if tax in ("POLICY_DERIVED", "TEMPLATE_BIAS"):
                continue          # annotations, not errors
            err[tax] += 1
            err_by_source[r.source][tax] += 1
            for op in (r.checks.get("plan_ops")
                       or [n["op"] for n in
                           next((x for x in records
                                 if x.get("id") == r.sample_id),
                                {}).get("plan_json", {})
                           .get("program", {}).get("nodes", [])] or []):
                err_by_skill[op][tax] += 1
    report["error_taxonomy"] = dict(err.most_common())
    report["error_taxonomy_by_source"] = {k: dict(v.most_common())
                                          for k, v in err_by_source.items()}
    report["error_taxonomy_by_skill"] = {k: dict(v.most_common())
                                         for k, v in err_by_skill.items()}
    report["top_error_patterns"] = [
        {"pattern": k, "count": v} for k, v in err.most_common(20)]

    out = ROOT / "data" / "reports" / "semantic_label_audit.json"
    detailed = {"summary": report,
                "results": [r.to_dict() for r in results]}
    out.write_text(json.dumps(detailed, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    md = ["# Semantic Label Audit (deterministic, pre-training)", "",
          f"total = {report['total']}; "
          f"verified {report['overall']['verified_pct']}% / "
          f"suspect {report['overall']['suspect_pct']}% / "
          f"unverifiable {report['overall']['unverifiable_pct']}%", "",
          "## by tier", "", "| tier | n | verified% | suspect% | unver% |",
          "|---|---:|---:|---:|---:|"]
    for t, a in report["by_tier"].items():
        md.append(f"| {t} | {a['n']} | {a['verified_pct']} "
                  f"| {a['suspect_pct']} | {a['unverifiable_pct']} |")
    md += ["", "## by source", "",
           "| source | n | verified% | suspect% | unver% |",
           "|---|---:|---:|---:|---:|"]
    for s, a in report["by_source"].items():
        md.append(f"| {s} | {a['n']} | {a['verified_pct']} "
                  f"| {a['suspect_pct']} | {a['unverifiable_pct']} |")
    md += ["", "## top error patterns", "", "| pattern | count |",
           "|---|---:|"]
    for e in report["top_error_patterns"]:
        md.append(f"| {e['pattern']} | {e['count']} |")
    (ROOT / "data" / "reports" / "semantic_label_audit.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"report -> {out}")

    # ---- Task 9: stratified judge sample --------------------------------
    rng = random.Random(args.seed)
    rec_by_id = {r["id"]: r for r in records}
    res_by_id = {r.sample_id: r for r in results}

    def strata(rec):
        ops = [n["op"] for n in rec["plan_json"]["program"]["nodes"]]
        return (rec["source"], rec["quality_tier"], tuple(sorted(set(ops))),
                len(ops) > 5)

    pools = defaultdict(list)
    for rec in records:
        if res_by_id[rec["id"]].status != "verified":
            continue
        pools[strata(rec)].append(rec)
    sample = []
    # Tier A target 500, Tier B target 1000, oversampling rare strata
    for tier, target in (("A", 500), ("B", 1000)):
        tier_pools = [p for s, p in pools.items() if s[1] == tier]
        total = sum(len(p) for p in tier_pools)
        if not total:
            continue
        for pool in tier_pools:                     # proportional baseline
            k = max(1, round(target * len(pool) / total))
            sample.extend(rng.sample(pool, min(k, len(pool))))
        # oversample rare strata (code/rtl/multi-action/heuristic)
        rare = [p for s, p in pools.items() if s[1] == tier and (
            s[0] in ("humaneval", "mbpp", "verilogeval") or s[3])]
        for pool in rare:
            extra = [r for r in pool if r not in sample]
            sample.extend(rng.sample(extra, min(max(2, target // 100),
                                                len(extra))))
    sample_ids = {r["id"] for r in sample}
    with open(ROOT / "data" / "reports" / "semantic_audit_sample.jsonl",
              "w", encoding="utf-8") as f:
        for rec in sample:
            raw = raw_idx.get(rec["id"], {})
            f.write(json.dumps({
                "sample_id": rec["id"], "source": rec["source"],
                "tier": rec["quality_tier"],
                "instruction": rec["instruction"],
                "raw_ground_truth_summary": {
                    "trajectory": (raw.get("trajectory") or
                                   raw.get("code") or raw.get("sql")
                                   or raw.get("input_text") or "")[:1500]
                    if isinstance(raw, dict) else ""},
                "taskir": rec["plan_target"],
                "provenance": rec.get("lowering"),
                "deterministic_audit": res_by_id[rec["id"]].to_dict(),
            }, ensure_ascii=False) + "\n")
    print(f"judge sample: {len(sample)} "
          f"(ids sampled from verified only)")

    # ---- Task 14: sanity overfit set ------------------------------------
    verified = [r for r in records
                if res_by_id[r["id"]].status == "verified"]
    by_source = defaultdict(list)
    for r in verified:
        by_source[r["source"]].append(r)
    sanity = []
    per = max(1, 256 // max(1, len(by_source)))
    for src, pool in sorted(by_source.items()):
        rng.shuffle(pool)
        sanity.extend(pool[:per])
    rng.shuffle(sanity)
    sanity = sanity[:256]
    sdir = ROOT / "data" / "sanity_overfit"
    sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / "samples.jsonl", "w", encoding="utf-8") as f:
        for r in sanity:
            f.write(json.dumps({
                "instruction": r["instruction"], "source": r["source"],
                "id": r["id"], "plan_target": r["plan_target"],
                "quality_tier": r["quality_tier"],
                "capabilities": r.get("capabilities") or []},
                ensure_ascii=False) + "\n")
    print(f"sanity_overfit: {len(sanity)} verified samples")

    # ---- Task 15: manifest ----------------------------------------------
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.strip()
    mdir = ROOT / "experiments" / "semantic_audit"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "manifest.json").write_text(json.dumps({
        "base_commit": "073ba99",
        "audit_run_commit": commit,
        "corpus": {s: hashlib.sha256(
            (corpus / f"{s}.jsonl").read_bytes()).hexdigest()
            for s in ("train", "val", "test")},
        "audit_code_sha256": hashlib.sha256(
            (ROOT / "src/audit/semantic_label.py").read_bytes()
        ).hexdigest(),
        "judge_sample_ids": sorted(sample_ids),
        "sanity_overfit_ids": [r["id"] for r in sanity],
        "random_seed": args.seed,
    }, indent=2), encoding="utf-8")
    print("manifest -> experiments/semantic_audit/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
