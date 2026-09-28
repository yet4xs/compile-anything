"""Build Compiler Corpus v3 — quality-gated (Phase 5B-0 Task 6).

Changes vs v2 (data quality gate):
  - dual targets per record: plan_target (pure lowering — first-round SFT
    target) and execution_target (compiler policy tail allowed)
  - quality_tier: A (no fallback, real benchmark, target from ground
    truth) / B (heuristic mappings, still semantic) / C (any fallback or
    heavily synthesized target). First-round SFT uses A+B; C is written to
    tier_c.jsonl for ablation/augmentation only.
  - group-aware split: near-duplicate families (MinHash LSH + same
    op-sequence) never cross splits; exact leakage must be 0.
  - capabilities field (normalized semantic capability context) for
    tool-use records (Task 5 view-B experiments).

    python scripts/build_real_corpus.py --v3                # default
    python scripts/build_real_corpus.py --v3 --with-synthetic
    python scripts/build_real_corpus.py --random-split      # v2 repro mode
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.pipeline import run_pipeline                # noqa: E402
from src.dataset.dedup import (near_duplicate_families,      # noqa: E402
                               group_split, cross_split_leakage)
from src.ir.taskir import module_to_dict, to_text            # noqa: E402

TOOLUSE_SOURCES = {"toolbench", "toolbench_static", "xlam", "apibank",
                   "agentbench"}


def quality_tier(source: str, lowering) -> str:
    if source.startswith("synthetic"):
        return "C"                       # heavily synthesized target
    if lowering is None:
        return "A"                       # direct lowering (sql/rtl/code)
    kinds = {c["mapping_kind"] for c in lowering}
    if "fallback" in kinds:
        return "C"
    if "heuristic" in kinds:
        return "B"
    return "A"


def make_v3_record(rec, exec_rec) -> dict:
    s, mod = rec["sample"], rec["module"]
    prov = mod.meta.get("provenance", {})
    out = {
        "instruction": s.get("input_text") or "",
        "source": s["source"],
        "id": s["id"],
        "plan_target": to_text(mod),
        "plan_json": module_to_dict(mod),
        "validator": {"pass": True, "warnings": rec.get("warnings", [])},
        "quality_tier": quality_tier(s["source"], prov.get("lowering")),
        "capabilities": (s.get("metadata") or {}).get("capabilities") or [],
        "lowering": prov.get("lowering"),
    }
    if exec_rec is not None and exec_rec.get("module") is not None:
        eres = exec_rec.get("result")
        out["execution_target"] = to_text(exec_rec["module"])
        out["execution"] = {
            "status": eres.status if eres else "not_simulated",
            "cost": {
                "latency_sequential_ms": eres.seq_latency_ms,
                "critical_path_ms": eres.critical_path_ms,
                "energy_j": eres.energy_j,
                "lm_calls": eres.lm_calls, "api_calls": eres.api_calls,
            } if eres else {},
        }
    else:
        out["execution_target"] = out["plan_target"]
        out["execution"] = {"status": "n/a", "cost": {}}
    return out


def load_synthetic_records(path: pathlib.Path, cap: int) -> list:
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(out) >= cap:
                break
            r = json.loads(line)
            text = r["taskir_text"]
            out.append({
                "instruction": r["input"]["text"],
                "source": f"synthetic:{r['source']['dataset']}",
                "id": r["source"]["id"],
                "plan_target": text, "plan_json": r["taskir"],
                "execution_target": text,
                "validator": {"pass": True, "warnings": []},
                "quality_tier": "C",
                "capabilities": [], "lowering": None,
                "execution": {"status": r["execution"]["trace"]["status"],
                              "cost": r["execution"]["cost"]},
            })
    return out


def build_v3(args) -> int:
    print("plan view (lift+validate)...")
    _, plan = run_pipeline(pathlib.Path(args.raw_root), args.source or None,
                           simulate=False, view="plan")
    print("execution view (lift+validate+simulate)...")
    _, exe = run_pipeline(pathlib.Path(args.raw_root), args.source or None,
                          simulate=True, view="execution")
    exe_by_id = {r["sample"]["id"]: r for r in exe}

    records, tier_c = [], []
    for r in plan:
        if r["module"] is None or not r.get("valid"):
            continue
        rec = make_v3_record(r, exe_by_id.get(r["sample"]["id"]))
        (tier_c if rec["quality_tier"] == "C" else records).append(rec)
    n_real_ab = len(records)
    n_real_c = len(tier_c)
    if args.with_synthetic:
        synth = load_synthetic_records(
            ROOT / "data" / "compiler_corpus" / "train.jsonl",
            args.synthetic_cap)
        tier_c.extend(synth)

    # group-aware split over A+B records
    texts = [r["instruction"] for r in records]
    op_seqs = ["|".join(node_op_seq(r)) for r in records]
    print("near-duplicate families (MinHash LSH)...")
    fams = near_duplicate_families(texts, op_seqs=op_seqs)
    splits = group_split(fams, seed=args.seed)
    for r, sp in zip(records, splits):
        r["split"] = sp

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    for r in records:
        buckets[r["split"]].append(r)
    for name in ("train", "val", "test"):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in buckets[name]:
                rr = dict(r)
                rr.pop("split", None)
                f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    with open(out / "tier_c.jsonl", "w", encoding="utf-8") as f:
        for r in tier_c:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("leakage check...")
    leak = cross_split_leakage(texts, splits)
    tiers = Counter(r["quality_tier"] for r in records) \
        + Counter(r["quality_tier"] for r in tier_c)
    stats = {
        "tiers": dict(tiers),
        "tier_A_B_real": n_real_ab,
        "tier_C": len(tier_c),
        "real_C_before_synthetic": n_real_c,
        "splits": {k: len(v) for k, v in buckets.items()},
        "leakage": leak,
        "tier_policy": "first-round SFT = A+B only; C (fallback or "
                       "synthetic) for ablation/augmentation",
        "targets": "plan_target is the SFT target (no policy tail); "
                   "execution_target kept for runtime-oriented experiments",
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2,
                                               ensure_ascii=False),
                                    encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0 if leak["cross_split_exact_duplicates"] == 0 else 1


def node_op_seq(rec) -> list:
    return [n["op"] for n in rec["plan_json"]["program"]["nodes"]]


def build_v2(args) -> int:
    """Legacy random-split mode (v2 reproduction)."""
    _, lifted = run_pipeline(pathlib.Path(args.raw_root), args.source or None)
    records = []
    for r in lifted:
        if r["module"] is None or not r.get("valid"):
            continue
        rec = make_v3_record(r, r)
        records.append({
            "instruction": rec["instruction"], "source": rec["source"],
            "id": rec["id"], "taskir_text": rec["execution_target"],
            "taskir_json": rec["plan_json"],
            "validator": rec["validator"], "execution": rec["execution"]})
    if args.with_synthetic:
        records.extend(load_synthetic_records(
            ROOT / "data" / "compiler_corpus" / "train.jsonl",
            args.synthetic_cap))
    rng = random.Random(args.seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val, n_test = int(len(records) * .05), int(len(records) * .05)
    val_ids = set(idx[:n_val])
    test_ids = set(idx[n_val:n_val + n_test])
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        rows = [r for i, r in enumerate(records)
                if (i in val_ids and name == "val")
                or (i in test_ids and name == "test")
                or (i not in val_ids and i not in test_ids and name == "train")]
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"v2-mode: train/val/test written -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--with-synthetic", action="store_true")
    ap.add_argument("--synthetic-cap", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "data" / "compiler_corpus_v3"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--random-split", action="store_true",
                    help="legacy v2 mode (random split, no tiers)")
    args = ap.parse_args()
    return build_v2(args) if args.random_split else build_v3(args)


if __name__ == "__main__":
    sys.exit(main())
