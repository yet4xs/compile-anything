"""Build Compiler Corpus v2 from REAL benchmark data (+ labeled synthetic).

    python scripts/build_real_corpus.py                     # real only
    python scripts/build_real_corpus.py --with-synthetic    # + 10k labeled

Pipeline per sample: adapter -> lifter -> validator -> simulator ->
corpus record. Splits: train 90% / val 5% / test 5% (seeded).
Output: data/compiler_corpus_v2/{train,val,test}.jsonl + stats.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.pipeline import run_pipeline, coverage_stats  # noqa: E402
from src.ir.taskir import module_to_dict, to_text              # noqa: E402


def make_record(rec) -> dict:
    s, mod = rec["sample"], rec["module"]
    res = rec.get("result")
    return {
        "instruction": s.get("input_text") or "",
        "source": s["source"],
        "id": s["id"],
        "taskir_text": to_text(mod),
        "taskir_json": module_to_dict(mod),
        "validator": {"pass": True,
                      "warnings": rec.get("warnings", [])},
        "execution": {
            "status": res.status if res else "not_simulated",
            "cost": {
                "latency_sequential_ms": res.seq_latency_ms,
                "critical_path_ms": res.critical_path_ms,
                "energy_j": res.energy_j,
                "memory_peak_mb": res.peak_memory_mb,
                "lm_calls": res.lm_calls, "api_calls": res.api_calls,
                "retries": res.retries,
            } if res else {},
        },
    }


def load_synthetic(path: pathlib.Path, cap: int) -> list:
    """Relabel the Phase-4 schema-faithful corpus as explicitly synthetic."""
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(out) >= cap:
                break
            r = json.loads(line)
            out.append({
                "instruction": r["input"]["text"],
                "source": f"synthetic:{r['source']['dataset']}",
                "id": r["source"]["id"],
                "taskir_text": r["taskir_text"],
                "taskir_json": r["taskir"],
                "validator": {"pass": True, "warnings": []},
                "execution": {"status": r["execution"]["trace"]["status"],
                              "cost": r["execution"]["cost"]},
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--with-synthetic", action="store_true",
                    help="append the labeled Phase-4 synthetic corpus")
    ap.add_argument("--synthetic-cap", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "data" / "compiler_corpus_v2"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    _, lifted = run_pipeline(pathlib.Path(args.raw_root), args.source or None)
    records = [make_record(r) for r in lifted
               if r["module"] is not None and r.get("valid")]
    n_real = len(records)
    n_synth = 0
    if args.with_synthetic:
        synth = load_synthetic(ROOT / "data" / "compiler_corpus" / "train.jsonl",
                               args.synthetic_cap)
        records.extend(synth)
        n_synth = len(synth)

    rng = random.Random(args.seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = int(len(records) * 0.05)
    n_test = int(len(records) * 0.05)
    val_ids = set(idx[:n_val])
    test_ids = set(idx[n_val:n_val + n_test])

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": [r for i, r in enumerate(records) if i not in val_ids | test_ids],
        "val": [r for i, r in enumerate(records) if i in val_ids],
        "test": [r for i, r in enumerate(records) if i in test_ids],
    }
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = coverage_stats(lifted)
    stats["CORPUS"] = {
        "real_records": n_real, "synthetic_records": n_synth,
        "total": len(records),
        "train": len(splits["train"]), "val": len(splits["val"]),
        "test": len(splits["test"]),
        "note": "synthetic records are explicitly labeled source=synthetic:*",
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2,
                                               ensure_ascii=False),
                                    encoding="utf-8")

    for name, rows in splits.items():
        print(f"{name:5s} {len(rows)}")
    print(f"real={n_real} synthetic(labeled)={n_synth} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
