"""xLAM -> TaskIR pipeline: load raw records, lift, validate, write dataset.

The validator is the quality gate: only valid TaskIR enters data/taskir/xlam/
(and the training pairs); invalid liftings are quarantined with their issue
reports for inspection.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.taskir import save_module, to_text, module_to_dict  # noqa: E402
from src.lifter.xlam import lift_record  # noqa: E402
from src.validator.validator import validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "raw" / "xlam_sample.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    src = pathlib.Path(args.input)
    outdir = ROOT / "data" / "taskir" / "xlam"
    quarantine = outdir / "_invalid"
    traindir = ROOT / "data" / "train"
    outdir.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)
    traindir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in
               open(src, encoding="utf-8").read().splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    n_ok = n_bad = 0
    pairs_path = traindir / "xlam_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as pf:
        for i, rec in enumerate(records, 1):
            mod = lift_record(rec)
            rep = validate(mod)
            if rep.valid:
                save_module(mod, outdir / f"xlam_{i:04d}.json")
                pf.write(json.dumps({
                    "task": mod.program.description,
                    "taskir_text": to_text(mod),
                    "taskir_json": module_to_dict(mod),
                }, ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                n_bad += 1
                issues = [{"code": e.code, "node": e.node, "msg": e.msg}
                          for e in rep.errors]
                save_module(mod, quarantine / f"xlam_{i:04d}.json")
                (quarantine / f"xlam_{i:04d}.issues.json").write_text(
                    json.dumps(issues, indent=2), encoding="utf-8")
                print(f"quarantined xlam_{i:04d}: {rep.summary()}")

    print(f"xLAM lift: {n_ok} valid -> {outdir}, {n_bad} quarantined; "
          f"pairs -> {pairs_path}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
