"""Prepare the SFT dataset from corpus v3 (chat format, ready for
train_lora.py / ms-swift).

    python scripts/prepare_sft.py                          # plan_target, A+B
    python scripts/prepare_sft.py --capability-context     # input view B
    python scripts/prepare_sft.py --target-field execution_target
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.external_guard import assert_not_external  # noqa: E402

from src.compiler.train.dataset import (records_to_chat,      # noqa: E402
                                        iter_corpus_records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data"
                                            / "compiler_corpus_v3"))
    ap.add_argument("--out", default=str(ROOT / "data" / "sft_v3"))
    ap.add_argument("--target-field", default="plan_target")
    ap.add_argument("--tiers", default="A,B")
    ap.add_argument("--capability-context", action="store_true")
    args = ap.parse_args()
    assert_not_external([args.corpus, args.out])   # firewall

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    corpus = pathlib.Path(args.corpus)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "val", "test"):
        src = corpus / f"{split}.jsonl"
        dst = out / f"{split}.jsonl"
        n = 0
        with open(dst, "w", encoding="utf-8") as f:
            for rec in records_to_chat(iter_corpus_records(
                    [src], target_field=args.target_field, tiers=tiers,
                    capability_context=args.capability_context)):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        counts[split] = n
        print(f"{split:5s} {n:6d} -> {dst}")
    print(f"target={args.target_field} tiers={tiers} "
          f"capability_context={args.capability_context}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
