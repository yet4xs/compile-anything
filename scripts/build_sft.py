"""Build the SFT dataset for the Qwen2B Compiler from TaskIR training pairs.

    python scripts/build_sft.py                 # uses all data/train/*_pairs.jsonl
    python scripts/build_sft.py --val-ratio 0.1
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.train_data import load_pairs, write_sft  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "train" / "sft"))
    args = ap.parse_args()

    pair_files = sorted((ROOT / "data" / "train").glob("*_pairs.jsonl"))
    if not pair_files:
        print("no *_pairs.jsonl under data/train/ — run the dataset pipeline first")
        return 1
    items = load_pairs(pair_files)
    stats = write_sft(items, pathlib.Path(args.out),
                      val_ratio=args.val_ratio, seed=args.seed)
    print(f"SFT dataset: {stats['train']} train / {stats['val']} val "
          f"records (from {len(items)} pairs) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
