"""Download real benchmark datasets per registry plan.

    python scripts/download_datasets.py --list
    python scripts/download_datasets.py                      # all available
    python scripts/download_datasets.py --dataset spider --dataset humaneval
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.registry import REGISTRY, available, unavailable  # noqa: E402
from src.dataset.downloader import download_dataset               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[],
                    help="repeatable; default = all available")
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, s in REGISTRY.items():
            status = "available" if s.plan else "gated/unreachable"
            print(f"{n:12s} {s.task_type:8s} ~{s.expected_size:6d}  "
                  f"[{status}]  {s.source_url}")
            if s.notes:
                print(f"             {s.notes}")
        return 0

    names = args.dataset or available()
    for n in names:
        if n not in REGISTRY:
            print(f"unknown dataset {n!r}; --list to see registry")
            return 1
        download_dataset(n, pathlib.Path(args.raw_root))
    print(f"\ngated/unreachable (registered, not fetched): {unavailable()}")
    print("metadata -> data/raw/metadata.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
