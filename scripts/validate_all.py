"""QC gate: validate every TaskIR file under data/taskir/ (excluding _invalid).

Exit code 1 if anything is invalid — run this before consuming the dataset.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.taskir import load_module  # noqa: E402
from src.validator.validator import validate  # noqa: E402


def main() -> int:
    base = ROOT / "data" / "taskir"
    files = [p for p in sorted(base.rglob("*.json"))
             if "_invalid" not in p.parts]
    n_ok = n_bad = 0
    for p in files:
        try:
            mod = load_module(p)
        except Exception as e:                       # malformed JSON etc.
            print(f"UNPARSEABLE {p}: {e}")
            n_bad += 1
            continue
        rep = validate(mod)
        if rep.valid:
            n_ok += 1
        else:
            n_bad += 1
            print(f"INVALID {p}")
            for i in rep.errors:
                print(f"    [{i.code}] {i.node or ''}: {i.msg}")
    print(f"validate_all: {n_ok} valid, {n_bad} invalid "
          f"({len(files)} files under {base})")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
