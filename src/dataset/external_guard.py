"""External-benchmark contamination firewall (Phase 5B-0.4 Task 11).

Everything under data/external_benchmarks/ is EVALUATION-ONLY. It must
never enter any training corpus. All corpus builders funnel their input
paths through assert_not_external(), which raises RuntimeError on any
external-benchmark path unless an explicit allow flag is passed — and
Phase 5B forbids passing it.
"""
from __future__ import annotations

import pathlib
from typing import Iterable

EXTERNAL_ROOTS = ("data/external_benchmarks", "external_benchmarks")
ALLOW_FLAG = "--allow-external-training"      # FORBIDDEN in Phase 5B


def is_external_path(path) -> bool:
    p = pathlib.Path(path)
    try:
        s = str(p.resolve()).replace("\\", "/").lower()
    except OSError:
        s = str(p).replace("\\", "/").lower()
    return any(root in s for root in EXTERNAL_ROOTS)


def assert_not_external(paths, allow_external: bool = False) -> None:
    """Raise RuntimeError if any path points into external benchmarks.

    allow_external exists only as the future explicit opt-in; Phase 5B
    code must never set it (see task spec)."""
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    for p in paths or []:
        if is_external_path(p) and not allow_external:
            raise RuntimeError(
                f"CONTAMINATION BLOCKED: {p} is external EVALUATION data "
                f"(data/external_benchmarks/**) and must never enter a "
                f"training corpus. If you truly need this in a future "
                f"phase, pass {ALLOW_FLAG} explicitly — forbidden in "
                f"Phase 5B.")
