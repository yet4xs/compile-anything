"""Dataset statistics over data/taskir/** — workload characterization.

Writes data/reports/dataset_stats.json and .md with per-subset and overall
distributions (see src/stats.py for metric definitions).
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.taskir import load_module  # noqa: E402
from src.stats import program_metrics, aggregate, markdown_summary  # noqa: E402

SUBSETS = ["examples", "synthetic", "xlam"]


def main() -> int:
    base = ROOT / "data" / "taskir"
    reports = ROOT / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    per_subset = {}
    for sub in SUBSETS:
        d = base / sub
        if not d.is_dir():
            continue
        metrics = []
        for f in sorted(d.glob("*.json")):
            metrics.append(program_metrics(load_module(f)))
        per_subset[sub] = aggregate(metrics, subset=sub)
        all_metrics.extend(metrics)

    overall = aggregate(all_metrics, subset="ALL")

    reports.joinpath("dataset_stats.json").write_text(
        json.dumps({"per_subset": per_subset, "overall": overall},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    md = ["# TaskIR Dataset Statistics", "",
          "Metric definitions in `src/stats.py`: depth = longest dependency "
          "chain; width = max nodes per depth level; branch_ratio = "
          "(guarded + SELECT) / nodes; parallel_ratio = 1 - critical_path_nodes"
          " / nodes.", ""]
    for sub in SUBSETS:
        if sub in per_subset:
            md.append(markdown_summary(per_subset[sub]))
    md.append(markdown_summary(overall))
    reports.joinpath("dataset_stats.md").write_text("\n".join(md), encoding="utf-8")

    print(f"programs: {overall['programs']}  "
          f"(nodes mean {overall['nodes']['mean']}, "
          f"depth mean {overall['depth']['mean']}, "
          f"width mean {overall['width']['mean']})")
    print(f"parallel programs: {overall['programs_with_parallelism']*100:.1f}%  "
          f"with retry: {overall['programs_with_retry']*100:.1f}%")
    print("reports -> data/reports/dataset_stats.{json,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
