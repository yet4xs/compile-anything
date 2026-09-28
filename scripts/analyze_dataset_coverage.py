"""Real dataset coverage analysis (Phase 5A Task 6).

    python scripts/analyze_dataset_coverage.py
    python scripts/analyze_dataset_coverage.py --source spider --source mbpp

Per source: lift coverage, skill coverage, reject-reason histogram,
TaskIR graph stats -> data/reports/dataset_coverage.{json,md}
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.pipeline import run_pipeline, coverage_stats  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "reports"
                                         / "dataset_coverage.json"))
    args = ap.parse_args()

    stats = coverage_stats(run_pipeline(pathlib.Path(args.raw_root),
                                        args.source or None)[1])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    md = out.with_suffix(".md")
    lines = ["# Real Dataset Coverage Analysis", ""]
    for src, s in sorted(stats.items()):
        lines += [f"## {src}", "",
                  f"- samples: **{s['samples']}**  lifted: {s['lifted']} "
                  f"({s['lift_coverage_pct']}%)  valid: {s['valid']} "
                  f"({s['valid_coverage_pct']}%)  executed/valid: "
                  f"{s['execution_pct_of_valid']}%", ""]
        lines += ["### reject reasons", "", "| reason | count |", "|---|---|"]
        for k, v in s["reject_reasons"].items():
            lines.append(f"| {k} | {v} |")
        lines += ["", "### skill coverage", "", "| skill | count |", "|---|---|"]
        for k, v in s["skill_coverage"].items():
            lines.append(f"| {k} | {v} |")
        g = s["taskir_graph"]
        lines += ["", f"### graph: nodes mean {g['nodes_mean']}, "
                  f"depth max {g['depth_max']}, width max {g['width_max']}, "
                  f"branch ratio {g['branch_ratio_mean']}, "
                  f"parallel ratio {g['parallel_ratio_mean']}", ""]
    md.write_text("\n".join(lines), encoding="utf-8")

    for src, s in sorted(stats.items()):
        print(f"{src:12s} n={s['samples']:5d} lifted={s['lift_coverage_pct']:6.2f}% "
              f"valid={s['valid_coverage_pct']:6.2f}% "
              f"exec={s['execution_pct_of_valid']:6.2f}% "
              f"rejects={s['reject_reasons']}")
    print(f"report -> {out} / {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
