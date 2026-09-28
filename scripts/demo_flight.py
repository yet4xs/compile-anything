"""Phase 1 acceptance demo: NL task -> TaskIR -> validator -> simulator
-> execution trace -> cost report.

Runs the three handwritten examples in data/taskir/examples/ and writes
data/reports/cost_report_flight.md. On the verified variant a verify-False
is injected once so the retry/rollback path is exercised in the trace.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.taskir import load_module, to_text                      # noqa: E402
from src.validator.validator import validate                        # noqa: E402
from src.runtime.simulator import Simulator                         # noqa: E402
from src.cost.model import console_report, markdown_report          # noqa: E402

EXAMPLES = ROOT / "data" / "taskir" / "examples"
REPORTS = ROOT / "data" / "reports"
FAIL_PLAN = {"%5": ["verify_false"]}    # first VERIFY attempt of the verified example


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    md_parts = ["# Phase 1 — End-to-End Demo Cost Reports",
                "", "Simulator: deterministic mock executors, nominal costs "
                "(see `spec/skill-isa.md` §4).", ""]
    rc = 0
    for path in sorted(EXAMPLES.glob("*.json")):
        mod = load_module(path)
        print("=" * 78)
        print(f"example: {path.name}")
        print("-" * 78)
        print(to_text(mod))
        rep = validate(mod)
        if not rep.valid:
            rc = 1
        sim = Simulator(mod, seed=f"demo:{path.stem}",
                        fail_plan=FAIL_PLAN if path.stem == "flight_verified" else None)
        res = sim.run()
        print(console_report(res, rep))
        md_parts.append(markdown_report(
            res, rep, title=f"{path.stem} — {mod.program.description}"))
        md_parts.append("")
    out = REPORTS / "cost_report_flight.md"
    out.write_text("\n".join(md_parts), encoding="utf-8")
    print("=" * 78)
    print(f"cost report written: {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
