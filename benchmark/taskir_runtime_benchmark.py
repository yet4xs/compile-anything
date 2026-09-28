"""TaskIR runtime benchmark protocol (Phase 3 Task 5).

For each input TaskIR program:
    validate -> list-schedule (bounded executor pools) -> simulate
    -> one JSON record with the protocol fields.

Output per program:
    { name, valid, status,
      latency_sequential_ms, makespan_scheduled_ms, critical_path_ms,
      energy_j, memory_peak_mb, lm_calls, api_calls, retries, skipped,
      flops, schedule: [ {node, op, executor, start_ms, end_ms} ] }

Protocol decisions (fixed for comparability):
    - simulator jitter = 0 (nominal registry latencies, deterministic)
    - executor pools default to 1 per resource class; override with --pool
    - 70B single-shot reference included for ratio columns

Usage:
    python benchmark/taskir_runtime_benchmark.py                 # examples/
    python benchmark/taskir_runtime_benchmark.py data/taskir/xlam/xlam_0001.json
    python benchmark/taskir_runtime_benchmark.py --glob 'data/taskir/synthetic/syn_000*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.taskir import load_module                       # noqa: E402
from src.validator.validator import validate                # noqa: E402
from src.optimizer.scheduler import ListScheduler           # noqa: E402
from src.runtime.simulator import Simulator                 # noqa: E402
from src.isa.registry import BIG_MODEL_BASELINE             # noqa: E402


def benchmark_one(path: pathlib.Path, pools: dict) -> dict:
    mod = load_module(path)
    rep = validate(mod)
    rec = {
        "name": mod.program.name,
        "file": str(path),
        "valid": rep.valid,
        "status": None,
        "latency_sequential_ms": None,
        "makespan_scheduled_ms": None,
        "critical_path_ms": None,
        "energy_j": None,
        "memory_peak_mb": None,
        "lm_calls": None,
        "api_calls": None,
        "flops": None,
        "retries": None,
        "skipped": None,
        "schedule": None,
    }
    if not rep.valid:
        rec["status"] = "invalid"
        return rec

    sched = ListScheduler(mod, pools=pools).schedule()
    res = Simulator(mod, seed=f"bench:{path.stem}", jitter=0).run()

    rec.update({
        "status": res.status,
        "latency_sequential_ms": res.seq_latency_ms,
        "makespan_scheduled_ms": sched.makespan_ms,
        "critical_path_ms": res.critical_path_ms,
        "energy_j": res.energy_j,
        "memory_peak_mb": res.peak_memory_mb,
        "lm_calls": res.lm_calls,
        "api_calls": res.api_calls,
        "flops": res.flops,
        "retries": res.retries,
        "skipped": res.skipped,
        "schedule": sched.to_dict()["schedule"],
        "schedule_ok": sched.ok,
    })
    return rec


def aggregate(records: list) -> dict:
    comp = [r for r in records if r["status"] == "completed"]
    base_lat = BIG_MODEL_BASELINE["latency_ms"]
    base_energy = BIG_MODEL_BASELINE["energy_j"]

    def mean(key):
        vals = [r[key] for r in comp if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "programs": len(records),
        "valid": sum(1 for r in records if r["valid"]),
        "completed": len(comp),
        "mean_latency_sequential_ms": mean("latency_sequential_ms"),
        "mean_makespan_scheduled_ms": mean("makespan_scheduled_ms"),
        "mean_critical_path_ms": mean("critical_path_ms"),
        "mean_energy_j": mean("energy_j"),
        "mean_memory_peak_mb": mean("memory_peak_mb"),
        "total_lm_calls": sum(r["lm_calls"] or 0 for r in comp),
        "total_api_calls": sum(r["api_calls"] or 0 for r in comp),
        "mean_energy_vs_70b_pct": round(
            mean("energy_j") / base_energy * 100, 3) if comp else None,
        "mean_latency_vs_70b_pct": round(
            mean("makespan_scheduled_ms") / base_lat * 100, 3) if comp else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*",
                    help="TaskIR json files (default: data/taskir/examples/)")
    ap.add_argument("--glob", dest="pattern", default=None)
    ap.add_argument("--pool", action="append", default=[],
                    help="resource pool override, e.g. --pool lm=2 (repeatable)")
    ap.add_argument("--out", default=str(ROOT / "data" / "reports"
                                         / "benchmark_results.json"))
    args = ap.parse_args()

    pools = {}
    for p in args.pool:
        cls, _, n = p.partition("=")
        pools[cls] = int(n)

    files = [pathlib.Path(f) for f in args.inputs]
    if args.pattern:
        files += [pathlib.Path(f) for f in sorted(glob.glob(args.pattern))]
    if not files:
        exdir = ROOT / "data" / "taskir" / "examples"
        files = sorted(exdir.glob("*.json"))
    if not files:
        print("no input programs found")
        return 1

    records = [benchmark_one(f, pools) for f in files]
    agg = aggregate(records)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"aggregate": agg, "programs": records},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = ["# TaskIR Runtime Benchmark", "",
             f"- programs: {agg['programs']} (valid {agg['valid']}, "
             f"completed {agg['completed']})",
             f"- executor pools: {pools or 'default (1 per resource class)'}",
             f"- simulator: nominal costs, jitter=0, deterministic",
             "- makespan schedules BOTH guarded branches optimistically;",
             "  critical-path and sequential reflect the actual execution", "",
             "| program | status | seq ms | makespan ms | crit-path ms | "
             "energy J | peak MB | lm | api/db | retries |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append(
            f"| {r['name']} | {r['status']} | "
            f"{r['latency_sequential_ms'] or '-'} | "
            f"{r['makespan_scheduled_ms'] or '-'} | "
            f"{r['critical_path_ms'] or '-'} | {r['energy_j'] or '-'} | "
            f"{r['memory_peak_mb'] or '-'} | {r['lm_calls'] if r['lm_calls'] is not None else '-'} | "
            f"{r['api_calls'] if r['api_calls'] is not None else '-'} | "
            f"{r['retries'] if r['retries'] is not None else '-'} |")
    lines += ["", "## Aggregate", "", "```json",
              json.dumps(agg, indent=2), "```", "",
              "70B single-shot reference: "
              f"~{BIG_MODEL_BASELINE['latency_ms']:g} ms, "
              f"~{BIG_MODEL_BASELINE['energy_j']:g} J, "
              f"~{BIG_MODEL_BASELINE['memory_mb']:g} MB.", ""]
    md.write_text("\n".join(lines), encoding="utf-8")

    for r in records:
        print(f"{r['name']:40s} {r['status']:10s} "
              f"makespan={r['makespan_scheduled_ms']}ms "
              f"energy={r['energy_j']}J")
    print(f"\naggregate: {json.dumps(agg)}")
    print(f"results -> {out} / {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
