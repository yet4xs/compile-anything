"""Compiler evaluation protocol (Phase 4 Task 7).

Measures a TaskIR producer (today: the corpus / lifters; Phase 5: a model)
on THREE escalating gates — exact match is explicitly NOT a metric:

  1. syntax accuracy    — can the TaskIR text be parsed? (src/ir/parser)
  2. validator pass rate — is the parsed IR a legal program? (V1-V6)
  3. semantic execution — does it run to completion in the simulator?

Inputs:
  --corpus  data/compiler_corpus/train.jsonl (default)
      evaluates each record's taskir_text (roundtrip through the parser);
  --predictions FILE
      {"input_text": str, "output_text": str} per line — the Phase-5 model
      interface: output_text is parsed, validated, executed.

Also aggregates cost (latency / energy / lm+api calls) per source.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.parser import parse_text, TaskIRSyntaxError   # noqa: E402
from src.validator.validator import validate               # noqa: E402
from src.runtime.simulator import Simulator                # noqa: E402


def evaluate_text(text: str, source: str, idx: int):
    """One (syntax -> validate -> execute) pipeline over a TaskIR text."""
    rec = {"source": source, "index": idx, "parsed": False,
           "valid": False, "executed": False,
           "errors": [], "cost": None}
    try:
        mod = parse_text(text)
    except TaskIRSyntaxError as e:
        rec["errors"].append(f"syntax: {e}")
        return rec
    rec["parsed"] = True

    rep = validate(mod)
    if not rep.valid:
        rec["errors"] = [f"{e.code}@{e.node}: {e.msg}" for e in rep.errors]
        return rec
    rec["valid"] = True

    res = Simulator(mod, seed=f"eval:{source}:{idx}", jitter=0).run()
    rec["executed"] = res.status == "completed"
    if res.status != "completed":
        rec["errors"].append(f"execution: {res.output_digest[:120]}")
    rec["cost"] = {
        "latency_sequential_ms": res.seq_latency_ms,
        "critical_path_ms": res.critical_path_ms,
        "energy_j": res.energy_j,
        "memory_peak_mb": res.peak_memory_mb,
        "lm_calls": res.lm_calls, "api_calls": res.api_calls,
    }
    return rec


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(records):
    by = defaultdict(lambda: {"n": 0, "parsed": 0, "valid": 0, "executed": 0,
                              "lat": [], "energy": [], "lm": [], "api": [],
                              "err_kinds": defaultdict(int)})
    for r in records:
        s = by[r["source"]]
        s["n"] += 1
        s["parsed"] += r["parsed"]
        s["valid"] += r["valid"]
        s["executed"] += r["executed"]
        if r["cost"] and r["executed"]:
            s["lat"].append(r["cost"]["latency_sequential_ms"])
            s["energy"].append(r["cost"]["energy_j"])
            s["lm"].append(r["cost"]["lm_calls"])
            s["api"].append(r["cost"]["api_calls"])
        for e in r["errors"]:
            s["err_kinds"][e.split(":")[0].split("@")[0]] += 1

    out = {}
    for src, s in sorted(by.items()):
        n = s["n"] or 1
        out[src] = {
            "n": s["n"],
            "syntax_accuracy_pct": round(100 * s["parsed"] / n, 2),
            "validator_pass_pct": round(100 * s["valid"] / n, 2),
            "execution_rate_pct": round(100 * s["executed"] / n, 2),
            "mean_latency_ms": _mean(s["lat"]),
            "mean_energy_j": _mean(s["energy"]),
            "mean_lm_calls": _mean(s["lm"]),
            "mean_api_calls": _mean(s["api"]),
            "error_kinds": dict(s["err_kinds"]),
        }
    total = len(records) or 1
    out["ALL"] = {
        "n": len(records),
        "syntax_accuracy_pct": round(100 * sum(r["parsed"] for r in records) / total, 2),
        "validator_pass_pct": round(100 * sum(r["valid"] for r in records) / total, 2),
        "execution_rate_pct": round(100 * sum(r["executed"] for r in records) / total, 2),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "compiler_corpus"
                                            / "train.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--predictions", default=None,
                    help="jsonl with {input_text, output_text} (model mode)")
    ap.add_argument("--out", default=str(ROOT / "data" / "reports"
                                         / "compiler_eval.json"))
    args = ap.parse_args()

    records = []
    if args.predictions:
        lines = [json.loads(l) for l in
                 open(args.predictions, encoding="utf-8").read().splitlines()
                 if l.strip()]
        if args.limit:
            lines = lines[: args.limit]
        for i, p in enumerate(lines):
            records.append(evaluate_text(p.get("output_text", ""),
                                         source="predictions", idx=i))
    else:
        path = pathlib.Path(args.corpus)
        if not path.exists():
            print(f"corpus not found: {path} — run "
                  f"scripts/build_compiler_corpus.py first")
            return 1
        lines = [json.loads(l) for l in
                 open(path, encoding="utf-8").read().splitlines() if l.strip()]
        if args.limit:
            lines = lines[: args.limit]
        for i, rec in enumerate(lines):
            records.append(evaluate_text(rec.get("taskir_text", ""),
                                         source=rec["source"]["dataset"], idx=i))

    summary = summarize(records)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    md = out.with_suffix(".md")
    lines = ["# Compiler Eval — syntax / validity / execution", "",
             "| source | n | syntax % | validator % | execution % | "
             "mean lat ms | mean energy J | lm | api/db |",
             "|---|---|---|---|---|---|---|---|---|"]
    for src, s in summary.items():
        lines.append(f"| {src} | {s['n']} | {s['syntax_accuracy_pct']} | "
                     f"{s['validator_pass_pct']} | {s['execution_rate_pct']} | "
                     f"{s.get('mean_latency_ms', '-')} | "
                     f"{s.get('mean_energy_j', '-')} | "
                     f"{s.get('mean_lm_calls', '-')} | "
                     f"{s.get('mean_api_calls', '-')} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for src, s in summary.items():
        print(f"{src:12s} n={s['n']:5d} syntax={s['syntax_accuracy_pct']:6.2f}% "
              f"valid={s['validator_pass_pct']:6.2f}% "
              f"exec={s['execution_rate_pct']:6.2f}%")
    print(f"report -> {out} / {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
