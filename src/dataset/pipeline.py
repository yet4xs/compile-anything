"""Shared real-data pipeline: adapters -> lifters -> validator -> simulator.

Used by scripts/analyze_dataset_coverage.py and scripts/build_real_corpus.py
(one code path, no duplication).
"""
from __future__ import annotations

import pathlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..ir.taskir import module_to_dict, to_text
from ..validator.validator import validate
from ..runtime.simulator import Simulator
from ..stats import program_metrics
from ..lifter.benchmark import lift_sample
from .schema import to_lifter_input
from .adapters import adapters_for


def run_pipeline(raw_root: pathlib.Path,
                 sources: Optional[List[str]] = None,
                 simulate: bool = True):
    """Returns (samples, lifted) where lifted is a list of dicts:
    {sample, module, valid, warnings, result, metrics}."""
    all_samples: List[Dict] = []
    lifted: List[Dict] = []
    for adapter in adapters_for(sources):
        try:
            samples = adapter.load(raw_root)
        except Exception as e:                       # noqa: BLE001
            print(f"  adapter {adapter.sources} load error: {e}")
            continue
        for s in samples:
            s["_lifter_input"] = to_lifter_input(s)
            all_samples.append(s)
            mod, reason, lifter = lift_sample(s["_lifter_input"])
            if mod is None:
                lifted.append({"sample": s, "module": None, "reason": reason,
                               "lifter": lifter})
                continue
            rep = validate(mod)
            rec = {"sample": s, "module": mod, "reason": None,
                   "lifter": lifter, "valid": rep.valid,
                   "warnings": [w.code for w in rep.warnings]}
            if rep.valid:
                rec["metrics"] = program_metrics(mod)
                if simulate:
                    rec["result"] = Simulator(
                        mod, seed=f"real:{s['id']}", jitter=0).run()
            lifted.append(rec)
    return all_samples, lifted


def coverage_stats(lifted: List[Dict]) -> Dict:
    by_source: Dict[str, Dict] = {}

    def bucket(source):
        return by_source.setdefault(
            source, {"n": 0, "lifted": 0, "valid": 0, "executed": 0,
                     "skill_frequency": Counter(), "reject_reasons": Counter(),
                     "metrics": []})

    for rec in lifted:
        src = rec["sample"]["source"]
        b = bucket(src)
        b["n"] += 1
        if rec["module"] is None:
            b["reject_reasons"][_coarse(rec["reason"])] += 1
            continue
        b["lifted"] += 1
        if not rec.get("valid"):
            b["reject_reasons"]["invalid_taskir"] += 1
            continue
        b["valid"] += 1
        b["skill_frequency"].update(rec["metrics"]["skills"])
        b["metrics"].append(rec["metrics"])
        if rec.get("result") is not None and rec["result"].status == "completed":
            b["executed"] += 1

    out = {}
    for src, b in by_source.items():
        n = b["n"] or 1
        agg_metrics = {
            "nodes_mean": round(sum(m["nodes"] for m in b["metrics"])
                                / len(b["metrics"]), 3) if b["metrics"] else 0,
            "depth_max": max((m["depth"] for m in b["metrics"]), default=0),
            "width_max": max((m["width"] for m in b["metrics"]), default=0),
            "branch_ratio_mean": round(
                sum(m["branch_ratio"] for m in b["metrics"])
                / len(b["metrics"]), 4) if b["metrics"] else 0.0,
            "parallel_ratio_mean": round(
                sum(m["parallel_ratio"] for m in b["metrics"])
                / len(b["metrics"]), 4) if b["metrics"] else 0.0,
        }
        out[src] = {
            "samples": b["n"],
            "lifted": b["lifted"],
            "valid": b["valid"],
            "executed": b["executed"],
            "lift_coverage_pct": round(100 * b["lifted"] / n, 2),
            "valid_coverage_pct": round(100 * b["valid"] / n, 2),
            "execution_pct_of_valid": round(
                100 * b["executed"] / b["valid"], 2) if b["valid"] else 0.0,
            "skill_coverage": dict(b["skill_frequency"].most_common()),
            "reject_reasons": dict(b["reject_reasons"].most_common()),
            "taskir_graph": agg_metrics,
        }
    return out


def _coarse(reason: Optional[str]) -> str:
    """Bucket detailed lifter reasons into analyzable classes."""
    if not reason:
        return "unknown"
    r = reason.lower()
    if "loop" in r or "recursion" in r:
        return "LOOP"
    if "pattern" in r or "unsupported" in r or "expr" in r or "builtin" in r \
            or "free variable" in r:
        return "PATTERN_UNSUPPORTED"
    if "trajectory" in r or "instruction" in r:
        return "EMPTY_TRAJECTORY"
    if "sql" in r or "select" in r:
        return "SQL_UNSUPPORTED"
    return reason[:40]
