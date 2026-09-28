"""Dataset statistics for TaskIR corpora — workload characterization.

Recorded from day one (DAC submission will need workload analysis):
  - number of nodes
  - graph depth  (longest dependency chain, data + after edges)
  - graph width  (max nodes sharing the same depth level ~= parallelism)
  - skill frequency
  - branch ratio ((guarded nodes + SELECTs) / nodes)
  - parallel ratio (1 - critical_path_nodes / nodes, unit-cost graph)
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Dict, List

from .ir.taskir import Module


def program_metrics(mod: Module) -> Dict:
    nodes = mod.program.nodes
    by_id = {n.id: n for n in nodes}

    def preds(n):
        refs = list(n.inputs) + list(n.after)
        if n.guard is not None:
            refs.append(n.guard.cond)
        return [r for r in refs if r in by_id]

    depth: Dict[str, int] = {}

    def depth_of(nid: str) -> int:
        if nid in depth:
            return depth[nid]
        p = preds(by_id[nid])
        depth[nid] = 1 + max((depth_of(x) for x in p), default=0)
        return depth[nid]

    max_depth = 0
    for n in nodes:
        max_depth = max(max_depth, depth_of(n.id))

    levels: Counter = Counter(depth[n.id] for n in nodes)
    width = max(levels.values(), default=0)

    # unit-cost critical path in nodes (longest chain)
    def chain(nid: str) -> int:
        p = preds(by_id[nid])
        return 1 + max((chain(x) for x in p), default=0)

    cp_nodes = chain(mod.program.output) if mod.program.output in by_id else 0

    skills = Counter(n.op for n in nodes)
    guarded = sum(1 for n in nodes if n.guard is not None)
    selects = skills.get("SELECT", 0)
    retries = sum(1 for n in nodes if n.retry is not None)

    n = len(nodes)
    return {
        "name": mod.program.name,
        "nodes": n,
        "depth": max_depth,
        "width": width,
        "skills": dict(skills),
        "guarded": guarded,
        "selects": selects,
        "retries": retries,
        "branch_ratio": round((guarded + selects) / n, 4) if n else 0.0,
        "parallel_ratio": round(1 - cp_nodes / n, 4) if n else 0.0,
        "has_parallelism": width > 1,
    }


def _dist(vals: List[float]) -> Dict:
    if not vals:
        return {"count": 0}
    s = sorted(vals)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    return {
        "count": len(vals),
        "mean": round(statistics.mean(vals), 3),
        "median": round(statistics.median(vals), 3),
        "p90": p90,
        "min": min(vals),
        "max": max(vals),
    }


def aggregate(metrics: List[Dict], subset: str = "") -> Dict:
    n_total = sum(m["nodes"] for m in metrics)
    skill_freq = Counter()
    for m in metrics:
        skill_freq.update(m["skills"])
    return {
        "subset": subset,
        "programs": len(metrics),
        "nodes": _dist([m["nodes"] for m in metrics]),
        "depth": _dist([m["depth"] for m in metrics]),
        "width": _dist([m["width"] for m in metrics]),
        "skill_frequency": dict(skill_freq.most_common()),
        "branch_ratio": _dist([m["branch_ratio"] for m in metrics]),
        "parallel_ratio": _dist([m["parallel_ratio"] for m in metrics]),
        "programs_with_parallelism": round(
            sum(1 for m in metrics if m["has_parallelism"]) / len(metrics), 4)
        if metrics else 0.0,
        "programs_with_retry": round(
            sum(1 for m in metrics if m["retries"] > 0) / len(metrics), 4)
        if metrics else 0.0,
        "total_nodes": n_total,
    }


def markdown_summary(agg: Dict) -> str:
    lines = [f"### subset: {agg['subset'] or 'all'}", ""]
    lines.append(f"- programs: **{agg['programs']}**  (total nodes: {agg['total_nodes']})")
    for key in ("nodes", "depth", "width", "branch_ratio", "parallel_ratio"):
        d = agg[key]
        if d.get("count"):
            lines.append(f"- {key}: mean {d['mean']}, median {d['median']}, "
                         f"p90 {d['p90']}, min {d['min']}, max {d['max']}")
    lines.append(f"- programs with width>1: {agg['programs_with_parallelism'] * 100:.1f}%")
    lines.append(f"- programs with retry:   {agg['programs_with_retry'] * 100:.1f}%")
    lines.append("")
    lines.append("| skill | count |")
    lines.append("|---|---|")
    for k, v in agg["skill_frequency"].items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"
