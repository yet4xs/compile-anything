"""Cost aggregation and reporting for simulated TaskIR executions.

All figures are nominal (see spec/skill-isa.md §4): they compare workloads,
they are not measurements. The 70B single-shot baseline gives the reference
point for the "small executors + compilation vs one big model" story.
"""
from __future__ import annotations

from typing import Optional

from ..isa.registry import BIG_MODEL_BASELINE, FLOPS_PER_TOKEN_70B
from ..runtime import executors as exec_mod
from ..runtime.simulator import ExecutionResult
from ..validator.validator import Report as ValidationReport


def baseline_flops() -> float:
    b = BIG_MODEL_BASELINE
    return (b["tokens_in"] + b["tokens_out"]) * FLOPS_PER_TOKEN_70B


def _executor_label(op: str) -> str:
    spec = exec_mod.spec_of(op)
    if spec is None:
        return "unknown"
    bound = spec.executors[0] if spec.executors else f"{spec.resource_class}:mock"
    return f"{bound} [{spec.resource_class}]"


def console_report(res: ExecutionResult, vrep: Optional[ValidationReport] = None) -> str:
    lines = []
    lines.append(f"TaskIR valid: {vrep.valid if vrep else 'n/a'}")
    if vrep and vrep.warnings:
        for w in vrep.warnings:
            lines.append(f"  warning [{w.code}] {w.node or ''}: {w.msg}")
    lines.append("")
    lines.append("Execution:")
    for e in res.events:
        lines.append(
            f"  {e.node} {e.op}"
            f"{f' (attempt {e.attempt})' if e.attempt > 1 else ''}"
            f"  [{e.status}]"
        )
        lines.append(f"    executor: {_executor_label(e.op)}")
        lines.append(f"    latency: {e.latency_ms:g}ms"
                     f"   tokens: {e.tokens_in}/{e.tokens_out}"
                     f"   flops: {e.flops:g}"
                     f"   energy: {e.energy_j:g}J   mem: {e.memory_mb:g}MB")
        if e.value_digest:
            lines.append(f"    value: {e.value_digest}")
        if e.note:
            lines.append(f"    note: {e.note}")
    cp = _critical_path_ops(res)
    lines.append("")
    lines.append(f"Status: {res.status}   output {res.output_id} = {res.output_digest}")
    lines.append(f"Total latency (sequential): {res.seq_latency_ms:g}ms")
    lines.append(f"Critical path (parallel):   {res.critical_path_ms:g}ms"
                 + (f"   [{' -> '.join(cp)}]" if cp else ""))
    lines.append(f"Tokens: {res.tokens_in} in / {res.tokens_out} out")
    lines.append(f"FLOPs: {res.flops:.3g}   Energy: {res.energy_j:g}J"
                 f"   Peak memory: {res.peak_memory_mb:g}MB (single node, "
                 f"no overlap modeled)")
    lines.append(f"Node calls: {res.node_calls} (lm {res.lm_calls}, api/db {res.api_calls})"
                 f"   retries: {res.retries}   skipped: {res.skipped}")
    lines.append(f"Reference 70B single-shot:  ~{BIG_MODEL_BASELINE['latency_ms']:g}ms, "
                 f"~{baseline_flops():.3g} FLOPs, ~{BIG_MODEL_BASELINE['energy_j']:g}J, "
                 f"~{BIG_MODEL_BASELINE['memory_mb']:g}MB")
    return "\n".join(lines) + "\n"


def _critical_path_ops(res: ExecutionResult, limit: int = 8) -> list:
    """Op names along the critical path (for the report header line)."""
    nodes = {n.id: n for n in res.module.program.nodes}
    best = {}

    def path_of(nid):
        if nid in best:
            return best[nid]
        node = nodes.get(nid)
        if node is None:
            return (0.0, [])
        lat = next((e.latency_ms for e in reversed(res.events)
                    if e.node == nid and e.status != "skipped"), 0.0)
        preds = [path_of(r) for r in list(node.inputs) + list(node.after)]
        if node.guard is not None:
            preds.append(path_of(node.guard.cond))
        if preds:
            w, chain = max(preds, key=lambda p: p[0])
            best[nid] = (lat + w, chain + [node.op])
        else:
            best[nid] = (lat, [node.op])
        return best[nid]

    out = res.module.program.output
    if out not in nodes:
        return []
    _, chain = path_of(out)
    return chain[:limit] if len(chain) <= limit else chain[:limit] + ["..."]


def markdown_report(res: ExecutionResult, vrep: Optional[ValidationReport] = None,
                    title: str = "Cost Report") -> str:
    p = res.module.program
    lines = [f"# {title}", ""]
    lines.append(f"- program: `{p.name}`")
    lines.append(f"- task: \"{p.description}\"")
    lines.append(f"- validation: {'**valid**' if (vrep is None or vrep.valid) else '**INVALID**'}"
                 + (f" ({len(vrep.warnings)} warnings)" if vrep and vrep.warnings else ""))
    lines.append(f"- status: {res.status}")
    lines.append("")
    lines.append("## Execution trace")
    lines.append("")
    lines.append("| seq | node | op | executor | attempt | status | latency ms | tokens | FLOPs | energy J | mem MB | value |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in res.events:
        val = e.value_digest.replace("|", "\\|")
        lines.append(f"| {e.seq} | {e.node} | {e.op} | `{_executor_label(e.op)}` "
                     f"| {e.attempt} | {e.status} "
                     f"| {e.latency_ms:g} | {e.tokens_in}/{e.tokens_out} "
                     f"| {e.flops:.3g} | {e.energy_j:g} | {e.memory_mb:g} | `{val}` |")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| sequential latency | {res.seq_latency_ms:g} ms |")
    lines.append(f"| critical-path latency (unlimited parallelism) | {res.critical_path_ms:g} ms |")
    lines.append(f"| tokens in / out | {res.tokens_in} / {res.tokens_out} |")
    lines.append(f"| FLOPs | {res.flops:.3g} |")
    lines.append(f"| energy | {res.energy_j:g} J |")
    lines.append(f"| peak memory (single node) | {res.peak_memory_mb:g} MB |")
    lines.append(f"| node calls (lm / api+db) | {res.node_calls} ({res.lm_calls} / {res.api_calls}) |")
    lines.append(f"| retries / skipped | {res.retries} / {res.skipped} |")
    b = BIG_MODEL_BASELINE
    lines.append(f"| 70B single-shot reference | ~{b['latency_ms']:g} ms, ~{baseline_flops():.3g} FLOPs, "
                 f"~{b['energy_j']:g} J, ~{b['memory_mb']:g} MB |")
    if baseline_flops() > 0:
        lines.append(f"| pipeline FLOPs vs 70B baseline | {res.flops / baseline_flops() * 100:.2f}% |")
        lines.append(f"| pipeline energy vs 70B baseline | {res.energy_j / b['energy_j'] * 100:.2f}% |")
    return "\n".join(lines) + "\n"
