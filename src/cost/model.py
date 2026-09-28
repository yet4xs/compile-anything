"""Cost aggregation and reporting for simulated TaskIR executions.

All figures are nominal (see spec/skill-isa.md §4): they compare workloads,
they are not measurements. The 70B single-shot baseline gives the reference
point for the "small executors + compilation vs one big model" story.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..isa.registry import BIG_MODEL_BASELINE, FLOPS_PER_TOKEN_70B
from ..runtime.simulator import ExecutionResult
from ..validator.validator import Report as ValidationReport


def baseline_flops() -> float:
    b = BIG_MODEL_BASELINE
    return (b["tokens_in"] + b["tokens_out"]) * FLOPS_PER_TOKEN_70B


def baseline_latency_ms() -> float:
    return BIG_MODEL_BASELINE["latency_ms"]


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
        lines.append(f"    latency: {e.latency_ms:g}ms"
                     f"   tokens: {e.tokens_in}/{e.tokens_out}"
                     f"   flops: {e.flops:g}")
        if e.value_digest:
            lines.append(f"    value: {e.value_digest}")
        if e.note:
            lines.append(f"    note: {e.note}")
    lines.append("")
    lines.append(f"Status: {res.status}   output {res.output_id} = {res.output_digest}")
    lines.append(f"Total latency (sequential): {res.seq_latency_ms:g}ms")
    lines.append(f"Critical path (parallel):   {res.critical_path_ms:g}ms")
    lines.append(f"Tokens: {res.tokens_in} in / {res.tokens_out} out")
    lines.append(f"FLOPs: {res.flops:.3g}")
    lines.append(f"Node calls: {res.node_calls} (lm {res.lm_calls}, api/db {res.api_calls})"
                 f"   retries: {res.retries}   skipped: {res.skipped}")
    lines.append(f"Reference 70B single-shot:  ~{baseline_latency_ms():g}ms, "
                 f"~{baseline_flops():.3g} FLOPs")
    return "\n".join(lines) + "\n"


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
    lines.append("| seq | node | op | attempt | status | latency ms | tokens | FLOPs | value |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for e in res.events:
        val = e.value_digest.replace("|", "\\|")
        lines.append(f"| {e.seq} | {e.node} | {e.op} | {e.attempt} | {e.status} "
                     f"| {e.latency_ms:g} | {e.tokens_in}/{e.tokens_out} "
                     f"| {e.flops:.3g} | `{val}` |")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| sequential latency | {res.seq_latency_ms:g} ms |")
    lines.append(f"| critical-path latency (unlimited parallelism) | {res.critical_path_ms:g} ms |")
    lines.append(f"| tokens in / out | {res.tokens_in} / {res.tokens_out} |")
    lines.append(f"| FLOPs | {res.flops:.3g} |")
    lines.append(f"| node calls (lm / api+db) | {res.node_calls} ({res.lm_calls} / {res.api_calls}) |")
    lines.append(f"| retries / skipped | {res.retries} / {res.skipped} |")
    lines.append(f"| 70B single-shot reference | ~{baseline_latency_ms():g} ms, ~{baseline_flops():.3g} FLOPs |")
    if baseline_flops() > 0:
        lines.append(f"| pipeline FLOPs vs 70B baseline | {res.flops / baseline_flops() * 100:.2f}% |")
    return "\n".join(lines) + "\n"
