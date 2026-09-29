"""RTL (VerilogEval) semantic audit (Task 7): the lifter is template-like
by design. Classify each op by its evidence source —
mandatory_from_ground_truth / policy_inserted / template_inserted — and
ANNOTATE (POLICY_DERIVED), not delete. Core task fidelity: the task asks
for an RTL fix/feature -> CODEGEN must be present and grounded in the
prompt."""
from __future__ import annotations

from ..semantic_label import (SemanticAuditResult, flag, annotate,
                              instruction_plan_consistency)

# ops justified by the raw task itself (prompt asks for a design fix on
# given artifacts)
MANDATORY = {"LOAD", "EXTRACT", "CODEGEN"}
# ops the compiler policy adds (lint/verification)
POLICY = {"VERIFY", "CALCULATE"}
# ops the template inserts without per-task evidence
TEMPLATE = {"SEARCH"}


def audit_rtl(record: dict, raw_prompt: str) -> SemanticAuditResult:
    res = SemanticAuditResult(sample_id=record.get("id", ""),
                              source=record.get("source", ""),
                              tier=record.get("quality_tier", ""))
    nodes = record["plan_json"]["program"]["nodes"]
    ops = [n["op"] for n in nodes]
    counts = {"mandatory_from_ground_truth": 0, "policy_inserted": 0,
              "template_inserted": 0}
    for op in ops:
        if op in MANDATORY:
            counts["mandatory_from_ground_truth"] += 1
        elif op in POLICY:
            counts["policy_inserted"] += 1
            annotate(res, "POLICY_DERIVED", op)
        elif op in TEMPLATE:
            counts["template_inserted"] += 1
            annotate(res, "TEMPLATE_BIAS", f"{op}(known_bug_db default)")
    res.checks["op_evidence"] = counts

    # core fidelity: fix/generate task without CODEGEN is wrong
    if "CODEGEN" not in ops:
        flag(res, "MISSING_ACTION", "RTL fix task without CODEGEN")
    # a SEARCH(known_bug_db) with no bug keywords in the prompt is pure
    # template — annotate only (never delete)
    instruction_plan_consistency(res, record.get("instruction", ""), ops)
    if res.status != "suspect":
        res.status = "verified"
        res.checks["verdict_basis"] = ("core ops grounded; policy/template "
                                       "ops annotated")
    return res
