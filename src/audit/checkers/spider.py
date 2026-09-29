"""Spider SQL semantic audit (Task 5): the conservative lowering keeps the
full SQL inside QUERY_DB — verify the payload survived intact (no
truncation, WHERE/LIMIT preserved) and the EXTRACT decomposition flag is
consistent."""
from __future__ import annotations

from typing import Dict

from ..semantic_label import (SemanticAuditResult, flag,
                              instruction_plan_consistency, _norm)


def audit_spider(record: dict, raw_sql: str) -> SemanticAuditResult:
    res = SemanticAuditResult(sample_id=record.get("id", ""),
                              source=record.get("source", ""),
                              tier=record.get("quality_tier", ""))
    nodes = record["plan_json"]["program"]["nodes"]
    ops = [n["op"] for n in nodes]
    qnode = next((n for n in nodes if n["op"] == "QUERY_DB"), None)

    if qnode is None:
        flag(res, "MISSING_ACTION", "no QUERY_DB node")
        return res

    stored = qnode.get("params", {}).get("query", "")
    raw_norm, stored_norm = _norm(raw_sql).rstrip(";"), _norm(stored).rstrip(";")
    preserved = raw_norm == stored_norm
    res.checks["sql_payload_preserved"] = preserved
    if not preserved:
        flag(res, "ARGUMENT_LOSS",
             f"sql altered: raw={raw_norm[:60]!r} stored={stored_norm[:60]!r}")

    # keyword features must survive
    for feat in ("where", "order by", "limit", "group by", "having", "join"):
        if feat in raw_norm and feat not in stored_norm:
            flag(res, "ARGUMENT_LOSS", f"clause lost: {feat}")

    # decomposition consistency: EXTRACT present iff plain-column select
    has_extract = "EXTRACT" in ops
    if has_extract and not qnode.get("params", {}).get("table"):
        flag(res, "WRONG_SKILL", "EXTRACT without table context")

    # keyword constraints live INSIDE the SQL payload. The plan is a
    # byte-faithful container of the benchmark's own SQL, so instruction-
    # keyword gaps reflect the BENCHMARK's phrasing (e.g. "average number
    # of employees" -> SQL avg(num_employees), "email" is a column name),
    # not label errors: annotate GROUND_TRUTH_AMBIGUOUS (same policy as
    # trajectory sources). Suspect is reserved for payload damage below.
    instruction_plan_consistency(res, record.get("instruction", ""), ops,
                                 mode="annotate", semantic_payload=stored)
    if res.status != "suspect":
        res.status = "verified"
        res.checks["verdict_basis"] = "SQL payload byte-faithful in QUERY_DB"
    return res
