"""Semantic label audit — is the TaskIR label faithful to the ORIGINAL
task, not merely legal (parser/validator/runtime already covered).

Principle: deterministic evidence from raw ground truth decides; keyword
detectors can only RAISE suspicion, never confirm error; suspect samples
are never auto-deleted or rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Task 12 — unified error taxonomy
TAXONOMY = [
    "MISSING_ACTION", "EXTRA_ACTION", "WRONG_SKILL", "ARGUMENT_LOSS",
    "WRONG_ORDER", "FALSE_DEPENDENCY", "MISSING_DEPENDENCY", "WRONG_OUTPUT",
    "MISSING_CONSTRAINT", "POLICY_DERIVED", "TEMPLATE_BIAS",
    "GROUND_TRUTH_AMBIGUOUS", "UNVERIFIABLE",
]


@dataclass
class SemanticAuditResult:
    sample_id: str
    source: str
    tier: str
    status: str = "unverifiable"     # verified | suspect | unverifiable
    checks: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"sample_id": self.sample_id, "source": self.source,
                "tier": self.tier, "status": self.status,
                "checks": self.checks, "reasons": self.reasons,
                "evidence": self.evidence}


def flag(res: SemanticAuditResult, reason: str, detail: str = "") -> None:
    """Record a suspect reason (taxonomy-tagged). Always escalates to
    suspect — the checker sets 'verified' only when NO flag fired."""
    res.reasons.append(f"{reason}" + (f":{detail}" if detail else ""))
    res.status = "suspect"


def annotate(res: SemanticAuditResult, reason: str, detail: str = "") -> None:
    """Record a NON-suspect annotation (e.g. POLICY_DERIVED evidence)."""
    res.reasons.append(f"{reason}" + (f":{detail}" if detail else ""))


# ---------------------------------------------------------------- Task 4
# instruction <-> plan consistency: keyword detectors, SUSPECT-only.
# A detector fires only when the keyword is present AND no semantically
# equivalent skill appears in the plan (equiv sets below).
KEYWORD_SKILLS = [
    # (regex on instruction, required-skill equivalence set, reason)
    (r"\b(cheapest|lowest price|lowest cost|minimum price|least expensive)\b",
     {"ARGMIN", "MIN"}, "MISSING_CONSTRAINT:optimization(min)"),
    (r"\b(most expensive|highest price|maximum|largest|highest)\b",
     {"ARGMAX", "MAX"}, "MISSING_CONSTRAINT:optimization(max)"),
    (r"\b(send|email|e-?mail)\b", {"SEND"}, "MISSING_CONSTRAINT:send"),
    (r"\btranslate\b", {"TRANSLATE"}, "MISSING_CONSTRAINT:translate"),
    (r"\bsummar(ize|y|ise)\b", {"SUMMARIZE"},
     "MISSING_CONSTRAINT:summarize"),
    (r"\bsort(ed|ing)?\b", {"SORT"}, "MISSING_CONSTRAINT:sort"),
    (r"\b(how many|count the|number of)\b", {"COUNT", "SUM"},
     "MISSING_CONSTRAINT:count"),
    (r"\b(average|mean of|avg)\b", {"AVG", "CALCULATE"},
     "MISSING_CONSTRAINT:average"),
]


def instruction_plan_consistency(res: SemanticAuditResult, instruction: str,
                                 ops: List[str],
                                 mode: str = "suspect",
                                 semantic_payload: str = "") -> None:
    """Keyword/constraint consistency — SUSPECT-only detector.

    mode="suspect": keyword hit without an equivalent skill flags suspect
    (used where no richer ground truth exists: code/rtl).
    mode="annotate": keyword hit is recorded as GROUND_TRUTH_AMBIGUOUS
    instead — used for trajectory sources where the LABEL's contract is
    faithfulness to the ground-truth calls (the gap then sits between
    instruction and the benchmark's own reference, not in our label).
    semantic_payload: extra text that may carry the semantics (e.g. the
    SQL inside QUERY_DB — "count(" there satisfies the count constraint).
    """
    import re
    text = (instruction or "").lower()
    payload = (semantic_payload or "").lower()
    ops_set = set(ops)
    res.checks["instruction_consistency"] = f"checked({mode})"
    for pattern, equiv, reason in KEYWORD_SKILLS:
        if not re.search(pattern, text):
            continue
        if ops_set & equiv:
            continue
        # payload-based satisfaction (e.g. SQL features inside QUERY_DB)
        if payload:
            feat = {"MISSING_CONSTRAINT:optimization(min)": ("min(", "order by"),
                    "MISSING_CONSTRAINT:optimization(max)": ("max(",),
                    "MISSING_CONSTRAINT:count": ("count(",),
                    "MISSING_CONSTRAINT:average": ("avg(", "average("),
                    "MISSING_CONSTRAINT:sort": ("order by",),
                    "MISSING_CONSTRAINT:translate": ("translate",),
                    "MISSING_CONSTRAINT:summarize": ("summar",)}
            if any(f in payload for f in feat.get(reason, ())):
                continue
        if mode == "annotate":
            annotate(res, "GROUND_TRUTH_AMBIGUOUS",
                     f"{reason} (instruction asks, reference trajectory "
                     f"does not provide it)")
        else:
            flag(res, reason,
                 f"keyword matched but none of {sorted(equiv)} in plan")


def _norm(s: Any) -> str:
    return " ".join(str(s).lower().split())


def value_preserved(value: Any, params_blob: str) -> bool:
    """Is a raw argument value (normalized) present in the node params?"""
    return _norm(value) in _norm(params_blob)
