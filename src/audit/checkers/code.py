"""Code (HumanEval/MBPP) semantic audit (Task 6): Python AST of the raw
solution is the ground truth; compare its operation sequence against the
plan's skill sequence (missing / extra / wrong op)."""
from __future__ import annotations

import ast
from typing import List

from ..semantic_label import SemanticAuditResult, flag, \
    instruction_plan_consistency


def expected_ops_from_code(code: str) -> List[str]:
    """Independent AST expectation of the plan's op sequence (mirrors the
    lifter's semantics: sorted->SORT, min/max+key->ARGMIN/ARGMAX,
    min/max->MIN/MAX, sum->SUM, len->COUNT, comprehension-if->FILTER,
    comprehension-elt->TRANSFORM, a+b lists->JOIN)."""
    tree = ast.parse(code)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fns:
        return []
    returns = [n for n in ast.walk(fns[0])
               if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return []
    return _expr_ops(returns[0].value)


def _expr_ops(e: ast.expr, out: List[str] = None) -> List[str]:
    out = out if out is not None else []
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
        fn = e.func.id
        kw = {k.arg for k in e.keywords}
        if fn == "sorted":
            out.append("SORT")
        elif fn == "min" and "key" in kw:
            out.append("ARGMIN")
        elif fn == "max" and "key" in kw:
            out.append("ARGMAX")
        elif fn == "min":
            out.append("MIN")
        elif fn == "max":
            out.append("MAX")
        elif fn == "sum":
            out.append("SUM")
        elif fn == "len":
            out.append("COUNT")
        for a in e.args:
            _expr_ops(a, out)
    elif isinstance(e, ast.ListComp) and len(e.generators) == 1:
        gen = e.generators[0]
        _expr_ops(gen.iter, out)
        if gen.ifs:
            out.append("FILTER")
        if not (isinstance(e.elt, ast.Name) and e.elt.id == gen.target.id):
            out.append("TRANSFORM")
    elif isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
        _expr_ops(e.left, out)
        _expr_ops(e.right, out)
        if not out or out[-1] not in ("SORT", "FILTER", "TRANSFORM"):
            out.append("JOIN")
    return out


def audit_code(record: dict, raw_code: str,
               strict_ops: bool = True) -> SemanticAuditResult:
    """strict_ops=True (HumanEval): AST op sequence must equal plan ops.
    strict_ops=False (MBPP): ground truth is a STATEMENT sequence, not a
    single return expression — op differences are recorded as
    GROUND_TRUTH_AMBIGUOUS annotations instead of suspect flags."""
    res = SemanticAuditResult(sample_id=record.get("id", ""),
                              source=record.get("source", ""),
                              tier=record.get("quality_tier", ""))
    ops = [n["op"] for n in record["plan_json"]["program"]["nodes"]]
    try:
        expected = expected_ops_from_code(raw_code)
    except SyntaxError:
        res.status = "unverifiable"
        res.reasons.append("UNVERIFIABLE:raw code unparseable")
        return res
    res.checks["expected_ops"] = expected
    res.checks["plan_ops"] = ops
    from ..semantic_label import annotate
    if expected != ops:
        from collections import Counter
        ce, co = Counter(expected), Counter(ops)
        missing = list((ce - co).elements())
        extra = list((co - ce).elements())
        detail = (f"missing {missing[:3]} extra {extra[:3]} "
                  f"expected={expected} plan={ops}")
        if strict_ops:
            if missing:
                flag(res, "MISSING_ACTION", f"missing {missing[:3]}")
            if extra:
                flag(res, "EXTRA_ACTION", f"extra {extra[:3]}")
            if not missing and not extra and expected != ops:
                flag(res, "WRONG_ORDER", detail)
        else:
            annotate(res, "GROUND_TRUTH_AMBIGUOUS",
                     f"statement-style ground truth; {detail}")
    # def-use sanity: inputs may be the program's declared globals
    # (@items/@a/@b for code tasks), not just @task
    nodes = record["plan_json"]["program"]["nodes"]
    prog_inputs = record["plan_json"]["program"].get("inputs") or []
    defined = {g.get("name") for g in prog_inputs} | {n["id"] for n in nodes}
    for n in nodes:
        for i in n["inputs"]:
            if i not in defined:
                flag(res, "MISSING_DEPENDENCY", f"{n['id']} uses {i}")
    out_node = next((n for n in nodes if n["id"] ==
                     record["plan_json"]["program"]["output"]), None)
    if out_node is None:
        flag(res, "WRONG_OUTPUT", "output node missing")
    instruction_plan_consistency(res, record.get("instruction", ""), ops)
    if res.status != "suspect":
        res.status = "verified"
        res.checks["verdict_basis"] = "AST op sequence == plan op sequence"
    return res
