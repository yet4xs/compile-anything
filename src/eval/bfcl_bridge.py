"""BFCL bridge — Neural Compiler TaskIR text -> BFCL function-call predictions.

Closes the evaluation loop for BFCL V4 (data/external_benchmarks/bfcl_v4):

    model output (TaskIR text)
      -> src.ir.parser.parse_text          (compiler frontend ingestion)
      -> taskir_to_bfcl_output             (this module: IR -> call dicts)
      -> predictions JSON                  ({"id", "result"} per case)
      -> run_bfcl_eval                     (scores vs possible_answer GT)

Conversion policy (inverse of src/eval/oracle_bfcl.py + src/lifter/toolmap.py):

  EXEC_ACTION   params.action IS the original function name (oracle keeps
                the exact BFCL name there); remaining params become the
                call arguments.
  SEARCH / SEND / CALCULATE / QUERY_DB / CONVERT / FETCH ...
                the original arguments ride in params under their original
                names (oracle) or merged next to the semantic scaffold keys
                domain/query/channel/expr (lifter) — they are preserved
                verbatim. The function NAME is recovered, in priority
                order, from: node.hints (func/function/func_name/tool/
                tool_name), a params name-key, meta.provenance.tools[i]
                (lifter writes raw tool names there, one per call node in
                program order), else the op itself lowercased ("search").
  GENERATE-only program (irrelevance) -> [] (model correctly declined).
  VERIFY / SELECT / MERGE / EXTRACT     -> not calls (control/bridge ops).

Output shapes (mirror bfcl_eval expectations, audited against
third_party/gorilla/.../bfcl_eval/eval_checker/):

  simple / parallel / multiple / irrelevance
        [{"func_name": {"param": value, ...}}, ...]   ([] when declined)
  multi_turn
        [per-turn lists of the same call dicts], grouped by node
        hints(turn=N) when the model emits turn markers; a GENERATE node
        carrying hints(turn=N) with no calls marks a declined (empty)
        turn. Without turn markers all calls land in a single turn
        (TaskIR chains turns via `after` edges, which carry ordering but
        not turn boundaries).

Scoring (`run_bfcl_eval`) reimplements the official AST-checker semantics
dependency-free (the vendored bfcl_eval pulls openai/anthropic/... at
import time): name match, required/optional params derived from the
possible-answer "" convention, value-in-possible-values with the official
string standardization, int->float promotion, no-order bipartite matching
for parallel categories (official substring rule: "parallel" in category),
single call for simple/multiple, empty for irrelevance (a syntax error
also passes irrelevance, as in the official decoder), non-empty for
live_relevance. Multi-turn is scored per turn by (name, args) multiset —
an approximation of the official stateful multi_turn_checker, which needs
the live execution environment. When the official ast_checker is
importable it is used instead for single-turn categories.

Known limitation: for SEARCH/SEND/... nodes whose original name was never
recorded (no action/hints/provenance), the fallback op name will not
match ground-truth names — arguments still round-trip. Note that the
REPRESENTABILITY ORACLE (src/eval/oracle_bfcl.py) itself drops
non-scalar arguments (possible-answer values are list-wrapped), so
oracle-generated TaskIR recovers names but not arguments and is not
expected to score on call categories; a trained model that emits the
arguments (the lifter training format merges original scalar args into
params under their original names) round-trips fully — the gold-model
check in _self_test asserts 100% through the whole loop.

Standalone:

    python -m src.eval.bfcl_bridge --preds preds.json \
        --bfcl data/external_benchmarks/bfcl_v4
"""
from __future__ import annotations

import argparse
import ast as pyast
import json
import pathlib
import re
import sys
import tempfile
import typing as t
from typing import Any, Dict, List, Optional, Tuple

from ..ir.parser import TaskIRSyntaxError, parse_text
from ..ir.taskir import Module, Node, to_text
from .adapters.bfcl import BFCL_DIR, CATEGORY_KIND

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Ops that never represent a tool call (control ops, IR bridges, LM-only
# answer synthesis). GENERATE is the irrelevance signal: answer without calls.
NON_CALL_OPS = {"GENERATE", "VERIFY", "SELECT", "MERGE", "EXTRACT"}

# Keys that may carry the original function name, in resolution order.
_NAME_HINT_KEYS = ("func", "function", "func_name", "tool", "tool_name")
_NAME_PARAM_KEYS = ("func", "function", "func_name", "tool", "tool_name")

# Fallback function name when the original was never recorded in the IR.
_OP_FALLBACK_NAMES = {
    "SEARCH": "search", "QUERY_DB": "query_db", "SEND": "send",
    "CALCULATE": "calculate", "CONVERT": "convert", "FETCH": "fetch",
}

# Category grouping for the official result-file layout (mirrors
# bfcl_eval.constants.category_mapping *_CATEGORY lists).
_GROUP_OF_CATEGORY = [
    (("live_",), "live"),
    (("multi_turn_",), "multi_turn"),
    (("memory", "web_search"), "agentic"),
    (("format_sensitivity",), "format_sensitivity"),
]


# ---------------------------------------------------------------- conversion

def _provenance_tools(module: Module) -> Optional[List[str]]:
    """meta.provenance.tools — lifter records raw tool names, one per call
    node in program order (chain.py). Present only if the model emitted
    the `; provenance.tools: [...]` header line."""
    prov = module.meta.get("provenance") if isinstance(module.meta, dict) else None
    if isinstance(prov, dict):
        tools = prov.get("tools")
        if isinstance(tools, list):
            return [x for x in tools if isinstance(x, str)]
    return None


def _resolve_func_name(node: Node, params: Dict[str, Any],
                        consumed: List[str],
                        provenance_tools: Optional[List[str]],
                        call_ordinal: int) -> str:
    """Priority: EXEC_ACTION action -> hints -> params name-key ->
    provenance.tools[ordinal] -> op fallback. Keys consumed as the name are
    collected into `consumed` and stripped from the emitted arguments."""
    if node.op == "EXEC_ACTION":
        action = params.get("action")
        if isinstance(action, str) and action.strip():
            consumed.append("action")
            return action.strip()
    for k in _NAME_HINT_KEYS:
        v = node.hints.get(k) if isinstance(node.hints, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in _NAME_PARAM_KEYS:
        v = params.get(k)
        if isinstance(v, str) and v.strip():
            consumed.append(k)
            return v.strip()
    if provenance_tools and 0 <= call_ordinal < len(provenance_tools):
        raw = provenance_tools[call_ordinal].strip()
        if raw:
            return raw
    return _OP_FALLBACK_NAMES.get(node.op, node.op.lower())


def _node_to_call(node: Node, call_ordinal: int,
                  provenance_tools: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """One TaskIR node -> {"func_name": {param: value}} or None (non-call).

    All params survive as arguments except the key(s) consumed as the
    function name — so SEARCH keeps its domain/query scaffold AND any
    original arguments merged by the lifter, EXEC_ACTION keeps everything
    besides `action`, and list/dict values round-trip as native JSON."""
    if node.op in NON_CALL_OPS:
        return None
    params = dict(node.params or {})
    consumed: List[str] = []
    name = _resolve_func_name(node, params, consumed, provenance_tools,
                              call_ordinal)
    args = {k: v for k, v in params.items() if k not in consumed}
    return {name: args}


def _turn_of(node: Node) -> Optional[int]:
    v = (node.hints or {}).get("turn") if isinstance(node.hints, dict) else None
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and re.match(r"^\d+$", v.strip()):
        return int(v.strip())
    return None


def taskir_to_bfcl_output(module: Module) -> List[Any]:
    """TaskIR Module -> BFCL function-call prediction.

    Returns [{"func": {"param": val}}, ...] for single-turn shapes, []
    when the model declined to call anything (GENERATE-only program), and
    [per-turn call lists] for multi-turn when nodes carry hints(turn=N).
    """
    provenance_tools = _provenance_tools(module)
    entries: List[Tuple[Optional[int], int, Dict[str, Any]]] = []  # turn, order, call
    declined_turns: set = set()
    ordinal = 0
    for node in module.program.nodes:
        call = _node_to_call(node, ordinal, provenance_tools)
        if call is None:
            # a GENERATE node with a turn marker declares an empty
            # (declined) multi-turn step — official protocol: the model
            # answers in text instead of calling
            if node.op == "GENERATE":
                t = _turn_of(node)
                if t is not None:
                    declined_turns.add(t)
            continue
        entries.append((_turn_of(node), ordinal, call))
        ordinal += 1

    turns = [e[0] for e in entries]
    if all(t is not None for t in turns) or (declined_turns and not turns):
        if entries or declined_turns:             # multi-turn grouping
            grouped: Dict[int, List[Dict[str, Any]]] = {}
            for turn, _, call in entries:
                grouped.setdefault(turn, []).append(call)
            every = sorted(set(grouped) | declined_turns)
            return [grouped.get(t, []) for t in every]
    return [call for _, _, call in entries]         # flat (single-turn)


def clean_taskir_text(text: str) -> str:
    """Strip markdown fences / chatter around the TaskIR body so a raw
    model response still parses (mirrors BFCL decode robustness)."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


def taskir_text_to_bfcl_output(text: str) -> List[Any]:
    """Parse TaskIR text (compiler frontend path) and convert to BFCL
    output. Raises TaskIRSyntaxError on malformed input."""
    return taskir_to_bfcl_output(parse_text(clean_taskir_text(text)))


# ---------------------------------------------------------------- predictions

_PRED_KEYS = ("taskir", "output", "response", "result", "prediction",
              "completion", "text")


def load_predictions(path: t.Union[str, pathlib.Path]) -> List[Dict[str, Any]]:
    """Load model predictions: JSON list of {"id", "taskir"}, JSONL of the
    same, or a JSON dict {id: taskir_text}. Field fallbacks cover the
    common output key spellings; the first non-id value key wins."""
    p = pathlib.Path(path)
    raw = p.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(l) for l in raw.splitlines() if l.strip()]

    entries: List[Dict[str, Any]] = []
    if isinstance(data, dict):                       # {id: text}
        entries = [{"id": k, "taskir": v} for k, v in data.items()]
    elif isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict) and rec.get("id") is not None:
                text = None
                for k in _PRED_KEYS:
                    if isinstance(rec.get(k), str):
                        text = rec[k]
                        break
                entries.append({"id": rec["id"], "taskir": text or "",
                                "raw": rec})
            elif isinstance(rec, (list, tuple)) and len(rec) == 2:
                entries.append({"id": rec[0], "taskir": rec[1] or ""})
    return entries


# ---------------------------------------------------------------- bfcl index

def _load_jsonl_or_doc(p: pathlib.Path) -> List[dict]:
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(l) for l in text.splitlines() if l.strip()]


def load_bfcl_index(bfcl_dir: t.Union[str, pathlib.Path]) -> Dict[str, Dict]:
    """id -> {category, kind, language, gt, functions} from the benchmark
    snapshot. Ground truth NEVER enters the model input (schema rule);
    here it is only read for scoring."""
    d = pathlib.Path(bfcl_dir)
    gt_index: Dict[str, dict] = {}
    pa_dir = d / "possible_answer"
    if pa_dir.is_dir():
        for pa in sorted(pa_dir.glob("BFCL_v4_*.json")):
            for rec in _load_jsonl_or_doc(pa):
                if isinstance(rec, dict) and rec.get("id"):
                    gt_index[rec["id"]] = rec

    index: Dict[str, Dict] = {}
    categories: set = set()
    for f in sorted(d.glob("BFCL_v4_*.json")):
        cat = f.stem.replace("BFCL_v4_", "")
        if cat == "format_sensitivity":              # id-list doc, not cases
            continue
        categories.add(cat)
        for rec in _load_jsonl_or_doc(f):
            sid = rec.get("id", "")
            if not sid:
                continue
            g = gt_index.get(sid, {})
            index[sid] = {
                "category": cat,
                "kind": CATEGORY_KIND.get(cat, cat),
                "language": ("java" if cat.endswith("_java") else
                             "javascript" if cat.endswith("_javascript")
                             else "python"),
                "gt": g.get("ground_truth"),
                "functions": rec.get("function"),
            }
    return index


def group_of_category(category: str) -> str:
    for prefixes, group in _GROUP_OF_CATEGORY:
        if any(category.startswith(p) for p in prefixes):
            return group
    return "non_live"


# ---------------------------------------------------------------- scoring
# Mirrors bfcl_eval/eval_checker/ast_eval/ast_checker.py without its
# heavy imports. Official ast_checker is used when importable.

def _standardize(s: str) -> str:
    """Official standardize_string: drop spaces + ",./-_*^", lowercase,
    single->double quotes."""
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", s).lower().replace("'", '"')


def _match_nested(value: Any, possible: Any) -> bool:
    """Value-vs-possible comparison for nested structures (one level, as
    in the official checker)."""
    if isinstance(value, str) and isinstance(possible, str):
        return _standardize(value) == _standardize(possible)
    if isinstance(value, bool) or isinstance(possible, bool):
        return value is possible
    if isinstance(value, (int, float)) and isinstance(possible, (int, float)):
        return float(value) == float(possible)
    if isinstance(value, list) and isinstance(possible, list):
        return len(value) == len(possible) and all(
            _match_nested(v, p) for v, p in zip(value, possible))
    if isinstance(value, dict) and isinstance(possible, dict):
        # possible is either {key: [vals]} (top-level param possible) or a
        # plain dict (nested inside a list possible)
        if all(isinstance(v, list) for v in possible.values()):
            for k, v in value.items():
                if k not in possible:
                    return False
                if not _possible_match(v, possible[k]):
                    return False
            for k, pv in possible.items():
                if k not in value and "" not in pv:
                    return False
            return True
        if set(value.keys()) != set(possible.keys()):
            return False
        return all(_match_nested(value[k], possible[k]) for k in value)
    return value == possible


def _possible_match(value: Any, possible_values: Any) -> bool:
    """Does `value` satisfy a possible-answer entry (list of alternatives
    or a bare value)?"""
    if not isinstance(possible_values, list):
        possible_values = [possible_values]
    return any(_match_nested(value, pv) for pv in possible_values
               if pv != "" or value == "")


def _call_matches(model_call: Dict[str, Any],
                  gt_call: Dict[str, Any]) -> bool:
    """One model call {func: {param: val}} vs one GT call
    {func: {param: [possibles]}} — official simple_function_checker."""
    if not isinstance(model_call, dict) or len(model_call) != 1:
        return False
    name = next(iter(model_call))
    args = model_call[name] or {}
    gt_name = next(iter(gt_call))
    if name != gt_name:
        return False
    gt_params = gt_call[gt_name] or {}
    if not isinstance(args, dict):
        return False
    for param, possibles in gt_params.items():
        required = isinstance(possibles, list) and "" not in possibles
        if required and param not in args:
            return False                        # missing required/optional-not-allowed
    for param, value in args.items():
        if param not in gt_params:
            return False                        # unexpected parameter
        if not _possible_match(value, gt_params[param]):
            return False
    return True


def _ast_mode(category: str) -> str:
    """Official ast_checker dispatch: 'parallel' in category (substring!)
    -> no-order; elif 'multiple' -> single call; else simple."""
    if "parallel" in category:
        return "parallel"
    if "multiple" in category:
        return "multiple"
    return "simple"


def _has_calls(calls: Any) -> bool:
    """Does the prediction contain at least one function call? Handles
    both the flat and the per-turn (multi-turn marker) shapes."""
    if not isinstance(calls, list) or not calls:
        return False
    if all(isinstance(x, list) for x in calls):
        return any(x for x in calls)
    return any(isinstance(x, dict) and x for x in calls)


def _score_ast(calls: List[Any], gt: List[Dict[str, Any]],
               category: str) -> bool:
    mode = _ast_mode(category)
    if not isinstance(calls, list):
        return False
    if calls and all(isinstance(x, list) for x in calls):
        calls = [c for turn in calls for c in turn]  # turn-marker habit
    if mode == "parallel":                      # bipartite, order-free
        if len(calls) != len(gt):
            return False
        unmatched = list(range(len(calls)))
        for g in gt:
            for i in list(unmatched):
                if _call_matches(calls[i], g):
                    unmatched.remove(i)
                    break
            else:
                return False
        return True
    if len(calls) != len(gt):                   # simple/multiple: 1 call
        return False
    return all(_call_matches(m, g) for m, g in zip(calls, gt))


def _parse_gt_call_string(s: str) -> Optional[Tuple[str, List[Any], Dict[str, Any]]]:
    """'cd(folder="document")' -> (name, positional_values, kwargs).
    Positional args (e.g. sort('final_report.pdf')) keep their values in
    order; names are not recoverable without the class schemas."""
    try:
        tree = pyast.parse(s.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None
    call = tree.body
    if not isinstance(call, pyast.Call) or not isinstance(call.func, pyast.Name):
        return None
    positional: List[Any] = []
    for a in call.args:
        try:
            positional.append(pyast.literal_eval(a))
        except (ValueError, SyntaxError):
            return None
    kwargs: Dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            return None
        try:
            kwargs[kw.arg] = pyast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return call.func.id, positional, kwargs


def _mt_call_matches(model_call: Dict[str, Any], gt_parsed) -> bool:
    name, gt_pos, gt_kw = gt_parsed
    if not isinstance(model_call, dict) or len(model_call) != 1:
        return False
    m_name = next(iter(model_call))
    args = model_call[m_name] or {}
    if m_name != name or not isinstance(args, dict):
        return False
    used = set()
    for k, v in gt_kw.items():
        if k not in args or not _match_nested(args[k], v):
            return False
        used.add(k)
    extra = [v for k, v in args.items() if k not in used]
    for v in gt_pos:                            # positional GT: match by value
        if not any(_match_nested(e, v) for e in extra):
            return False
    return True


def _score_multi_turn(turns: List[List[Dict[str, Any]]],
                      gt_turns: List[List[str]]) -> bool:
    """Per-turn (name, args) multiset comparison — approximation of the
    official stateful multi_turn_checker (which executes against the
    multi-turn environment). Empty GT turns must map to empty turns.
    A flat call list (no turn markers) is treated as a single turn."""
    if not isinstance(turns, list):
        return False
    if turns and all(isinstance(x, dict) for x in turns):
        turns = [turns]
    if len(turns) != len(gt_turns):
        return False
    for model_turn, gt_turn in zip(turns, gt_turns):
        parsed = [p for p in (_parse_gt_call_string(c)
                              for c in gt_turn or []) if p]
        if len(parsed) != len(gt_turn or []):
            return False                        # unparsable GT string
        if not isinstance(model_turn, list):
            return False
        if not parsed:
            if model_turn:
                return False                    # should have declined
            continue
        if len(model_turn) != len(parsed):
            return False
        unmatched = list(range(len(model_turn)))
        for g in parsed:
            for i in list(unmatched):
                if _mt_call_matches(model_turn[i], g):
                    unmatched.remove(i)
                    break
            else:
                return False
    return True


def _try_official_checker():
    """Import the vendored official ast_checker when its heavy deps are
    installed; None otherwise (scored by the builtin mirror instead)."""
    root = ROOT / "third_party" / "gorilla" / "berkeley-function-call-leaderboard"
    if not root.is_dir():
        return None
    sys.path.insert(0, str(root))
    try:
        from bfcl_eval.constants.enums import Language             # noqa: F401
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        return ast_checker
    except Exception:
        return None
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


_OFFICIAL_AST = None
_OFFICIAL_LANG = None


def _official_score(calls, gt, category, functions):
    """Official ast_checker path for single-turn categories, or None to
    fall back to the builtin scorer."""
    global _OFFICIAL_AST, _OFFICIAL_LANG
    if _OFFICIAL_AST is None:
        _OFFICIAL_AST = _try_official_checker() or False
        if _OFFICIAL_AST:
            from bfcl_eval.constants.enums import Language
            _OFFICIAL_LANG = Language
    if not _OFFICIAL_AST or not functions:
        return None
    if calls and all(isinstance(x, list) for x in calls):
        calls = [c for turn in calls for c in turn]  # turn-marker habit
    lang = {"java": _OFFICIAL_LANG.JAVA,
            "javascript": _OFFICIAL_LANG.JAVASCRIPT}.get(
        "java" if category.endswith("_java")
        else "javascript" if category.endswith("_javascript") else "python",
        _OFFICIAL_LANG.PYTHON)
    try:
        res = _OFFICIAL_AST(functions, calls, gt, lang, category,
                            "neural-compiler/TaskIR")
        return bool(res.get("valid"))
    except Exception:
        return None


def _score_entry(calls: List[Any], rec: Dict[str, Any]) -> Tuple[bool, str]:
    """Score one prediction against its ground truth. Returns (correct,
    error_type) — error_type "" when correct."""
    category = rec["category"]
    kind = rec["kind"]
    gt = rec["gt"]

    if kind in ("memory", "web_search"):
        return False, "unsupported:agentic_environment"
    if kind == "irrelevance":
        expect_calls = category != "live_relevance"
        return _has_calls(calls) != expect_calls, ""
    if kind == "multi_turn":
        ok = _score_multi_turn(calls, gt or [])
        return ok, "" if ok else "multi_turn:turn_mismatch"
    # single-turn AST categories
    if not isinstance(gt, list):
        return False, "missing_ground_truth"
    official = _official_score(calls, gt, category, rec["functions"])
    if official is not None:
        return official, "" if official else "ast_checker:invalid"
    ok = _score_ast(calls, gt, category)
    return ok, "" if ok else f"ast:{_ast_mode(category)}:mismatch"


def run_bfcl_eval(predictions_file: t.Union[str, pathlib.Path],
                  bfcl_data_dir: t.Union[str, pathlib.Path] = None) -> Dict[str, Any]:
    """Score a TaskIR-text predictions file against the BFCL V4 snapshot.

    predictions_file: {"id", "taskir"} entries (see load_predictions).
    Returns {per_category, group_accuracy, overall, counts, errors,
    missing_categories, notes}."""
    preds = load_predictions(predictions_file)
    index = load_bfcl_index(pathlib.Path(bfcl_data_dir) if bfcl_data_dir
                            else BFCL_DIR)

    per_cat: Dict[str, Dict[str, int]] = {}
    syntax_errors: List[str] = []
    unknown_ids: List[str] = []
    error_kinds: Dict[str, int] = {}

    for pred in preds:
        sid = str(pred["id"])
        rec = index.get(sid)
        if rec is None:
            unknown_ids.append(sid)
            continue
        cat = rec["category"]
        c = per_cat.setdefault(cat, {"correct": 0, "total": 0})
        c["total"] += 1
        try:
            calls = taskir_text_to_bfcl_output(pred.get("taskir") or "")
        except (TaskIRSyntaxError, ValueError):
            # official irrelevance semantics: undecodable output contains
            # no function call, which is exactly what irrelevance wants
            if (rec["kind"] == "irrelevance"
                    and rec["category"] != "live_relevance"):
                c["correct"] += 1
            else:
                syntax_errors.append(sid)
                error_kinds["ast_decoder:decoder_failed"] = \
                    error_kinds.get("ast_decoder:decoder_failed", 0) + 1
            continue
        ok, err = _score_entry(calls, rec)
        if ok:
            c["correct"] += 1
        elif err:
            error_kinds[err] = error_kinds.get(err, 0) + 1

    per_category: Dict[str, Dict[str, float]] = {}
    for cat, c in sorted(per_cat.items()):
        per_category[cat] = {"accuracy": c["correct"] / c["total"] if c["total"] else 0.0,
                             **c}
    group_acc: Dict[str, Dict[str, float]] = {}
    for cat, s in per_category.items():
        g = group_of_category(cat)
        gc = group_acc.setdefault(g, {"correct": 0, "total": 0})
        gc["correct"] += s["correct"]
        gc["total"] += s["total"]
    for g, gc in group_acc.items():
        gc["accuracy"] = gc["correct"] / gc["total"] if gc["total"] else 0.0

    total = sum(c["total"] for c in per_cat.values())
    correct = sum(c["correct"] for c in per_cat.values())
    present_cats = set(per_cat)
    all_cats = {r["category"] for r in index.values()}
    return {
        "predictions_file": str(predictions_file),
        "bfcl_data_dir": str(bfcl_data_dir or BFCL_DIR),
        "num_predictions": len(preds),
        "num_scored": total,
        "num_correct": correct,
        "num_invalid_syntax": len(syntax_errors),
        "per_category": per_category,
        "group_accuracy": group_acc,
        "overall": {"accuracy": correct / total if total else 0.0,
                    "correct": correct, "total": total},
        "errors": {"syntax_error_ids": syntax_errors[:50],
                   "unknown_ids": unknown_ids[:50],
                   "error_kinds": error_kinds},
        "missing_categories": sorted(all_cats - present_cats),
        "notes": [
            "multi_turn scored per-turn by (name,args) multiset — "
            "approximation of the official stateful checker",
            "memory/web_search require the agentic environment and are "
            "reported as unsupported",
            "SEARCH/SEND/... nodes without action/hints/provenance name "
            "provenance fall back to the op name and will not match GT "
            "names (arguments still round-trip)",
        ],
    }


# ------------------------------------------------------- official-format emit

def _py_literal(v: Any) -> str:
    """Python-call-syntax literal (the official OSSHandler decode_ast /
    decode_execute parse this after wrapping in [...])."""
    return repr(v)


def _call_to_python(call: Dict[str, Any]) -> str:
    name = next(iter(call))
    args = call[name] or {}
    inner = ", ".join(f"{k}={_py_literal(v)}" for k, v in args.items())
    return f"{name}({inner})"


def _result_for_official(calls: List[Any], category: str, kind: str) -> Any:
    """Render the BFCL `result` field so the official evaluator can
    consume it: single-turn -> one python-syntax string (decode_ast
    wraps it in [...] and ast-parses); multi-turn -> per-turn lists of
    step strings (decode_execute per step)."""
    def flat(lst) -> str:
        return ", ".join(_call_to_python(c) for c in lst
                         if isinstance(c, dict) and c)

    if calls and all(isinstance(x, list) for x in calls):
        if kind == "multi_turn":
            return [[flat(turn)] for turn in calls]
        calls = [c for turn in calls for c in turn]  # turn-marker habit
    if kind == "multi_turn":
        return [[flat(calls)]] if calls else []
    return flat(calls) if calls else "[]"


def emit_bfcl_results(scored_ids: List[Tuple[str, List[Any]]],
                      bfcl_dir: t.Union[str, pathlib.Path],
                      out_dir: t.Union[str, pathlib.Path],
                      model_name: str = "neural_compiler") -> List[pathlib.Path]:
    """Write predictions in the official result-file layout:
    <out_dir>/<model>/<group>/BFCL_v4_<category>_result.json with
    {"id", "result"} entries (bfcl_eval eval_runner.py contract)."""
    index = load_bfcl_index(bfcl_dir)
    files: Dict[pathlib.Path, List[Dict[str, Any]]] = {}
    for sid, calls in scored_ids:
        rec = index.get(str(sid))
        if rec is None:
            continue
        cat = rec["category"]
        group = group_of_category(cat)
        path = (pathlib.Path(out_dir) / model_name / group /
                f"BFCL_v4_{cat}_result.json")
        files.setdefault(path, []).append(
            {"id": str(sid),
             "result": _result_for_official(calls, cat, rec["kind"])})
    written = []
    for path, entries in sorted(files.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        written.append(path)
    return written


# ---------------------------------------------------------------- cli

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bridge Neural Compiler TaskIR output to BFCL V4 "
                    "function-call predictions and scores.")
    ap.add_argument("--preds", required=True,
                    help="predictions JSON: [{'id', 'taskir'}, ...] / JSONL "
                         "/ {id: text}")
    ap.add_argument("--bfcl", default=str(BFCL_DIR),
                    help="BFCL V4 snapshot directory "
                         "(default: data/external_benchmarks/bfcl_v4)")
    ap.add_argument("--scores", default=None,
                    help="where to write the scores JSON (default: "
                         "<preds stem>_scores.json)")
    ap.add_argument("--emit-bfcl", default=None, dest="emit_bfcl",
                    help="directory for official-layout BFCL result files "
                         "(default: <preds stem>_bfcl; '' disables)")
    args = ap.parse_args(argv)

    scores = run_bfcl_eval(args.preds, args.bfcl)

    preds_path = pathlib.Path(args.preds)
    scores_path = (pathlib.Path(args.scores) if args.scores
                   else preds_path.with_name(preds_path.stem + "_scores.json"))
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    emit_dir = args.emit_bfcl
    if emit_dir is None:
        emit_dir = str(preds_path.with_name(preds_path.stem + "_bfcl"))
    n_files = 0
    if emit_dir != "":
        scored = []
        for pred in load_predictions(args.preds):
            try:
                scored.append((pred["id"],
                               taskir_text_to_bfcl_output(pred.get("taskir") or "")))
            except (TaskIRSyntaxError, ValueError):
                continue
        n_files = len(emit_bfcl_results(scored, args.bfcl, emit_dir))

    print(f"predictions : {args.preds} ({scores['num_predictions']} entries)")
    print(f"bfcl data   : {args.bfcl}")
    for cat, s in scores["per_category"].items():
        print(f"  {cat:<28} {s['accuracy']:7.2%}  "
              f"({s['correct']}/{s['total']})")
    o = scores["overall"]
    print(f"  {'OVERALL':<28} {o['accuracy']:7.2%}  ({o['correct']}/{o['total']})")
    if scores["num_invalid_syntax"]:
        print(f"  invalid TaskIR syntax: {scores['num_invalid_syntax']}")
    if scores["errors"]["unknown_ids"]:
        print(f"  unknown ids (not in BFCL snapshot): "
              f"{len(scores['errors']['unknown_ids'])}")
    print(f"scores      : {scores_path}")
    print(f"bfcl results: {emit_dir if emit_dir != '' else '(disabled)'}"
          f"{f' ({n_files} files)' if n_files else ''}")
    return 0


# ---------------------------------------------------------------- self-test

def _mod(nodes: List[Node], name: str = "program", meta: Dict = None) -> Module:
    from ..ir.taskir import Program
    return Module(program=Program(name=name, description="test",
                                  inputs=[{"name": "@task", "type": "Str"}],
                                  nodes=nodes,
                                  output=nodes[-1].id if nodes else ""),
                  meta=meta or {"name": name})


def _roundtrip(module: Module) -> List[Any]:
    """to_text -> parse_text -> convert: the exact production path for a
    trained model's output."""
    return taskir_to_bfcl_output(parse_text(to_text(module)))


def _pick_alternative(alt: Any) -> Any:
    """One possible-answer alternative -> the concrete value a model would
    emit: dicts unwrap per-key possible lists (dropping optional-"" keys),
    lists recurse, scalars pass through. Test fixture helper."""
    if isinstance(alt, dict):
        out: Dict[str, Any] = {}
        for k, pv in alt.items():
            if isinstance(pv, list) and pv:
                if pv[0] == "":
                    continue                      # optional key, omit
                out[k] = _pick_alternative(pv[0])
            elif pv == "":
                continue
            else:
                out[k] = pv
        return out
    if isinstance(alt, list):
        return [_pick_alternative(x) for x in alt]
    return alt


def _gold_module(sample) -> Module:
    """Ground truth -> the TaskIR a perfectly trained compiler would emit
    (EXEC_ACTION with the exact name + unwrapped arguments, turn hints
    for multi-turn, GENERATE-only for irrelevance). Test fixture ONLY —
    it is the inverse of taskir_to_bfcl_output and never touches real
    model output."""
    from ..ir.taskir import Program
    kind = sample.metadata.get("kind")
    gt = sample.reference_actions
    nodes: List[Node] = []
    if kind == "irrelevance":
        nodes = [Node(id="%ans", op="GENERATE", inputs=["@task"],
                      params={"role": "final_answer"})]
    elif kind == "multi_turn":
        for turn, lst in enumerate(gt or []):
            if not lst:                     # declined turn -> marker
                nodes.append(Node(id=f"%g{turn}", op="GENERATE",
                                  params={"role": "final_answer"},
                                  hints={"turn": turn}))
                continue
            for j, call in enumerate(lst):
                parsed = _parse_gt_call_string(call) if isinstance(
                    call, str) else None
                if parsed is None:
                    continue
                name, positional, kwargs = parsed
                args = dict(kwargs)
                args.update({f"arg{i}": v for i, v in enumerate(positional)})
                nodes.append(Node(id=f"%c{turn}_{j}", op="EXEC_ACTION",
                                  params={"action": name, **args},
                                  hints={"turn": turn}))
    else:
        i = 0
        for g in gt or []:
            for name, poss in (g or {}).items():
                args = {k: _pick_alternative(p[0])
                        for k, p in (poss or {}).items()
                        if isinstance(p, list) and p and p[0] != ""}
                nodes.append(Node(id=f"%c{i}", op="EXEC_ACTION",
                                  inputs=["@task"],
                                  params={"action": name, **args}))
                i += 1
    if not nodes:
        nodes = [Node(id="%ans", op="GENERATE", inputs=["@task"],
                      params={"role": "final_answer"})]
    return Module(program=Program(
        name="gold", description=sample.instruction or "",
        inputs=[{"name": "@task", "type": "Str"}],
        nodes=nodes, output=nodes[-1].id), meta={"name": "gold"})


def _self_test() -> int:
    from ..ir.taskir import Program

    checks: List[Tuple[str, bool]] = []

    def check(label: str, ok: bool):
        checks.append((label, ok))

    # 1. EXEC_ACTION: action as function name, args survive
    m = _mod([Node(id="%c0", op="EXEC_ACTION", inputs=["@task"],
                   params={"action": "cd", "folder": "document"})])
    check("exec_action -> call", _roundtrip(m) == [{"cd": {"folder": "document"}}])

    # 2. SEARCH with domain/query kept + name from hints
    m = _mod([Node(id="%c0", op="SEARCH", inputs=["@task"],
                   params={"domain": "weather", "query": "Tel Aviv"},
                   hints={"func": "get_current_weather"})])
    check("search keeps domain/query + hints name",
          _roundtrip(m) == [{"get_current_weather":
                             {"domain": "weather", "query": "Tel Aviv"}}])

    # 3. SEARCH without any name source -> op fallback
    m = _mod([Node(id="%c0", op="SEARCH", inputs=["@task"],
                   params={"domain": "web", "query": "q"})])
    check("search fallback name",
          _roundtrip(m) == [{"search": {"domain": "web", "query": "q"}}])

    # 4. irrelevance: GENERATE-only -> []
    m = _mod([Node(id="%ans", op="GENERATE", inputs=["@task"],
                   params={"role": "final_answer"})])
    check("irrelevance -> []", _roundtrip(m) == [])

    # 5. parallel: independent calls, order preserved, nested values survive
    m = _mod([
        Node(id="%c0", op="EXEC_ACTION", inputs=["@task"],
             params={"action": "uber.ride", "loc": "Berkeley", "time": 600}),
        Node(id="%c1", op="EXEC_ACTION", inputs=["@task"],
             params={"action": "send_email", "to": ["a@x.com", "b@x.com"],
                     "body": {"greeting": "hi"}}),
    ])
    check("parallel two calls",
          _roundtrip(m) == [{"uber.ride": {"loc": "Berkeley", "time": 600}},
                            {"send_email": {"to": ["a@x.com", "b@x.com"],
                                            "body": {"greeting": "hi"}}}])

    # 6. multi-turn: hints(turn=N) -> per-turn lists
    m = _mod([
        Node(id="%c0", op="EXEC_ACTION",
             params={"action": "cd", "folder": "document"}, hints={"turn": 0}),
        Node(id="%c1", op="EXEC_ACTION",
             params={"action": "mkdir", "dir_name": "temp"}, hints={"turn": 0}),
        Node(id="%c2", op="EXEC_ACTION",
             params={"action": "grep", "file_name": "f.pdf",
                     "pattern": "budget"}, hints={"turn": 1}),
    ])
    check("multi_turn per-turn lists", _roundtrip(m) == [
        [{"cd": {"folder": "document"}}, {"mkdir": {"dir_name": "temp"}}],
        [{"grep": {"file_name": "f.pdf", "pattern": "budget"}}]])

    # 7. declined (empty) multi-turn step via GENERATE turn marker
    m = _mod([
        Node(id="%c0", op="EXEC_ACTION", params={"action": "ls"},
             hints={"turn": 0}),
        Node(id="%g0", op="GENERATE", params={"role": "final_answer"},
             hints={"turn": 1}),
        Node(id="%c1", op="EXEC_ACTION",
             params={"action": "cat", "file_name": "a"}, hints={"turn": 2}),
    ])
    check("declined turn marker", _roundtrip(m) ==
          [[{"ls": {}}], [], [{"cat": {"file_name": "a"}}]])

    # 8. provenance.tools name recovery (lifter header line)
    text = ("; TaskIR v0.1  module=program\n"
            "; provenance.tools: [\"get_user_info\"]\n"
            "%c0 = SEARCH(@task, user_id=7890)\n"
            "return %c0\n")
    check("provenance.tools name recovery",
          taskir_to_bfcl_output(parse_text(text)) ==
          [{"get_user_info": {"user_id": 7890}}])

    # 9. fenced / chatty model output still parses
    fenced = "```taskir\n" + to_text(_mod([Node(
        id="%c0", op="EXEC_ACTION", params={"action": "ls", "a": True})])) + "```"
    check("fenced output", taskir_text_to_bfcl_output(fenced) ==
          [{"ls": {"a": True}}])

    # 10. official-format rendering: python-call syntax strings
    calls = [{"cd": {"folder": "document"}}, {"mkdir": {"dir_name": "temp"}}]
    check("official render single-turn",
          _result_for_official(calls, "simple", "single") ==
          "cd(folder='document'), mkdir(dir_name='temp')")
    check("official render irrelevance",
          _result_for_official([], "irrelevance", "irrelevance") == "[]")
    check("official render multi_turn",
          _result_for_official([calls[:1], calls[1:]], "multi_turn_base",
                               "multi_turn") ==
          [["cd(folder='document')"], ["mkdir(dir_name='temp')"]])

    # 11. scoring end-to-end on a synthetic BFCL snapshot
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        (d / "possible_answer").mkdir()
        (d / "BFCL_v4_simple.json").write_text(json.dumps([
            {"id": "simple_0", "question": [[{"role": "user", "content": "q0"}]],
             "function": [{"name": "cd"}]},
            {"id": "simple_1", "question": [[{"role": "user", "content": "q1"}]],
             "function": [{"name": "cd"}]},
            {"id": "simple_2", "question": [[{"role": "user", "content": "q2"}]],
             "function": [{"name": "cd"}]},
        ]), encoding="utf-8")
        (d / "possible_answer" / "BFCL_v4_simple.json").write_text(
            '{"id": "simple_0", "ground_truth": '
            '[{"cd": {"folder": ["document"]}}]}\n'
            '{"id": "simple_1", "ground_truth": '
            '[{"cd": {"folder": ["elsewhere"]}}]}\n'
            '{"id": "simple_2", "ground_truth": '
            '[{"cd": {"folder": ["anywhere"]}}]}\n', encoding="utf-8")
        (d / "BFCL_v4_irrelevance.json").write_text(json.dumps([
            {"id": "irrelevance_0",
             "question": [[{"role": "user", "content": "q"}]],
             "function": [{"name": "bmi"}]},
        ]), encoding="utf-8")
        (d / "BFCL_v4_multi_turn_base.json").write_text(json.dumps([
            {"id": "multi_turn_base_0", "question": [[{"role": "user",
                                                       "content": "q"}]]}],
        ), encoding="utf-8")
        (d / "possible_answer" / "BFCL_v4_multi_turn_base.json").write_text(
            '{"id": "multi_turn_base_0", "ground_truth": '
            '[["cd(folder=\'document\')", "mkdir(dir_name=\'temp\')"], '
            '["sort(\'final_report.pdf\')"]]}\n', encoding="utf-8")

        preds = [
            {"id": "simple_0", "taskir": to_text(_mod([Node(
                id="%c0", op="EXEC_ACTION",
                params={"action": "cd", "folder": "document"})]))},
            {"id": "simple_1", "taskir": to_text(_mod([Node(
                id="%c0", op="EXEC_ACTION",
                params={"action": "cd", "folder": "wrong"})]))},
            {"id": "irrelevance_0", "taskir": to_text(_mod([Node(
                id="%ans", op="GENERATE", params={"role": "final_answer"})],
                name="x"))},
            {"id": "multi_turn_base_0", "taskir": to_text(_mod([
                Node(id="%c0", op="EXEC_ACTION",
                     params={"action": "cd", "folder": "document"},
                     hints={"turn": 0}),
                Node(id="%c1", op="EXEC_ACTION",
                     params={"action": "mkdir", "dir_name": "temp"},
                     hints={"turn": 0}),
                Node(id="%c2", op="EXEC_ACTION",
                     params={"action": "sort", "file_name": "final_report.pdf"},
                     hints={"turn": 1}),
            ]))},
            {"id": "simple_2", "taskir": "not taskir at all"},
        ]
        preds_file = d / "preds.json"
        preds_file.write_text(json.dumps(preds), encoding="utf-8")
        scores = run_bfcl_eval(preds_file, d)
        check("simple 1/3",
              (scores["per_category"]["simple"]["correct"],
               scores["per_category"]["simple"]["total"]) == (1, 3))
        check("irrelevance 1/1",
              scores["per_category"]["irrelevance"]["accuracy"] == 1.0)
        check("multi_turn positional match",
              scores["per_category"]["multi_turn_base"]["accuracy"] == 1.0)
        check("syntax error counted",
              scores["num_invalid_syntax"] == 1
              and "simple_2" in scores["errors"]["syntax_error_ids"])

        emitted = emit_bfcl_results(
            [(p["id"], taskir_text_to_bfcl_output(p["taskir"]))
             for p in preds[:4]], d, d / "out")
        rel = {str(f.relative_to(d / "out")).replace("\\", "/") for f in emitted}
        check("emit layout", rel == {
            "neural_compiler/non_live/BFCL_v4_simple_result.json",
            "neural_compiler/non_live/BFCL_v4_irrelevance_result.json",
            "neural_compiler/multi_turn/BFCL_v4_multi_turn_base_result.json"})
        mt = json.loads((d / "out" / "neural_compiler" / "multi_turn" /
                         "BFCL_v4_multi_turn_base_result.json")
                        .read_text(encoding="utf-8"))
        check("emit multi_turn shape", mt[0]["result"] ==
              [["cd(folder='document'), mkdir(dir_name='temp')"],
               ["sort(file_name='final_report.pdf')"]])

    # 12. real snapshot, gold model: GT -> TaskIR text (what a perfectly
    # trained compiler emits) -> bridge -> score must be 100% everywhere
    # (except cases whose ground truth contains an empty possible-list
    # [], which the official checker semantics make unsatisfiable)
    if (ROOT / "data" / "external_benchmarks" / "bfcl_v4").is_dir():
        gold_cats = {"simple_python", "live_simple", "live_parallel",
                     "live_parallel_multiple", "multi_turn_base",
                     "irrelevance", "live_irrelevance"}

        def _winnable(gt) -> bool:
            # unsatisfiable under official semantics: a param (or an
            # inner dict key) whose possible-alternatives list is empty —
            # omitting fails "missing", including fails "value not in []"
            for call in gt or []:
                if not isinstance(call, dict):
                    continue
                for params in call.values():
                    if not isinstance(params, dict):
                        continue
                    for poss in params.values():
                        if poss == []:
                            return False
                        for alt in (poss if isinstance(poss, list) else []):
                            if isinstance(alt, dict) and any(
                                    v == [] for v in alt.values()):
                                return False
            return True

        try:
            from .adapters import bfcl as bfcl_adapter
            from .oracle_bfcl import oracle
            samples = [s for s in bfcl_adapter.load()
                       if s.metadata.get("category") in gold_cats]
            winnable = {s.case_id for s in samples
                        if s.metadata.get("kind") in ("irrelevance",
                                                      "multi_turn")
                        or _winnable(s.reference_actions)}
            with tempfile.TemporaryDirectory() as td:
                pf = pathlib.Path(td) / "gold_preds.json"
                pf.write_text(json.dumps(
                    [{"id": s.case_id, "taskir": to_text(_gold_module(s))}
                     for s in samples]), encoding="utf-8")
                sc = run_bfcl_eval(pf, None)
                for cat in sorted(gold_cats):
                    pc = sc["per_category"].get(cat)
                    want = sum(1 for s in samples
                               if s.metadata.get("category") == cat
                               and s.case_id in winnable)
                    check(f"gold model ceiling [{cat}]",
                          pc is not None and pc["correct"] == want)
            # oracle-derived TaskIR only recovers names (its lowering
            # drops list-wrapped args) — must still run, score <= gold
            with tempfile.TemporaryDirectory() as td:
                pf = pathlib.Path(td) / "oracle_preds.json"
                pf.write_text(json.dumps(
                    [{"id": s.case_id, "taskir": to_text(oracle(s)["module"])}
                     for s in samples if oracle(s).get("module")]),
                    encoding="utf-8")
                sc = run_bfcl_eval(pf, None)
                ok = all(0.0 <= v["accuracy"] <= 1.0
                         for v in sc["per_category"].values())
                check("oracle-derived predictions run", ok)
        except Exception as e:                          # pragma: no cover
            check(f"real snapshot runs (failed: {e})", False)
    else:
        print("  [skip] real BFCL snapshot not found")

    failed = [lbl for lbl, ok in checks if not ok]
    for lbl, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {lbl}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(_self_test())
    raise SystemExit(main())
