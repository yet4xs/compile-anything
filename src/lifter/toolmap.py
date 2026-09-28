"""Tool-to-semantic-skill mapping (first lowering layer).

Concrete tool names from agent/tool-use datasets (e.g. xLAM) are matched
against domain patterns and lowered to Skill ISA instructions. Raw tool
names NEVER enter the TaskIR semantic part — the lifter stores them in
meta.provenance for auditing instead.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

Rule = Tuple[str, str, object, str]   # (regex, skill, param maker, mapping kind)


def _q(*parts: Any) -> str:
    return " ".join(str(p) for p in parts if p)


# mapping_kind:
#   exact     — tool name identifies a specific semantic domain/verb
#   heuristic — verb-level match only (generic search / send / calc ...)
#   fallback  — nothing matched (only via map_tool's default branch)
RULES: List[Rule] = [
    (r"flight", "SEARCH", lambda a: {"domain": "flight",
        "query": _q(a.get("origin"), a.get("destination"), a.get("date"),
                    a.get("airline"))}, "exact"),
    (r"hotel|lodging|accommodation", "SEARCH", lambda a: {"domain": "hotel",
        "query": _q(a.get("location"), a.get("check_in"), a.get("check_out"))},
     "exact"),
    (r"weather|forecast", "SEARCH", lambda a: {"domain": "weather",
        "query": _q(a.get("city"), a.get("location"), a.get("date"))}, "exact"),
    (r"news|article", "SEARCH", lambda a: {"domain": "news",
        "query": _q(a.get("query"), a.get("topic"), a.get("keyword"))}, "exact"),
    (r"map|direction|route|navigat", "SEARCH", lambda a: {"domain": "maps",
        "query": _q(a.get("origin"), a.get("destination"))}, "exact"),
    (r"restaurant|food|recipe", "SEARCH", lambda a: {"domain": "product",
        "query": _q(a.get("cuisine"), a.get("location"), a.get("query"))},
     "exact"),
    (r"stock|market|ticker|quote", "QUERY_DB", lambda a: {"table": "market",
        "query": _q(a.get("symbol"), a.get("ticker"), a.get("market"))},
     "exact"),
    (r"currency|convert.*money|exchange.*rate", "CONVERT",
     lambda a: {"from": a.get("from_currency", a.get("base", "USD")),
                "to": a.get("to_currency", a.get("target", "EUR")),
                "amount": a.get("amount", 1)}, "exact"),
    (r"calendar|event|meeting|appointment|schedule", "EXEC_ACTION",
     lambda a: {"action": "create_event",
                "query": _q(a.get("title"), a.get("date"), a.get("time"))},
     "exact"),
    (r"book|reserv|order|purchase|buy|checkout", "EXEC_ACTION",
     lambda a: {"action": "book", "query": _q(a.get("item"), a.get("id"))},
     "exact"),
    (r"translat", "TRANSLATE", lambda a: {"lang": a.get("language",
        a.get("target_lang", "en"))}, "exact"),
    (r"summar", "SUMMARIZE", lambda a: {}, "exact"),
    (r"entit|ner", "EXTRACT_ENTITIES", lambda a: {}, "exact"),
    (r"classif|categor|label", "CLASSIFY", lambda a: {}, "exact"),
    (r"code|program|script", "CODEGEN", lambda a: {}, "exact"),
    (r"search|find|lookup", "SEARCH", lambda a: {"domain": "web",
        "query": _q(a.get("query"), a.get("keyword"), a.get("q"))}, "heuristic"),
    (r"email|mail", "SEND", lambda a: {"channel": "email",
        "query": _q(a.get("to"), a.get("subject"))}, "heuristic"),
    (r"sms|text|message|notify", "SEND", lambda a: {"channel": "sms",
        "query": _q(a.get("to"))}, "heuristic"),
    (r"calculat|math|comput|evaluate|arithmetic", "CALCULATE",
     lambda a: {"expr": a.get("expression", a.get("expr", ""))}, "heuristic"),
    (r"database|sql|db|repo.*query", "QUERY_DB", lambda a: {"table": "generic",
        "query": _q(a.get("query"), a.get("sql"))}, "heuristic"),
    (r"fetch|download|get_|http|url", "FETCH", lambda a: {"url": a.get("url", "")},
     "heuristic"),
]

# domains for which SEARCH gets an explicit output type (List[<Domain>])
_TYPED_DOMAINS = {"flight", "hotel", "weather", "news", "maps", "product", "web"}


CONFIDENCE = {"exact": 0.95, "heuristic": 0.70, "fallback": 0.30}


def map_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Lower one concrete tool call to a semantic skill invocation.

    Returns {skill, params, mapping_kind, matched_rule, confidence}.
    mapping_kind: exact (domain-identifying rule) | heuristic (verb-level
    rule) | fallback (unknown tool -> generic EXEC_ACTION). The provenance
    fields must NEVER enter the TaskIR semantic body — they ride in
    meta.provenance only."""
    name_l = (tool_name or "").lower()
    args = args or {}
    for pattern, skill, mk, kind in RULES:
        if re.search(pattern, name_l):
            params = {k: v for k, v in (mk(args) or {}).items()
                      if v not in (None, "", [])}
            return {"skill": skill, "params": params,
                    "mapping_kind": kind, "matched_rule": pattern,
                    "confidence": CONFIDENCE[kind]}
    # fallback: generic action with a sanitized verb phrase from the tool name
    # (skip a leading vendor/service prefix: "SomeVendor.do_thing" -> "do thing")
    segs = [s for s in re.split(r"[.:_\-\s]+", name_l.strip()) if s]
    verb = " ".join(segs[1:]) or (segs[0] if segs else "call")
    return {"skill": "EXEC_ACTION", "params": {"action": verb,
                                               "query": _q(*args.values())},
            "mapping_kind": "fallback", "matched_rule": None,
            "confidence": CONFIDENCE["fallback"]}


def search_output_type(skill: str, params: Dict[str, Any]) -> str | None:
    """Explicit output type for SEARCH nodes (List[<Domain>]) or None."""
    if skill == "SEARCH":
        domain = str(params.get("domain", "web"))
        if domain in _TYPED_DOMAINS or domain.isalpha():
            return f"List[{domain.capitalize()}]"
    return None
