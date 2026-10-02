"""τ³ tool → semantic Skill ISA oracle (deterministic, from tool definitions only).

Maps concrete τ³ actions to canonical Skill ISA ops + effect classes.
Input: tool name, tool description, parameter schema, Skill ISA spec.
Never reads: model predictions, ground truth answers, scores.
"""
import re
from typing import Optional, Tuple

# ── Effect classification (heuristic baseline; tool_def evidence preferred) ──
READ_PAT = re.compile(r"^(get|find|search|query|list|retrieve|lookup|check|show|view|read|calculate|compare|track|locate|display|fetch)", re.I)
IRREV_PAT = re.compile(r"(cancel|refund|purchase|pay|book|reserv|send|delete|remove|grant|revoke|reboot|reset|refuel|unlock|charge|deduct|withdraw|transfer|purchase|subscribe|unsubscribe)", re.I)
REV_PAT = re.compile(r"(toggle|enable|disable|set|update|change|modify|edit|switch|turn_on|turn_off|add|create|insert|adjust|configure|provision|deactivate|activate)", re.I)


def classify_effect(action_name: str, tool_desc: str = "", source: str = "heuristic") -> Tuple[str, str]:
    """Returns (effect_class, evidence_source)."""
    name = action_name or ""
    desc = (tool_desc or "").lower()

    # Try tool description first if available
    if desc:
        if any(w in desc for w in ("cancel", "irreversible", "cannot be undone",
                                     "permanent", "send", "payment", "charge")):
            return "irreversible_world", "tool_description"
        if any(w in desc for w in ("read", "retrieve", "get", "query", "list",
                                     "search", "lookup", "check")):
            if not any(w in desc for w in ("toggle", "set", "update", "change",
                                             "create", "add", "enable", "disable")):
                return "read_only", "tool_description"

    # Fall back to name-based heuristic
    if READ_PAT.match(name):
        return "read_only", source
    if IRREV_PAT.search(name):
        return "irreversible_world", source
    if REV_PAT.search(name):
        return "reversible_state", source
    return "other", source


def tool_to_semantic_skill(action_name: str, tool_desc: str = "",
                            tool_params: Optional[dict] = None) -> dict:
    """Lower a concrete τ³ tool action to a canonical Skill ISA op.

    Returns: {"skill": str, "params": dict, "effect_class": str, "evidence": str}
    """
    name = action_name or ""
    nl = name.lower()
    desc = (tool_desc or "").lower()

    # Read-only retrieval family
    if READ_PAT.match(nl):
        if "search" in nl or "find" in nl or "lookup" in nl:
            skill = "SEARCH"
        elif "query" in nl or "database" in nl or "db" in nl:
            skill = "QUERY_DB"
        else:
            skill = "FETCH"
        eff, ev = classify_effect(name, tool_desc)
        return {"skill": skill, "params": {"query": name},
                "effect_class": eff, "evidence": ev}

    # Communication / irreversible actions
    if "send" in nl or "email" in nl or "notify" in nl or "message" in nl:
        eff, ev = classify_effect(name, tool_desc)
        return {"skill": "SEND", "params": {"channel": "system", "query": name},
                "effect_class": "irreversible_world", "evidence": ev}

    # State persistence
    if "save" in nl or "store" in nl or "persist" in nl:
        return {"skill": "SAVE", "params": {},
                "effect_class": "reversible_state", "evidence": "isa_spec"}

    # Computation
    if any(w in nl for w in ("calculate", "compute", "convert", "translate")):
        eff, ev = classify_effect(name, tool_desc)
        return {"skill": "CALCULATE" if "calc" in nl else "CONVERT",
                "params": {"expr": name},
                "effect_class": "read_only", "evidence": ev}

    # LM operations (rare in τ³ but possible)
    if any(w in nl for w in ("summarize", "summarise")):
        return {"skill": "SUMMARIZE", "params": {},
                "effect_class": "read_only", "evidence": "isa_spec"}
    if "translate" in nl and "text" in (desc or ""):
        return {"skill": "TRANSLATE", "params": {},
                "effect_class": "read_only", "evidence": "isa_spec"}
    if "classify" in nl or "categorize" in nl:
        return {"skill": "CLASSIFY", "params": {},
                "effect_class": "read_only", "evidence": "isa_spec"}

    # Default: all state-changing / domain-specific actions → EXEC_ACTION
    # This is the VALID canonical lowering for domain-specific operations
    eff, ev = classify_effect(name, tool_desc)
    return {
        "skill": "EXEC_ACTION",
        "params": {"action": name, **({"description": desc[:100]} if desc else {})},
        "effect_class": eff,
        "evidence": ev,
    }


def lower_reference_actions(ref_actions: list) -> list:
    """Lower a list of τ³ reference actions (dicts) to semantic skill list."""
    lowered = []
    for ra in ref_actions:
        if isinstance(ra, dict):
            name = ra.get("name", ra.get("action_id", ""))
            args = ra.get("arguments", {})
        else:
            name = str(ra)
            args = {}
        result = tool_to_semantic_skill(name)
        result["original_name"] = name
        result["original_args"] = args
        lowered.append(result)
    return lowered


def semantic_match(pred_skill: str, pred_params: dict,
                    ref_lowered: dict) -> bool:
    """Check if a predicted TaskIR node semantically matches a lowered reference."""
    ref_skill = ref_lowered["skill"]
    ref_params = ref_lowered["params"]

    # Same skill op = match
    if pred_skill == ref_skill:
        # For EXEC_ACTION, also check action name
        if pred_skill == "EXEC_ACTION":
            paction = str(pred_params.get("action", "")).lower()
            raction = str(ref_params.get("action", "")).lower()
            if paction and raction:
                return paction in raction or raction in paction or \
                       paction.split("_")[0] == raction.split("_")[0]
        return True

    # Equivalent retrieval family
    retrieval = {"SEARCH", "FETCH", "QUERY_DB", "LOAD"}
    if pred_skill in retrieval and ref_skill in retrieval:
        return True

    # EXEC_ACTION can match any non-read skill (it's the generic lowering)
    if pred_skill == "EXEC_ACTION" and ref_skill in (
            "EXEC_ACTION", "SEND", "SAVE"):
        paction = str(pred_params.get("action", "")).lower()
        raction = str(ref_params.get("action", "")).lower()
        if paction and raction:
            return paction in raction or raction in paction or \
                   paction.split("_")[0] == raction.split("_")[0]
        return True  # generic match if no action names to compare

    return False
