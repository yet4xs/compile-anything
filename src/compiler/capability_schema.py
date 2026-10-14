"""Phase 8D-0.2 Task 1: Unified Capability Schema contract.

Normalizes tool input/output schemas across Skeleton Lowering, Validator,
and Mock Runtime so that all three use the same registry format.

Canonical format per tool:
    {
        "input_schema": {"param_name": "Type", ...},     # REQUIRED params
        "optional_inputs": {"param_name": "Type", ...},  # optional
        "output_schema": {"field_name": "Type", ...},
        "effect_class": "read_only" | "state_update_reversible" | ...
    }

Rules:
- Unknown/missing schema entries are NEVER treated as "empty" (no params).
- runtime_ref producers MUST have output_schema; consumers MUST have
  input_schema for the bound param.
- The same normalized registry is passed to lower_skeleton(),
  validate(), and DataflowRuntime.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def normalize_registry(raw: Dict[str, Any]) -> Dict[str, Dict]:
    """Normalize heterogeneous schema formats into the canonical contract.

    Accepts:
    - Corpus format: {"params": [{"name": ..., "optional": ...}],
                      "schema": "props"|"explicit_empty"|"unknown",
                      "required": [...], "properties": {...}}
    - v0.2 format: {"input_schema": {...}, "output_schema": {...},
                     "effect_class": ...}
    - Mixed with both

    Returns canonical dict mapping tool_id → normalized entry.
    Tools with genuinely unknown schemas get schema_status="unknown"
    and are REJECTED in strict mode.
    """
    out = {}
    for tool_id, entry in raw.items():
        if not isinstance(entry, dict):
            out[tool_id] = {"schema_status": "unknown",
                            "input_schema": None, "output_schema": None,
                            "optional_inputs": {}, "effect_class": None}
            continue

        # input_schema: from v0.2 format or derived from corpus format
        input_schema = entry.get("input_schema")
        if input_schema is None:
            # corpus format: derive from required + properties
            req = entry.get("required") or []
            props = entry.get("properties") or {}
            schema_kind = entry.get("schema", entry.get("schema_status", ""))
            if schema_kind == "props" or (props and req):
                input_schema = {k: props.get(k, "Any") for k in req}
            elif entry.get("params"):
                # lowering format: [{"name": ..., "optional": False}]
                input_schema = {}
                for p in entry["params"]:
                    if not p.get("optional"):
                        input_schema[p["name"]] = "Any"
            else:
                input_schema = None  # unknown

        # output_schema: from v0.2 format or hints
        output_schema = entry.get("output_schema")
        if output_schema is None:
            output_schema = entry.get("output_fields")
            if not isinstance(output_schema, dict):
                output_schema = None

        # optional inputs
        optional = entry.get("optional_inputs")
        if optional is None and entry.get("params"):
            optional = {p["name"]: "Any" for p in entry["params"]
                        if p.get("optional")}
        if not isinstance(optional, dict):
            optional = {}

        # schema_status
        if input_schema is not None and output_schema is not None:
            status = "complete"
        elif input_schema is not None or output_schema is not None:
            status = "partial"
        else:
            status = "unknown"

        out[tool_id] = {
            "schema_status": status,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "optional_inputs": optional,
            "effect_class": entry.get("effect_class"),
        }
    return out


def validate_runtime_ref(registry: Dict[str, Dict],
                         producer_tool: str, field_name: str) -> Optional[str]:
    """Check that a runtime_ref's producer has the field in its output
    schema. Returns field type on success, None on failure."""
    entry = registry.get(producer_tool)
    if not entry or entry.get("schema_status") == "unknown":
        return None
    os = entry.get("output_schema")
    if not isinstance(os, dict) or field_name not in os:
        return None
    return os[field_name]


def validate_param_binding(registry: Dict[str, Dict],
                           consumer_tool: str,
                           param_name: str,
                           ref_type: Optional[str]) -> Optional[str]:
    """Check that a param_ref's type is compatible with the consumer's
    input schema. Returns error message or None (ok)."""
    from src.ir.types import is_compatible
    entry = registry.get(consumer_tool)
    if not entry:
        return f"unknown tool {consumer_tool!r}"
    ischema = entry.get("input_schema")
    if not isinstance(ischema, dict):
        return f"tool {consumer_tool!r} has unknown input schema"
    if param_name not in ischema:
        opts = entry.get("optional_inputs") or {}
        if param_name not in opts:
            return (f"param {param_name!r} not in {consumer_tool} "
                    f"input schema {sorted(ischema.keys())}")
        expected = opts[param_name]
    else:
        expected = ischema[param_name]
    if ref_type and ref_type != "Any":
        if not is_compatible(expected, ref_type):
            return (f"param {param_name} type {ref_type} incompatible "
                    f"with {consumer_tool} expected {expected}")
    return None
