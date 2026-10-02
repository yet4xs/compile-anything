"""Canonical capability input format for Phase 5C schema-conditioned training.

Formats available capabilities into the user prompt alongside the task,
maintaining a consistent structure across all sources.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def format_capabilities(caps: Optional[List[Any]]) -> str:
    """Format capability list into canonical text block.

    Accepts:
    - List of concrete tool dicts: [{"name": ..., "description": ..., "parameters": ...}]
    - List of semantic skill strings: ["SEARCH", "SEND"]
    - None/empty: returns "<not provided>"
    """
    if not caps:
        return "<not provided>"

    parts = []
    for i, cap in enumerate(caps[:15], 1):  # cap at 15 to control prompt length
        if isinstance(cap, dict):
            name = cap.get("name", f"tool_{i}")
            desc = (cap.get("description", "") or "")[:120]
            params = cap.get("parameters", {})
            req = params.get("required", []) if isinstance(params, dict) else []
            param_str = ", ".join(req[:8]) if req else ""
            entry = f"[{i}] {name}"
            if desc:
                entry += f": {desc}"
            if param_str:
                entry += f" (params: {param_str})"
            parts.append(entry)
        elif isinstance(cap, str):
            parts.append(f"[{i}] {cap}")
        else:
            parts.append(f"[{i}] {str(cap)[:100]}")

    return "\n".join(parts)


def build_capability_prompt(instruction: str,
                            capabilities: Optional[List[Any]]) -> str:
    """Build the full user prompt with capability context."""
    cap_text = format_capabilities(capabilities)
    return f"{instruction}\n\nAvailable capabilities:\n{cap_text}"


def format_db_schema(tables: Optional[List[Dict]]) -> str:
    """Format database schema for SQL-domain sources."""
    if not tables:
        return "<not provided>"
    parts = []
    for t in tables[:10]:
        name = t.get("name", "")
        cols = t.get("columns", [])
        col_str = ", ".join(str(c) for c in cols[:10])
        parts.append(f"- {name}({col_str})" if col_str else f"- {name}")
    return "\n".join(parts) if parts else "<not provided>"
