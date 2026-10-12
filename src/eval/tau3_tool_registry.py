"""tau3 runtime tool registry — built ONLY from the official tool
definitions (AST parse of third_party/tau3-bench domains' tools.py).

Phase 8A Task 1: the agent-visible candidate pool must come from the
runtime registry (@is_tool, non-discoverable), never from reference
actions / ground truth / model predictions. Each entry carries name,
ToolType, mutates_state, description, and the typed parameter schema.

Discoverable tools (@is_discoverable_tool) exist in the runtime but are
NOT placed in the agent's system prompt; they are recorded separately
and excluded from the default candidate pool.
"""
from __future__ import annotations

import ast
import os
import re
from typing import Dict, List, Optional

DOMAINS_ROOT = os.path.join(
    "third_party", "tau3-bench", "src", "tau2", "domains")
DOMAINS = ["airline", "banking_knowledge", "retail", "telecom"]


def _annotation_str(node) -> str:
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _parse_docstring(doc: str):
    """First paragraph = description; 'Args:' section = param docs."""
    if not doc:
        return "", {}
    lines = doc.strip().splitlines()
    desc_lines = []
    for ln in lines:
        if ln.strip().lower().startswith(("args:", "returns:", "raises:")):
            break
        desc_lines.append(ln.strip())
    description = " ".join(x for x in desc_lines if x)
    params = {}
    in_args = False
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("args:"):
            in_args = True
            continue
        if s.lower().startswith(("returns:", "raises:")):
            in_args = False
            continue
        if in_args:
            m = re.match(r"^(\w+)\s*(\([^)]*\))?\s*:\s*(.+)$", s)
            if m:
                params[m.group(1)] = m.group(3)
    return description, params


def _decorator_info(dec) -> Optional[dict]:
    """Extract is_tool/is_discoverable_tool info from a decorator node."""
    name = None
    if isinstance(dec, ast.Name):
        name = dec.id
    elif isinstance(dec, ast.Attribute):
        name = dec.attr
    elif isinstance(dec, ast.Call):
        f = dec.func
        name = f.attr if isinstance(f, ast.Attribute) else (
            f.id if isinstance(f, ast.Name) else None)
    if name not in ("is_tool", "is_discoverable_tool"):
        return None
    info = {"discoverable": name == "is_discoverable_tool",
            "tool_type": "read", "mutates_state": None}
    if isinstance(dec, ast.Call):
        for kw in dec.keywords:
            if kw.arg == "mutates_state":
                v = kw.value
                if isinstance(v, ast.Constant):
                    info["mutates_state"] = bool(v.value)
        if dec.args:
            a = dec.args[0]
            txt = _annotation_str(a)          # e.g. ToolType.WRITE
            m = re.search(r"ToolType\.(\w+)", txt)
            if m:
                info["tool_type"] = m.group(1).lower()
    if info["mutates_state"] is None:
        info["mutates_state"] = info["tool_type"] == "write"
    return info


def parse_domain_tools(path: str) -> List[dict]:
    """All @is_tool / @is_discoverable_tool functions in a tools.py file."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    tools = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        infos = [d for d in (_decorator_info(x) for x in node.decorator_list)
                 if d]
        if not infos:
            continue
        info = infos[-1]
        description, param_docs = _parse_docstring(ast.get_docstring(node))
        params = []
        for a in node.args.args + node.args.kwonlyargs:
            if a.arg in ("self", "cls"):
                continue
            has_default = False
            defaults = node.args.defaults[-len(node.args.args):] \
                if node.args.defaults else []
            # positional defaults align to the tail of args.args
            idx = node.args.args.index(a) if a in node.args.args else -1
            n_pos = len(node.args.args)
            if idx >= 0 and defaults and idx >= n_pos - len(defaults):
                has_default = True
            if a in node.args.kw_defaults:
                has_default = node.args.kw_defaults[
                    node.args.kwonlyargs.index(a)] is not None
            params.append({
                "name": a.arg,
                "type": _annotation_str(a.annotation),
                "doc": param_docs.get(a.arg, ""),
                "optional": has_default,
            })
        tools.append({
            "name": node.name,
            "tool_type": info["tool_type"],
            "mutates_state": info["mutates_state"],
            "discoverable": info["discoverable"],
            "description": description,
            "params": params,
            "evidence": f"AST:{os.path.basename(path)}:{node.lineno}",
        })
    return tools


def build_registry(domains_root: str = DOMAINS_ROOT,
                   domains=None) -> Dict[str, List[dict]]:
    """Scan every module under each domain (tools.py, user_tools.py,
    environment.py, ...) so nothing callable is missed; entries carry the
    source file for side classification (agent tools.py vs user_tools.py).
    """
    domains = domains or DOMAINS
    reg = {}
    for d in domains:
        pdir = os.path.join(domains_root, d)
        tools = []
        for root, _dirs, files in os.walk(pdir):
            if any(x in root for x in ("tasks", "__pycache__", "mock",
                                       "tests")):
                continue
            for fn in sorted(files):
                if not fn.endswith(".py") or fn == "__init__.py":
                    continue
                fpath = os.path.join(root, fn)
                try:
                    parsed = parse_domain_tools(fpath)
                except SyntaxError:
                    continue
                rel = os.path.relpath(fpath, pdir).replace("\\", "/")
                for t in parsed:
                    t["source_file"] = rel
                    t["side"] = ("user" if "user_tools" in rel
                                 else "agent")
                tools.extend(parsed)
        by_name = {}
        for t in tools:
            by_name[t["name"]] = t
        reg[d] = sorted(by_name.values(), key=lambda t: t["name"])
    return reg


def runtime_pool(registry: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Agent-visible candidate pool: non-discoverable @is_tool entries on
    the agent side (user_tools.py holds USER-side device tools the agent
    can only request, never call)."""
    return {d: [t for t in ts
                if not t["discoverable"] and t.get("side") == "agent"]
            for d, ts in registry.items()}
