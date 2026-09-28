"""TaskIR text-form parser (v0.1).

Parses the format emitted by `src/ir/taskir.py:to_text` back into a Module.
This is the compiler frontend's ingestion path: a (future) Qwen compiler
emits TaskIR text, this parser + the validator gate it into the pipeline.

Grammar (see spec/taskir-spec.md §5.2):

    ; comment / header lines
    inputs: @name: Type, ...
    %id = OP(ref, ..., key=json_literal, ...)  [-> Type] [after(...)]
                                                  [guard(%c == true)]
                                                  [retry(on=%v, max=3)]
                                                  [hints(k=v, ...)]
    return %id

Malformed input raises TaskIRSyntaxError (with line number).
"""
from __future__ import annotations

import ast as pyast
import json
import re
from typing import Any, Dict, List

from .taskir import Guard, Module, Node, Program, Retry

_NODE_RE = re.compile(r"^(%[A-Za-z0-9_.\-]+)\s*=\s*([A-Za-z_]\w*)\s*\(")
_ANN_RE = re.compile(r"(guard|retry|after|hints)\(([^()]*)\)")
_TYPE_RE = re.compile(r"->\s*([A-Za-z_]\w*(?:\[[^\]]*\])?)")


class TaskIRSyntaxError(ValueError):
    pass


def _split_top(s: str) -> List[str]:
    """Split on top-level commas (quote- and bracket-aware)."""
    parts, depth, cur, i, n = [], 0, "", 0, len(s)
    in_str = None
    while i < n:
        ch = s[i]
        if in_str:
            cur += ch
            if ch == "\\" and i + 1 < n:
                cur += s[i + 1]
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
            cur += ch
        elif ch in "[{(":
            depth += 1
            cur += ch
        elif ch in "]})":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
        i += 1
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _match_paren(s: str, start: int) -> int:
    """Index of the ')' matching the '(' at s[start] (quote-aware). -1 if none."""
    depth, i, n = 0, start, len(s)
    in_str = None
    while i < n:
        ch = s[i]
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _parse_value(tok: str) -> Any:
    """A value slot: value ref (%x/@x), JSON literal, or bare token."""
    tok = tok.strip()
    if re.match(r"^[%@][A-Za-z0-9_.\-]+$", tok):
        return tok
    try:
        return json.loads(tok)
    except (json.JSONDecodeError, ValueError):
        pass
    try:                                    # python reprs in provenance lines
        return pyast.literal_eval(tok)
    except (ValueError, SyntaxError):
        return tok                          # bare token (e.g. retry on=error)


def _parse_kv(items: List[str]) -> Dict[str, Any]:
    out = {}
    for it in items:
        if not it:
            continue
        key, _, val = it.partition("=")
        out[key.strip()] = _parse_value(val)
    return out


def parse_text(text: str) -> Module:
    name, description = "program", ""
    provenance: Dict[str, Any] = {}
    inputs: List[Dict[str, str]] = []
    nodes: List[Node] = []
    output = ""

    lines = text.splitlines()
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith(";"):
            if line.startswith("; TaskIR"):
                m = re.search(r"module=(\S+)", line)
                if m:
                    name = m.group(1)
            elif line.startswith("; task:"):
                description = line[len("; task:"):].strip()
            elif line.startswith("; provenance."):
                key, _, val = line[len("; provenance."):].partition(":")
                provenance[key.strip()] = _parse_value(val.strip())
            continue
        if line.startswith("inputs:"):
            body = line[len("inputs:"):].strip()
            for item in _split_top(body):
                nm, _, ty = item.partition(":")
                inputs.append({"name": nm.strip(),
                               "type": ty.strip() or "Any"})
            continue
        if line.startswith("return"):
            output = line[len("return"):].strip()
            if not re.match(r"^[%@][A-Za-z0-9_.\-]+$", output):
                raise TaskIRSyntaxError(
                    f"line {lineno}: bad return target {output!r}")
            continue

        m = _NODE_RE.match(line)
        if not m:
            raise TaskIRSyntaxError(
                f"line {lineno}: cannot parse node line: {line!r}")
        nid, op = m.group(1), m.group(2)
        open_paren = line.index("(", m.end() - 1)
        close_paren = _match_paren(line, open_paren)
        if close_paren < 0:
            raise TaskIRSyntaxError(
                f"line {lineno}: unbalanced parentheses in {op} node")
        args_body = line[open_paren + 1: close_paren]
        trailing = line[close_paren + 1:]

        inputs_list: List[str] = []
        params: Dict[str, Any] = {}
        for item in _split_top(args_body):
            if not item:
                continue
            if "=" in item and not re.match(r"^[%@]", item):
                key, _, val = item.partition("=")
                if re.match(r"^[A-Za-z_]\w*$", key.strip()):
                    params[key.strip()] = _parse_value(val)
                    continue
            inputs_list.append(_parse_value(item))

        node = Node(id=nid, op=op,
                    inputs=[x for x in inputs_list if isinstance(x, str)],
                    params=params)

        tm = _TYPE_RE.search(trailing)
        if tm:
            node.output_type = tm.group(1)

        for kind, body in _ANN_RE.findall(trailing):
            if kind == "guard":
                cond, _, expect = body.partition("==")
                node.guard = Guard(cond=cond.strip(),
                                   expect=_parse_value(expect))
            elif kind == "retry":
                kv = _parse_kv(_split_top(body))
                node.retry = Retry(max_attempts=int(kv.get("max", 1)),
                                   on=kv.get("on", "error"))
            elif kind == "after":
                node.after = [x.strip() for x in body.split(",") if x.strip()]
            elif kind == "hints":
                node.hints = _parse_kv(_split_top(body))
        nodes.append(node)

    if not nodes:
        raise TaskIRSyntaxError("no nodes found")
    if not output:
        raise TaskIRSyntaxError("missing return statement")

    prog = Program(name=name, description=description, inputs=inputs,
                   nodes=nodes, output=output)
    meta: Dict[str, Any] = {"name": name}
    if provenance:
        meta["provenance"] = provenance
    return Module(program=prog, meta=meta)
