"""Tool-use dataset adapters: ToolBench (real answer files), API-Bank,
AgentBench (registered; data gated)."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

from ..schema import make_sample


def _find_tool_calls(node: Any, out: List[Dict[str, Any]],
                     depth: int = 0) -> None:
    """Recursively collect tool-call shaped dicts from a ToolBench record
    (answer_generation messages etc.): {"name"/"api_name", "arguments"} or
    function_call wrappers."""
    if depth > 12:
        return
    if isinstance(node, dict):
        name = node.get("name") or node.get("api_name") or node.get("tool")
        args = node.get("arguments") or node.get("parameters") or {}
        if isinstance(name, str) and name and isinstance(args, dict) \
                and (args or node.get("arguments") is not None
                     or node.get("parameters") is not None):
            out.append({"tool": name,
                        "args": {k: v for k, v in args.items()
                                 if isinstance(v, (str, int, float, bool))}})
        if isinstance(node.get("function_call"), dict):
            fc = node["function_call"]
            out.append({"tool": fc.get("name", ""),
                        "args": _json_args(fc.get("arguments"))})
        for v in node.values():
            _find_tool_calls(v, out, depth + 1)
    elif isinstance(node, list):
        for v in node:
            _find_tool_calls(v, out, depth + 1)


def _json_args(x) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, dict) else {"raw": x}
        except json.JSONDecodeError:
            return {"raw": x}
    return {}


class ToolBenchAdapter:
    sources = ["toolbench"]

    def load(self, raw_dir: pathlib.Path) -> List[Dict[str, Any]]:
        out = []
        for sub in sorted((raw_dir / "toolbench" / "answer").glob("*")):
            if not sub.is_dir():
                continue
            for f in sorted(sub.glob("*.json")):
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                out.append(self.normalize(rec, f"{sub.name}/{f.stem}"))
        return [s for s in out if s]

    def normalize(self, rec: Dict[str, Any], sid: str) -> Dict[str, Any]:
        ag = rec.get("answer_generation") or {}
        instruction = (rec.get("instruction") or rec.get("query")
                       or ag.get("query") or "")
        if not instruction:
            return {}
        calls: List[Dict[str, Any]] = []
        _find_tool_calls(rec.get("answer_generation", rec), calls)
        # keep call order, drop duplicates from retry-echoes in the trajectory
        seen, traj = set(), []
        for c in calls:
            key = json.dumps(c, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                traj.append(c)
        return make_sample(id=f"toolbench-{sid}", source="toolbench",
                           input_text=instruction, trajectory=traj,
                           metadata={"split": rec.get("split", "")},
                           raw_payload={"instruction": instruction})


def _parse_tool_defs(x) -> list:
    """Normalize a raw tool list; JSON-encoded strings must be parsed first.

    Regression: xLAM raw records carry tools as a JSON STRING. Before the
    Phase 6A fix, semantic_capabilities iterated the string CHARACTER by
    character, every char fell to map_tool's fallback, and the capability
    context collapsed to the constant ['EXEC_ACTION'] on all 19,700 xLAM
    records (see docs/phase6/phase5c-capability-audit.md).
    """
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else []
        except json.JSONDecodeError:
            return []
    return x if isinstance(x, list) else []


def semantic_capabilities(tool_defs) -> List[str]:
    """Normalize a raw tool list into semantic capability phrases (Task 5B:
    the compiler's capability context). Concrete API names stay out."""
    from ...lifter import toolmap
    seen, out = set(), []
    for t in _parse_tool_defs(tool_defs):
        name = ""
        if isinstance(t, str):
            name = t
        elif isinstance(t, dict):
            name = t.get("name") or t.get("api_name") or ""
        if not name:
            continue
        m = toolmap.map_tool(name, {})
        cap = m["skill"]
        if m["params"].get("domain"):
            cap += f"({m['params']['domain']})"
        if cap not in seen:
            seen.add(cap)
            out.append(cap)
    return out


class XlamAdapter:
    """REAL Salesforce/xLAM function-calling 60k (via ModelScope mirror).
    Record shape: {id, query, tools[...], answers: "<json string of
    [{name, arguments}]>"} — the trajectory lives in `answers`."""
    sources = ["xlam"]

    def load(self, raw_dir: pathlib.Path) -> List[Dict[str, Any]]:
        import glob as _glob
        files = sorted(_glob.glob(str(raw_dir / "xlam" / "*.json*")))
        out: List[Dict[str, Any]] = []
        for fp in files:
            text = pathlib.Path(fp).read_text(encoding="utf-8",
                                              errors="replace").strip()
            if not text:
                continue
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            for rec in data:
                s = self.normalize(rec)
                if s:
                    out.append(s)
        return out

    def normalize(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        question = rec.get("query") or rec.get("question") or ""
        raw_answers = rec.get("answers")
        calls: List[Dict[str, Any]] = []
        if isinstance(raw_answers, str):
            try:
                raw_answers = json.loads(raw_answers)
            except json.JSONDecodeError:
                raw_answers = []
        for a in raw_answers if isinstance(raw_answers, list) else []:
            if isinstance(a, dict) and a.get("name"):
                args = a.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                calls.append({"tool": a["name"],
                              "args": args if isinstance(args, dict) else {}})
        if not question or not calls:
            return {}
        return make_sample(
            id=f"xlam-{rec.get('id', '')}", source="xlam",
            input_text=question, trajectory=calls,
            metadata={"capabilities": semantic_capabilities(rec.get("tools"))},
            raw_payload={"answer": str(rec.get("answers", ""))[:200]})


_ACTION_RE = None


def _parse_react_target(target: str) -> List[Dict[str, Any]]:
    """ToolBench-Static ground truth is ReAct text:
    'Action: <tool>\\nAction Input: {json}' (repeated for multi-step)."""
    import re as _re
    global _ACTION_RE
    if _ACTION_RE is None:
        _ACTION_RE = _re.compile(
            r"Action:\s*([A-Za-z0-9_.\-]+)\s*\nAction Input:\s*(\{.*?\})\s*(?=\n|$)",
            _re.S)
    calls = []
    for name, argstr in _ACTION_RE.findall(target or ""):
        try:
            args = json.loads(argstr)
        except json.JSONDecodeError:
            args = {}
        calls.append({"tool": name,
                      "args": args if isinstance(args, dict) else {}})
    return calls


class ToolBenchStaticAdapter:
    """REAL ToolBench static eval subset (via ModelScope mirror): messages
    [system(tools), user(instruction)] + ReAct-format ground truth target."""
    sources = ["toolbench_static"]

    def load(self, raw_dir: pathlib.Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fname in ("in_domain.jsonl", "out_of_domain.jsonl"):
            f = raw_dir / "toolbench_static" / fname
            if not f.exists():
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8",
                                                 errors="replace").splitlines()):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = self.normalize(rec, f"{fname}-{i}")
                if s:
                    out.append(s)
        return out

    def normalize(self, rec: Dict[str, Any], sid: str) -> Dict[str, Any]:
        instruction = ""
        for m in reversed(rec.get("messages") or []):
            if m.get("role") == "user":
                instruction = m.get("content") or ""
                break
        calls = _parse_react_target(rec.get("target") or "")
        if not instruction or not calls:
            return {}
        return make_sample(id=f"tbs-{sid}", source="toolbench_static",
                           input_text=instruction, trajectory=calls,
                           metadata={"capabilities":
                                     semantic_capabilities(rec.get("tools"))},
                           raw_payload={})


class ApiBankAdapter:
    """Registered for when a mirror is located; tolerant parser kept."""
    sources = ["apibank"]

    def load(self, raw_dir: pathlib.Path) -> List[Dict[str, Any]]:
        root = raw_dir / "apibank"
        if not root.is_dir():
            return []
        out = []
        for f in sorted(root.rglob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            samples = data.get("samples") if isinstance(data, dict) else data
            if not isinstance(samples, list):
                continue
            for i, rec in enumerate(samples):
                s = self.normalize(rec, f"{f.stem}-{i}")
                if s:
                    out.append(s)
        return out

    def normalize(self, rec: Dict[str, Any], sid: str) -> Dict[str, Any]:
        # API-Bank level records: initial_session text + "golden function calls"
        text = rec.get("initial_session") or rec.get("instruction") or ""
        calls_raw = (rec.get("golden function calls")
                     or rec.get("golden_function_calls") or [])
        calls = []
        for c in calls_raw if isinstance(calls_raw, list) else []:
            if isinstance(c, dict):
                name = c.get("name") or c.get("API") or c.get("api") or ""
                args = c.get("arguments") or c.get("args") or c.get("params") or {}
                if name:
                    calls.append({"tool": name,
                                  "args": args if isinstance(args, dict) else {}})
        if not text or not calls:
            return {}
        return make_sample(id=f"apibank-{sid}", source="apibank",
                           input_text=str(text), trajectory=calls,
                           raw_payload={})
