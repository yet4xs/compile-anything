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
