"""Phase 9B-2.4 Task 2: verified capability registry from OFFICIAL sources.

Input schema: the tau2 tool functions' own signatures (inspect).
Output schema: the functions' return type annotations, expanded
recursively over pydantic models — never derived from any model
prediction, never from our old handwritten guesses.
Effect class: from the toolkit's own mutation metadata.

The single JSON this emits is the ONE schema version consumed by
validator / lowering / runtime / scorer / gold reconstruction
(verified_registry_manifest.json).

Type descriptor format (stable, shared by all consumers):
  {"kind": "scalar", "type": "Str"}         leaf primitive
  {"kind": "scalar_root"}                   whole output is a bare value
  {"kind": "list", "elem": <descriptor>}
  {"kind": "struct", "fields": {name: <descriptor>}}
  {"kind": "dict", "elem": <descriptor>}    homogeneous dict (variants)

Run with the tau3 conda env:
  D:/miniconda3/envs/tau3/python.exe scripts/build_verified_registry.py
"""

import inspect
import json
import sys
import typing
from pathlib import Path

from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
TAU2_SRC = Path("D:/ccfa-2026/tau3-bench-src")
OUT = REPO / "results" / "phase9" / "verified_registry_manifest.json"

PRIM = {"str": "Str", "int": "Int", "float": "Float", "bool": "Bool",
        "Any": "Any"}


def descriptor(tp, depth=0) -> dict:
    if depth > 6:
        return {"kind": "scalar", "type": "Any"}
    origin = typing.get_origin(tp)
    if origin is None:
        if tp in PRIM:
            return {"kind": "scalar", "type": PRIM[tp]}
        name = getattr(tp, "__name__", str(tp))
        if name in PRIM:
            return {"kind": "scalar", "type": PRIM[name]}
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            return struct_descriptor(tp, depth)
        return {"kind": "scalar", "type": "Any"}
    args = typing.get_args(tp)
    if origin in (list, typing.List):
        return {"kind": "list", "elem": descriptor(args[0], depth + 1)}
    if origin in (dict, typing.Dict):
        return {"kind": "dict", "elem": descriptor(args[-1], depth + 1)}
    if origin in (typing.Optional, typing.Union):
        nonnone = [a for a in args if a is not type(None)]
        return descriptor(nonnone[0], depth + 1) if nonnone else \
            {"kind": "scalar", "type": "Any"}
    return {"kind": "scalar", "type": "Any"}


def struct_descriptor(model: type, depth=0) -> dict:
    fields = {}
    for name, f in model.model_fields.items():
        fields[name] = descriptor(f.annotation, depth + 1)
    return {"kind": "struct", "fields": fields}


def main():
    sys.path.insert(0, str(TAU2_SRC / "src"))
    from tau2.domains.airline.environment import get_environment as airline_env
    from tau2.domains.retail.environment import get_environment as retail_env

    registry = {}
    for dom, ctor in (("airline", airline_env), ("retail", retail_env)):
        env = ctor()
        tools = env.tools
        for name, fn in tools.tools.items():
            sig = inspect.signature(fn)
            ret = sig.return_annotation
            # bare scalar returns (find_user_*) are scalar_root regardless
            # of the cosmetic `-> str` annotation
            out_desc = descriptor(ret)
            if out_desc.get("kind") == "scalar":
                out_desc = {"kind": "scalar_root"}
            params = {}
            for pname, p in sig.parameters.items():
                if pname in ("self",):
                    continue
                d = descriptor(p.annotation)
                params[pname] = d if d["kind"] != "scalar" else d["type"]
            try:
                mutates = tools.tool_mutates_state(name)
            except Exception:
                mutates = None
            key = f"{dom}.{name}"
            registry[key] = {
                "domain": dom,
                "input_schema": params,
                "output_schema": out_desc,
                # official metadata only says mutates-or-not; reversibility
                # is NOT in official sources, so we do not fake it
                "effect_class": ("mutating" if mutates else "read_only")
                if mutates is not None else "unknown",
                "provenance": {
                    "input": "inspect(tool function signature)",
                    "output": "return annotation, pydantic-expanded",
                    "effect": "toolkit.tool_mutates_state (bool only)",
                },
            }

    OUT.write_text(json.dumps(
        {"n_tools": len(registry), "tools": registry},
        indent=1, ensure_ascii=False), encoding="utf-8")
    # print a compact view
    for k in sorted(registry):
        r = registry[k]
        print(f"{k}: effect={r['effect_class']} out={json.dumps(r['output_schema'])[:110]}")


if __name__ == "__main__":
    main()
