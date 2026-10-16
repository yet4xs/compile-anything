"""Prompt schema renderer v2 — nested types, shared by train & eval.

Renders verified-registry descriptors as compact type strings:
  {"kind":"struct","fields":{...}}        -> {a: Str, b: List[...]}
  {"kind":"list","elem":T}                -> List[T]
  {"kind":"dict","elem":T}                -> Dict[T]
  {"kind":"scalar","type":"Str"}          -> Str
  {"kind":"scalar_root"}                  -> (whole output)
"""


def render_type(desc, depth=0):
    if depth > 4:
        return "Any"
    k = desc.get("kind")
    if k == "scalar":
        return desc.get("type", "Any")
    if k == "scalar_root":
        return "Str"
    if k == "list":
        return f"List[{render_type(desc.get('elem', {}), depth + 1)}]"
    if k == "dict":
        return f"Dict[{render_type(desc.get('elem', {}), depth + 1)}]"
    if k == "struct":
        inner = ", ".join(
            f"{n}: {render_type(d, depth + 1)}"
            for n, d in (desc.get("fields") or {}).items())
        return "{" + inner + "}"
    return "Any"


def render_tool(name, entry):
    ins = ", ".join(f"{k}: {render_type(v) if isinstance(v, dict) else v}"
                    for k, v in entry["input_schema"].items())
    out = render_type(entry["output_schema"])
    eff = entry.get("effect_class", "unknown")
    return f"- {name}({ins}) -> {out} | effect: {eff}"


def render_registry(schemas):
    return "\n".join(render_tool(n, e) for n, e in sorted(schemas.items()))
