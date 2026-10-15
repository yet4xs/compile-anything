"""Typed nested field access for TaskIR runtime bindings — PROTOTYPE.

Phase 9B-2.2 Task 4 design artifact. NOT wired into the validator or the
lowering pipeline yet; the production EXTRACT_FIELD semantics are unchanged
until the design is accepted.

A FieldPath is a compile-time-checked sequence of accessors:

    Field("orders") -> Index(0) -> Field("order_id")

Checked against a declared schema type language (the same primitive names
the capability registry already uses: Str/Int/Float/Bool/Any plus
Dict(field: type) / List(elem: type) / Struct(fields)). Compilation:

  check_path(schema, path) resolves the path to a result type or raises
  PathTypeError with a precise reason (FIELD_NOT_FOUND, INDEX_ON_NON_LIST,
  FIELD_ON_PRIMITIVE, INDEX_OUT_OF_RANGE_STATIC, ...).

Design boundary, stated honestly: when a producer's schema declares
``List[Any]`` (as tau3's handwritten registries do for ``orders``), static
checking CANNOT know the element structure — the element access is a
DYNAMIC CHECK BOUNDARY: compilation succeeds with ``check=dynamic`` and the
runtime must re-validate the path against the actual value. We never fake a
static proof over ``Any``.
"""

from dataclasses import dataclass, field as dc_field
from typing import List as PyList, Optional, Union


# ── schema type language ──

@dataclass
class ListT:
    elem: "Type"


@dataclass
class DictT:
    fields: dict  # name -> Type


@dataclass
class StructT:
    name: str
    fields: dict  # name -> Type (ordered)


Type = Union[str, ListT, DictT, StructT]  # str = primitive name ("Str", "Any", ...)

PRIMITIVES = {"Str", "Int", "Float", "Bool", "Any"}


class PathTypeError(Exception):
    def __init__(self, code: str, msg: str,
                 at_step: Optional[int] = None, dynamic: bool = False):
        super().__init__(msg)
        self.code = code
        self.at_step = at_step
        self.dynamic = dynamic


# ── path AST ──

@dataclass
class Field:
    name: str


@dataclass
class Index:
    i: int


Step = Union[Field, Index]


@dataclass
class CheckedPath:
    steps: PyList[Step]
    result_type: str
    check: str = "static"          # "static" | "dynamic"
    dynamic_from_step: Optional[int] = None  # where static knowledge ends


def parse_path(text: str) -> PyList[Step]:
    """Parse 'orders[0].order_id' into [Field, Index, Field].

    Raises PathTypeError(PARSE_ERROR) on malformed input — this is the field
    SYNTAX gate that the 9B-2.1 classifier was missing.
    """
    steps: PyList[Step] = []
    tok = ""
    i = 0
    expect = "name"       # "name" | "index"
    after_index = False   # just closed ']' — a '.' separator is allowed
    while i < len(text):
        c = text[i]
        if c == ".":
            if after_index:
                after_index = False  # separator between ] and next name
            elif expect != "name" or not tok:
                raise PathTypeError("PARSE_ERROR",
                                    f"unexpected '.' at {i} in {text!r}", i)
            else:
                steps.append(Field(tok)); tok = ""
        elif c == "[":
            after_index = False
            if expect != "name" or (tok == "" and not steps):
                raise PathTypeError("PARSE_ERROR",
                                    f"unexpected '[' at {i} in {text!r}", i)
            if tok:
                steps.append(Field(tok)); tok = ""
            expect = "index"
        elif c == "]":
            after_index = False
            if expect != "index":
                raise PathTypeError("PARSE_ERROR",
                                    f"unexpected ']' at {i} in {text!r}", i)
            if not tok.isdigit():
                raise PathTypeError("PARSE_ERROR",
                                    f"non-integer index {tok!r} in {text!r}", i)
            steps.append(Index(int(tok))); tok = ""
            expect = "name"
            after_index = True
        else:
            after_index = False
            if expect == "index" and c in " -":
                raise PathTypeError("PARSE_ERROR",
                                    f"bad index char {c!r} in {text!r}", i)
            tok += c
        i += 1
    if expect == "index":
        raise PathTypeError("PARSE_ERROR", f"unterminated index in {text!r}",
                            i)
    if after_index and (tok or steps and isinstance(steps[-1], Index)):
        pass  # path may legitimately end right after an index
    if tok:
        steps.append(Field(tok))
    elif not steps:
        raise PathTypeError("PARSE_ERROR", "empty path", 0)
    elif not after_index and text.rstrip()[-1:] == ".":
        raise PathTypeError("PARSE_ERROR", f"trailing '.' in {text!r}", i)
    return steps


def check_path(declared: Type, steps: PyList[Step]) -> CheckedPath:
    """Type-check a parsed path against a declared schema type.

    Static checking stops at ``Any``: the remainder of the path is marked as
    a dynamic check boundary rather than invented knowledge.
    """
    cur: Type = declared
    dynamic_from = None
    result = "Any"

    for k, st in enumerate(steps):
        if dynamic_from is not None:
            continue  # beyond static knowledge; runtime re-checks
        if isinstance(st, Field):
            if isinstance(cur, str):
                if cur == "Any":
                    dynamic_from = k
                    continue
                raise PathTypeError(
                    "FIELD_ON_PRIMITIVE",
                    f"field '{st.name}' on primitive {cur}", k)
            fields = cur.fields if isinstance(cur, (DictT, StructT)) else {}
            if st.name not in fields:
                raise PathTypeError(
                    "FIELD_NOT_FOUND",
                    f"field '{st.name}' not in {sorted(fields) or 'schema'}",
                    k)
            cur = fields[st.name]
        elif isinstance(st, Index):
            if isinstance(cur, str):
                if cur == "Any":
                    dynamic_from = k
                    continue
                raise PathTypeError(
                    "INDEX_ON_NON_LIST", f"index on primitive {cur}", k)
            if not isinstance(cur, ListT):
                raise PathTypeError(
                    "INDEX_ON_NON_LIST",
                    f"index on {type(cur).__name__}", k)
            cur = cur.elem
        else:  # pragma: no cover
            raise PathTypeError("PARSE_ERROR", f"unknown step {st!r}", k)

    # resolve final type name
    if dynamic_from is not None:
        result, check = "Any", "dynamic"
    elif isinstance(cur, str):
        result, check = cur, "static"
    elif isinstance(cur, ListT):
        result, check = "List", "static"
    else:
        result, check = "Struct", "static"
    return CheckedPath(steps=steps, result_type=result, check=check,
                       dynamic_from_step=dynamic_from)


def resolve_path(value, steps: PyList[Step]):
    """Runtime resolution with the same failure codes (the dynamic half of
    the boundary: list bounds, real dict keys, real object fields)."""
    cur = value
    for k, st in enumerate(steps):
        if st is not None and isinstance(st, Field):
            if isinstance(cur, dict):
                if st.name not in cur:
                    raise PathTypeError(
                        "FIELD_NOT_FOUND",
                        f"runtime: '{st.name}' not in dict", k)
                cur = cur[st.name]
            else:
                if not hasattr(cur, st.name):
                    raise PathTypeError(
                        "FIELD_NOT_FOUND",
                        f"runtime: '{st.name}' not on "
                        f"{type(cur).__name__}", k)
                cur = getattr(cur, st.name)
        elif isinstance(st, Index):
            if not isinstance(cur, (list, tuple)):
                raise PathTypeError(
                    "INDEX_ON_NON_LIST",
                    f"runtime: index on {type(cur).__name__}", k)
            if not (0 <= st.i < len(cur)):
                raise PathTypeError(
                    "INDEX_OUT_OF_RANGE",
                    f"runtime: index {st.i} of {len(cur)}", k)
            cur = cur[st.i]
    return cur
