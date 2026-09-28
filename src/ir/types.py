"""TaskIR v0.1 type system utilities.

Types are nominal strings with parametric syntax (List[T], Map[K, V]).
Compatibility rules (see spec/taskir-spec.md §3):
  1. Any/Unknown is bidirectionally compatible with everything.
  2. Parametric types compare recursively (List[Any] ~ List[Flight]).
  3. Otherwise exact match after canonicalization.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

OPEN_TYPES = {"Any", "Unknown"}


def normalize(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _split_args(s: str) -> list:
    """Split 'K, V' on top-level commas (bracket-aware)."""
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def parse(t: str) -> Tuple[str, Optional[list]]:
    """'List[Flight]' -> ('List', ['Flight']); 'Flight' -> ('Flight', None)."""
    t = normalize(t)
    m = re.match(r"^([A-Za-z_]\w*)\[(.+)\]$", t)
    if not m:
        return t, None
    return m.group(1), _split_args(m.group(2))


def is_compatible(declared: str, actual: str) -> bool:
    """Can a value of type `actual` flow into a slot declared as `declared`?"""
    d, a = normalize(declared), normalize(actual)
    if not d or not a:
        return False
    if d in OPEN_TYPES or a in OPEN_TYPES:
        return True
    dh, dargs = parse(d)
    ah, aargs = parse(a)
    if dargs is None and aargs is None:
        return d == a
    if dargs is None or aargs is None or dh != ah or len(dargs) != len(aargs):
        return False
    return all(is_compatible(x, y) for x, y in zip(dargs, aargs))


def item_of(t: str) -> str:
    """Element type of List[T]/Set[T]; Any otherwise."""
    head, args = parse(t)
    if args and len(args) == 1 and head in ("List", "Set"):
        return args[0]
    return "Any"


def list_of(t: str) -> str:
    return f"List[{normalize(t)}]"
