"""HumanEval / MBPP style code samples -> TaskIR.

Goal is NOT a Python IR: we extract the *algorithmic core* of the
canonical solution into Skill ISA ops (the "abstract algorithm semantics"):

    sorted(x, key=...)      -> SORT
    min/max(x, key=...)     -> ARGMIN/ARGMAX (MIN/MAX without key)
    sum(x)                  -> SUM
    len(x)                  -> COUNT
    [e for e in x if p]     -> FILTER
    [f(e) for e in x]       -> TRANSFORM
    a + b (two lists)       -> JOIN

Unsupported constructs (loops, recursion, control flow, string ops, dict
building, arbitrary calls) return None with a reason — coverage gaps feed
docs/missing-skills.md. Code tasks get NO lm tail: their output is the
computed value, which keeps the corpus workload mix honest (0 lm calls).
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

from ...ir.taskir import Module, Node, Program
from .base import BenchmarkLifter, register


class _Unsupported(Exception):
    pass


class _ExprCompiler:
    """Compiles one Python expression AST into TaskIR nodes."""

    def __init__(self, params: List[str]):
        self.params = params            # function arguments -> @inputs
        self.nodes: List[Node] = []

    def _new(self, op: str, inputs: List[str], params: Dict, out_t: str) -> str:
        nid = f"%c{len(self.nodes)}"
        self.nodes.append(Node(id=nid, op=op, inputs=inputs, params=params,
                               output_type=out_t))
        return nid

    # ------------------------------------------------------------------ expr
    def compile(self, e: ast.expr) -> str:
        if isinstance(e, ast.Name):
            if e.id in self.params:
                return f"@{e.id}"
            raise _Unsupported(f"free variable {e.id!r}")
        if isinstance(e, ast.Call):
            return self._call(e)
        if isinstance(e, ast.ListComp):
            return self._listcomp(e)
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
            left, right = self.compile(e.left), self.compile(e.right)
            if left.startswith("@") or right.startswith("@"):
                pass
            return self._new("JOIN", [left, right], {}, "List[Any]")
        raise _Unsupported(f"expr {type(e).__name__}")

    def _key_name(self, kw: ast.keyword) -> Optional[str]:
        """key=lambda x: x['price'] / x.price -> 'price'."""
        v = kw.value
        if isinstance(v, ast.Lambda) and len(v.args.args) == 1:
            body = v.body
            if isinstance(body, ast.Subscript) and isinstance(body.value, ast.Name):
                sl = body.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    return sl.value
            if isinstance(body, ast.Attribute) and isinstance(body.value, ast.Name):
                return body.attr
        return None

    def _call(self, e: ast.Call) -> str:
        f = e.func
        if not (isinstance(f, ast.Name) and
                all(kw.arg in ("key", "reverse") for kw in e.keywords)):
            raise _Unsupported(f"call to {ast.dump(e.func)[:40]}")
        fname = f.id
        kwargs = {kw.arg: kw for kw in e.keywords}

        if fname == "sorted" and len(e.args) == 1:
            src = self.compile(e.args[0])
            p = {"order": "desc" if "reverse" in kwargs else "asc"}
            if "key" in kwargs:
                k = self._key_name(kwargs["key"])
                if k is None:
                    raise _Unsupported("non-trivial sorted key")
                p["key"] = k
            return self._new("SORT", [src], p, "List[Any]")

        if fname in ("min", "max") and len(e.args) == 1:
            src = self.compile(e.args[0])
            if "key" in kwargs:
                k = self._key_name(kwargs["key"])
                if k is None:
                    raise _Unsupported(f"non-trivial {fname} key")
                op = "ARGMIN" if fname == "min" else "ARGMAX"
                return self._new(op, [src], {"key": k}, "Any")
            op = "MIN" if fname == "min" else "MAX"
            return self._new(op, [src], {}, "Any")

        if fname == "sum" and len(e.args) == 1:
            return self._new("SUM", [self.compile(e.args[0])], {}, "Float")
        if fname == "len" and len(e.args) == 1:
            return self._new("COUNT", [self.compile(e.args[0])], {}, "Int")

        raise _Unsupported(f"builtin {fname!r}")

    def _listcomp(self, e: ast.ListComp) -> str:
        if len(e.generators) != 1:
            raise _Unsupported("nested comprehension")
        gen = e.generators[0]
        src = self.compile(gen.iter)

        cur = src
        if gen.ifs:
            pred = " AND ".join(ast.unparse(i) for i in gen.ifs)
            cur = self._new("FILTER", [cur], {"predicate": pred}, "List[Any]")

        elt = e.elt
        if isinstance(elt, ast.Name) and elt.id == gen.target.id:
            if cur == src and src.startswith("@"):
                # identity over the input: still needs a defining node
                return self._new("TRANSFORM", [cur],
                                 {"op": "identity"}, "List[Any]")
            return cur
        desc = ast.unparse(elt)
        return self._new("TRANSFORM", [cur], {"op": desc}, "List[Any]")


@register
class HumanEvalLifter(BenchmarkLifter):
    name = "humaneval"
    dataset = "humaneval/mbpp"

    def can_handle(self, sample: Dict) -> bool:
        code = sample.get("code") or sample.get("canonical_solution")
        return bool(sample.get("prompt") or sample.get("text")) and bool(code)

    def lift(self, sample: Dict) -> Optional[Module]:
        code = sample.get("code") or sample.get("canonical_solution") or ""
        prompt = sample.get("prompt") or sample.get("text") or ""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not fns:
            return None
        fn = fns[0]
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)
                   and n.value is not None]
        if not returns:
            return None
        params = [a.arg for a in fn.args.args]
        comp = _ExprCompiler(params)
        try:
            out = comp.compile(returns[0].value)
        except _Unsupported:
            return None
        if out.startswith("@"):
            out = comp._new("TRANSFORM", [out], {"op": "identity"},
                            "List[Any]")
        if not comp.nodes:
            return None
        prog = Program(
            name=f"code_{sample.get('task_id', 'task')}",
            description=prompt.strip()[:200],
            inputs=[{"name": f"@{p}", "type": "List[Any]"} for p in params],
            nodes=comp.nodes, output=out)
        return Module(program=prog, meta={
            "name": prog.name,
            "provenance": {"source": self.name,
                           "task_id": str(sample.get("task_id", ""))}})

    def lift_with_reason(self, sample: Dict) -> Tuple[Optional[Module], Optional[str]]:
        code = sample.get("code") or sample.get("canonical_solution") or ""
        if not (sample.get("prompt") or sample.get("text")):
            return None, "missing prompt"
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None, "python syntax error"
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not fns:
            return None, "no function definition"
        fn = fns[0]
        for n in ast.walk(fn):
            if isinstance(n, (ast.For, ast.While)):
                return None, "loop (no LOOP op in TaskIR v0.1)"
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                    and n.func.id in (fn.name, "recurse"):
                if n.func.id == fn.name:
                    return None, "recursion (no LOOP op in v0.1)"
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)
                   and n.value is not None]
        if not returns:
            return None, "no return"
        mod = self.lift(sample)
        if mod is None:
            try:
                _ExprCompiler([a.arg for a in fn.args.args]).compile(
                    returns[0].value)
            except _Unsupported as e:
                return None, f"pattern: {e}"
            return None, "unsupported"
        return mod, None
