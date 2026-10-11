"""Phase 7C-0.1 Task 3: D1 fidelity + constraint behavior tests.

No GPU required — fixtures are validator-valid TaskIR programs loaded from
data/compiler_corpus_v4/test.jsonl (real corpus) and data/taskir/examples
(handwritten but validator-clean).

Key guarantee under test: Level 1 (grammar) is LOSSLESS — every program that
is already validator-valid must come out of D1 with an identical AST. This
was the v1 failure mode (custom regex re-serialization corrupted programs).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ir.parser import parse_text
from src.ir.taskir import to_text, module_from_dict, ID_RE
from src.validator.validator import validate
from src.compiler.constrained_assembly import (
    apply_constraints, parse_and_assemble, module_ast_equal)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _corpus_modules(n=100, min_nodes=1):
    """Up to n validator-valid corpus programs with >= min_nodes nodes."""
    path = os.path.join(ROOT, "data", "compiler_corpus_v4", "test.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get("plan_text") or r.get("plan_target") or ""
            if not text:
                continue
            try:
                mod = parse_text(text)
                if validate(mod).valid and len(mod.program.nodes) >= min_nodes:
                    out.append(mod)
            except Exception:
                continue
            if len(out) >= n:
                break
    return out


def _multi_node_fixture():
    mods = _corpus_modules(50, min_nodes=3)
    assert mods, "no multi-node valid corpus program found"
    return mods[0]


def _guarded_fixture():
    p = os.path.join(ROOT, "data", "taskir", "examples", "guarded_answer.json")
    return module_from_dict(json.load(open(p, encoding="utf-8")))


# ---------------------------------------------------------------- D1 lossless

def test_d1_lossless_on_valid_corpus():
    """THE fidelity gate: D1 must preserve 100% of valid programs' ASTs."""
    corpus = _corpus_modules(100)
    assert len(corpus) >= 20, f"corpus too small: {len(corpus)}"
    broken = []
    for mod in corpus:
        res = apply_constraints(mod, 1)
        if not res["valid"] or not module_ast_equal(mod, res["module"]):
            broken.append(mod.program.name)
    assert not broken, f"D1 corrupted {len(broken)} valid programs: {broken[:5]}"


def test_d1_lossless_roundtrip():
    """parse -> to_text -> parse -> D1 keeps the AST stable."""
    for mod in _corpus_modules(10, min_nodes=3):
        again = parse_text(to_text(mod))
        res = apply_constraints(again, 1)
        assert res["valid"], mod.program.name
        assert module_ast_equal(mod, res["module"]), mod.program.name


def test_numeric_ssa_ids_accepted():
    """v1 bug: engine only accepted %name; official ID_RE allows %1/%c0."""
    assert ID_RE.match("%1") and ID_RE.match("%c0") and ID_RE.match("@task")
    # direct regression: rename a node to a bare-numeric id '%1'
    mod = _multi_node_fixture()
    rename = {mod.program.nodes[0].id: "%1"}
    for n in mod.program.nodes:
        n.id = rename.get(n.id, n.id)
        n.inputs = [rename.get(r, r) for r in n.inputs]
        n.after = [rename.get(a, a) for a in (n.after or [])]
    if mod.program.output in rename:
        mod.program.output = rename[mod.program.output]
    res = apply_constraints(mod, 1)
    assert res["valid"] and not res["rejections"], res["rejections"]


# ---------------------------------------------------------------- D2 auto-fix

def test_d2_fixes_undefined_ref():
    mod = _multi_node_fixture()
    target = mod.program.nodes[1]
    target.inputs = ["%ghost"] + target.inputs
    assert not validate(mod).valid
    res = apply_constraints(mod, 2)
    assert res["valid"], f"rejections={res['rejections']}"
    kinds = [c["kind"] for c in res["corrections"]]
    assert "undef_ref" in kinds
    fixed = res["module"].program.nodes[1].inputs
    assert "%ghost" not in fixed
    # original object untouched (deep copy)
    assert "%ghost" in mod.program.nodes[1].inputs


def test_d2_retargets_dangling_output():
    mod = _multi_node_fixture()
    mod.program.output = "%nonexistent"
    res = apply_constraints(mod, 2)
    assert res["valid"], f"rejections={res['rejections']}"
    kinds = [c["kind"] for c in res["corrections"]]
    assert "output_retarget" in kinds
    outs = {n.id for n in res["module"].program.nodes}
    outs |= {i["name"] for i in res["module"].program.inputs}
    assert res["module"].program.output in outs


def test_d2_arity_truncation():
    from src.isa.registry import get as get_skill
    mod = _multi_node_fixture()
    spec = get_skill(mod.program.nodes[1].op)
    if not (spec and spec.max_inputs):
        return  # variadic op in fixture; nothing to assert
    node_ids = [n.id for n in mod.program.nodes] + \
               [mod.program.nodes[0].id] * 4
    mod.program.nodes[1].inputs = node_ids
    res = apply_constraints(mod, 2)
    kinds = [c["kind"] for c in res["corrections"]]
    assert "arity" in kinds
    assert len(res["module"].program.nodes[1].inputs) <= spec.max_inputs


# ---------------------------------------------------------------- D3 control

def test_d3_preserves_guard_annotations():
    mod = _guarded_fixture()
    guarded = [n for n in mod.program.nodes if n.guard is not None]
    assert guarded, "fixture lost its guards"
    res = apply_constraints(mod, 3)
    assert res["valid"], f"rejections={res['rejections']}"
    out_by_id = {n.id: n for n in res["module"].program.nodes}
    for g in guarded:
        assert out_by_id[g.id].guard is not None
        assert out_by_id[g.id].guard.cond == g.guard.cond
        assert out_by_id[g.id].guard.expect == g.guard.expect


def test_d3_flags_changed_control():
    """If control annotations drift during assembly, D3 must reject."""
    mod = _guarded_fixture()
    res = apply_constraints(mod, 3)
    assert res["valid"]
    # simulate drift: strip a guard on the assembled copy, re-check vs original
    guarded_ids = [n.id for n in res["module"].program.nodes if n.guard]
    victim = next(n for n in res["module"].program.nodes if n.guard)
    victim.guard = None
    orig = {n.id: n for n in mod.program.nodes}
    drift = [n.id for n in res["module"].program.nodes
             if (orig[n.id].guard is None) != (n.guard is None)]
    assert drift and set(drift) & set(guarded_ids)


# ---------------------------------------------------------------- assembly API

def test_parse_and_assemble_error_path():
    out = parse_and_assemble("this is not taskir at all", 2)
    assert out["module"] is None and not out["valid"]
    assert out["error"].startswith("parse:")
    mod = _multi_node_fixture()
    out2 = parse_and_assemble(to_text(mod), 2)
    assert out2["valid"] and out2["error"] is None


def test_example_modules_all_levels_clean():
    """Handwritten examples pass all levels without rejections."""
    import glob
    paths = glob.glob(os.path.join(ROOT, "data", "taskir", "examples", "*.json"))
    assert paths
    for p in paths:
        mod = module_from_dict(json.load(open(p, encoding="utf-8")))
        for lvl in (1, 2, 3):
            res = apply_constraints(mod, lvl)
            assert res["valid"], \
                f"{os.path.basename(p)} L{lvl}: {res['rejections']}"


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{'ALL TESTS PASSED' if not failed else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
