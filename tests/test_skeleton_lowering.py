"""Phase 8B-0: skeleton validation + lowering tests (CPU)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.plan_skeleton import skeleton_from_dict, validate_skeleton
from src.compiler.skeleton_lowering import lower_skeleton
from src.ir.parser import parse_text
from src.validator.validator import validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCHEMAS = {
    "get_reservation_details": {"params": [
        {"name": "reservation_id", "optional": False}]},
    "update_reservation_flights": {"params": [
        {"name": "reservation_id", "optional": False},
        {"name": "cabin", "optional": False},
        {"name": "flights", "optional": False},
        {"name": "payment_id", "optional": False}]},
    "get_user_details": {"params": [
        {"name": "user_id", "optional": False}]},
}

GOOD = {
    "task_id": "airline-44",
    "nodes": [
        {"id": "p0", "capability_id": "get_user_details",
         "skill": "FETCH", "depends_on": [],
         "arguments": {"user_id": "raj_sanchez_7340"}},
        {"id": "p1", "capability_id": "get_reservation_details",
         "skill": "FETCH", "depends_on": ["p0"],
         "arguments": {"reservation_id": "Q69X3R"}},
        {"id": "p2", "capability_id": "update_reservation_flights",
         "skill": "EXEC_ACTION", "depends_on": ["p1"],
         "arguments": {"reservation_id": "Q69X3R", "cabin": "economy",
                       "flights": "[]", "payment_id": "credit_card_1"}},
    ],
}


def test_valid_skeleton_lowering_passes_validator():
    sk = skeleton_from_dict(GOOD)
    errs = validate_skeleton(sk, "airline", SCHEMAS)
    assert not errs, errs
    out = lower_skeleton(sk, "airline", SCHEMAS)
    assert out["errors"] == [] and out["unresolved"] == []
    mod = parse_text(out["taskir"])
    assert validate(mod).valid
    # tool identity preserved
    exec_nodes = [n for n in mod.program.nodes
                  if n.op == "EXEC_ACTION"]
    assert exec_nodes[0].params["action"] == "update_reservation_flights"


def test_missing_required_param_is_unresolved_not_fabricated():
    bad = {"task_id": "x", "nodes": [
        {"id": "p0", "capability_id": "update_reservation_flights",
         "skill": "EXEC_ACTION", "depends_on": [],
         "arguments": {"reservation_id": "Q"}}]}
    out = lower_skeleton(skeleton_from_dict(bad), "airline", SCHEMAS)
    assert out["taskir"] is None
    kinds = {u["missing_param"] for u in out["unresolved"]}
    assert {"cabin", "flights", "payment_id"} <= kinds


def test_unknown_capability_rejected():
    bad = {"task_id": "x", "nodes": [
        {"id": "p0", "capability_id": "no_such_tool", "skill": "FETCH",
         "depends_on": [], "arguments": {}}]}
    errs = validate_skeleton(skeleton_from_dict(bad), "airline", SCHEMAS)
    assert any("not in runtime registry" in e for e in errs)


def test_mutating_capability_must_keep_identity():
    bad = {"task_id": "x", "nodes": [
        {"id": "p0", "capability_id": "update_reservation_flights",
         "skill": "FETCH", "depends_on": [],
         "arguments": {"reservation_id": "Q", "cabin": "economy",
                       "flights": "[]", "payment_id": "c1"}}]}
    errs = validate_skeleton(skeleton_from_dict(bad), "airline", SCHEMAS)
    assert any("must use EXEC_ACTION" in e for e in errs)


def test_read_tool_with_exec_action_identity_is_allowed():
    sk = {"task_id": "x", "nodes": [
        {"id": "p0", "capability_id": "get_user_details",
         "skill": "EXEC_ACTION", "depends_on": [],
         "arguments": {"user_id": "u1"}}]}
    errs = validate_skeleton(skeleton_from_dict(sk), "airline", SCHEMAS)
    assert not errs


def test_dependency_cycle_rejected():
    bad = {"task_id": "x", "nodes": [
        {"id": "p0", "capability_id": "get_user_details",
         "skill": "FETCH", "depends_on": ["p1"],
         "arguments": {"user_id": "u1"}},
        {"id": "p1", "capability_id": "get_user_details",
         "skill": "FETCH", "depends_on": ["p0"],
         "arguments": {"user_id": "u2"}}]}
    errs = validate_skeleton(skeleton_from_dict(bad), "airline", SCHEMAS)
    assert any("cycle" in e for e in errs)


def test_wrong_plan_not_silently_repaired():
    """A semantically wrong but structurally valid skeleton lowers to a
    legal program carrying the SAME wrong semantics (no repair)."""
    wrong = {"task_id": "x", "nodes": [
        {"id": "p0", "capability_id": "get_user_details",
         "skill": "FETCH", "depends_on": [],
         "arguments": {"user_id": "WRONG_USER"}}]}
    out = lower_skeleton(skeleton_from_dict(wrong), "airline", SCHEMAS)
    assert out["errors"] == []
    mod = parse_text(out["taskir"])
    assert validate(mod).valid
    # the wrong argument survived lowering untouched
    node = [n for n in mod.program.nodes if n.op == "FETCH"][0]
    # args appear as key=value params
    assert "WRONG_USER" in out["taskir"]


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
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
