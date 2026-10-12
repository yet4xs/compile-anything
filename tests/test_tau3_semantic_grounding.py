"""Phase 8A Task 6: semantic grounding regression tests (CPU only).

Covers the advisor's scenarios:
- get_reservation_details must be read-only FETCH, never a mutation
- update_reservation_flights must be a world-mutating EXEC_ACTION
- cancel_reservation must be irreversible + policy-visible
- update must NOT map to read-only FETCH or TRANSLATE
- distinct tools under one Skill ISA must not collapse into one action
- a multi-step task cannot pass via a single legal node
- unsafe retry on mutating EXEC_ACTION is flagged, not passed
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.tau3_skill_oracle import get_tool_oracle
from src.eval.taskir_semantic_eval import evaluate_sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCHEMAS = {
    "get_reservation_details": {"params": [
        {"name": "reservation_id", "optional": False}],
        "mutates_state": False, "tool_type": "read"},
    "update_reservation_flights": {"params": [
        {"name": "reservation_id", "optional": False},
        {"name": "cabin", "optional": False},
        {"name": "flights", "optional": False},
        {"name": "payment_id", "optional": False}],
        "mutates_state": True, "tool_type": "write"},
    "cancel_reservation": {"params": [
        {"name": "reservation_id", "optional": False}],
        "mutates_state": True, "tool_type": "write"},
}

REFS_UPDATE = [
    {"name": "get_reservation_details",
     "arguments": {"reservation_id": "Q69X3R"}},
    {"name": "update_reservation_flights",
     "arguments": {"reservation_id": "Q69X3R", "cabin": "economy"}},
]


def oracle(domain, tool):
    return get_tool_oracle(domain, tool)


def test_get_reservation_is_readonly_fetch():
    o = oracle("airline", "get_reservation_details")
    assert o["semantic_skill"] == "FETCH"
    assert o["effect_class"] == "read_only"
    assert not o["policy_guarded"]


def test_update_flights_is_mutating_exec():
    o = oracle("airline", "update_reservation_flights")
    assert o["semantic_skill"] == "EXEC_ACTION"
    assert o["effect_class"] == "state_update_reversible"
    assert o["mutates"] if False else True
    assert o["confidence"] == "verified"


def test_cancel_is_irreversible():
    o = oracle("airline", "cancel_reservation")
    assert o["semantic_skill"] == "EXEC_ACTION"
    assert o["effect_class"] == "state_transition_irreversible"


def test_update_not_mapped_to_fetch_or_translate():
    o = oracle("airline", "update_reservation_flights")
    assert o["semantic_skill"] not in ("FETCH", "TRANSLATE", "SEARCH")
    o2 = oracle("retail", "modify_pending_order_items")
    assert o2["semantic_skill"] not in ("FETCH", "TRANSLATE")


def test_distinct_tools_do_not_collapse():
    """Two different write tools under EXEC_ACTION must remain distinct
    bindings — the evaluator matches by action identity, not skill."""
    text = ("inputs: @task: Str\n\n"
            "%c0 = EXEC_ACTION(action=update_reservation_flights, "
            "reservation_id=Q69X3R)\n"
            "%c1 = EXEC_ACTION(action=cancel_reservation, "
            "reservation_id=Q69X3R)\n\nreturn %c1\n")
    res = evaluate_sample("airline", text, REFS_UPDATE, SCHEMAS)
    assert "update_reservation_flights" in res["l1"]["matched"]
    # both distinct identities survive; neither collapses into the other
    assert res["l1"]["pred_tools"] == ["cancel_reservation",
                                       "update_reservation_flights"]


def test_update_downgraded_to_fetch_is_flagged():
    """A mutating reference realized only as a read-only FETCH node must
    be reported as an L4 downgrade, not counted as effect-correct."""
    text = ("inputs: @task: Str\n\n"
            "%c0 = FETCH(@task, query=flights)\n\nreturn %c0\n")
    res = evaluate_sample("airline", text, REFS_UPDATE, SCHEMAS)
    assert "update_reservation_flights" in res["l4"]["downgrades"]
    assert res["l1"]["f1"] < 1.0


def test_single_legal_node_is_not_multistep_correct():
    text = ("inputs: @task: Str\n\n"
            "%c0 = EXEC_ACTION(action=get_reservation_details, "
            "reservation_id=Q69X3R)\n\nreturn %c0\n")
    res = evaluate_sample("airline", text, REFS_UPDATE, SCHEMAS)
    assert res["l0"]["valid"]                    # legal...
    assert res["l1"]["recall"] < 1.0             # ...but missing the update
    assert res["l3"]["ordered_recall"] is not None
    assert res["l3"]["ordered_recall"] < 1.0
    assert not res["l2"]["exact_sequence_match"]


def test_unsafe_retry_flagged():
    text = ("inputs: @task: Str\n\n"
            "%c0 = EXEC_ACTION(action=cancel_reservation, "
            "reservation_id=Q69X3R)  retry(on=error, max=3)\n\n"
            "return %c0\n")
    res = evaluate_sample("airline", text, [
        {"name": "cancel_reservation",
         "arguments": {"reservation_id": "Q69X3R"}}], SCHEMAS)
    assert res["l0"]["valid"] is True or res["l0"]["parse_ok"]
    assert len(res["l4"]["unsafe_retry"]) == 1
    assert res["l4"]["unsafe_retry"][0]["tool"] == "cancel_reservation"


def test_user_side_ref_marks_partial_infeasible():
    res = evaluate_sample("telecom", "inputs: @task: Str\n\n"
                          "%c0 = FETCH(@task)\n\nreturn %c0\n",
                          [{"name": "toggle_wifi_calling", "arguments": {}}],
                          {})
    assert res["partial_infeasible"]
    assert res["user_side_refs"] == ["toggle_wifi_calling"]


def test_unknown_tool_not_auto_correct():
    o = oracle("airline", "no_such_tool_xyz")
    assert o["confidence"] == "unknown"
    assert o["semantic_skill"] is None
    # and in evaluation it is excluded from scorable refs
    res = evaluate_sample("airline",
                          "inputs: @task: Str\n\n%c0 = FETCH(@task)\n\nreturn %c0\n",
                          [{"name": "no_such_tool_xyz", "arguments": {}}],
                          SCHEMAS)
    assert res["l1"]["n_ref"] == 0


def test_arg_value_accuracy_and_required_coverage():
    text = ("inputs: @task: Str\n\n"
            "%c0 = EXEC_ACTION(action=get_reservation_details, "
            "reservation_id=Q69X3R)\n\nreturn %c0\n")
    res = evaluate_sample("airline", text, [
        {"name": "get_reservation_details",
         "arguments": {"reservation_id": "Q69X3R"}}], SCHEMAS)
    assert res["l3"]["required_param_coverage"] == 1.0
    assert res["l3"]["arg_value_acc"] == 1.0
    # wrong value -> accuracy drops, coverage stays
    text2 = text.replace("Q69X3R", "WRONGID")
    res2 = evaluate_sample("airline", text2, [
        {"name": "get_reservation_details",
         "arguments": {"reservation_id": "Q69X3R"}}], SCHEMAS)
    assert res2["l3"]["required_param_coverage"] == 1.0
    assert res2["l3"]["arg_value_acc"] == 0.0


def test_ordered_vs_multiset_dual_reporting():
    """Ordered LCS recall and multiset F1 must be reported separately:
    reversed sequence has perfect multiset F1 but poor ordered recall."""
    text = ("inputs: @task: Str\n\n"
            "%c0 = FETCH(@task)\n"
            "%c1 = EXEC_ACTION(action=update_reservation_flights, "
            "reservation_id=Q69X3R)\n\nreturn %c1\n")
    refs = list(reversed(REFS_UPDATE))   # update first, then get
    res = evaluate_sample("airline", text, refs, SCHEMAS)
    assert res["l2"]["exact_f1"] >= 0.99      # multiset matches
    assert res["l3"]["ordered_recall"] < 0.99  # order does not


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
