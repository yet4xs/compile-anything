"""Phase 9B-2.3 Task 3: gold binding scorer — bijection, instance
distinction, reorder tolerance, root-scalar equivalence."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.gold_binding_scorer import (
    score_prediction, calls_from_skeleton_nodes,
    bindings_from_skeleton_nodes, gold_bindings_from_program)

# Reconstructed-gold-style program: find_user -> get_order(order_id ref)
# -> exchange(item_ids multi-ref, payment ref)
GOLD_PROGRAM = [
    {"step": 0, "tool": "find_user_id_by_name_zip",
     "arguments": {"first_name": {"class": "LITERAL", "value": "A"},
                   "last_name": {"class": "LITERAL", "value": "B"},
                   "zip": {"class": "LITERAL", "value": "19122"}}},
    {"step": 1, "tool": "get_user_details",
     "arguments": {"user_id": {"class": "runtime_ref", "producer_step": 0,
                                "path": ""}}},
    {"step": 2, "tool": "get_order_details",
     "arguments": {"order_id": {"class": "runtime_ref", "producer_step": 1,
                                 "path": "orders[0]"}}},
    {"step": 3, "tool": "exchange_delivered_order_items",
     "arguments": {"order_id": {"class": "runtime_ref",
                                 "producer_step": 2, "path": "order_id"},
                    "item_ids": {"class": "runtime_ref_multi",
                                  "elements": [
                                      {"producer_step": 2,
                                       "path": "items[0].item_id"},
                                      {"producer_step": 2,
                                       "path": "items[1].item_id"}]},
                    "payment_method_id": {"class": "runtime_ref",
                                           "producer_step": 2,
                                           "path":
                                           "payment_history[0].payment_method_id"}}},
]


class ScorerTest(unittest.TestCase):
    def test_perfect_match_with_reordering(self):
        """Prediction in a different node order still matches: positions
        never enter the comparison, only the literal-signature bijection."""
        pred_nodes = [  # reversed order of the same program, as skeleton
            {"id": "q0", "capability_id": "exchange_delivered_order_items",
             "arguments": {
                 "order_id": {"kind": "runtime_ref", "producer": "q2",
                               "field": "order_id"}}},
            {"id": "q1", "capability_id": "find_user_id_by_name_zip",
             "arguments": {"first_name": "A", "last_name": "B",
                            "zip": "19122"}},
            {"id": "q2", "capability_id": "get_order_details",
             "arguments": {"order_id": {"kind": "runtime_ref",
                                          "producer": "q3",
                                          "field": "orders[0]"}}},
            {"id": "q3", "capability_id": "get_user_details",
             "arguments": {"user_id": {"kind": "runtime_ref",
                                          "producer": "q1",
                                          "field": "user_id"}}},
        ]
        # note: the exchange node's item_ids/payment params are absent in
        # the prediction (the skeleton format cannot express multi-refs);
        # the literal-signature bijection therefore matches it to the gold
        # exchange call, whose sourced params contribute no literals either
        calls = calls_from_skeleton_nodes(pred_nodes)
        bindings = bindings_from_skeleton_nodes(pred_nodes)
        r = score_prediction(calls, bindings, GOLD_PROGRAM)
        self.assertTrue(r["calls_bijection_complete"], r)
        self.assertEqual(r["n_bindings_exact"], 2)
        self.assertEqual(r["n_bindings_root_scalar_equiv"], 1)

    def test_wrong_producer_not_matched(self):
        pred_nodes = [
            {"id": "q0", "capability_id": "find_user_id_by_name_zip",
             "arguments": {"first_name": "A", "last_name": "B",
                            "zip": "19122"}},
            {"id": "q1", "capability_id": "get_user_details",
             "arguments": {"user_id": {"kind": "runtime_ref",
                                          "producer": "q0",
                                          "field": "user_id"}}},
            {"id": "q2", "capability_id": "get_order_details",
             "arguments": {"order_id": {"kind": "runtime_ref",
                                          "producer": "q0",
                                          "field": "orders[0]"}}},
        ]
        calls = calls_from_skeleton_nodes(pred_nodes)
        bindings = bindings_from_skeleton_nodes(pred_nodes)
        r = score_prediction(calls, bindings, GOLD_PROGRAM)
        self.assertFalse(r["strict_binding_exact"])
        self.assertEqual(r["n_bindings_exact"], 0)
        self.assertEqual(r["n_bindings_root_scalar_equiv"], 1)

    def test_same_tool_instances_distinguished_by_literals(self):
        """Two get_order_details with different literal args must each
        map to their own gold instance through the bijection."""
        gold = [
            {"step": 0, "tool": "get_order_details",
             "arguments": {"order_id": {"class": "LITERAL",
                                         "value": "#W1"}}},
            {"step": 1, "tool": "get_order_details",
             "arguments": {"order_id": {"class": "LITERAL",
                                         "value": "#W2"}}},
            {"step": 2, "tool": "return_delivered_order_items",
             "arguments": {"order_id": {"class": "runtime_ref",
                                         "producer_step": 0,
                                         "path": "order_id"}}},
        ]
        pred_calls = [("get_order_details", {"order_id": "#W2"}),
                      ("get_order_details", {"order_id": "#W1"}),
                      ("return_delivered_order_items", {})]
        pred_bindings = [(2, "order_id", 1, "order_id")]
        r = score_prediction(pred_calls, pred_bindings, gold)
        self.assertTrue(r["calls_bijection_complete"])
        self.assertEqual(r["n_bindings_exact"], 1)
        self.assertTrue(r["strict_binding_exact"], r)

    def test_unsourced_gold_params_reported_not_passed(self):
        gold = [{"step": 0, "tool": "t",
                 "arguments": {"x": {"class": "AMBIGUOUS", "value": 1}}}]
        r = score_prediction([("t", {})], [], gold)
        self.assertEqual(r["gold_unsourced_params"], 1)
        self.assertFalse(r["strict_binding_exact"])

    def test_root_scalar_requires_top_level_path(self):
        """A root-scalar gold path never matches a nested pred path."""
        gold = [
            {"step": 0, "tool": "find", "arguments": {}},
            {"step": 1, "tool": "use",
             "arguments": {"uid": {"class": "runtime_ref",
                                    "producer_step": 0, "path": ""}}},
        ]
        r = score_prediction(
            [("find", {}), ("use", {})],
            [(1, "uid", 0, "orders[0].user_id")], gold)
        self.assertEqual(r["n_bindings_exact"], 0)
        self.assertEqual(r["n_bindings_root_scalar_equiv"], 0)

    def test_gold_bindings_multi_expansion(self):
        b = gold_bindings_from_program(GOLD_PROGRAM)
        names = sorted((c, p) for c, p, _, _ in b)
        self.assertIn((3, "item_ids[0]"), names)
        self.assertIn((3, "item_ids[1]"), names)
        self.assertEqual(len(b), 6)

    def test_equivalent_instances_resolved_by_bindings(self):
        """Two calls with IDENTICAL literal signatures, distinguished
        only by their dataflow: the bijection must pick the mapping that
        satisfies the bindings (greedy could pick the swapped one)."""
        gold = [
            {"step": 0, "tool": "find", "arguments": {}},
            {"step": 1, "tool": "get", "arguments":
                {"uid": {"class": "runtime_ref", "producer_step": 0,
                          "path": ""}}},
            {"step": 2, "tool": "get", "arguments":
                {"uid": {"class": "runtime_ref", "producer_step": 0,
                          "path": ""}}},
            # two identical write calls, each bound to a DIFFERENT get
            {"step": 3, "tool": "use", "arguments":
                {"x": {"class": "runtime_ref", "producer_step": 1,
                        "path": "v"}}},
            {"step": 4, "tool": "use", "arguments":
                {"x": {"class": "runtime_ref", "producer_step": 2,
                        "path": "v"}}},
        ]
        # prediction swaps the two use-calls relative to gold order
        pred_calls = [("find", {}), ("get", {}), ("get", {}),
                      ("use", {}), ("use", {})]
        # pred use#3 binds to pred get#2; pred use#4 binds to pred get#1
        # (reversed vs gold's step order) — legal reorder; the get.uid
        # root-scalar bindings are present too (path "" both sides)
        pred_bindings = [(1, "uid", 0, ""), (2, "uid", 0, ""),
                         (3, "x", 2, "v"), (4, "x", 1, "v")]
        r = score_prediction(pred_calls, pred_bindings, gold)
        self.assertEqual(r["bijection_method"], "backtracking", r)
        self.assertTrue(r["value_match"], r)
        self.assertTrue(r["exact_binding_match"], r)
        self.assertTrue(r["strict_binding_exact"], r)

    def test_gold_edges_consumed_one_to_one(self):
        """Two pred bindings cannot both claim the same gold edge."""
        gold = [
            {"step": 0, "tool": "find", "arguments": {}},
            {"step": 1, "tool": "use", "arguments":
                {"x": {"class": "runtime_ref", "producer_step": 0,
                        "path": ""}}},
        ]
        r = score_prediction(
            [("find", {}), ("use", {}), ("use", {})],
            [(1, "x", 0, "any"), (2, "x", 0, "any")], gold)
        # only one pred binding can consume the single gold edge
        self.assertEqual(r["n_bindings_exact"]
                         + r["n_bindings_root_scalar_equiv"], 1)
        self.assertFalse(r["strict_binding_exact"])


if __name__ == "__main__":
    unittest.main()
