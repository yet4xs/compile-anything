"""Regression test: canonical_dataflow_matcher return-schema contract.

Phase 9B-2.2 Task 0.2. The server evaluation crash of Phase 9B-2 was caused
by scoring code reading a nonexistent ``n_correct`` key (actual contract:
``n_correct_bindings``). This test pins the matcher's return schema so any
future rename fails here instead of on a GPU server mid-run.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.canonical_dataflow_matcher import (
    canonical_from_skeleton, canonical_from_taskir,
    canonical_call_binding_match, canonical_dag_match)

MATCH_KEYS = {
    "calls_match", "bindings_match",
    "n_ref_calls", "n_pred_calls",
    "n_ref_bindings", "n_pred_bindings",
    "n_correct_bindings", "extra_bindings",
}
DAG_KEYS = {
    "data_exact", "extra_data", "missing_data",
    "data_precision", "data_recall", "data_f1",
}

SKELETON = {
    "task_id": "t",
    "nodes": [
        {"id": "p0", "capability_id": "get_user", "skill": "EXEC_ACTION",
         "depends_on": [], "order_on": [],
         "arguments": {"name": "alice"}},
        {"id": "p1", "capability_id": "get_order", "skill": "EXEC_ACTION",
         "depends_on": ["p0"], "order_on": [],
         "arguments": {"user_id": {"kind": "runtime_ref", "producer": "p0",
                                    "field": "user_id"}}},
    ],
}


class MatcherContractTest(unittest.TestCase):
    def test_match_result_schema(self):
        ref = canonical_from_skeleton(SKELETON["nodes"])
        pred = canonical_from_skeleton(SKELETON["nodes"])
        self.assertTrue(ref and pred)
        a = canonical_call_binding_match(ref, pred)
        self.assertEqual(set(a.keys()), MATCH_KEYS,
                         "canonical_call_binding_match return keys changed")
        # Perfect match must be reported through the pinned fields.
        self.assertTrue(a["calls_match"])
        self.assertTrue(a["bindings_match"])
        self.assertEqual(a["n_ref_bindings"], 1)
        self.assertEqual(a["n_correct_bindings"], 1)
        self.assertEqual(a["extra_bindings"], 0)

    def test_dag_result_schema(self):
        ref = canonical_from_skeleton(SKELETON["nodes"])
        pred = canonical_from_skeleton(SKELETON["nodes"])
        d = canonical_dag_match(ref, pred)
        self.assertEqual(set(d.keys()), DAG_KEYS,
                         "canonical_dag_match return keys changed")
        self.assertTrue(d["data_exact"])

    def test_n_correct_key_is_the_binding_count(self):
        """The exact crash of Phase 9B-2: scoring code must use
        n_correct_bindings, and no n_correct key may exist."""
        ref = canonical_from_skeleton(SKELETON["nodes"])
        pred = canonical_from_skeleton(SKELETON["nodes"])
        a = canonical_call_binding_match(ref, pred)
        self.assertNotIn("n_correct", a)
        self.assertEqual(a["n_correct_bindings"], 1)

    def test_taskir_roundtrip_binding_count(self):
        taskir = (
            "; TaskIR v0.1  module=skeleton_lowered\n"
            "; task: t\ninputs: @task: Str\n\n"
            "%p0 = EXEC_ACTION(action=get_user, name=\"alice\")\n"
            "%ext0 = EXTRACT_FIELD(%p0, field=\"user_id\")\n"
            "%p1 = EXEC_ACTION(action=get_order, %p0)  bind(user_id=%ext0)\n"
            "\nreturn %p1\n"
        )
        pred = canonical_from_taskir(taskir)
        self.assertTrue(pred)
        a = canonical_call_binding_match(canonical_from_skeleton(
            SKELETON["nodes"]), pred)
        self.assertEqual(a["n_correct_bindings"], 1)
        self.assertTrue(a["bindings_match"])


if __name__ == "__main__":
    unittest.main()
