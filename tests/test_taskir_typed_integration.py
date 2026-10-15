"""Phase 9B-2.4 Task 4: typed nested access — production integration tests.

Covers: skeleton runtime_ref with nested path lowers to a path-carrying
EXTRACT_FIELD; runtime_ref_multi lowers to EXTRACT_FIELD chain + MAKE_LIST;
validator checks paths against typed descriptors (static / dynamic-Any /
refuted); old flat schemas stay compatible; parse->validate roundtrip.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ir.parser import parse_text
from src.compiler.plan_skeleton import skeleton_from_dict
from src.compiler.skeleton_lowering import lower_skeleton
from src.validator.validator import validate

# Typed (verified-registry-style) schemas
TYPED = {
    "get_user_details": {
        "input_schema": {"user_id": "Str"},
        "output_schema": {"kind": "struct", "fields": {
            "user_id": {"kind": "scalar", "type": "Str"},
            "orders": {"kind": "list", "elem": {"kind": "struct", "fields": {
                "order_id": {"kind": "scalar", "type": "Str"},
                "status": {"kind": "scalar", "type": "Str"}}}},
            "payment_methods": {"kind": "list", "elem": {
                "kind": "scalar", "type": "Any"}}}},
    },
    "get_order_details": {
        "input_schema": {"order_id": "Str"},
        "output_schema": {"kind": "struct", "fields": {
            "order_id": {"kind": "scalar", "type": "Str"},
            "items": {"kind": "list", "elem": {"kind": "struct", "fields": {
                "item_id": {"kind": "scalar", "type": "Str"}}}}}},
    },
    "find_user": {
        "input_schema": {"first_name": "Str", "last_name": "Str",
                          "zip": "Str"},
        "output_schema": {"kind": "scalar_root"},
    },
    "exchange": {
        "input_schema": {"order_id": "Str", "item_ids": "List[Any]",
                          "payment_method_id": "Str"},
        "output_schema": {"kind": "struct", "fields": {
            "order_id": {"kind": "scalar", "type": "Str"}}},
    },
}
FLAT = {t: {"input_schema": s["input_schema"],
            "output_schema": {"user_id": "Str", "order_id": "Str"}}
        for t, s in TYPED.items()}


def lower(nodes):
    sk = skeleton_from_dict({"task_id": "t", "nodes": nodes})
    return lower_skeleton(sk, None, TYPED, strict=True)


NODES_NESTED = [
    {"id": "p0", "capability_id": "find_user", "skill": "EXEC_ACTION",
     "depends_on": [], "order_on": [],
     "arguments": {"first_name": "A", "last_name": "B", "zip": "19122"}},
    {"id": "p1", "capability_id": "get_user_details", "skill": "EXEC_ACTION",
     "depends_on": ["p0"], "order_on": [],
     "arguments": {"user_id": {"kind": "runtime_ref", "producer": "p0",
                                "field": "user_id"}}},
    {"id": "p2", "capability_id": "get_order_details", "skill": "EXEC_ACTION",
     "depends_on": ["p1"], "order_on": [],
     "arguments": {"order_id": {"kind": "runtime_ref", "producer": "p1",
                                 "field": "orders[0].order_id"}}},
]

NODES_MULTI = NODES_NESTED + [
    {"id": "p3", "capability_id": "exchange", "skill": "EXEC_ACTION",
     "depends_on": ["p2"], "order_on": [],
     "arguments": {"order_id": {"kind": "runtime_ref", "producer": "p2",
                                 "field": "order_id"},
                    "item_ids": {"kind": "runtime_ref_multi",
                                  "elements": [
                                      {"producer": "p2",
                                       "field": "items[0].item_id"},
                                      {"producer": "p2",
                                       "field": "items[1].item_id"}]},
                    "payment_method_id": {
                        "kind": "runtime_ref", "producer": "p1",
                        "field": "payment_methods[0]"}}},
]


class LoweringTest(unittest.TestCase):
    def test_nested_path_lowers_and_validates(self):
        low = lower(NODES_NESTED)
        self.assertTrue(low.get("taskir"), low.get("errors"))
        self.assertIn('field="orders[0].order_id"', low["taskir"])
        mod = parse_text(low["taskir"])
        rep = validate(mod, capability_registry=TYPED)
        self.assertTrue(rep.valid, [i.msg for i in rep.errors])

    def test_multi_ref_lowers_to_make_list(self):
        low = lower(NODES_MULTI)
        self.assertTrue(low.get("taskir"), low.get("errors"))
        self.assertIn("MAKE_LIST(", low["taskir"])
        mod = parse_text(low["taskir"])
        rep = validate(mod, capability_registry=TYPED)
        self.assertTrue(rep.valid, [i.msg for i in rep.errors])
        # the bound param references the MAKE_LIST result
        ex = [n for n in mod.program.nodes if n.op == "EXEC_ACTION"
              and n.params.get("action") == "exchange"][0]
        self.assertIn("item_ids", ex.param_refs)

    def test_roundtrip_print_parse(self):
        low = lower(NODES_MULTI)
        mod = parse_text(low["taskir"])
        from src.ir.taskir import to_text
        mod2 = parse_text(to_text(mod))
        self.assertEqual(len(mod2.program.nodes), len(mod.program.nodes))


class ValidatorTypedTest(unittest.TestCase):
    def lower_taskir(self, nodes):
        low = lower(nodes)
        self.assertTrue(low.get("taskir"), low.get("errors"))
        return parse_text(low["taskir"])

    def test_wrong_root_field_refuted(self):
        nodes = [NODES_NESTED[0],
                 {"id": "p1", "capability_id": "get_user_details",
                  "skill": "EXEC_ACTION", "depends_on": ["p0"],
                  "order_on": [],
                  "arguments": {"user_id": {
                      "kind": "runtime_ref", "producer": "p0",
                      "field": "user_id"}}},
                 {"id": "p2", "capability_id": "get_order_details",
                  "skill": "EXEC_ACTION", "depends_on": ["p1"],
                  "order_on": [],
                  "arguments": {"order_id": {
                      "kind": "runtime_ref", "producer": "p1",
                      "field": "membership_level"}}}]
        sk = skeleton_from_dict({"task_id": "t", "nodes": nodes})
        low = lower_skeleton(sk, None, TYPED, strict=True)
        # fail-closed AT COMPILE TIME: the wrong field never reaches IR
        self.assertIsNone(low["taskir"])
        self.assertTrue(any("FIELD_NOT_FOUND" in e for e in low["errors"]),
                        low["errors"])

    def test_any_list_boundary_is_warning_not_error(self):
        nodes = [NODES_NESTED[0],
                 {"id": "p1", "capability_id": "get_user_details",
                  "skill": "EXEC_ACTION", "depends_on": ["p0"],
                  "order_on": [],
                  "arguments": {"user_id": {
                      "kind": "runtime_ref", "producer": "p0",
                      "field": "user_id"}}},
                 {"id": "p2", "capability_id": "get_order_details",
                  "skill": "EXEC_ACTION", "depends_on": ["p1"],
                  "order_on": [],
                  "arguments": {"order_id": {
                      "kind": "runtime_ref", "producer": "p1",
                      "field": "payment_methods[0].card_number"}}}]
        rep = validate(self.lower_taskir(nodes), capability_registry=TYPED)
        self.assertTrue(rep.valid, [i.msg for i in rep.errors])
        self.assertTrue(any(i.code == "DYNAMIC_CHECK_BOUNDARY"
                            for i in rep.warnings))

    def test_malformed_path_rejected(self):
        taskir = (
            "; TaskIR v0.1\ninputs: @task: Str\n\n"
            "%p0 = EXEC_ACTION(action=find_user, first_name=\"A\")\n"
            "%ext0 = EXTRACT_FIELD(%p0, field=\"orders[0]id\")\n"
            "%p1 = EXEC_ACTION(action=get_order_details, %p0)  "
            "bind(order_id=%ext0)\n\nreturn %p1\n")
        rep = validate(parse_text(taskir), capability_registry=TYPED)
        self.assertFalse(rep.valid)
        self.assertTrue(any(i.code == "FIELD_PATH_SYNTAX"
                            for i in rep.errors))

    def test_flat_schema_backward_compatible(self):
        low = lower([NODES_NESTED[0],
                     {"id": "p1", "capability_id": "get_user_details",
                      "skill": "EXEC_ACTION", "depends_on": ["p0"],
                      "order_on": [],
                      "arguments": {"user_id": {
                          "kind": "runtime_ref", "producer": "p0",
                          "field": "user_id"}}}])
        rep = validate(parse_text(low["taskir"]),
                       capability_registry=FLAT)
        self.assertTrue(rep.valid, [i.msg for i in rep.errors])


if __name__ == "__main__":
    unittest.main()
