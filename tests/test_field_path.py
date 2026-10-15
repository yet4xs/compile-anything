"""Phase 9B-2.2 Task 4: typed nested access — positive and negative tests.

The compiler must (a) accept and type-check genuine nested dynamic access
(tau3_retail_11's orders[0].order_id) and (b) REJECT wrong-source fields
(tau3_retail_48's find_user_id_by_name_zip.default_address), wrong types,
nonexistent fields, and out-of-range indices. List[Any] must surface as a
declared dynamic boundary, never a fake static proof.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.field_path import (
    Field, Index, ListT, StructT, PathTypeError,
    parse_path, check_path, resolve_path)


# Real tau3 retail shapes (from the official replay, Phase 9B-2.2 Task 3):
# get_user_details -> User(user_id: Str, orders: List[Order], address: ...)
# Order(order_id: Str, items: List[Item], status: Str)
ORDER = StructT("Order", {"order_id": "Str", "items": "List", "status": "Str"})
USER = StructT("User", {"user_id": "Str", "orders": ListT(ORDER),
                        "address": "Str"})
# find_user_id_by_name_zip returns a bare scalar in the real environment
FIND_USER = "Str"
# handwritten-registry-style untyped list
USER_ANY = StructT("User", {"user_id": "Str", "orders": ListT("Any")})


class Positive(unittest.TestCase):
    def test_parse_nested_path(self):
        self.assertEqual(parse_path("orders[0].order_id"),
                         [Field("orders"), Index(0), Field("order_id")])

    def test_genuine_nested_access_static(self):
        cp = check_path(USER, parse_path("orders[0].order_id"))
        self.assertEqual(cp.check, "static")
        self.assertEqual(cp.result_type, "Str")

    def test_top_level_field(self):
        cp = check_path(USER, parse_path("user_id"))
        self.assertEqual((cp.check, cp.result_type), ("static", "Str"))

    def test_list_any_is_declared_dynamic(self):
        cp = check_path(USER_ANY, parse_path("orders[0].order_id"))
        self.assertEqual(cp.check, "dynamic")
        self.assertEqual(cp.dynamic_from_step, 2)  # Field('order_id') unknown
        self.assertEqual(cp.result_type, "Any")

    def test_runtime_resolution(self):
        value = {"orders": [{"order_id": "#W1", "status": "delivered"}]}
        self.assertEqual(
            resolve_path(value, parse_path("orders[0].order_id")), "#W1")


class Negative(unittest.TestCase):
    def test_wrong_source_field_is_rejected(self):
        """tau3_retail_48: find_user_id_by_name_zip has no default_address —
        the producer is a bare Str; relaxing the IR must NOT let this pass."""
        with self.assertRaises(PathTypeError) as cm:
            check_path(FIND_USER, parse_path("default_address"))
        self.assertEqual(cm.exception.code, "FIELD_ON_PRIMITIVE")

    def test_nonexistent_field_in_struct(self):
        with self.assertRaises(PathTypeError) as cm:
            check_path(USER, parse_path("membership_level"))
        self.assertEqual(cm.exception.code, "FIELD_NOT_FOUND")

    def test_field_on_primitive_rejected(self):
        with self.assertRaises(PathTypeError) as cm:
            check_path("Str", parse_path("anything"))
        self.assertEqual(cm.exception.code, "FIELD_ON_PRIMITIVE")

    def test_index_on_non_list(self):
        with self.assertRaises(PathTypeError) as cm:
            check_path(USER, parse_path("user_id[0]"))
        self.assertEqual(cm.exception.code, "INDEX_ON_NON_LIST")

    def test_nested_nonexistent_field(self):
        with self.assertRaises(PathTypeError) as cm:
            check_path(USER, parse_path("orders[0].tracking_number"))
        self.assertEqual(cm.exception.code, "FIELD_NOT_FOUND")

    def test_malformed_paths(self):
        for bad in ("orders[[0]", "orders[0", "orders[x]", ".orders",
                    "orders..id", "orders[0].", ""):
            with self.assertRaises(PathTypeError, msg=bad):
                parse_path(bad)

    def test_runtime_index_out_of_range(self):
        with self.assertRaises(PathTypeError) as cm:
            resolve_path({"orders": []}, parse_path("orders[0].order_id"))
        self.assertEqual(cm.exception.code, "INDEX_OUT_OF_RANGE")

    def test_runtime_field_missing_on_real_value(self):
        with self.assertRaises(PathTypeError) as cm:
            resolve_path({"orders": [{"status": "x"}]},
                         parse_path("orders[0].order_id"))
        self.assertEqual(cm.exception.code, "FIELD_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
