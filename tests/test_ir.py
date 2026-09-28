import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ir.taskir import (Module, Node, Program, module_to_dict,
                           module_from_dict, to_text)
from src.ir import types as ty


def tiny_module() -> Module:
    return Module(
        program=Program(
            name="t", description="d",
            inputs=[{"name": "@task", "type": "Str"}],
            nodes=[
                Node(id="%1", op="SEARCH", inputs=["@task"],
                     params={"domain": "flight"}, output_type="List[Flight]"),
                Node(id="%2", op="FILTER", inputs=["%1"],
                     params={"predicate": "p"}),
            ],
            output="%2"),
        meta={"name": "t"},
    )


class TestIR(unittest.TestCase):
    def test_json_roundtrip(self):
        m = tiny_module()
        m2 = module_from_dict(module_to_dict(m))
        self.assertEqual(to_text(m), to_text(m2))
        self.assertEqual(m2.program.output, "%2")

    def test_type_compat(self):
        self.assertTrue(ty.is_compatible("List[Any]", "List[Flight]"))
        self.assertTrue(ty.is_compatible("Any", "Flight"))
        self.assertTrue(ty.is_compatible("Flight", "Flight"))
        self.assertFalse(ty.is_compatible("Flight", "Hotel"))
        self.assertFalse(ty.is_compatible("List[Flight]", "List[Hotel]"))
        self.assertTrue(ty.is_compatible("Map[Str, Any]", "Map[Str, Float]"))
        self.assertTrue(ty.is_compatible("Str", "Any"))

    def test_item_of(self):
        self.assertEqual(ty.item_of("List[Flight]"), "Flight")
        self.assertEqual(ty.item_of("Flight"), "Any")


if __name__ == "__main__":
    unittest.main()
