import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ir.taskir import to_text
from src.isa.registry import ALL_OPS
from src.lifter.toolmap import map_tool
from src.lifter.xlam import lift_record, parse_record
from src.validator.validator import validate

REC = {
    "task_id": "t-1",
    "question": "Find flights from SFO to Tokyo and convert the price to JPY.",
    "steps": [
        "Looking up flights. ```json\n"
        '[{"name": "FlightSearch.search", "arguments": '
        '{"origin": "SFO", "destination": "Tokyo", "date": "2026-10-05"}}]\n```',
        "Now converting. ```json\n"
        '[{"name": "CurrencyConverter.convert", "arguments": '
        '{"amount": 540, "from_currency": "USD", "to_currency": "JPY"}}]\n```',
    ],
    "answer": "The cheapest flight is 78,200 JPY.",
}


class TestLifter(unittest.TestCase):
    def test_parse_steps(self):
        parsed = parse_record(REC)
        self.assertEqual(parsed["task_id"], "t-1")
        self.assertEqual(len(parsed["calls"]), 2)
        self.assertEqual(parsed["calls"][0]["name"], "FlightSearch.search")
        self.assertEqual(parsed["calls"][1]["arguments"]["to_currency"], "JPY")

    def test_semantic_only_ops(self):
        mod = lift_record(REC)
        ops = [n.op for n in mod.program.nodes]
        for op in ops:
            self.assertIn(op, ALL_OPS)
            self.assertNotIn(".", op)                # no raw tool names
        self.assertIn("SEARCH", ops)
        self.assertIn("CONVERT", ops)

    def test_provenance_keeps_tool_names(self):
        mod = lift_record(REC)
        prov = mod.meta["provenance"]
        self.assertEqual(prov["source"], "xlam")
        self.assertIn("FlightSearch.search", prov["tools"])
        # tool names must not appear in the semantic node ops
        for n in mod.program.nodes:
            self.assertNotIn("FlightSearch", n.op)
            self.assertNotIn(".", n.op)

    def test_bridge_inserted_for_type_mismatch(self):
        mod = lift_record(REC)
        ops = [n.op for n in mod.program.nodes]
        # SEARCH(List[Flight]) -> CONVERT(Float) 需要 EXTRACT 桥接
        self.assertLess(ops.index("EXTRACT"), ops.index("CONVERT"))

    def test_lifted_program_valid(self):
        rep = validate(lift_record(REC))
        self.assertTrue(rep.valid, rep.summary())

    def test_toolmap_fallback(self):
        out = map_tool("SomeVendor.do_random_thing", {"x": 1})
        self.assertEqual(out["skill"], "EXEC_ACTION")
        self.assertEqual(out["params"]["action"], "do random thing")

    def test_toolmap_domains(self):
        self.assertEqual(map_tool("WeatherAPI.forecast", {"city": "Rome"})["skill"],
                         "SEARCH")
        self.assertEqual(map_tool("StockQuote.get_price", {"symbol": "AAPL"})["skill"],
                         "QUERY_DB")
        self.assertEqual(map_tool("EmailClient.send", {"to": "a@b.c"})["skill"],
                         "SEND")


if __name__ == "__main__":
    unittest.main()
