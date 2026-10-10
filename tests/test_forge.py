"""Test Forge: TaskIR → LangGraph execution engine."""
import sys, os
sys.path.insert(0, ".")

from src.ir.taskir import Module, Program, Node
from src.runtime.forge import Forge, PythonExecutor, APIExecutor, compile_and_execute


def make_simple_program():
    """Create: search → filter → sort → calculate"""
    nodes = [
        Node(id="%s0", op="SEARCH", inputs=[{"name": "task", "type": "Str"}],
             params={"query": "flights NYC to LA"}, output_type="List[Any]"),
        Node(id="%f1", op="FILTER", inputs=["%s0"],
             params={"key": "price", "value": None}, output_type="List[Any]"),
        Node(id="%c2", op="CALCULATE", inputs=["%f1"],
             params={"expression": "sum(v0) / len(v0)"}, output_type="Float"),
    ]
    return Module(program=Program(name="test", inputs=[{"name": "task", "type": "Str"}], nodes=nodes))


def make_parallel_program():
    """Create: two independent FETCHes → MERGE"""
    nodes = [
        Node(id="%a", op="FETCH", inputs=[{"name": "task", "type": "Str"}], params={"url": "api/weather"},
             output_type="Json"),
        Node(id="%b", op="FETCH", inputs=[{"name": "task", "type": "Str"}], params={"url": "api/stocks"},
             output_type="Json"),
        Node(id="%m", op="MERGE", inputs=["%a", "%b"], params={},
             output_type="Json"),
    ]
    return Module(program=Program(name="test", inputs=[{"name": "task", "type": "Str"}], nodes=nodes))


def make_pipeline_program():
    """Create: SEARCH → EXTRACT (as TRANSFORM) → SORT → GENERATE (as JOIN)"""
    nodes = [
        Node(id="%search", op="SEARCH", inputs=[{"name": "task", "type": "Str"}],
             params={"query": "tallest buildings"}, output_type="List[Any]"),
        Node(id="%sort", op="SORT", inputs=["%search"],
             params={"key": "height", "descending": True}, output_type="List[Any]"),
        Node(id="%extract", op="FILTER", inputs=["%sort"],
             params={}, output_type="List[Any]"),
        Node(id="%report", op="JOIN", inputs=["%extract"],
             params={"separator": "; "}, output_type="Str"),
    ]
    return Module(program=Program(name="test", inputs=[{"name": "task", "type": "Str"}], nodes=nodes))


def test_forge():
    forge = Forge()

    # Test 1: Simple pipeline
    print("=== Test 1: Simple pipeline ===")
    mod = make_simple_program()
    result = forge.run(mod)
    print(f"  values: {list(result.get('values', {}).keys())}")
    print(f"  traces: {len(result.get('traces', []))} steps")
    print(f"  errors: {result.get('errors', [])}")
    assert not result.get("errors"), f"Unexpected errors: {result['errors']}"
    assert "%s0" in result["values"]
    print("  ✓ PASSED")

    # Test 2: Parallel execution
    print("\n=== Test 2: Parallel branches ===")
    mod2 = make_parallel_program()
    result2 = forge.run(mod2)
    print(f"  values: {list(result2.get('values', {}).keys())}")
    print(f"  traces: {len(result2.get('traces', []))} steps")
    assert not result2.get("errors"), f"Errors: {result2['errors']}"
    assert "%m" in result2["values"]
    print("  ✓ PASSED")

    # Test 3: Full pipeline with cost
    print("\n=== Test 3: Full pipeline ===")
    mod3 = make_pipeline_program()
    result3 = compile_and_execute(mod3)
    print(f"  status: {result3['status']}")
    print(f"  validation: {result3['validation']['valid']}")
    print(f"  trace steps: {len(result3.get('trace', []))}")
    print(f"  cost: {result3.get('cost_estimate', {})}")
    assert result3["status"] == "success"
    print("  ✓ PASSED")

    print("\n=== ALL FORGE TESTS PASSED ===")


if __name__ == "__main__":
    test_forge()
