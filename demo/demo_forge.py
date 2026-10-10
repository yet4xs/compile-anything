"""
Forge 端到端 Demo — 自然语言 → TaskIR → 验证 → 调度 → 执行 → 结果

展示完整的编译器管线：
  1. 构建 TaskIR 程序（模拟神经前端输出）
  2. V1-V6 静态验证
  3. Forge lowering（TaskIR → LangGraph StateGraph）
  4. 执行（含并行分支）
  5. 输出结果 + 执行 trace + 成本估算
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ir.taskir import Module, Program, Node, Guard, Retry
from src.validator.validator import validate
from src.runtime.forge import Forge, PythonExecutor, APIExecutor, ActionExecutor


# ── Demo 1: 数据处理管线（搜索→过滤→计算） ──

def demo_pipeline():
    print("=" * 60)
    print("DEMO 1: 数据处理管线 (SEARCH → FILTER → SUM)")
    print("=" * 60)

    nodes = [
        Node(id="%s0", op="SEARCH", inputs=["@query"],
             params={"query": "stock prices AAPL GOOG MSFT"},
             output_type="List[Any]"),
        Node(id="%f1", op="FILTER", inputs=["%s0"],
             params={"key": "symbol", "value": "AAPL"},
             output_type="List[Any]"),
        Node(id="%c2", op="SUM", inputs=["%f1"],
             params={},
             output_type="Float"),
    ]
    prog = Program(
        name="stock_pipeline",
        inputs=[{"name": "@query", "type": "Str"}],
        nodes=nodes,
        output="%c2",
    )
    mod = Module(program=prog, meta={"description": "Sum AAPL stock prices"})

    # Step 1: Validate
    report = validate(mod)
    print(f"\n[1] Validation: {'✓ PASS' if report.valid else '✗ FAIL'}")

    # Step 2: Forge execution
    forge = Forge()
    result = forge.run(mod, {"@query": "stock prices"})

    print(f"\n[2] Execution Result:")
    print(f"    Final output (%c2) = {result['values'].get('%c2')}")

    print(f"\n[3] Execution Trace:")
    for t in result.get("traces", []):
        status = "✓" if t["status"] == "ok" else t["status"]
        print(f"    {t['node']:8s} {t['op']:10s} {status} ({t['duration_ms']:.1f}ms)")

    print(f"\n[4] Total time: {result.get('total_time_ms', 0):.1f}ms")
    return result


# ── Demo 2: 并行分支（两路 FETCH → MERGE） ──

def demo_parallel():
    print("\n" + "=" * 60)
    print("DEMO 2: 并行分支 (FETCH ∥ FETCH → MERGE)")
    print("=" * 60)

    nodes = [
        # Two independent branches — should run in parallel
        Node(id="%weather", op="FETCH", inputs=["@city"],
             params={"url": "api/weather"},
             output_type="Str"),
        Node(id="%stocks", op="FETCH", inputs=["@city"],
             params={"url": "api/stocks"},
             output_type="Str"),
        # Join point
        Node(id="%merged", op="MERGE", inputs=["%weather", "%stocks"],
             params={},
             output_type="Str"),
    ]
    prog = Program(
        name="parallel_fetch",
        inputs=[{"name": "@city", "type": "Str"}],
        nodes=nodes,
        output="%merged",
    )
    mod = Module(program=prog, meta={"description": "Fetch weather+stocks in parallel"})

    report = validate(mod)
    print(f"\n[1] Validation: {'✓ PASS' if report.valid else '✗ FAIL'}")

    forge = Forge()
    result = forge.run(mod, {"@city": "New York"})

    print(f"\n[2] Execution Result:")
    print(f"    Merged output keys: {list(result['values'].get('%merged', {}).keys())}")

    print(f"\n[3] Execution Trace (note parallel execution):")
    for t in result.get("traces", []):
        print(f"    {t['node']:12s} {t['op']:8s} {t['status']} ({t['duration_ms']:.1f}ms)")

    print(f"\n[4] DAG structure:")
    print("    @city ─┬─ %weather ─┬─ %merged")
    print("           └─ %stocks  ─┘   (parallel)")

    return result


# ── Demo 3: 带 VERIFY + Retry 的容错管线 ──

def demo_verify_retry():
    print("\n" + "=" * 60)
    print("DEMO 3: 容错管线 (SEARCH → VERIFY + RETRY → CALCULATE)")
    print("=" * 60)

    nodes = [
        Node(id="%search", op="SEARCH", inputs=["@query"],
             params={"query": "flight prices"},
             output_type="List[Any]"),
        Node(id="%filter", op="FILTER", inputs=["%search"],
             params={"key": "price"},
             output_type="List[Any]"),
        Node(id="%verify", op="VERIFY", inputs=["%filter"],
             params={},
             output_type="Bool"),
        Node(id="%calc", op="SUM", inputs=["%filter"],
             params={},
             output_type="Float",
             after=["%verify"]),
    ]
    prog = Program(
        name="verified_pipeline",
        inputs=[{"name": "@query", "type": "Str"}],
        nodes=nodes,
        output="%calc",
    )
    mod = Module(program=prog)

    report = validate(mod)
    print(f"\n[1] Validation: {'✓ PASS' if report.valid else '✗ FAIL'}")
    if not report.valid:
        for e in report.errors:
            print(f"    E: {e.msg}")

    forge = Forge()
    result = forge.run(mod, {"@query": "flights"})

    print(f"\n[2] Execution Result:")
    print(f"    Verified price (%calc) = {result['values'].get('%calc')}")

    print(f"\n[3] Execution Trace:")
    for t in result.get("traces", []):
        print(f"    {t['node']:10s} {t['op']:10s} {t['status']} ({t['duration_ms']:.1f}ms)")

    return result


# ── Demo 4: 完整管线 (compile_and_execute) ──

def demo_full_pipeline():
    print("\n" + "=" * 60)
    print("DEMO 4: 完整编译器管线 (compile → validate → execute → report)")
    print("=" * 60)

    nodes = [
        Node(id="%search", op="SEARCH", inputs=["@topic"],
             params={"query": "renewable energy statistics"},
             output_type="List[Any]"),
        Node(id="%filter", op="FILTER", inputs=["%search"],
             params={"key": "year"},
             output_type="List[Any]"),
        Node(id="%sort", op="SORT", inputs=["%filter"],
             params={"key": "value", "descending": True},
             output_type="List[Any]"),
        Node(id="%count", op="COUNT", inputs=["%sort"],
             params={},
             output_type="Int"),
    ]
    prog = Program(
        name="energy_analysis",
        inputs=[{"name": "@topic", "type": "Str"}],
        nodes=nodes,
        output="%count",
    )
    mod = Module(program=prog)

    from src.runtime.forge import compile_and_execute
    result = compile_and_execute(mod)

    print(f"\n  Status: {result['status']}")
    print(f"  Validation: {result['validation']['valid']}")
    print(f"  Output (%count): {result['output'].get('%count')}")
    print(f"  Trace steps: {len(result['trace'])}")
    print(f"  Cost estimate: {result.get('cost_estimate', {})}")
    print(f"  Total time: {result.get('total_time_ms', 0):.1f}ms")

    print(f"\n  Trace:")
    for t in result["trace"]:
        print(f"    {t['node']:10s} {t['op']:10s} {t['status']}")

    return result


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Forge: TaskIR → LangGraph Execution Engine Demo       ║")
    print("║   Compile Anything — Complete Compiler Pipeline          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_pipeline()
    demo_parallel()
    demo_verify_retry()
    demo_full_pipeline()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
