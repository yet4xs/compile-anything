"""Phase 8D-0 Task 4: Mock Capability Registry + Runtime Dataflow Executor.

Provides injectable mock tools with input/output schemas and deterministic
executors. The runtime executes a TaskIR v0.2 program in topological order:
  1. EXTRACT_FIELD reads from actual upstream results
  2. param_refs resolve SSA values at execution time
  3. Each tool call validates input params against its schema
  4. Each tool output is validated against its output_schema
  5. Full execution trace records: node, tool, input, output, timing
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.ir.parser import parse_text
from src.ir.taskir import Module
from src.validator.validator import validate


@dataclass
class MockCapability:
    capability_id: str
    input_schema: Dict[str, str]          # param_name -> type
    output_schema: Dict[str, str]         # field_name -> type
    effect_class: str = "read_only"       # read_only | state_update_reversible | ...
    mock_executor: Optional[Callable] = None  # fn(params: dict) -> dict
    call_count: int = 0                   # track repeated executions


class MockCapabilityRegistry:
    """Injectable registry of mock tools for runtime testing."""

    def __init__(self):
        self._caps: Dict[str, MockCapability] = {}

    def register(self, cap: MockCapability):
        self._caps[cap.capability_id] = cap

    def register_simple(self, capability_id: str,
                        input_schema: Dict[str, str],
                        output_schema: Dict[str, str],
                        effect_class: str = "read_only",
                        executor: Optional[Callable] = None):
        """Convenience: register with default or custom executor."""
        if executor is None:
            # default: return values matching output_schema keys
            def default_executor(params):
                return {k: f"{capability_id}_{k}_val" for k in output_schema}
            executor = default_executor
        self.register(MockCapability(
            capability_id, input_schema, output_schema,
            effect_class, executor))

    def get(self, capability_id: str) -> Optional[MockCapability]:
        return self._caps.get(capability_id)

    def execute(self, capability_id: str, params: Dict[str, Any]) -> Dict:
        cap = self._caps.get(capability_id)
        if cap is None:
            raise RuntimeError(f"Unknown capability: {capability_id}")
        cap.call_count += 1
        # validate input params
        for k, expected_type in cap.input_schema.items():
            if k not in params:
                raise RuntimeError(
                    f"{capability_id}: missing required param {k!r}")
        result = cap.mock_executor(params)
        # validate output schema
        for field_name in cap.output_schema:
            if field_name not in result:
                raise RuntimeError(
                    f"{capability_id}: output missing field {field_name!r}")
        return result

    @property
    def capabilities(self):
        return dict(self._caps)


@dataclass
class ExecutionStep:
    node_id: str
    op: str
    tool_id: Optional[str] = None
    input_params: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    duration_ms: float = 0.0


@dataclass
class ExecutionTrace:
    steps: List[ExecutionStep] = field(default_factory=list)
    final_result: Any = None
    success: bool = False
    error: Optional[str] = None


class DataflowRuntime:
    """Executes a TaskIR v0.2 program with real dataflow."""

    def __init__(self, registry: MockCapabilityRegistry):
        self.registry = registry

    def execute(self, module: Module) -> ExecutionTrace:
        trace = ExecutionTrace()
        results: Dict[str, Any] = {}       # SSA id -> value
        # include program inputs
        for inp in module.program.inputs:
            results[inp["name"]] = inp.get("type", "Any")  # placeholder

        try:
            for node in module.program.nodes:
                t0 = time.time()
                step = ExecutionStep(
                    node_id=node.id, op=node.op,
                    tool_id=node.params.get("action")
                    or (node.hints or {}).get("tool"))

                if node.op == "EXTRACT_FIELD":
                    # read from actual upstream result
                    if not node.inputs:
                        raise RuntimeError(
                            f"{node.id}: EXTRACT_FIELD needs 1 input")
                    src_id = node.inputs[0]
                    if src_id not in results:
                        raise RuntimeError(
                            f"{node.id}: source {src_id} not yet executed")
                    field_name = node.params.get("field")
                    src_val = results[src_id]
                    if isinstance(src_val, dict):
                        if field_name not in src_val:
                            raise RuntimeError(
                                f"{node.id}: field {field_name!r} not in "
                                f"{src_id} output {list(src_val.keys())}")
                        value = src_val[field_name]
                    else:
                        value = src_val  # scalar passthrough
                    results[node.id] = value
                    step.input_params = {"field": field_name, "src": src_id}
                    step.output = value

                elif node.op == "EXEC_ACTION":
                    # merge compile-time params with resolved param_refs
                    resolved = dict(node.params)
                    for pk, ref in node.param_refs.items():
                        if ref not in results:
                            raise RuntimeError(
                                f"{node.id}: param_ref {pk}={ref} "
                                f"not yet executed")
                        resolved[pk] = results[ref]
                    step.input_params = {k: v for k, v in resolved.items()
                                         if k != "action"}
                    tool_id = node.params.get("action")
                    if tool_id:
                        result = self.registry.execute(tool_id, resolved)
                        results[node.id] = result
                        step.output = result
                    else:
                        # no tool binding — just pass through
                        results[node.id] = resolved
                        step.output = resolved

                else:
                    # other ops (FETCH, SEARCH, etc.) — use tool from hints
                    tool_id = (node.hints or {}).get("tool")
                    if tool_id:
                        resolved = dict(node.params)
                        for pk, ref in node.param_refs.items():
                            if ref in results:
                                resolved[pk] = results[ref]
                        step.input_params = {k: v for k, v in resolved.items()}
                        result = self.registry.execute(tool_id, resolved)
                        results[node.id] = result
                        step.output = result
                    else:
                        # pass-through node
                        if node.inputs and node.inputs[0] in results:
                            results[node.id] = results[node.inputs[0]]
                            step.output = results[node.id]
                        else:
                            results[node.id] = node.params
                            step.output = node.params

                step.duration_ms = round((time.time() - t0) * 1000, 2)
                trace.steps.append(step)

            # final result
            output_ref = module.program.output
            trace.final_result = results.get(output_ref)
            trace.success = True

        except Exception as e:
            trace.error = str(e)
            trace.success = False

        return trace

    def execute_text(self, taskir_text: str) -> ExecutionTrace:
        """Parse + validate + execute."""
        mod = parse_text(taskir_text)
        rep = validate(mod)
        if not rep.valid:
            trace = ExecutionTrace()
            trace.error = f"validation failed: {rep.errors[:3]}"
            return trace
        return self.execute(mod)
