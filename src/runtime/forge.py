"""
Forge — TaskIR 到 LangGraph 的执行引擎（编译器后端）

命名: Forge（锻造）——将 TaskIR 锻造为可执行的计算图。
类比: LLVM 的 codegen 阶段（IR → 目标平台机器码）。

架构:
  TaskIR (SSA IR) → Forge.lower() → LangGraph StateGraph → .invoke() → 执行结果

每个 TaskIR 节点被降级为一个 LangGraph 节点函数，
数据依赖通过共享 state dict 传递（对应 SSA value），
控制依赖通过 conditional edge 表达（对应 after/guard）。
"""
from __future__ import annotations
import json, time, copy, logging
from typing import Any, Dict, List, Optional, Callable, Annotated
from dataclasses import dataclass, field
from operator import add as list_add

from langgraph.graph import StateGraph, END


def merge_dicts(a: Dict, b: Dict) -> Dict:
    """Reducer: merge dict updates from parallel nodes."""
    a.update(b)
    return a


class GraphState(dict):
    """LangGraph state schema with reducers for parallel node updates."""
    pass

from src.ir.taskir import Module, Program, Node
from src.validator.validator import validate


# ── Execution State (shared across all nodes, like SSA memory) ──

@dataclass
class ForgeState:
    """Shared state passed through the graph — analogous to SSA value store."""
    values: Dict[str, Any] = field(default_factory=dict)    # SSA %id → value
    traces: List[Dict] = field(default_factory=list)         # execution trace
    errors: List[str] = field(default_factory=list)          # collected errors
    start_time: float = field(default_factory=time.time)


# ── Executors (the "ISAs" that Forge targets) ──

class Executor:
    """Base executor for a Skill ISA resource class."""
    resource_class: str = "base"

    def execute(self, op: str, params: Dict, inputs: List[Any]) -> Any:
        raise NotImplementedError


class PythonExecutor(Executor):
    """Executes deterministic computation skills using Python."""
    resource_class = "python"

    def execute(self, op: str, params: Dict, inputs: List[Any]) -> Any:
        data = inputs[0] if inputs else params.get("data")

        if op == "CALCULATE":
            return self._safe_calc(params.get("expression", ""), inputs)
        elif op == "CONVERT":
            return self._convert(data, params)
        elif op == "FILTER":
            return self._filter(data, params)
        elif op == "SORT":
            return self._sort(data, params)
        elif op == "TRANSFORM":
            return self._transform(data, params)
        elif op == "DEDUP":
            return list(dict.fromkeys(data)) if isinstance(data, list) else data
        elif op in ("MIN", "MAX", "SUM", "AVG", "COUNT"):
            return self._aggregate(op, data)
        elif op in ("ARGMIN", "ARGMAX"):
            return self._argext(op, data, params)
        elif op == "COMPARE":
            return self._compare(inputs, params)
        elif op == "JOIN":
            return self._join(inputs, params)
        elif op == "MERGE":
            return self._merge(inputs)
        else:
            return self._generic(op, params, inputs)

    def _safe_calc(self, expr: str, inputs: List) -> Any:
        """Safe arithmetic expression evaluation."""
        import asteval
        ae = asteval.Interpreter()
        if inputs:
            for i, v in enumerate(inputs):
                ae.symtable[f"v{i}"] = v
        return ae.eval(expr)

    def _convert(self, data, params):
        rate = params.get("rate", 1.0)
        return data * rate

    def _filter(self, data, params):
        if not isinstance(data, list):
            return data
        key = params.get("key")
        value = params.get("value")
        if key and isinstance(data[0], dict):
            return [x for x in data if x.get(key) == value]
        return [x for x in data if x is not None]

    def _sort(self, data, params):
        if not isinstance(data, list):
            return data
        key = params.get("key")
        reverse = params.get("descending", False)
        if key and data and isinstance(data[0], dict):
            return sorted(data, key=lambda x: x.get(key, 0), reverse=reverse)
        return sorted(data, reverse=reverse)

    def _transform(self, data, params):
        if isinstance(data, list):
            return [self._apply_map(x, params) for x in data]
        return self._apply_map(data, params)

    def _apply_map(self, item, params):
        if isinstance(item, dict):
            return {k: params.get("default", v) if k in params.get("overwrite", [])
                    else v for k, v in item.items()}
        return item

    def _aggregate(self, op, data):
        if not isinstance(data, list) or not data:
            return 0
        if op == "MIN": return min(data)
        if op == "MAX": return max(data)
        if op == "SUM": return sum(data)
        if op == "AVG": return sum(data) / len(data)
        if op == "COUNT": return len(data)

    def _argext(self, op, data, params):
        if not isinstance(data, list) or not data:
            return None
        key = params.get("key")
        if key and isinstance(data[0], dict):
            rev = (op == "ARGMAX")
            return sorted(data, key=lambda x: x.get(key, 0), reverse=rev)[0]
        return max(data) if op == "ARGMAX" else min(data)

    def _compare(self, inputs, params):
        a, b = (inputs + [None, None])[:2]
        op = params.get("operator", "==")
        return {"==": a == b, "!=": a != b, "<": a < b, ">": a > b,
                "<=": a <= b, ">=": a >= b}.get(op, False)

    def _join(self, inputs, params):
        sep = params.get("separator", " ")
        return sep.join(str(x) for x in inputs if x is not None)

    def _merge(self, inputs):
        result = {}
        for inp in inputs:
            if isinstance(inp, dict):
                result.update(inp)
            elif isinstance(inp, list):
                result.setdefault("items", []).extend(inp)
        return result

    def _generic(self, op, params, inputs):
        return {"op": op, "params": params, "result": "executed"}


class LLMExecutor(Executor):
    """Executes language model skills using a local model."""
    resource_class = "lm"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto")

    def execute(self, op: str, params: Dict, inputs: List[Any]) -> Any:
        self._load()
        prompt = self._build_prompt(op, params, inputs)
        inputs_enc = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with __import__("torch").no_grad():
            output = self._model.generate(
                **inputs_enc, max_new_tokens=512,
                do_sample=False, temperature=None,
                pad_token_id=self._tokenizer.pad_token_id)
        return self._tokenizer.decode(
            output[0][inputs_enc["input_ids"].shape[1]:],
            skip_special_tokens=True)

    def _build_prompt(self, op, params, inputs):
        context = "\n".join(str(x) for x in inputs if x is not None)
        if op == "GENERATE":
            return params.get("prompt", context)
        elif op == "SUMMARIZE":
            return f"Summarize the following:\n\n{context}"
        elif op == "TRANSLATE":
            lang = params.get("target_lang", "English")
            return f"Translate to {lang}:\n\n{context}"
        elif op == "CLASSIFY":
            return f"Classify the following:\n\n{context}"
        elif op == "EXTRACT_ENTITIES":
            return f"Extract named entities from:\n\n{context}"
        elif op == "CODEGEN":
            return f"Write code for:\n\n{params.get('specification', context)}"
        elif op == "PLAN":
            return f"Create a plan for:\n\n{context}"
        return context


class APIExecutor(Executor):
    """Executes retrieval skills via HTTP API calls."""
    resource_class = "api"

    def __init__(self, base_url: str = "", api_key: str = ""):
        self.base_url = base_url
        self.api_key = api_key

    def execute(self, op: str, params: Dict, inputs: List[Any]) -> Any:
        query = params.get("query") or params.get("url") or (inputs[0] if inputs else "")
        if op in ("SEARCH", "FETCH"):
            # In production: make real HTTP call
            # For demo: return structured mock result
            return {
                "query": query,
                "results": [
                    {"title": f"Result for {query}", "content": f"Information about {query}..."}
                ],
                "source": "api_executor"
            }
        elif op == "QUERY_DB":
            return {"rows": [], "query": params.get("sql", str(query))}
        return {"op": op, "query": query}


class ActionExecutor(Executor):
    """Executes side-effect skills (bookings, emails, etc.)."""
    resource_class = "action"

    def execute(self, op: str, params: Dict, inputs: List[Any]) -> Any:
        if op == "EXEC_ACTION":
            return {"status": "completed", "action": params.get("action", op), "params": params}
        elif op == "SEND":
            return {"status": "sent", "to": params.get("to"), "content": str(inputs[0] if inputs else "")}
        elif op == "SAVE":
            return {"status": "saved", "data": str(inputs[0] if inputs else "")}
        return {"status": "executed", "op": op}


class VerifyExecutor(Executor):
    """Executes VERIFY nodes."""
    resource_class = "control"

    def execute(self, op: str, params: Dict, inputs: List[Any]) -> Any:
        if op == "VERIFY":
            value = inputs[0] if inputs else True
            return bool(value)
        elif op == "SELECT":
            branches = params.get("branches", [])
            selected = params.get("selected", 0)
            return branches[selected] if branches and selected < len(branches) else None
        return True


# ── Forge: the compiler backend ──

class Forge:
    """Compile TaskIR to LangGraph StateGraph and execute.

    The "codegen" stage of the Compile Anything pipeline:
        TaskIR → validate → Forge.lower() → LangGraph graph → execute
    """

    def __init__(self, executors: Optional[List[Executor]] = None):
        if executors is None:
            executors = [
                PythonExecutor(),
                APIExecutor(),
                ActionExecutor(),
                VerifyExecutor(),
                # LLMExecutor() is heavy — add explicitly when needed
            ]
        self.executors = {e.resource_class: e for e in executors}
        # Also index by op name for direct lookup
        self.op_executors = {}
        for e in executors:
            self.op_executors[e.resource_class] = e

    def _get_executor(self, op: str) -> Optional[Executor]:
        """Find the executor for a given skill op."""
        from src.isa.registry import get as get_skill
        spec = get_skill(op)
        if spec:
            rc = spec.resource_class
            if rc in self.executors:
                return self.executors[rc]
        # Fallback: try each executor
        return self.executors.get("python")

    def lower(self, module: Module) -> StateGraph:
        """Lower TaskIR Module to a LangGraph StateGraph.

        This is the "codegen" step — analogous to LLVM's llc.
        """
        report = validate(module)
        if not report.valid:
            errors = [e.msg for e in report.errors]
            raise ValueError(f"TaskIR failed validation: {errors}")

        # Use Annotated state schema with reducers for parallel-safe updates
        from typing import TypedDict
        StateSchema = TypedDict("StateSchema", {
            "values": Annotated[Dict[str, Any], merge_dicts],
            "traces": Annotated[List[Dict], list_add],
            "errors": Annotated[List[str], list_add],
        })
        graph = StateGraph(StateSchema)
        nodes = module.program.nodes

        # Find entry points (nodes with no inputs or after)
        all_deps = set()
        for n in nodes:
            all_deps.update(n.inputs)
            all_deps.update(n.after or [])
        node_ids = {n.id for n in nodes}
        entries = [n for n in nodes
                   if not (set(n.inputs) & node_ids)
                   and not (set(n.after or []) & node_ids)]

        # Add virtual entry node
        graph.add_node("__entry__", lambda state: state)
        graph.set_entry_point("__entry__")

        # Add each TaskIR node as a LangGraph node
        for node in nodes:
            fn = self._make_node_function(node)
            graph.add_node(node.id, fn)

        # Wire edges based on dependencies
        for node in entries:
            graph.add_edge("__entry__", node.id)

        for node in nodes:
            # Data dependencies become edges
            for dep in node.inputs:
                if dep in node_ids:
                    graph.add_edge(dep, node.id)
            # Control dependencies become edges
            for dep in (node.after or []):
                if dep in node_ids:
                    graph.add_edge(dep, node.id)

        # Nodes with no downstream dependencies connect to END
        has_downstream = set()
        for node in nodes:
            for dep in node.inputs:
                has_downstream.add(dep)
            for dep in (node.after or []):
                has_downstream.add(dep)
        for node in nodes:
            if node.id not in has_downstream:
                graph.add_edge(node.id, END)

        return graph.compile()

    def _make_node_function(self, node: Node) -> Callable:
        """Convert a TaskIR Node to a LangGraph node function.

        Returns a DELTA (only the changes) — LangGraph merges deltas from
        parallel nodes using the reducers on GraphState.
        """
        executor = self._get_executor(node.op)
        node_id = node.id
        node_op = node.op
        node_params = dict(node.params) if node.params else {}
        node_inputs = list(node.inputs)
        node_guard = node.guard
        node_retry = node.retry

        def execute_node(state: Dict) -> Dict:
            """Execute this TaskIR node, return delta for LangGraph to merge."""
            start = time.time()

            # Check guard (predication)
            if node_guard is not None:
                cond_val = state.get("values", {}).get(node_guard.cond)
                if cond_val is not None and not bool(cond_val):
                    return {
                        "values": {node_id: None},
                        "traces": [{"node": node_id, "op": node_op,
                                    "status": "skipped_guard", "duration_ms": 0}],
                    }

            # Gather inputs from state (SSA value lookup)
            input_values = [state.get("values", {}).get(dep)
                            for dep in node_inputs]

            # Execute
            try:
                result = executor.execute(node_op, node_params, input_values)
                status = "ok"
                values_delta = {node_id: result}
            except Exception as e:
                status = "error"
                values_delta = {node_id: None}

            duration = (time.time() - start) * 1000
            trace_entry = {
                "node": node_id, "op": node_op,
                "status": status,
                "duration_ms": round(duration, 2),
                "input_count": len([v for v in input_values if v is not None]),
            }

            delta = {"values": values_delta, "traces": [trace_entry]}
            if status == "error":
                delta["errors"] = [f"{node_id}: {executor.__class__.__name__} failed"]
            return delta

        return execute_node

    def run(self, module: Module, initial_state: Optional[Dict] = None) -> Dict:
        """Compile and execute a TaskIR program."""
        app = self.lower(module)
        state = initial_state or {}
        state.setdefault("values", {})
        state.setdefault("traces", [])
        state.setdefault("errors", [])
        result = app.invoke(state)
        result["total_time_ms"] = round((time.time() - state.get("_start", time.time())) * 1000, 2)
        return result


# ── Convenience: full pipeline ──

def compile_and_execute(module: Module,
                        executors: Optional[List[Executor]] = None) -> Dict:
    """One-shot: TaskIR → validate → lower → execute → result + trace + cost."""
    forge = Forge(executors)

    # Compile-time: validate
    report = validate(module)
    validation = {
        "valid": report.valid,
        "errors": [e.msg for e in report.errors],
        "warnings": [w.msg for w in report.warnings],
    }

    if not report.valid:
        return {"status": "compilation_error", "validation": validation}

    # Compile-time: cost estimate (from ISA registry)
    from src.isa.registry import get as get_skill
    from src.cost.model import baseline_flops
    total_latency = sum(get_skill(n.op).cost.latency_ms
                        for n in module.program.nodes if get_skill(n.op))
    cost_estimate = {
        "estimated_latency_ms": total_latency,
        "node_count": len(module.program.nodes),
        "baseline_flops": baseline_flops(),
    }

    # Runtime: execute
    result = forge.run(module)

    return {
        "status": "success" if not result.get("errors") else "runtime_warning",
        "validation": validation,
        "cost_estimate": cost_estimate,
        "output": result.get("values", {}),
        "trace": result.get("traces", []),
        "errors": result.get("errors", []),
        "total_time_ms": result.get("total_time_ms", 0),
    }
