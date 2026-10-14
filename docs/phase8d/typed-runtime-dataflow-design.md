# TaskIR v0.2: Typed Runtime Dataflow — Design Document

**Phase**: 8D-0 | **Date**: 2026-10-14 | **Status**: Design

## 1. Problem

TaskIR v0.1 expresses **operation ordering** but not **runtime value flow**:

```
%p0 = EXEC_ACTION(action="lookup_user", name="Alice")
%p1 = EXEC_ACTION(action="get_account")     ← user_id 从哪来?
```

The model must fabricate `user_id="U123"` — a value that only exists after
`lookup_user` executes. This is a **representation gap**: the IR cannot
express "bind parameter X to field Y of upstream node Z's output".

## 2. Design

### 2.1 Two Parameter Kinds

| | `params` | `param_refs` |
|--|----------|-------------|
| Nature | Compile-time literal | Runtime SSA reference |
| Value known at | Compile time | Only after upstream executes |
| Static check | Type check vs input schema | SSA def-use + type check |
| Serialization | `key=value` | `key=%ssa_id` via `bind()` |

### 2.2 EXTRACT_FIELD (IR Intrinsic)

```
%uid = EXTRACT_FIELD(%u, field="user_id") -> Str
```

- **No side effect** — pure projection from an executed node's structured result
- Input: SSA reference to any node with a Dict/Record output type
- Output: the named field's type (from the producer's output_schema)
- Validator checks: field exists in upstream output_schema; type compatible

### 2.3 bind() Annotation

```
%acc = EXEC_ACTION(action="get_account")
       bind(user_id=%uid)
```

- `bind()` maps **named parameters** to **SSA value references**
- Parser stores as `Node.param_refs: Dict[str, str]`
- Validator: referenced SSA must be defined earlier; no duplicate binding
  with `params`; type check against consumer's input_schema

### 2.4 CapabilityIR Output Schema

```json
{
  "capability_id": "lookup_user",
  "input_schema": {"name": "Str"},
  "output_schema": {"user_id": "Str", "display_name": "Str", "email": "Str"},
  "effect_class": "read_only"
}
```

- `output_schema` declares the fields the tool's result contains
- EXTRACT_FIELD validates against this; Runtime reads actual values

### 2.5 Grammar Extension

```
node_line := ssa_id '=' op '(' args ')' annotations
annotations := [ '->' type ] [ 'after' '(' refs ')' ]
              [ 'bind' '(' key=ssa_ref ',' ... ')' ]
              [ 'guard' '(' ... ')' ] [ 'retry' '(' ... ')' ]
              [ 'hints' '(' ... ')' ]

EXTRACT_FIELD node:
  %uid = EXTRACT_FIELD(%u, field="user_id") -> Str
```

### 2.6 Compatibility with v0.1

- v0.1 programs are valid v0.2 (no `bind()` / `EXTRACT_FIELD` = all params literal)
- Parser accepts both; Validator adds new checks only when v0.2 features present
- `to_text()` / `parse_text()` roundtrip preserves `param_refs`

## 3. Runtime Execution Model

```
for node in topological_order(program):
    if node.op == "EXTRACT_FIELD":
        value = results[node.inputs[0]][node.params["field"]]
        results[node.id] = value
    else:
        resolved_params = merge(node.params, resolve_refs(node.param_refs, results))
        result = mock_registry.execute(node.tool_id, resolved_params)
        validate_output_schema(result, node.tool_id)
        results[node.id] = result
```

- Each tool's `mock_executor` returns a deterministic result
- `EXTRACT_FIELD` reads from the **actual returned dict**
- `param_refs` resolve to actual values at execution time
- Full execution trace records: node, tool, input params, output, timing

## 4. Three Dependency Kinds

| | DATA | ORDER | EFFECT |
|--|------|-------|--------|
| Representation | `param_refs` + `EXTRACT_FIELD` inputs | `after()` | Conservative serialization |
| Semantic | Value flows producer→consumer | Execution order only | Side-effect sequencing |
| Static check | Type + field existence | Acyclic | Mutating nodes ordered |
| Runtime check | Value present + type match | Timestamp ordering | No unintended re-execution |

## 5. What v0.2 Enables

1. **True dataflow DAG**: edges represent value flow, not just ordering
2. **Runtime verification**: assert downstream received upstream's actual output
3. **Scheduling**: critical path, communication cost, data locality become meaningful
4. **Graph partitioning**: KaHyPar/HEFT on semantically correct dataflow graphs
5. **Retry safety**: EXTRACT_FIELD re-reads (no side effect); EXEC_ACTION retry has explicit policy
