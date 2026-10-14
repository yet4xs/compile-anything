# Phase 8D-0: Typed Runtime Dataflow — Results

**Date**: 2026-10-14 | **Gate**: PASS

## What TaskIR v0.2 Adds

| Capability | v0.1 | v0.2 |
|-----------|------|------|
| Parameter binding | Compile-time literals only (`key=value`) | **+ Runtime SSA references** (`bind(key=%ssa)`) |
| Field extraction | Not expressible | **`EXTRACT_FIELD(%node, field="name")`** |
| Data flow representation | Ordering only (after) | **True value flow** (producer→consumer) |
| Type safety for dataflow | None | **Field type check + param type check** |
| Cycle prevention | After() ordering | **param_refs def-use + cycle check** |
| Runtime execution | Not supported | **Mock Runtime with actual value propagation** |

## Case That v0.1 Cannot Express

```
Task: "Look up user Alice, find her account, get the latest transaction."

v0.1 (impossible — model must fabricate IDs):
  %c0 = EXEC_ACTION(action="lookup_user", name="Alice")
  %c1 = EXEC_ACTION(action="get_account", user_id="???")  ← fabrication

v0.2 (real dataflow):
  %u = EXEC_ACTION(action="lookup_user", name="Alice")
  %uid = EXTRACT_FIELD(%u, field="user_id") -> Str
  %acc = EXEC_ACTION(action="get_account")  bind(user_id=%uid)
  %aid = EXTRACT_FIELD(%acc, field="account_id") -> Str
  %tx = EXEC_ACTION(action="get_latest_transaction")  bind(account_id=%aid)
```

## Test Results

**13/13 runtime dataflow tests passed** (`tests/test_runtime_field_binding.py`):

| Test | Verifies |
|------|----------|
| linear_3step | lookup → account: user_id flows correctly |
| linear_4step | lookup → account → transaction → status |
| fork_multi_consumer | Same upstream output → two consumers |
| multi_source_params | Params from two different upstream nodes |
| read_then_write | Read chain → mutating operation |
| missing_field | Non-existent field rejected at runtime |
| forward_ref_rejected | Validator catches USE_BEFORE_DEF |
| param_conflict | Both literal and SSA for same key rejected |
| v01_compat | v0.1 programs still validate and execute |
| roundtrip | to_text → parse preserves param_refs |
| 8step_deep_chain | 8-node linear chain with 4 field extractions |
| side_effect_once | Mutating tool called exactly once |
| trace_proves_dataflow | Execution trace shows actual value flow |

**Regression**: 51/51 relevant tests pass (1 pre-existing forge failure unrelated).

## Verified Runtime Dataflow

```
Step 0: %u    EXEC_ACTION  action=lookup_user  params={name: "Alice"}
         → output: {user_id: "U_Alice", email: "Alice@test.com"}

Step 1: %uid  EXTRACT_FIELD  field=user_id  src=%u
         → output: "U_Alice"  ← READ FROM ACTUAL UPSTREAM RESULT

Step 2: %acc  EXEC_ACTION  action=get_account  bind(user_id=%uid)
         → input_params: {user_id: "U_Alice"}  ← RESOLVED FROM SSA
         → output: {account_id: "A_U_Alice", balance: 100.0}
```

The execution trace **proves** that downstream parameters received actual
upstream tool outputs, not fabricated values.

## Go/No-Go (Task 7)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | v0.2 Parser/Printer Roundtrip | ✅ PASS |
| 2 | v0.1 regression tests | ✅ 51/51 (1 pre-existing unrelated) |
| 3 | All legal Mock Dataflow tasks execute correctly | ✅ 13/13 |
| 4 | Missing field / wrong type / illegal ref rejected | ✅ 3/3 |
| 5 | DATA/ORDER/EFFECT invariants | ✅ |
| 6 | Runtime trace proves dataflow | ✅ |
| 7 | Retry safety (no unintended re-execution) | ✅ |

**GATE: PASS**

## What This Enables

1. **True dataflow DAG**: edges carry values, enabling meaningful scheduling
2. **KaHyPar / HEFT**: graph partitioning on semantically correct dataflow
3. **End-to-end verification**: assert downstream received upstream's output
4. **Next phase (8D-1)**: Build training data with runtime dataflow, retrain M0/M1
