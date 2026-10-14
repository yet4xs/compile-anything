# Positioning Note: Why Runtime Dataflow is a Compiler Problem

**Phase**: 9B-0 Task 6 | **Date**: 2026-10-15

## 1. Why Existing Benchmarks Don't Cover Runtime Dataflow

Existing tool-use benchmarks (BFCL, ToolBench, xLAM) evolved from a **function calling** paradigm:

> "Given a user request, select the right function and extract the right parameters from the user's text."

This paradigm assumes a **single decision point**: the model sees the request, calls one (or a few independent) functions, and returns the result. All information needed is assumed to be in the user's message.

**Runtime dataflow** is fundamentally different:

> "Given a user request that requires chaining tools, determine which upstream tool outputs must flow into which downstream tool inputs — values that don't exist until execution time."

This is not an extension of parameter extraction. It requires:
1. **Planning** (which tools to call, in what order)
2. **Binding** (which output field feeds which input parameter)
3. **Deferred resolution** (values not available at generation time)

No existing benchmark tests this because the function calling paradigm doesn't model it. It's not a data gap — it's a **conceptual gap**.

## 2. Why TaskIR v0.2 Needs EXTRACT_FIELD + bind()

Without typed field extraction and named parameter binding, the model must:

1. **Invent** runtime IDs it hasn't seen (e.g., generate "U123" for a user_id that doesn't exist yet)
2. **Memorize** output formats (which field contains the ID from which tool)
3. **Coordinate** SSA naming across a multi-step program

Our Phase 8D-2B results prove this is impossible for a 3B model in ANY output format (text: 0%, JSON: 0%). The Semantic Skeleton sidesteps this by letting the model express **intent** (tool A's "user_id" feeds tool B's "user_id") without inventing values, while the compiler's deterministic lowering generates the actual SSA chains.

**This is the compiler's value proposition**: move impossible-for-model work (SSA construction, field resolution, type checking) into a deterministic pass, leaving the model to do what it can (intent expression, tool selection, dependency reasoning).

## 3. Why This is a Compiler Problem, Not an Agent Problem

An **agent** approach would have the LLM execute tools step by step, observe outputs, and decide the next call. This works for interactive scenarios but has problems:

1. **Latency**: Each step requires a full LLM inference
2. **Non-determinism**: The agent might choose different tools each run
3. **No static guarantees**: Can't validate the plan before executing (side effects!)
4. **No optimization**: Can't reorder independent calls or parallelize

A **compiler** approach:
1. **Generates the full plan upfront** (one inference)
2. **Statically validates** before any execution (no wasted side effects)
3. **Deterministically lowers** from intent to executable SSA
4. **Enables optimization**: DAG scheduling, parallel execution, critical path analysis
5. **Provides auditability**: The compilation trace explains every decision

The distinction matters for **safety-critical workflows** (banking, healthcare, e-commerce): you want to validate that "send notification" only happens after "approve loan" BEFORE executing either. A compiler can prove this; an agent discovers it after the fact.

## 4. Paper Positioning

### What We Claim

1. **Problem definition**: Runtime dataflow compilation is a distinct challenge from function calling
2. **Representation**: Semantic Skeleton IR enables neural models to express dynamic bindings
3. **Compiler**: Deterministic lowering from Skeleton to typed SSA TaskIR with field extraction
4. **Evidence**: Three-arm ablation proves representation effect (M1: 100% vs M0-T/M0-J: 0%)
5. **Benchmark gap**: No existing benchmark evaluates this capability

### What We Don't Claim

- We don't claim SOTA on function calling (that's BFCL, not our problem)
- We don't claim general agent capabilities (no multi-turn conversation)
- We don't claim hardware scheduling results (HEFT is future work)
- We don't claim the 25.5% makespan improvement (withdrawn after safety audit)

### DAC Fit

| Aspect | DAC Concern | Our Answer |
|--------|------------|-----------|
| **Novelty** | Is this just LLM prompting? | No: IR design + compiler pass + typed SSA |
| **Architecture** | What does the compiler do? | Field extraction, type checking, SSA construction, validation, scheduling-ready DAG |
| **Evaluation** | Is the benchmark meaningful? | First runtime dataflow benchmark; no existing benchmark covers this |
| **Systems** | Where's the hardware angle? | DAG → HEFT scheduling → execution plan (Phase 9C) |

## 5. Key Insight for the Paper

> The hardest part of AI task compilation is not selecting tools or extracting parameters — it's **expressing dynamic dataflow between tools whose values only exist at runtime**. Neural models cannot directly generate this in low-level IR (0% binding exact across text and JSON formats). The solution is not a better model or prompt, but a **semantic abstraction layer** (Skeleton) that captures intent, paired with a **deterministic compiler** (Lowering) that constructs the executable SSA program.
