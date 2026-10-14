# Runtime Dataflow Benchmark — Design Document

**Phase**: 9B-0 | **Date**: 2026-10-15 | **Status**: Protocol Draft

## 1. Benchmark Schema

### 1.1 Task Record

```json
{
  "task_id": "rdf_00001",
  "user_instruction": "Look up user Alice, find her latest booking, and modify the departure date.",
  "available_tools": [
    {
      "tool_id": "lookup_user",
      "description": "Find a user by name",
      "input_schema": {"name": "Str"},
      "output_schema": {"user_id": "Str", "email": "Str"},
      "effect_class": "read_only"
    },
    {
      "tool_id": "get_bookings",
      "description": "Retrieve bookings for a user",
      "input_schema": {"user_id": "Str"},
      "output_schema": {"latest_booking_id": "Str", "booking_count": "Int"},
      "effect_class": "read_only"
    },
    {
      "tool_id": "modify_booking",
      "description": "Change a booking's departure",
      "input_schema": {"booking_id": "Str", "new_date": "Str"},
      "output_schema": {"success": "Bool"},
      "effect_class": "state_update_reversible"
    }
  ],
  "reference_execution_trace": [
    {
      "step": 0,
      "tool_id": "lookup_user",
      "input_arguments": {"name": "Alice"},
      "param_sources": {"name": "user_text"},
      "output_values": {"user_id": "U_ALICE_9F3K", "email": "alice@test.com"}
    },
    {
      "step": 1,
      "tool_id": "get_bookings",
      "input_arguments": {"user_id": "U_ALICE_9F3K"},
      "param_sources": {"user_id": {"type": "runtime_dep", "producer_step": 0, "producer_tool": "lookup_user", "producer_field": "user_id"}},
      "output_values": {"latest_booking_id": "BK_2024_8812", "booking_count": 3}
    },
    {
      "step": 2,
      "tool_id": "modify_booking",
      "input_arguments": {"booking_id": "BK_2024_8812", "new_date": "2024-12-25"},
      "param_sources": {
        "booking_id": {"type": "runtime_dep", "producer_step": 1, "producer_tool": "get_bookings", "producer_field": "latest_booking_id"},
        "new_date": "user_text"
      },
      "output_values": {"success": true}
    }
  ],
  "dependency_level": 3,
  "runtime_dependencies": [
    {
      "producer_tool": "lookup_user",
      "producer_field": "user_id",
      "consumer_tool": "get_bookings",
      "consumer_argument": "user_id"
    },
    {
      "producer_tool": "get_bookings",
      "producer_field": "latest_booking_id",
      "consumer_tool": "modify_booking",
      "consumer_argument": "booking_id"
    }
  ],
  "target_skeleton": { "...": "See Phase 8D skeleton format" },
  "target_taskir": "TaskIR v0.2 text with EXTRACT_FIELD + bind()",
  "provenance": {
    "source": "human_curated | adapted_from | synthetic_template",
    "annotator": "name/id",
    "review_status": "pending | reviewed | frozen"
  }
}
```

### 1.2 Dependency Level Classification

| Level | Definition | Example |
|-------|-----------|---------|
| **L0** | Single tool call | `get_weather(city="NYC")` |
| **L1** | Multi-tool, all independent | `get_weather("NYC")` + `get_news()` |
| **L2** | ORDER dependency (B after A, no value flows) | `authenticate()` then `get_profile()` |
| **L3** | Runtime VALUE dependency (B consumes A's output) | `search()` → `extract booking_id` → `modify(id=booking_id)` |

### 1.3 Param Source Taxonomy

For every parameter in every tool call, annotate:

| Source | Code | Description |
|--------|------|-------------|
| User text | `user_text` | Value is extractable from user instruction |
| Runtime dep | `runtime_dep` | Value comes from a prior tool's output field |
| Environment | `environment` | Value from hidden state (DB, context, session) |
| Default | `default` | Hardcoded/model-chosen default value |
| Unknown | `unknown` | Cannot determine origin (see §3.4) |

**Only `runtime_dep` counts as a runtime dataflow dependency.**

### 1.4 Runtime Dependency Annotation Format

```json
{
  "producer_step": 0,
  "producer_tool": "lookup_user",
  "producer_field": "user_id",
  "consumer_step": 1,
  "consumer_tool": "get_bookings",
  "consumer_argument": "user_id",
  "confidence": "verified | inferred | unknown"
}
```

---

## 2. Leakage Rules

### 2.1 Split Dimensions

| Dimension | Split Policy | Rationale |
|-----------|-------------|-----------|
| **Tool family** | STRICT: no tool appears in both train and test | Tests generalization to unseen APIs |
| **Task template** | STRICT: no shared workflow pattern | Tests composition, not memorization |
| **Execution trace** | STRICT: no identical trace | Prevents answer memorization |
| **Hidden values** | STRICT: no shared user IDs, booking IDs, etc. | Prevents value leakage |

### 2.2 Allowed Shared Fields

| Field | Shareable? | Note |
|-------|-----------|------|
| Tool type schema (Str/Int/Float) | ✅ | Generic types |
| Effect class taxonomy | ✅ | Read_only/mutation/etc. |
| Annotation format | ✅ | Protocol itself |
| **Tool IDs/names** | ❌ | Must be unseen in test |
| **User entity names** | ❌ | Different users across splits |
| **Workflow patterns** | ❌ | Different dependency chains |
| **Output value formats** | ❌ | Different ID patterns (UUID vs numeric) |

### 2.3 Split Verification Protocol

Before freezing, run automated checks:

1. **Tool overlap**: set(train_tools) ∩ set(test_tools) = ∅
2. **Template overlap**: no pair (train_task, test_task) with same tool sequence
3. **Value leakage**: no test output value appears in train data
4. **Near-duplicate check**: 8-gram Jaccard < 0.3 on user instructions
5. **Depth balance**: test covers same L3 depth range as train

---

## 3. Annotation Protocol

### 3.1 Per-Parameter Source Judgment

For each tool call argument, the annotator answers:

> "Where does the value of this parameter come from?"

| Answer | Code | Test |
|--------|------|------|
| User explicitly stated it | `user_text` | Value string appears in user instruction |
| Previous tool returned it | `runtime_dep` | Value matches a field in a prior step's output |
| System/environment knows it | `environment` | Value not in user text or prior outputs |
| Model chose it as a default | `default` | No evidence-based derivation |
| Cannot determine | `unknown` | Ambiguous or insufficient info |

### 3.2 Decision Tree

```
Parameter value V in tool call at step N:

Q1: Does V appear verbatim in the user instruction?
  YES → user_text
  NO  → Q2

Q2: Does V match any output field from steps 0..N-1?
  YES → runtime_dep (record producer_step, producer_tool, producer_field)
  NO  → Q3

Q3: Does V match a system variable (session_id, current_date, etc.)?
  YES → environment
  NO  → Q4

Q4: Is V a reasonable default (page=1, limit=10, etc.)?
  YES → default
  NO  → unknown
```

### 3.3 Inter-Annotator Agreement

- Minimum 2 annotators per task
- Cohen's κ ≥ 0.8 required for `runtime_dep` classification
- Disagreements resolved by third annotator
- Final labels frozen before any model evaluation

### 3.4 Ambiguity Handling

| Situation | Rule |
|-----------|------|
| Value partially matches user text (e.g., "NYC" vs "New York City") | `user_text` with note |
| Value could come from either tool A or tool B | Mark both; prefer earlier step |
| Value is a transformation of prior output (e.g., ID truncated) | `runtime_dep` with `transform` field |
| Annotator unsure | `unknown` — **never force-classify as runtime_dep** |

### 3.5 Quality Gates

| Check | Threshold |
|-------|-----------|
| Runtime dep κ (inter-annotator) | ≥ 0.80 |
| Unknown rate | ≤ 15% of all params |
| Trace completeness | 100% (all steps have input+output) |
| Schema coverage | 100% (all tools have input+output schema) |
| Verified runtime deps | ≥ 90% confidence for published tasks |

---

## 4. Existing Dataset Feasibility Review

### 4.1 Field Mapping

| Dataset | user_instruction | tool_schema | input_args | output_values | runtime_dep_annotatable |
|---------|-----------------|-------------|------------|---------------|----------------------|
| BFCL V4 | ✅ | ✅ (parameters) | ❌ (only in target) | ❌ | ❌ (0 multi-tool) |
| ToolBench | ✅ | ✅ (partial) | ✅ (ReAct format) | ❌ | ❌ (0 runtime deps) |
| τ³-bench | ✅ | ✅ (tools.py) | ❌ (names only) | ❌ | **MAYBE** (multi-step, NL suggests deps) |
| xLAM | ✅ | ✅ | ✅ (some) | ❌ | ❌ (single-turn) |

### 4.2 Gap Analysis

| Missing Field | BFCL | ToolBench | τ³ | xLAM |
|---------------|------|-----------|-----|------|
| Per-call arguments | ❌ | ✅ (in target) | ❌ | ✅ (some) |
| Tool output values | ❌ | ❌ | ❌ | ❌ |
| Output schema | ❌ | ❌ | ❌ | ❌ |
| Multi-step chains | ❌ | 77 tasks | ✅ many | ❌ |
| Runtime dep labels | ❌ | ❌ | ❌ | ❌ |

### 4.3 Annotation Cost Estimate

| Task | Per-Task Effort | 200 Tasks |
|------|----------------|-----------|
| Add output schema to tools | ~5 min/tool × 8 tools | ~13 hours |
| Trace per-call arguments | ~3 min/call × 4 calls | ~40 hours |
| Verify runtime deps | ~2 min/dep × 3 deps | ~20 hours |
| Quality review (double annotation) | × 1.5 overhead | ~110 hours total |
| **Total** | | **~183 hours (~4.5 weeks single person)** |

**Recommendation**: Start with 50-100 tasks for feasibility, then scale to 200+ if quality is high.

---

## 5. Candidate Real Workflow Sources

### 5.1 Public Datasets/Tools

| Source | Domain | Tools | License | Runtime Deps? | Schema Quality |
|--------|--------|-------|---------|--------------|----------------|
| τ³-bench | airline/banking/retail/telecom | 14-68/domain | MIT | Likely (multi-step) | tools.py has types |
| ToolBench | general (16k APIs) | thousands | Apache 2.0 | Unlikely (L2 only) | Partial |
| BFCL V4 | general function calling | varies | MIT | No (single-tool) | Good (input only) |
| ToolAlpaca | general | ~400 | Apache 2.0 | Unknown | Poor |
| API-Bank | banking | 73 | Research | Unknown | Good |
| WorkArena | web automation | ~700 | MIT | Likely (web flows) | DOM-based |

### 5.2 Most Promising: τ³-bench

**Why**: Multi-step trajectories already exist. Tool schemas in Python source. Four domains. MIT license.

**What's needed**:
1. Execute reference trajectories in τ³'s actual runtime (it has a simulator)
2. Capture actual output values per step
3. Compare output values to downstream input arguments
4. Annotate param sources

**Estimated yield**: 100-300 L3 tasks from ~1,500 multi-step trajectories

### 5.2 Alternative: Synthetic-from-Real

Take real tool schemas (from τ³, ToolBench, RapidAPI) but construct realistic multi-step workflows manually. This is NOT "synthetic chains" — it's human-curated realistic workflows using real API schemas.

---

## 6. Benchmark Evaluation Protocol

### 6.1 Metrics (frozen from Phase 8D)

| Metric | Definition |
|--------|-----------|
| **Runtime Binding Exact** | All runtime_dep params correctly bound (producer, field, consumer, arg) |
| **Canonical Tool Call Match** | Correct tool instances with correct literal args |
| **DATA DAG Exact** | Exact dependency edge match |
| **Execution Success** | Mock runtime executes without error |
| **Strict Joint** | All of the above + validator pass |
| **Generated Tokens** | Output length (efficiency) |

### 6.2 Evaluation Protocols

| Protocol | Tool Pool | Tests |
|----------|-----------|-------|
| A: Oracle Pool | Only gold tools | Pure dataflow/binding ability |
| B: Distractor Pool | Gold + similar distractors | Tool selection + dataflow |

### 6.3 Comparison Arms

| Arm | Output | Compiler Path |
|-----|--------|--------------|
| M0-T | TaskIR v0.2 text | Parse → Validate |
| M0-J | TaskIR v0.2 JSON AST | Deserialize → Validate |
| M1 | Semantic Skeleton JSON | Lowering → Validate |
| Oracle | Reference skeleton | Lowering (compiler upper bound) |

---

## 7. Freeze Checklist

Before any model evaluation:

- [ ] All tasks pass annotation quality gates (κ ≥ 0.8)
- [ ] All tools have input AND output schemas
- [ ] All runtime deps verified with confidence ≥ 90%
- [ ] Split leakage checks pass (tool, template, value, near-dup)
- [ ] Depth/complexity distribution reported
- [ ] Evaluation metrics frozen (no post-hoc changes)
- [ ] Oracle skeleton + TaskIR generated and validated for all tasks
- [ ] Mock runtime traces verified for all L3 tasks
- [ ] Provenance recorded (annotator, source, review status)
