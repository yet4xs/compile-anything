# LLMCompiler DAG Executor — Code Research (for Phase 6 real runtime)

Source: `D:\ccfa-2026\_papers\LLMCompiler-main\LLMCompiler-main\` (snapshot Jul 2024).
Scope: `src/llm_compiler/` (orchestrator, planner, parser, task fetching unit),
`src/executors/` (ReAct baseline), `src/callbacks/`.

## 1. Architecture (text diagram)

```
                    LLMCompiler._acall()  [llm_compiler.py]  replan loop (max_replans)
                    ┌───────────────────────────────────────────────────────────┐
                    │                                                           │
 user query ──► Planner ──stream tokens──► LLMCompilerCallback                 │
                    │                    on_llm_new_token →                    │
                    │                    StreamingGraphParser (regex per       │
                    │                    newline) → Task objects                │
                    │                              │                            │
                    │                        asyncio.Queue                     │
                    │                              ▼                            │
                    │                 TaskFetchingUnit.aschedule(queue)        │
                    │                 [task_fetching_unit.py]                  │
                    │                              │                            │
                    │   fire asyncio.create_task(_run_task) per ready task;    │
                    │   one asyncio.Event per task; 10 ms polling loop         │
                    │                              ▼                            │
                    │           all tasks done (incl. join barrier)            │
                    │                              ▼                            │
                    │   agent_scratchpad = concat of Thought/Action/           │
                    │   Observation per task  →  join(): agent LLM answers     │
                    │   "Action: Finish(ans)"  or  "Action: Replan(thought)"   │
                    │        │ Finish ──► return answer                         │
                    │        └ Replan ──► next iteration (with context)        │
                    └───────────────────────────────────────────────────────────┘
```

Non-streaming mode: `Planner.plan()` batch-parses the whole response into
`Dict[int, Task]` (`LLMCompilerPlanParser.parse`), then `set_tasks` + `schedule()`.

## 2. DAG construction: parsing planner output

Planner emits numbered function calls; dependencies are **inferred syntactically**
from `$id` / `${id}` references inside argument strings.

`src/llm_compiler/output_parser.py`:

```python
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
ID_PATTERN = r"\$\{?(\d+)\}?"

def default_dependency_rule(idx, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers

def _get_dependencies_from_graph(idx, tool_name, args):
    if tool_name == "join":
        dependencies = list(range(1, idx))          # join depends on ALL prior tasks
    else:
        dependencies = [i for i in range(1, idx) if default_dependency_rule(i, args)]
    return dependencies
```

- Args parsed with `ast.literal_eval` (`_parse_llm_compiler_action_args`); on
  failure the raw string passes through — no hard schema enforcement. Unknown tool
  name raises `OutputParserException` at parse time (`_find_tool`). `join` gets a
  no-op tool (`tool_func = lambda x: None`, `is_join=True`).
- **No cycle detection**: acyclicity holds only by convention — the prompt demands
  "strictly increasing" IDs and dependencies can only point backwards
  (`range(1, idx)`).

Task unit (`src/llm_compiler/task_fetching_unit.py`):

```python
@dataclass
class Task:
    idx: int; name: str; tool: Callable; args: Collection[Any]
    dependencies: Collection[int]
    stringify_rule: Optional[Callable] = None; thought: Optional[str] = None
    observation: Optional[str] = None; is_join: bool = False

    async def __call__(self) -> Any:
        return await self.tool(*self.args)   # direct await of tool coroutine
```

Note the DAG is *untyped*: the only dataflow value is `observation: Optional[str]`;
everything is stringified when substituted into downstream args.

## 3. DAG executor: TaskFetchingUnit

`src/llm_compiler/task_fetching_unit.py` — the core scheduler:

```python
SCHEDULING_INTERVAL = 0.01  # seconds

class TaskFetchingUnit:
    tasks: Dict[str, Task]
    tasks_done: Dict[str, asyncio.Event]     # one Event per task idx
    remaining_tasks: set[str]

    def _get_all_executable_tasks(self):
        return [task_name for task_name in self.remaining_tasks
                if all(self.tasks_done[d].is_set()
                       for d in self.tasks[task_name].dependencies)]

    async def _run_task(self, task: Task):
        self._preprocess_args(task)          # late-bind $id -> observation
        if not task.is_join:
            observation = await task()
            task.observation = observation
        self.tasks_done[task.idx].set()

    async def schedule(self):
        """Run all tasks in parallel, respecting dependencies."""
        while not self._all_tasks_done():
            executable_tasks = self._get_all_executable_tasks()
            for task_name in executable_tasks:
                asyncio.create_task(self._run_task(self.tasks[task_name]))
                self.remaining_tasks.remove(task_name)
            await asyncio.sleep(SCHEDULING_INTERVAL)
```

Dependency resolution happens **at dispatch time**, not parse time — string
interpolation of placeholders with the dependency's observation:

```python
def _replace_arg_mask_with_real_value(args, dependencies, tasks):
    ...  # recursive into list/tuple; for str:
    elif isinstance(args, str):
        for dependency in sorted(dependencies, reverse=True):
            # consider both ${1} and $1 (in case planner makes a mistake)
            for arg_mask in ["${" + str(dependency) + "}", "$" + str(dependency)]:
                if arg_mask in args and tasks[dependency].observation is not None:
                    args = args.replace(arg_mask, str(tasks[dependency].observation))
    return args
```

Design points (and flaws):
- **Readiness = polling.** Every 10 ms the executor rescans `remaining_tasks` and
  launches every task whose dependency Events are all set — up to one interval of
  added latency per dependency layer, and busy-wait CPU while idle.
- **Fire-and-forget tasks.** `asyncio.create_task` results are not stored, never
  awaited, exceptions never retrieved.
- **Failure = livelock.** `_run_task` has no try/except: if a tool raises, its
  Event is never set and `schedule()` spins forever.
- Recursive substitution into list/tuple args is supported; observation is always
  `str(...)`-ed. Annotations are loose (`Dict[str, Task]` with `int` keys).

## 4. Streaming plan generation (task fetching unit input side)

`src/llm_compiler/planner.py` — streaming is implemented as a **LangChain callback
that parses tokens incrementally and pushes finished Tasks onto a queue**:

```python
class LLMCompilerCallback(AsyncCallbackHandler):
    async def on_llm_new_token(self, token, *, run_id, ...):
        parsed_data = self._parser.ingest_token(token)
        if parsed_data:
            await self._queue.put(parsed_data)
            if parsed_data.is_join:
                await self._queue.put(None)       # sentinel

    async def on_llm_end(self, response, *, run_id, ...):
        parsed_data = self._parser.finalize()
        if parsed_data:
            await self._queue.put(parsed_data)
        await self._queue.put(None)
```

`StreamingGraphParser` keeps a line buffer; on every `\n` it regex-matches the
buffer against `THOUGHT_PATTERN` (stores the thought for the next action) or
`ACTION_PATTERN` (emits an `instantiate_task` result). Tasks are therefore
**scheduled while the planner is still generating** — this overlap is the paper's
headline latency win.

Consumer side (`TaskFetchingUnit.aschedule`): same dispatch loop as `schedule()`
(see §3) with one addition — it first `await task_queue.get()` (blocking on the
planner stream), sets `no_more_tasks = True` on the `None` sentinel, registers
each arriving task via `set_tasks({task.idx: task})`, and breaks only when
`no_more_tasks and self._all_tasks_done()`.

Wiring in `LLMCompiler._acall` (`src/llm_compiler/llm_compiler.py`):

```python
task_fetching_unit = TaskFetchingUnit()
if self.planner_stream:
    task_queue = asyncio.Queue()
    asyncio.create_task(self.planner.aplan(     # planner runs concurrently
        inputs=inputs, task_queue=task_queue, ...))
    await task_fetching_unit.aschedule(task_queue=task_queue, func=lambda x: None)
else:
    tasks = await self.planner.plan(...)
    task_fetching_unit.set_tasks(tasks)
    await task_fetching_unit.schedule()
```

## 5. Join mechanism

`join` is a **barrier marker, not a data node**:

1. In the DAG it is a pseudo-task depending on `range(1, idx)` (all prior tasks)
   whose "execution" is skipped (`if not task.is_join: ... await task()`), then its
   Event is set — so it only guarantees `schedule()`/`aschedule()` doesn't return
   until every real task has finished.
2. The actual synthesis is a separate LLM call after the barrier —
   `LLMCompiler.join()` (`llm_compiler.py`): builds
   `prompt = joinner_prompt + "Question: ..." + agent_scratchpad` and runs the
   agent LLM (`stop=["<END_OF_RESPONSE>"]`), then parses the reply.

3. Joinner output format is ReAct-like, parsed line-by-line:
   `Action: Finish(answer)` vs `Action: Replan(...)` (constants in
   `src/llm_compiler/constants.py`); on the final iteration `is_replan` is forced
   to False.
4. **Replan path**: on Replan, executed Thought-Action-Observation triples plus the
   joinner's Thought are formatted into a "Previous Plan" context
   (`_generate_context_for_replanner` / `_format_contexts`) and fed to
   `Planner.aplan(is_replan=True)`, which uses a separate system prompt whose
   guidelines say "NEVER repeat already-executed actions"; loop bounded by
   `max_replans`.

Notable: the joinner prompt carries the few-shot examples (e.g.
`configs/hotpotqa/gpt_prompts.py::OUTPUT_PROMPT`) and the scratchpad is the *only*
data channel — no structured values cross the join boundary.

## 6. asyncio patterns inventory

| Pattern | Where | Notes |
|---|---|---|
| `asyncio.create_task` (fire-and-forget) | `TaskFetchingUnit.schedule/aschedule`; `LLMCompiler._acall` (planner task) | return values discarded; no exception retrieval |
| `asyncio.Event` per task | `TaskFetchingUnit.tasks_done` | completion flag polled by scheduler |
| `asyncio.Queue` + `None` sentinel | planner stream → `aschedule` | producer is a callback handler |
| `asyncio.sleep(0.01)` polling loop | both `schedule` and `aschedule` | readiness via rescan, not `Event.wait` |
| `asyncio.gather` | `src/executors/agent_executor.py::_atake_next_step` (parallel tool calls, ReAct baseline only — not the compiler path) | |
| `asyncio_timeout` | ReAct executor `_acall` only | no timeout in DAG executor |
| No TaskGroup / `asyncio.wait` / Semaphore | — | no concurrency caps or rate limiting anywhere |
| per-phase stats callbacks | `AsyncStatsCallbackHandler` attached separately to planner and executor LLM calls (`src/callbacks/callbacks.py`): token counts + wall times; streaming mode counts tokens by hand (tiktoken input, +1 per streamed token) | |

Driver (`run_llm_compiler.py`): examples run sequentially (`run_until_complete`,
`await` per question); parallelism exists only *within* one question's DAG.

## 7. What to adopt for our Phase 6 real runtime

1. **Streaming pipeline shape** (the crown jewel): LLM token stream → incremental
   line/regex parser → `asyncio.Queue` → scheduler that starts independent tasks
   before planning finishes. Maps onto our lift/compile stage: emit TaskIR nodes
   incrementally (e.g. per statement) and start the ready set immediately.
2. **Late binding of inputs**: keep `$id` placeholders through planning and resolve
   them at dispatch time from producer observations. In TaskIR terms: resolve SSA
   value references only when the node is dispatched (which we already do); their
   dual-form tolerance (`${1}` and `$1`) is a cheap robustness trick for LLM output.
3. **Event-per-node completion tracking** — but replace their 10 ms polling with
   dependency-conjunction waits: `await asyncio.gather(*(events[d].wait() for d in deps))`
   inside the node task, or a central `asyncio.wait(..., FIRST_COMPLETED)` reactor.
   Same data structures, zero polling latency, no busy-wait CPU.
4. **Join-as-barrier + separate synthesis LLM call + Replan loop**: the
   Finish/Replan decision with a formatted "Previous Plan" context and bounded
   `max_replans` is a proven control loop; our VERIFY op can be an extra Replan
   trigger besides the joinner's own judgment.
5. **Per-phase instrumentation callbacks** (planner vs executor separately) — matches
   our trace design; their streaming-mode manual token counting (tiktoken for input,
   +1 per streamed token) is a useful trick when the provider doesn't return usage
   on streamed calls.
6. **Sentinel-terminated queue** (`None` = end-of-plan) — simplest correct way to
   coordinate "planner done" with "all scheduled work done" without extra futures.

## 8. What's missing vs our TaskIR runtime

| Capability | LLMCompiler | Our TaskIR runtime (`src/runtime/`, `src/validator/`, `src/cost/`) |
|---|---|---|
| Static validation | None. Acyclicity by prompt convention ("strictly increasing IDs"); no schema/type check on args (`ast.literal_eval` silently falls back to raw string); unknown tool = parse-time crash | V1–V6 validator: structural, typing, effect ordering, retry-edge legality; data gate for everything entering `data/taskir/` |
| Retry / failure handling | Single attempt, no backoff. Failed tool ⇒ Event never set ⇒ scheduler spins forever (livelock bug); unhandled task exceptions rot in unreferenced tasks | `retry(on=%v)` is an IR-level structure; runtime does attempt tracking, retry exhaustion marking, memo invalidation + dependent rollback (A→B→C chain re-execution) |
| Cost model | None. Callbacks record token counts and wall times only; no per-node latency/cost estimation, no scheduling input | `src/cost/model.py`: per-node cost, critical-path analysis, resource classes (lm/api/db), console + markdown reports |
| Typed dataflow | Everything is `str`; observation `str()`-ed into arg strings; no producer/consumer type contract | Typed SSA values with type checking in validator |
| Control flow | Flat DAG only: no conditionals/guards, no branches, no loops | Guard nodes, branch semantics, consume-skipped/chosen-branch-skipped statuses |
| Determinism / replay; concurrency control | None: logs only; unbounded fan-out, no semaphore/rate limit/per-class caps | Trace events + seeded simulator (fuzzed for defined-failure-only); resource classes, v0.2 scheduler interface |
| Timeouts | None in DAG path | Executor timeouts by design (their ReAct path had `asyncio_timeout`; ours is per design) |

### Concrete hazards to avoid (found in their code)

- `_run_task` has no `try/except` → one tool exception hangs `schedule()`/`aschedule()`
  forever (`task_fetching_unit.py:117-122`). Phase 6 must set completion events in a
  `finally` and record the failure as a trace event.
- `asyncio.create_task` without keeping a reference → tasks can be garbage-collected
  mid-flight (CPython gotcha) and exceptions are never observed. Keep a task set (or
  use `TaskGroup`).
- Polling scheduler adds ~10 ms per dependency layer and burns loop iterations while
  idle; use event-driven readiness instead.
- `Dict[str, Task]` annotation vs `int` keys — sloppy typing; key nodes by opaque
  string ids from the start.

## 9. File map (repo-relative)

- `src/llm_compiler/llm_compiler.py` — orchestrator, replan loop, join(), scratchpad
- `src/llm_compiler/planner.py` — prompts, `StreamingGraphParser`, `LLMCompilerCallback`
- `src/llm_compiler/output_parser.py` — batch parser, `$id` dependency rule
- `src/llm_compiler/task_fetching_unit.py` — `Task`, `TaskFetchingUnit` (the executor)
- `src/llm_compiler/constants.py` — `<END_OF_PLAN>`, Finish/Replan tokens
- `src/executors/agent_executor.py` — ReAct baseline; only `asyncio.gather` + timeouts
- `src/callbacks/callbacks.py` — `AsyncStatsCallbackHandler` (token/latency stats)
- `run_llm_compiler.py` — driver; sequential over dataset, `run_until_complete`
