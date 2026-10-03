# τ³-bench Evaluation Mechanics — Research Notes

Sources: `data/external_benchmarks/tau3_bench/domains/*` (tasks, DBs, policies) and
`third_party/tau3-bench/src/tau2/` (evaluator, environment, user simulator).
2546 tasks total: retail 114, airline 50, telecom 2285, banking_knowledge 97.

## 1. Task structure

Every task (`tasks.json`, one list per domain) has:

- `user_scenario` — what the simulated user knows/wants (see §4)
- `initial_state` — optional env setup: `initialization_data` (DB patches),
  `initialization_actions` (env function calls, e.g. `user:turn_roaming_off`),
  `message_history` (pre-existing conversation). Telecom tasks always use
  `initialization_actions` on both `user` and `assistant` environments.
- `evaluation_criteria`:
  - `actions[]`: `{action_id, requestor: "assistant"|"user", name, arguments, info?, compare_args?}`.
    Includes READ actions (e.g. `find_user_id_by_name_zip`) — they are the expected call graph.
  - `env_assertions[]` (telecom): `{env_type, func_name, arguments, assert_value, message}` —
    boolean functions run against the final environment (e.g. `user:assert_internet_speed{expected_speed:200}`).
  - `communicate_info[]`: strings that must appear in agent messages (deprecated but still scored).
  - `nl_assertions[]`: natural-language outcomes judged by an LLM.
  - `reward_basis`: subset of `DB | ENV_ASSERTION | ACTION | COMMUNICATE | NL_ASSERTION`.

Per-domain `reward_basis`: retail `DB+NL_ASSERTION` (112/114), airline
`DB+COMMUNICATE+NL_ASSERTION` (50/50), telecom `ENV_ASSERTION` (+`ACTION`),
banking_knowledge `DB+ACTION`.

## 2. Reward computation (src/tau2/evaluator/evaluator.py)

```
if termination_reason not in {AGENT_STOP, USER_STOP}: reward = 0.0     # premature death = fail
R = prod(component_rewards for components in task.reward_basis)         # multiplicative, each in {0,1}
component rewards:
  R_env    = R_db * R_env_assertions       # each only if in reward_basis
  R_db     = 1 if hash(gold_db) == hash(pred_db) AND hash(gold_user_db) == hash(pred_user_db) else 0
  R_assert = prod(1 if assertion() == assert_value else 0 for each env_assertion)
  R_action = 1 if every golden action matches some tool call in the trajectory else 0
  R_comm   = 1 if every communicate_info string is found in some agent text message else 0
  R_nl     = 1 if LLM judge passes every nl_assertion else 0
```

Binary, all-or-nothing at every level (one wrong argument ⇒ R_action=0; one
unread info string ⇒ R_comm=0). Reward is 1.0 if `evaluation_criteria` is None.

### 2.1 DB check (evaluator_env.py) — replay-and-diff

Two fresh environments are constructed from the same constructor:

- **gold env**: apply `task.initial_state`, then execute each golden action
  (`env.make_tool_call(name, requestor, **arguments)`); errors are logged, not fatal.
- **predicted env**: apply `task.initial_state`, then **replay the recorded trajectory**:
  extract `(tool_call, tool_response)` pairs from the message history and re-execute
  only *mutating* (WRITE) tools; READ/THINK tools are skipped (no state effect, avoids
  nondeterministic output comparison); unknown/hallucinated tool names are skipped
  and reported. Replayed response content must equal the recorded response, else ValueError.

Both agent DB and user DB (telecom device mock) are compared via
`get_pydantic_hash()` = hash of the canonical pydantic `model_dump`. Exact
structural equality — no partial credit for close states.

### 2.2 Action check (evaluator_action.py, data_model/tasks.py)

`Action.compare_with_tool_call`: tool name must match exactly; if `compare_args`
is set, only those argument keys are compared (subset match), else **all arguments
of the tool call** are compared for equality (`tool_args == action_args` on the
intersecting key set). `requestor` is NOT compared — user-side golden actions
(e.g. telecom `toggle_roaming`) can be satisfied by either participant's call.
Every golden action must be found somewhere in the trajectory (order-insensitive).

### 2.3 Communicate check (evaluator_communicate.py)

For each `info_str`: substring test `info_str.lower() in message.content.lower().replace(",", "")`
over assistant text messages. Brittle: the exact value (e.g. `"10"`) must be spoken.

### 2.4 NL assertions (evaluator_nl_assertions.py)

Single LLM call (default Gemini, `DEFAULT_LLM_NL_ASSERTIONS`): system prompt asks
to grade each expected outcome against the full transcript; response is JSON
`{"results": [{"expectedOutcome", "metExpectation", "reasoning"}]}`. All must pass.
Nondeterministic — the only non-programmatic component.

## 3. Expected DB state per write action (retail tools.py; others analogous)

Tools are typed `@is_tool(ToolType.READ|WRITE|GENERIC)`; only WRITE mutates state.

| action | state mutation |
|---|---|
| `cancel_pending_order` | `status: pending→cancelled`, `cancel_reason=reason`; append `refund` entry per `payment_history` item; gift-card balances credited immediately (rounded 2dp) |
| `exchange_delivered_order_items` | `status: delivered→"exchange requested"`, `exchange_items=sorted(item_ids)`, `exchange_new_items=sorted(new_item_ids)`, `exchange_payment_method_id`, `exchange_price_difference=round(Σ(new−old),2)` |
| `return_delivered_order_items` | `status→"return requested"`, `return_items=sorted(item_ids)`, `return_payment_method_id` (must be original method or gift card) |
| `modify_pending_order_items` | swap items in `order.items`; append `payment`+`refund` entries for the diff; gift-card balances adjusted |
| `modify_pending_order_payment` | append `payment`(new)+`refund`(old) of same amount; gift-card balances adjusted |
| `modify_pending_order_address`, `modify_user_address` | replace address object wholesale |
| `transfer_to_human_agents` | no DB change (GENERIC) |
| reads (`get_*`, `find_user_id_*`, `list_all_product_types`, `calculate`) | none |

Airline WRITE: `book_reservation`, `cancel_reservation` (refund + gift-card),
`update_reservation_flights/baggages/passengers`, `send_certificate`.
Telecom agent WRITE: `suspend_line`, `resume_line`, `send_payment_request`,
`enable_roaming`/`disable_roaming` (`line.is_roaming_enabled`), `refuel_data`
(data balance + charge). Telecom user env WRITE: device flags
(`turn_roaming_on/off`, `turn_data_off`, `set_network_mode_preference`,
`break_apn_settings`, `lock_sim_card`, …). Banking: `apply_for_credit_card`, etc.

Hard preconditions live in tool code (`raise ValueError` on wrong status,
insufficient balance, non-matching items) — a policy violation surfaces as a tool
error, never as a silent success. Soft policy (policy.md) is prompt-level only.

## 4. User simulator (src/tau2/user/)

`UserScenario = {persona?, instructions}`; instructions are
`{task_instructions, domain, reason_for_call, known_info, unknown_info}` or a
plain string. System prompt = global guidelines (`data/tau2/user_simulator/
simulation_guidelines[_tools].md`) + runtime persona config + the scenario wrapped
in `<scenario>...</scenario>`. Guidelines enforce: one message at a time, never
invent facts beyond the scenario, disclose info progressively (wait for agent to ask),
ground tool results (telecom: never fabricate `run_speed_test` output).

Mechanics: the simulator keeps its own history with **roles flipped** (it plays
"assistant" to the agent's "user"), generates a `UserMessage` per turn, and may
emit **user tool calls** (`requestor="user"`, telecom mock-phone tools).
Termination tokens in user content: `###STOP###` (goal satisfied),
`###TRANSFER###`, `###OUT-OF-SCOPE###`. A run must end in `USER_STOP`/`AGENT_STOP`
or reward is 0 (max-turns abort = failure).

## 5. Evaluating a TaskIR program against τ³ criteria

τ³ is interactive and stateful, so a TaskIR program can only be scored through a
trajectory + final state, not by static graph comparison. Required pipeline:

1. **Lower golden actions to TaskIR** and keep concrete tool names in
   `meta.provenance` (IR semantics use skills only) — the action check needs the
   literal `name` + `arguments`, so provenance is load-bearing for evaluation.
2. **Run the program against a τ³ environment loop**: user simulator turn →
   TaskIR execution (reads feed dataflow, WRITE nodes mutate env) → agent message.
   Record (tool_call, response) pairs with requestor tags; only WRITE nodes
   contribute to state replay.
3. **Score with the τ³ formula** (§2): replay WRITE nodes' effects into a fresh
   env initialized from `db.json` + `initial_state`; hash-compare against a gold
   env built from golden actions; match actions by (name, args); substring-check
   communicate_info in emitted messages; LLM-judge nl_assertions.
4. **Determinism caveat**: NL assertions need an LLM judge (cache per
   transcript); everything else is deterministic given the same tool semantics.
   Our executor must reproduce tau2 tool mutations *exactly* (field names,
   rounding, `sorted()` on id lists) or the hash diff fails even when the
   business intent is right. Safest option: reuse `third_party/tau3-bench`
   tool classes as the execution backend instead of reimplementing.

## 6. Runtime requirements checklist

- **State management**: initialize env from `db.json`/`db.toml` +
  `initialization_data` + `initialization_actions`; canonical-hashable DB objects
  (pydantic-style dumps); two synchronized stores (agent CRM + user device mock in
  telecom — `sync_tools()` keeps them consistent); deep-copy snapshots for
  gold/predicted forks.
- **Effect typing**: every skill/tool classified READ vs WRITE (τ³ replays only
  WRITE; our effect audit already classifies actions — align with `ToolType`).
- **Policy enforcement**: hard rules as preconditions that raise errors (guard/
  VERIFY nodes in TaskIR map naturally); soft rules injected as agent context
  (policy.md). Policy violations must not mutate state.
- **Trajectory logging**: message list with tool calls paired to responses by id;
  needed for action matching, replay, and communication checks.
- **Termination handling**: detect STOP/TRANSFER/OUT-OF-SCOPE tokens; abort on
  turn/latency budget counts as premature termination ⇒ reward 0.
- **User-simulator loop**: turn-based LLM with role-flipped history, optional user
  tools, seedable for reproducibility.
- **LLM judge**: pluggable judge for `nl_assertions` with JSON-mode output parsing.
