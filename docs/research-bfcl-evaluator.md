# BFCL V4 Official Evaluator — Technical Summary

Source: `third_party/gorilla/berkeley-function-call-leaderboard/` (BFCL v4, `VERSION_PREFIX = "BFCL_v4"`).
Key files (all paths relative to `bfcl_eval/`):

| Concern | File |
|---|---|
| Entry scoring dispatch | `eval_checker/eval_runner.py` |
| AST matching + type/value checks | `eval_checker/ast_eval/ast_checker.py` |
| Multi-turn checkers | `eval_checker/multi_turn_eval/multi_turn_checker.py`, `multi_turn_utils.py` |
| Leaderboard aggregation / formulas | `eval_checker/eval_runner_helper.py` |
| Output decoding (`ast_parse`) | `model_handler/utils.py` |
| Format predicates | `utils.py` |
| Model registry (handlers) | `constants/model_config.py`, `constants/category_mapping.py` |

## 1. Scoring algorithm

Every entry is scored **binary (valid=True/False)**; category accuracy = correct / total entries
(`save_eval_results`: `accuracy = correct_count / len(model_result)`). All-or-nothing per entry.

### 1.1 Single-turn AST categories (simple, multiple, parallel, parallel_multiple + live_* variants)

Pipeline per entry (`_evaluate_single_ast_entry`):
1. `handler.decode_ast(raw_string)` → Python `list[dict]` of `{func_name: {param: value}}`.
   Decode failure → fail (`ast_decoder:decoder_failed`).
2. `is_function_calling_format_output` must hold: a list where every item is a 1-key dict whose
   value is a dict. Violation → fail (`decoder_wrong_output_format`).
3. `ast_checker` dispatch on category name substring:
   - `"parallel" in category` → `parallel_function_checker_no_order`
   - `"multiple" in category` → `multiple_function_checker`
   - else → `simple_function_checker` (model output must have exactly **1** call, else `wrong_count`)

`simple_function_checker(func_description, model_output={name: params}, possible_answer)`:
1. Function name must be a key of the decoded dict (exact match; `convert_func_name` rewrites
   `a.b` → `a_b` only for models with `underscore_to_dot=True`, i.e. OpenAI-style FC handlers).
2. Every `required` param must be present.
3. Every model-provided param must exist in both the function spec and `possible_answer`.
4. Per-param checks (Python language path):
   - Type check against schema type (`PYTHON_TYPE_MAPPING`; nested item type checked 1 level deep
     for `array`/`tuple`; `tuple` values are coerced to `list` before comparison; `int` auto-coerces
     to `float` when schema says float).
   - If schema type is `str` → `string_checker` (standardized membership, see gotchas).
   - If `dict` → `dict_checker` (keys must match exactly; values standardized-membership;
     missing key fails unless GT marks it optional via `""`).
   - If `list` of `dict` → `list_dict_checker` (order-sensitive per dict).
   - If `list` → `list_checker` (whole-list standardized membership; order matters).
   - Else raw `value in possible_answer[param]` (numeric/bool exact).
5. Any GT param the model omitted must have `""` among its acceptable values (optional) else fail.

`possible_answer` format (JSONL `data/possible_answer/BFCL_v4_<cat>.json`):
`{"id": ..., "ground_truth": [{func: {param: [acceptable values...]}}]}` — each param maps to a
**list of acceptable values**; `""` in that list means "may be omitted".

- **simple** (python/java/js): GT = 1 dict → exactly 1 call checked as above.
- **multiple**: GT = list of exactly 1 dict → model must emit exactly **1** call (`len` must equal 1).
- **parallel / parallel_multiple**: GT = list of N dicts → model must emit exactly N calls.
  Matching is order-insensitive greedy elimination: for each GT call (in GT order), scan unused
  model outputs until one passes `simple_function_checker`; unmatched GT call → fail.
  (An `enforce_order` variant exists but is not used by the runner.)
- Java/JS: all decoded param values must be strings; `java_type_converter`/`js_type_converter`
  parse them into typed values before checking.

### 1.2 Irrelevance / Relevance

No ground truth. Pass iff:
- **irrelevance** (non-live + live): `decode_ast` fails OR decodes to "empty" (`[]`, `[{}]`,
  or anything not in FC format). I.e. the model must NOT produce a parseable function call.
- **relevance** (live only): decode succeeds AND output is non-empty FC format.

### 1.3 Multi-turn (base, miss_func, miss_param, long_context)

Per entry: model result is `list[turn] of list[step] of raw responses`. Decode each step with
`decode_execute` (skips undecodable/chat steps). Fail immediately if #turns != #GT turns
(`multi_turn:force_terminated` — mirrors the MAXIMUM_STEP_LIMIT=20 force-quit at generation).

`multi_turn_checker` re-executes the whole conversation deterministically:
- For each class in `involved_classes`, instantiate fresh from `initial_config` (module-global
  instances keyed by `model_name + test_entry_id + class`; ground truth gets its own instances
  with `_ground_truth` suffix, eval model gets `_eval` suffix).
- Model calls for the turn are executed via Python `eval()` after prefixing method names with
  the instance name; GT calls executed the same way. `kill/exit/quit/remove/unlink/popen/Popen/run`
  are blacklisted.
- If GT turn is non-empty but the model produced no calls this turn → fail (`empty_turn_model_response`).
- If GT turn is empty (miss_func/miss_param irrelevance turns): model calls are still executed
  (state consequences carry to next turn), but no direct penalty here; irrelevance is enforced
  implicitly — wrong extra calls mutate state and fail the next `state_checker`.
- After each non-empty GT turn, two checks:
  1. `state_checker`: all public (non `_`-prefixed) attributes of every model instance must
     equal the ground-truth instance's attributes.
  2. `response_checker`: GT turn execution results must be an **unordered subsequence** of ALL
     model execution results accumulated so far (across turns — a call made in an earlier turn
     counts for a later turn's GT).
- (A `method_invoke_order_checker` exists but is commented out; order within a turn is free.)

### 1.4 Agentic (web_search, memory) — not requested in detail

Model loops with real backends (search API / memory files); only the last non-FC message is
checked for containing the expected final answer(s).

### 1.5 Aggregation formulas (`generate_leaderboard_csv`)

- Non-Live overall = unweighted mean of [simple (itself mean of python/java/js), multiple, parallel, parallel_multiple].
- Live overall = **entry-weighted** (by total_count) mean of the 4 live AST categories.
- Multi-turn overall = unweighted mean of the 4 multi-turn categories.
- Agentic overall = unweighted mean of [web_search summary, memory summary].
- Total irrelevance = unweighted mean of [irrelevance, live_irrelevance].
- **Total score = percentage-weighted: non-live 10, live 10, irrelevance 10, multi-turn 30, agentic 40.**
- Unevaluated categories count as 0 in overall (displayed "N/A").

## 2. Expected model output format

Default prompting handler (`base_oss_handler` → `default_decode_ast_prompting`):
- A single string of Python-syntax calls, comma-separated, optionally wrapped in `[...]`:
  `[func1(param1=val1, param2=val2), func2(x=1)]` — backticks/whitespace stripped, outer
  `[` `]` auto-added if missing, parsed with Python `ast` (keyword args only; nested calls as
  argument values are resolved; `ast.BinOp` argument expressions are `eval`-ed).
- Function names keep dots: `math.factorial(number=5)` → key `"math.factorial"`.
- Multi-turn/agentic use `decode_execute`: same decode, then each call rendered back to
  executable `func(k=repr(v), ...)` strings.
- FC-mode handlers (OpenAI etc.) decode native tool-call objects instead; `underscore_to_dot`
  models get `.`→`_` name normalization.

Result file schema (input to the evaluator): JSONL at
`result/<model_name_with_underscores>/<group>/BFCL_v4_<category>_result.json`
(`group` = `non_live` | `live` | `multi_turn` | `agentic[/memory/<backend>]`), one entry per line:
`{"id": "<dataset id>", "result": <raw response string | multi-turn nested list>, "input_token_count": .., "output_token_count": .., "latency": ..}`.
Multi-turn `result` = list per turn of list per step of raw responses. Entries are re-sorted by id
on load; ids must match the dataset exactly.

## 3. Running it on Neural Compiler TaskIR output

The evaluator only reads the result JSONLs; generation is a separate step. Plan:

1. **TaskIR → BFCL call-string backend**: emit TaskIR calls as
   `[fn(p1=v1, p2=v2), ...]` (Python keyword-arg syntax, one comma-separated list). This is the
   only surface the default decoders need. Keep dotted function names verbatim (no `_` rewrite,
   since our handler will set `underscore_to_dot=False`).
2. **Irrelevance entries**: TaskIR must emit *no* call — output a plain natural-language string
   (any non-parseable text passes; `[]` or `[{}]` also count as "no call" and pass irrelevance).
3. **Multi-turn**: we cannot just emit calls once — the official generation loop feeds execution
   results back to the model each step (max 20 steps/turn). Simplest compliant path: implement a
   thin handler subclassing `BaseOSSHandler`-style prompting handler whose query step calls our
   compiler with the running conversation, then run the official `openfunctions_evaluation.py`
   generation (which also executes the sandbox classes and writes correctly-shaped results).
   Alternatively synthesize result files ourselves, but then we must re-implement step-loop state
   feeding or accept eval-time re-execution semantics (evaluator re-executes all calls from
   `initial_config`, so our recorded turns must be self-consistent with execution outputs).
4. **Register a model**: add a `ModelConfig` entry in `constants/model_config.py`
   (model_handler = a handler whose `decode_ast`/`decode_execute` are the defaults — copy
   `base_oss_handler.py` pattern), plus `supported_models.py`. Name must match the result
   subdirectory (with `/`→`_`).
5. **Run**: `bfcl evaluate --model our_model --test-category <cat ...> --result-dir ... --score-dir ...`
   (or `python -m bfcl_eval.eval_checker.eval_runner ...`). `--partial-eval` allows scoring a
   subset of ids. Score JSONs land in `score/<model>/<group>/BFCL_v4_<cat>_score.json` with
   per-entry error details (great for debugging our compiler's failures).

## 4. Gotchas

- **String matching** (`standardize_string`): case-insensitive; strips all spaces AND the
  characters `, . / - _ * ^`; converts `'`→`"`. So `"April 1, 2024"` == `"april 1 2024"`, and
  `a-b_c` == `abC`. Applies to str params, str list elements, and str dict values.
- **Variables**: if GT values are all strings but the schema type is non-string ("variable"
  placeholder case), the model value must be a string and is compared by **exact** membership
  (case-sensitive) — the standardizer is skipped.
- Optional param semantics: omit only if GT acceptable-values list contains `""`; providing a
  param whose GT value list has no matching entry fails (`missing_optional` / `value_error`).
- `int`→`float` auto-coercion (Python only); `tuple` GT is compared as `list` (lists pass where
  tuples expected — known false positive). Nested array item types checked only one level deep.
- Duplicate/extra calls: exact call count must match GT (parallel: N; multiple/simple: 1) —
  emitting a harmless extra call fails the entry.
- For parallel, greedy first-match elimination: an incorrect early match can starve a later GT
  call even if a global assignment exists (rare).
- Java/JS params must decode as JSON strings (e.g. `"[1,2]"`), then get converted.
- Multi-turn: whole-entry fail on any turn's state mismatch or missing GT response; model
  responses from previous turns satisfy later turns' response checks (unordered, duplicate-aware).
- Blacklisted eval names (`kill`, `run`, `remove`, ...) throw → recorded as error strings, which
  then can't match GT responses.
- `live_*` accuracy is entry-count weighted when aggregated; non-live is unweighted — don't
  average raw category accuracies and call it "BFCL score" (official total: 10/10/10/30/40).
- Evaluator skips exec/rest/sql/chatable categories in V4; format_sensitivity is non-scoring.
- `--model` must be a registered name (`MODEL_CONFIG_MAPPING`); result subdir name = model name
  with `/`→`_`.
