# Phase 1 — End-to-End Demo Cost Reports

Simulator: deterministic mock executors, nominal costs (see `spec/skill-isa.md` §4).

# flight_cheapest — Find the cheapest flight arriving before 8 PM

- program: `find_cheapest_flight`
- task: "Find the cheapest flight arriving before 8 PM"
- validation: **valid**
- status: completed

## Execution trace

| seq | node | op | attempt | status | latency ms | tokens | FLOPs | value |
|---|---|---|---|---|---|---|---|---|
| 0 | %1 | SEARCH | 1 | ok | 124.587 | 0/0 | 0 | `[{'carrier': 'flight_carrier_0A', 'price': 546.58, 'depart_time': 'flight_depart_time_0...` |
| 1 | %2 | FILTER | 1 | ok | 4.56 | 0/0 | 1e+06 | `[{'carrier': 'flight_carrier_0A', 'price': 546.58, 'depart_time': 'flight_depart_time_0...` |
| 2 | %3 | ARGMIN | 1 | ok | 1.061 | 0/0 | 1e+06 | `{'carrier': 'flight_carrier_4E', 'price': 157.41, 'depart_time': 'flight_depart_time_4G...` |

## Totals

| metric | value |
|---|---|
| sequential latency | 130.208 ms |
| critical-path latency (unlimited parallelism) | 130.208 ms |
| tokens in / out | 0 / 0 |
| FLOPs | 2e+06 |
| node calls (lm / api+db) | 3 (0 / 1) |
| retries / skipped | 0 / 0 |
| 70B single-shot reference | ~2500 ms, ~2.52e+14 FLOPs |
| pipeline FLOPs vs 70B baseline | 0.00% |


# flight_verified — Find the cheapest flight arriving before 8 PM, explain it, and verify the answer before returning

- program: `find_cheapest_flight_verified`
- task: "Find the cheapest flight arriving before 8 PM, explain it, and verify the answer before returning"
- validation: **valid**
- status: completed

## Execution trace

| seq | node | op | attempt | status | latency ms | tokens | FLOPs | value |
|---|---|---|---|---|---|---|---|---|
| 0 | %1 | SEARCH | 1 | ok | 108.322 | 0/0 | 0 | `[{'carrier': 'flight_carrier_0A', 'price': 277.15, 'depart_time': 'flight_depart_time_0...` |
| 1 | %2 | FILTER | 1 | ok | 5.201 | 0/0 | 1e+06 | `[{'carrier': 'flight_carrier_0A', 'price': 277.15, 'depart_time': 'flight_depart_time_0...` |
| 2 | %3 | ARGMIN | 1 | ok | 0.933 | 0/0 | 1e+06 | `{'carrier': 'flight_carrier_0A', 'price': 277.15, 'depart_time': 'flight_depart_time_0C...` |
| 3 | %4 | GENERATE | 1 | ok | 634.596 | 800/300 | 4.4e+12 | `"[final_answer] Based on {'carrier': 'flight_carrier_0A', 'pri...; 'Str': synthesized r...` |
| 4 | %5 | VERIFY | 1 | ok | 54.199 | 400/1 | 1.6e+12 | `False` |
| 5 | %4 | GENERATE | 2 | ok | 611.129 | 800/300 | 4.4e+12 | `"[final_answer] Based on {'carrier': 'flight_carrier_0A', 'pri...; 'Str': synthesized r...` |
| 6 | %5 | VERIFY | 2 | ok | 47.466 | 400/1 | 1.6e+12 | `True` |

## Totals

| metric | value |
|---|---|
| sequential latency | 1461.85 ms |
| critical-path latency (unlimited parallelism) | 725.585 ms |
| tokens in / out | 2400 / 602 |
| FLOPs | 1.2e+13 |
| node calls (lm / api+db) | 7 (4 / 1) |
| retries / skipped | 1 / 0 |
| 70B single-shot reference | ~2500 ms, ~2.52e+14 FLOPs |
| pipeline FLOPs vs 70B baseline | 4.77% |


# guarded_answer — Answer from retrieved facts if verification passes, otherwise fall back to a generic answer (branch via guard + SELECT)

- program: `branch_select_demo`
- task: "Answer from retrieved facts if verification passes, otherwise fall back to a generic answer (branch via guard + SELECT)"
- validation: **valid**
- status: completed

## Execution trace

| seq | node | op | attempt | status | latency ms | tokens | FLOPs | value |
|---|---|---|---|---|---|---|---|---|
| 0 | %1 | SEARCH | 1 | ok | 124.735 | 0/0 | 0 | `[{'title': 'news_title_0A', 'source': 'news_source_0B', 'date': 'news_date_0C', 'catego...` |
| 1 | %2 | EXTRACT | 1 | ok | 8.098 | 0/0 | 1e+06 | `[{'title': 'news_title_0A', 'source': 'news_source_0B', 'date': 'news_date_0C'}, {'titl...` |
| 2 | %3 | GENERATE | 1 | ok | 598.39 | 800/300 | 4.4e+12 | `"[draft] Based on [{'title': 'news_title_0A', 'source':...; 'Str': synthesized result #...` |
| 3 | %4 | VERIFY | 1 | ok | 53.803 | 400/1 | 1.6e+12 | `True` |
| 4 | %5 | GENERATE | 1 | ok | 648.841 | 800/300 | 4.4e+12 | `'[polished_answer] Based on "[draft] Based on [{\'title\': \'news_ti...: synthesized re...` |
| 5 | %6 | GENERATE | 0 | skipped | 0 | 0/0 | 0 | `<skipped>` |
| 6 | %7 | SELECT | 1 | ok | 0.106 | 0/0 | 0 | `'[polished_answer] Based on "[draft] Based on [{\'title\': \'news_ti...: synthesized re...` |

## Totals

| metric | value |
|---|---|
| sequential latency | 1433.97 ms |
| critical-path latency (unlimited parallelism) | 1433.97 ms |
| tokens in / out | 2000 / 601 |
| FLOPs | 1.04e+13 |
| node calls (lm / api+db) | 6 (3 / 1) |
| retries / skipped | 0 / 1 |
| 70B single-shot reference | ~2500 ms, ~2.52e+14 FLOPs |
| pipeline FLOPs vs 70B baseline | 4.13% |

