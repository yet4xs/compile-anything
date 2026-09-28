# Phase 1 — End-to-End Demo Cost Reports

Simulator: deterministic mock executors, nominal costs (see `spec/skill-isa.md` §4).

# financial_analysis — Analyze the company's annual report, find the fastest growing business segment and explain why it grew

- program: `financial_analysis`
- task: "Analyze the company's annual report, find the fastest growing business segment and explain why it grew"
- validation: **valid**
- status: completed

## Execution trace

| seq | node | op | executor | attempt | status | latency ms | tokens | FLOPs | energy J | mem MB | value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | %1 | SEARCH | `tool:search_api [api]` | 1 | ok | 126.144 | 0/0 | 0 | 0.005 | 0 | `[{'name': 'product_name_0A', 'price': 821.12, 'rating': 126.46, 'stock': 'product_stock...` |
| 1 | %2 | EXTRACT | `python:dict [python]` | 1 | ok | 7.984 | 0/0 | 1e+06 | 0.005 | 30 | `[{'segment': None, 'revenue': None, 'growth_rate': None, 'region': None}, {'segment': N...` |
| 2 | %3 | FILTER | `python:listcomp [python]` | 1 | ok | 4.9 | 0/0 | 1e+06 | 0.005 | 30 | `[{'segment': None, 'revenue': None, 'growth_rate': None, 'region': None}, {'segment': N...` |
| 3 | %4 | ARGMAX | `python:mock [python]` | 1 | ok | 0.97 | 0/0 | 1e+06 | 0.005 | 30 | `{'segment': None, 'revenue': None, 'growth_rate': None, 'region': None}` |
| 4 | %5 | GENERATE | `lm:qwen2b-instruct [lm]` | 1 | ok | 615.835 | 800/300 | 4.4e+12 | 22 | 4600 | `"[explanation] Based on {'segment': None, 'revenue': None, 'g...; [{'segment': None, 'r...` |
| 5 | %6 | VERIFY | `lm:qwen2b-verifier [lm]` | 1 | ok | 45.094 | 400/1 | 1.6e+12 | 8.02 | 4600 | `True` |

## Totals

| metric | value |
|---|---|
| sequential latency | 800.927 ms |
| critical-path latency (unlimited parallelism) | 755.833 ms |
| tokens in / out | 1200 / 301 |
| FLOPs | 6e+12 |
| energy | 30.04 J |
| peak memory (single node) | 4600 MB |
| node calls (lm / api+db) | 6 (2 / 1) |
| retries / skipped | 0 / 0 |
| 70B single-shot reference | ~2500 ms, ~2.52e+14 FLOPs, ~900 J, ~140000 MB |
| pipeline FLOPs vs 70B baseline | 2.38% |
| pipeline energy vs 70B baseline | 3.34% |


# flight_cheapest — Find the cheapest flight arriving before 8 PM

- program: `find_cheapest_flight`
- task: "Find the cheapest flight arriving before 8 PM"
- validation: **valid**
- status: completed

## Execution trace

| seq | node | op | executor | attempt | status | latency ms | tokens | FLOPs | energy J | mem MB | value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | %1 | SEARCH | `tool:search_api [api]` | 1 | ok | 124.587 | 0/0 | 0 | 0.005 | 0 | `[{'carrier': 'flight_carrier_0A', 'price': 546.58, 'depart_time': 'flight_depart_time_0...` |
| 1 | %2 | FILTER | `python:listcomp [python]` | 1 | ok | 4.56 | 0/0 | 1e+06 | 0.005 | 30 | `[{'carrier': 'flight_carrier_0A', 'price': 546.58, 'depart_time': 'flight_depart_time_0...` |
| 2 | %3 | ARGMIN | `python:mock [python]` | 1 | ok | 1.061 | 0/0 | 1e+06 | 0.005 | 30 | `{'carrier': 'flight_carrier_4E', 'price': 157.41, 'depart_time': 'flight_depart_time_4G...` |

## Totals

| metric | value |
|---|---|
| sequential latency | 130.208 ms |
| critical-path latency (unlimited parallelism) | 130.208 ms |
| tokens in / out | 0 / 0 |
| FLOPs | 2e+06 |
| energy | 0.015 J |
| peak memory (single node) | 30 MB |
| node calls (lm / api+db) | 3 (0 / 1) |
| retries / skipped | 0 / 0 |
| 70B single-shot reference | ~2500 ms, ~2.52e+14 FLOPs, ~900 J, ~140000 MB |
| pipeline FLOPs vs 70B baseline | 0.00% |
| pipeline energy vs 70B baseline | 0.00% |


# flight_verified — Find the cheapest flight arriving before 8 PM, explain it, and verify the answer before returning

- program: `find_cheapest_flight_verified`
- task: "Find the cheapest flight arriving before 8 PM, explain it, and verify the answer before returning"
- validation: **valid**
- status: completed

## Execution trace

| seq | node | op | executor | attempt | status | latency ms | tokens | FLOPs | energy J | mem MB | value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | %1 | SEARCH | `tool:search_api [api]` | 1 | ok | 108.322 | 0/0 | 0 | 0.005 | 0 | `[{'carrier': 'flight_carrier_0A', 'price': 277.15, 'depart_time': 'flight_depart_time_0...` |
| 1 | %2 | FILTER | `python:listcomp [python]` | 1 | ok | 5.201 | 0/0 | 1e+06 | 0.005 | 30 | `[{'carrier': 'flight_carrier_0A', 'price': 277.15, 'depart_time': 'flight_depart_time_0...` |
| 2 | %3 | ARGMIN | `python:mock [python]` | 1 | ok | 0.933 | 0/0 | 1e+06 | 0.005 | 30 | `{'carrier': 'flight_carrier_0A', 'price': 277.15, 'depart_time': 'flight_depart_time_0C...` |
| 3 | %4 | GENERATE | `lm:qwen2b-instruct [lm]` | 1 | ok | 634.596 | 800/300 | 4.4e+12 | 22 | 4600 | `"[final_answer] Based on {'carrier': 'flight_carrier_0A', 'pri...; '<task input text>':...` |
| 4 | %5 | VERIFY | `lm:qwen2b-verifier [lm]` | 1 | ok | 54.199 | 400/1 | 1.6e+12 | 8.02 | 4600 | `False` |
| 5 | %4 | GENERATE | `lm:qwen2b-instruct [lm]` | 2 | ok | 611.129 | 800/300 | 4.4e+12 | 22 | 4600 | `"[final_answer] Based on {'carrier': 'flight_carrier_0A', 'pri...; '<task input text>':...` |
| 6 | %5 | VERIFY | `lm:qwen2b-verifier [lm]` | 2 | ok | 47.466 | 400/1 | 1.6e+12 | 8.02 | 4600 | `True` |

## Totals

| metric | value |
|---|---|
| sequential latency | 1461.85 ms |
| critical-path latency (unlimited parallelism) | 725.585 ms |
| tokens in / out | 2400 / 602 |
| FLOPs | 1.2e+13 |
| energy | 60.055 J |
| peak memory (single node) | 4600 MB |
| node calls (lm / api+db) | 7 (4 / 1) |
| retries / skipped | 1 / 0 |
| 70B single-shot reference | ~2500 ms, ~2.52e+14 FLOPs, ~900 J, ~140000 MB |
| pipeline FLOPs vs 70B baseline | 4.77% |
| pipeline energy vs 70B baseline | 6.67% |


# guarded_answer — Answer from retrieved facts if verification passes, otherwise fall back to a generic answer (branch via guard + SELECT)

- program: `branch_select_demo`
- task: "Answer from retrieved facts if verification passes, otherwise fall back to a generic answer (branch via guard + SELECT)"
- validation: **valid**
- status: completed

## Execution trace

| seq | node | op | executor | attempt | status | latency ms | tokens | FLOPs | energy J | mem MB | value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | %1 | SEARCH | `tool:search_api [api]` | 1 | ok | 124.735 | 0/0 | 0 | 0.005 | 0 | `[{'title': 'news_title_0A', 'source': 'news_source_0B', 'date': 'news_date_0C', 'catego...` |
| 1 | %2 | EXTRACT | `python:dict [python]` | 1 | ok | 8.098 | 0/0 | 1e+06 | 0.005 | 30 | `[{'title': 'news_title_0A', 'source': 'news_source_0B', 'date': 'news_date_0C'}, {'titl...` |
| 2 | %3 | GENERATE | `lm:qwen2b-instruct [lm]` | 1 | ok | 598.39 | 800/300 | 4.4e+12 | 22 | 4600 | `"[draft] Based on [{'title': 'news_title_0A', 'source':...; '<task input text>': synthe...` |
| 3 | %4 | VERIFY | `lm:qwen2b-verifier [lm]` | 1 | ok | 53.803 | 400/1 | 1.6e+12 | 8.02 | 4600 | `True` |
| 4 | %5 | GENERATE | `lm:qwen2b-instruct [lm]` | 1 | ok | 648.841 | 800/300 | 4.4e+12 | 22 | 4600 | `'[polished_answer] Based on "[draft] Based on [{\'title\': \'news_ti...: synthesized re...` |
| 5 | %6 | GENERATE | `lm:qwen2b-instruct [lm]` | 0 | skipped | 0 | 0/0 | 0 | 0 | 0 | `<skipped>` |
| 6 | %7 | SELECT | `runtime:mock [runtime]` | 1 | ok | 0.106 | 0/0 | 0 | 0.005 | 30 | `'[polished_answer] Based on "[draft] Based on [{\'title\': \'news_ti...: synthesized re...` |

## Totals

| metric | value |
|---|---|
| sequential latency | 1433.97 ms |
| critical-path latency (unlimited parallelism) | 1433.97 ms |
| tokens in / out | 2000 / 601 |
| FLOPs | 1.04e+13 |
| energy | 52.035 J |
| peak memory (single node) | 4600 MB |
| node calls (lm / api+db) | 6 (3 / 1) |
| retries / skipped | 0 / 1 |
| 70B single-shot reference | ~2500 ms, ~2.52e+14 FLOPs, ~900 J, ~140000 MB |
| pipeline FLOPs vs 70B baseline | 4.13% |
| pipeline energy vs 70B baseline | 5.78% |


# rtl_debug — The FIFO overflow assertion in fifo_ctrl.sv fires in simulation; load the design artifacts and waveform, find the failing window, retrieve similar known bugs, and generate a verified RTL fix

- program: `rtl_debug`
- task: "The FIFO overflow assertion in fifo_ctrl.sv fires in simulation; load the design artifacts and waveform, find the failing window, retrieve similar known bugs, and generate a verified RTL fix"
- validation: **valid**
- status: completed

## Execution trace

| seq | node | op | executor | attempt | status | latency ms | tokens | FLOPs | energy J | mem MB | value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | %1 | LOAD | `python:fs [python]` | 1 | ok | 19.368 | 0/0 | 1e+06 | 0.005 | 30 | `{'source': 'fifo_ctrl.sv + waveform.vcd', 'rows': [{'name': 'product_name_0A', 'price':...` |
| 1 | %2 | EXTRACT | `python:dict [python]` | 1 | ok | 8.765 | 0/0 | 1e+06 | 0.005 | 30 | `{'assertion_failures': None, 'fifo_count': None, 'depth': None}` |
| 2 | %3 | SEARCH | `tool:search_api [api]` | 1 | ok | 112.936 | 0/0 | 0 | 0.005 | 0 | `[{'name': 'product_name_0A', 'price': 732.47, 'rating': 75.71, 'stock': 'product_stock_...` |
| 3 | %4 | CODEGEN | `lm:qwen2b-coder [lm]` | 1 | ok | 966.636 | 1000/600 | 6.4e+12 | 32 | 4600 | `'def solution_4():\n    return 41'` |
| 4 | %5 | VERIFY | `lm:qwen2b-verifier [lm]` | 1 | ok | 53.192 | 400/1 | 1.6e+12 | 8.02 | 4600 | `True` |

## Totals

| metric | value |
|---|---|
| sequential latency | 1160.9 ms |
| critical-path latency (unlimited parallelism) | 1079.57 ms |
| tokens in / out | 1400 / 601 |
| FLOPs | 8e+12 |
| energy | 40.035 J |
| peak memory (single node) | 4600 MB |
| node calls (lm / api+db) | 5 (2 / 1) |
| retries / skipped | 0 / 0 |
| 70B single-shot reference | ~2500 ms, ~2.52e+14 FLOPs, ~900 J, ~140000 MB |
| pipeline FLOPs vs 70B baseline | 3.18% |
| pipeline energy vs 70B baseline | 4.45% |

