# TaskIR Dataset Statistics

Metric definitions in `src/stats.py`: depth = longest dependency chain; width = max nodes per depth level; branch_ratio = (guarded + SELECT) / nodes; parallel_ratio = 1 - critical_path_nodes / nodes.

### subset: examples

- programs: **5**  (total nodes: 26)
- nodes: mean 5.2, median 5, p90 7, min 3, max 7
- depth: mean 4.8, median 5, p90 6, min 3, max 6
- width: mean 1.4, median 1, p90 2, min 1, max 2
- branch_ratio: mean 0.086, median 0.0, p90 0.4286, min 0.0, max 0.4286
- parallel_ratio: mean 0.182, median 0.167, p90 0.4, min 0.0, max 0.4
- programs with width>1: 40.0%
- programs with retry:   60.0%

| skill | count |
|---|---|
| SEARCH | 5 |
| GENERATE | 5 |
| VERIFY | 4 |
| EXTRACT | 3 |
| FILTER | 3 |
| ARGMIN | 2 |
| ARGMAX | 1 |
| SELECT | 1 |
| LOAD | 1 |
| CODEGEN | 1 |

### subset: synthetic

- programs: **1000**  (total nodes: 4744)
- nodes: mean 4.744, median 5.0, p90 6, min 3, max 7
- depth: mean 4.264, median 4.0, p90 5, min 3, max 6
- width: mean 1.48, median 1.0, p90 2, min 1, max 2
- branch_ratio: mean 0.08, median 0.0, p90 0.5, min 0.0, max 0.5
- parallel_ratio: mean 0.173, median 0.183, p90 0.2857, min 0.0, max 0.4
- programs with width>1: 48.0%
- programs with retry:   41.7%

| skill | count |
|---|---|
| GENERATE | 1318 |
| SEARCH | 823 |
| VERIFY | 576 |
| CALCULATE | 330 |
| EXTRACT_ENTITIES | 189 |
| FETCH | 168 |
| SUMMARIZE | 168 |
| LOAD | 165 |
| EXTRACT | 165 |
| COMPARE | 165 |
| SELECT | 159 |
| MERGE | 156 |
| DEDUP | 76 |
| FILTER | 63 |
| SORT | 60 |
| ARGMAX | 52 |
| MAX | 39 |
| MIN | 36 |
| ARGMIN | 36 |

### subset: xlam

- programs: **100**  (total nodes: 520)
- nodes: mean 5.2, median 5.0, p90 6, min 4, max 7
- depth: mean 5.2, median 5.0, p90 6, min 4, max 7
- width: mean 1, median 1.0, p90 1, min 1, max 1
- branch_ratio: mean 0.0, median 0.0, p90 0.0, min 0.0, max 0.0
- parallel_ratio: mean 0.198, median 0.2, p90 0.25, min 0.1429, max 0.25
- programs with width>1: 0.0%
- programs with retry:   100.0%

| skill | count |
|---|---|
| GENERATE | 100 |
| VERIFY | 100 |
| SEARCH | 90 |
| EXTRACT | 90 |
| SEND | 60 |
| EXEC_ACTION | 20 |
| TRANSLATE | 20 |
| CONVERT | 20 |
| QUERY_DB | 10 |
| CALCULATE | 10 |

### subset: ALL

- programs: **1105**  (total nodes: 5290)
- nodes: mean 4.787, median 5, p90 6, min 3, max 7
- depth: mean 4.351, median 4, p90 5, min 3, max 7
- width: mean 1.436, median 1, p90 2, min 1, max 2
- branch_ratio: mean 0.072, median 0.0, p90 0.5, min 0.0, max 0.5
- parallel_ratio: mean 0.175, median 0.2, p90 0.2857, min 0.0, max 0.4
- programs with width>1: 43.6%
- programs with retry:   47.1%

| skill | count |
|---|---|
| GENERATE | 1423 |
| SEARCH | 918 |
| VERIFY | 680 |
| CALCULATE | 340 |
| EXTRACT | 258 |
| EXTRACT_ENTITIES | 189 |
| FETCH | 168 |
| SUMMARIZE | 168 |
| LOAD | 166 |
| COMPARE | 165 |
| SELECT | 160 |
| MERGE | 156 |
| DEDUP | 76 |
| FILTER | 66 |
| SORT | 60 |
| SEND | 60 |
| ARGMAX | 53 |
| MAX | 39 |
| ARGMIN | 38 |
| MIN | 36 |
| EXEC_ACTION | 20 |
| TRANSLATE | 20 |
| CONVERT | 20 |
| QUERY_DB | 10 |
| CODEGEN | 1 |
