# TaskIR Dataset Statistics

Metric definitions in `src/stats.py`: depth = longest dependency chain; width = max nodes per depth level; branch_ratio = (guarded + SELECT) / nodes; parallel_ratio = 1 - critical_path_nodes / nodes.

### subset: examples

- programs: **3**  (total nodes: 15)
- nodes: mean 5, median 5, p90 7, min 3, max 7
- depth: mean 4.667, median 5, p90 6, min 3, max 6
- width: mean 1.333, median 1, p90 2, min 1, max 2
- branch_ratio: mean 0.143, median 0.0, p90 0.4286, min 0.0, max 0.4286
- parallel_ratio: mean 0.114, median 0.143, p90 0.2, min 0.0, max 0.2
- programs with width>1: 33.3%
- programs with retry:   33.3%

| skill | count |
|---|---|
| GENERATE | 4 |
| SEARCH | 3 |
| FILTER | 2 |
| ARGMIN | 2 |
| VERIFY | 2 |
| EXTRACT | 1 |
| SELECT | 1 |

### subset: synthetic

- programs: **1000**  (total nodes: 4761)
- nodes: mean 4.761, median 5.0, p90 6, min 3, max 7
- depth: mean 4.271, median 4.0, p90 5, min 3, max 6
- width: mean 1.49, median 1.0, p90 2, min 1, max 2
- branch_ratio: mean 0.086, median 0.0, p90 0.5, min 0.0, max 0.5
- parallel_ratio: mean 0.173, median 0.167, p90 0.2857, min 0.0, max 0.4
- programs with width>1: 49.0%
- programs with retry:   41.6%

| skill | count |
|---|---|
| GENERATE | 1344 |
| SEARCH | 837 |
| VERIFY | 588 |
| CALCULATE | 324 |
| EXTRACT_ENTITIES | 191 |
| SELECT | 172 |
| LOAD | 162 |
| EXTRACT | 162 |
| COMPARE | 162 |
| FETCH | 157 |
| SUMMARIZE | 157 |
| MERGE | 156 |
| SORT | 70 |
| FILTER | 63 |
| DEDUP | 54 |
| MAX | 54 |
| MIN | 37 |
| ARGMAX | 37 |
| ARGMIN | 34 |

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

- programs: **1103**  (total nodes: 5296)
- nodes: mean 4.801, median 5, p90 6, min 3, max 7
- depth: mean 4.356, median 4, p90 6, min 3, max 7
- width: mean 1.445, median 1, p90 2, min 1, max 2
- branch_ratio: mean 0.078, median 0.0, p90 0.5, min 0.0, max 0.5
- parallel_ratio: mean 0.175, median 0.2, p90 0.2857, min 0.0, max 0.4
- programs with width>1: 44.5%
- programs with retry:   46.9%

| skill | count |
|---|---|
| GENERATE | 1448 |
| SEARCH | 930 |
| VERIFY | 690 |
| CALCULATE | 334 |
| EXTRACT | 253 |
| EXTRACT_ENTITIES | 191 |
| SELECT | 173 |
| LOAD | 162 |
| COMPARE | 162 |
| FETCH | 157 |
| SUMMARIZE | 157 |
| MERGE | 156 |
| SORT | 70 |
| FILTER | 65 |
| SEND | 60 |
| DEDUP | 54 |
| MAX | 54 |
| MIN | 37 |
| ARGMAX | 37 |
| ARGMIN | 36 |
| EXEC_ACTION | 20 |
| TRANSLATE | 20 |
| CONVERT | 20 |
| QUERY_DB | 10 |
