# Real Dataset Coverage Analysis

## humaneval

- samples: **164**  lifted: 12 (7.32%)  valid: 12 (7.32%)  executed/valid: 100.0%

### reject reasons

| reason | count |
|---|---|
| LOOP | 87 |
| PATTERN_UNSUPPORTED | 65 |

### skill coverage

| skill | count |
|---|---|
| FILTER | 6 |
| TRANSFORM | 5 |
| SUM | 2 |
| COUNT | 1 |
| JOIN | 1 |

### graph: nodes mean 1.25, depth max 3, width max 1, branch ratio 0.0, parallel ratio 0.0

## mbpp

- samples: **974**  lifted: 32 (3.29%)  valid: 32 (3.29%)  executed/valid: 100.0%

### reject reasons

| reason | count |
|---|---|
| PATTERN_UNSUPPORTED | 503 |
| LOOP | 439 |

### skill coverage

| skill | count |
|---|---|
| TRANSFORM | 27 |
| SUM | 3 |
| FILTER | 2 |
| MIN | 1 |
| COUNT | 1 |

### graph: nodes mean 1.062, depth max 2, width max 1, branch ratio 0.0, parallel ratio 0.0

## spider

- samples: **8034**  lifted: 8034 (100.0%)  valid: 8034 (100.0%)  executed/valid: 100.0%

### reject reasons

| reason | count |
|---|---|

### skill coverage

| skill | count |
|---|---|
| QUERY_DB | 8034 |
| GENERATE | 8034 |
| VERIFY | 8034 |
| EXTRACT | 2211 |

### graph: nodes mean 3.275, depth max 4, width max 1, branch ratio 0.0, parallel ratio 0.3104

## toolbench

- samples: **15**  lifted: 15 (100.0%)  valid: 15 (100.0%)  executed/valid: 100.0%

### reject reasons

| reason | count |
|---|---|

### skill coverage

| skill | count |
|---|---|
| EXEC_ACTION | 95 |
| SEARCH | 34 |
| EXTRACT | 19 |
| GENERATE | 15 |
| VERIFY | 15 |
| FETCH | 15 |
| QUERY_DB | 2 |
| CODEGEN | 1 |
| EXTRACT_ENTITIES | 1 |

### graph: nodes mean 13.133, depth max 20, width max 1, branch ratio 0.0, parallel ratio 0.0892

## verilogeval

- samples: **312**  lifted: 312 (100.0%)  valid: 312 (100.0%)  executed/valid: 100.0%

### reject reasons

| reason | count |
|---|---|

### skill coverage

| skill | count |
|---|---|
| LOAD | 312 |
| EXTRACT | 312 |
| SEARCH | 312 |
| CODEGEN | 312 |
| VERIFY | 312 |

### graph: nodes mean 5.0, depth max 4, width max 2, branch ratio 0.0, parallel ratio 0.4
