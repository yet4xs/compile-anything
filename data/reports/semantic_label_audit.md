# Semantic Label Audit (deterministic, pre-training)

total = 31215; verified 35.406% / suspect 64.594% / unverifiable 0.0%

## by tier

| tier | n | verified% | suspect% | unver% |
|---|---:|---:|---:|---:|
| A | 15755 | 56.084 | 43.916 | 0.0 |
| B | 15460 | 14.334 | 85.666 | 0.0 |

## by source

| source | n | verified% | suspect% | unver% |
|---|---:|---:|---:|---:|
| humaneval | 12 | 66.667 | 33.333 | 0.0 |
| mbpp | 32 | 75.0 | 25.0 | 0.0 |
| spider | 8034 | 92.42 | 7.58 | 0.0 |
| toolbench_static | 1089 | 46.832 | 53.168 | 0.0 |
| verilogeval | 312 | 95.513 | 4.487 | 0.0 |
| xlam | 21736 | 12.822 | 87.178 | 0.0 |

## top error patterns

| pattern | count |
|---|---:|
| ARGUMENT_LOSS | 19528 |
| GROUND_TRUTH_AMBIGUOUS | 1016 |
| MISSING_CONSTRAINT | 662 |
| WRONG_ORDER | 2 |
| EXTRA_ACTION | 1 |
