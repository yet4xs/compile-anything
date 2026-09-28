# Training Corpus Quality Audit

## Lowering provenance

- exact: **15.58%**  heuristic: 29.22%  fallback: **55.2%**
- EXEC_ACTION: intentional 2382 vs unknown-fallback 56588

## Per dataset

| source | samples | calls | exact | heuristic | fallback % | semantic_mapping_coverage % |
|---|---|---|---|---|---|---|
| toolbench | 15 | 148 | 10 | 43 | 64.19 | 0.0 |
| toolbench_static | 2356 | 2356 | 496 | 593 | 53.78 | 46.22 |
| xlam | 60000 | 100011 | 15464 | 29321 | 55.22 | 36.23 |
| humaneval | 12 | 0 | 0 | 0 | 0.0 | 100.0 |
| mbpp | 32 | 0 | 0 | 0 | 0.0 | 100.0 |
| spider | 8034 | 0 | 0 | 0 | 0.0 | 100.0 |
| verilogeval | 312 | 0 | 0 | 0 | 0.0 | 100.0 |

xLAM detail: {"total": 60000, "zero_fallback": 21736, "some_fallback": 10190, "all_fallback": 28074}

## Policy tail effect (GENERATE/VERIFY)

- plan view GENERATE count: 0, VERIFY: 312
- with tail GENERATE: 70405, VERIFY: 70717

## Instruction-only ambiguity (Task 5)

- exact groups: 58743, ambiguous: 499 (0.85%)
- near-dup families: 10381, ambiguous: 386 (3.72%)
- ambiguous w/ differing capability context: 0; same context: 499
