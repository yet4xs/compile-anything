# External Overlap Audit (training contamination, report-only)

v3 train = 28093 instructions. Overlapping samples are NOT removed.

| benchmark | n | exact % | near pairs | tool-family % | composition % | classification |
|---|---:|---:|---:|---:|---:|---|
| bfcl_v4 | 4696 | 0.021 | 4 | 100.0 | 57.08 | cross-dataset (same tool families) |
| tau3_bench | 2546 | 0.0 | 0 | None | None | strong OOD |
| agentboard_tools | 100 | 0.0 | 0 | None | None | strong OOD |
| agentboard_webshop | 251 | 0.0 | 0 | None | None | strong OOD |
| rtl_repo | 1174 | 0.0 | 0 | None | None | strong OOD |
| bird_mini_dev | 500 | 0.0 | 0 | None | None | strong OOD |
