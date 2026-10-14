# Phase 9A-0: Real Runtime Dataflow Feasibility Audit

**Date**: 2024-10-14 | **CPU-only, no training, no synthetic data**

## Audit Results

| Source | Total | Multi-tool | Runtime Value Dep (L3) |
|--------|-------|-----------|----------------------|
| BFCL V4 | 3,621 | **0** | **0** |
| ToolBench (in) | 1,588 | 50 | **0** |
| ToolBench (out) | 781 | 27 | **0** |
| τ³-bench | ~2,546 | many | UNKNOWN (args not stored) |
| xLAM | ~2,800 | few | **0** |
| **Total** | **~11,300** | — | **0 confirmed** |

## Key Finding

**No existing tool-use benchmark evaluates runtime value dependencies.** BFCL tests single-tool selection + parameter extraction. ToolBench's multi-tool tasks are independent parallel calls. This is a structural gap, not a rarity.

## Q1-Q3 Answers

Q1: No confirmed runtime_ref in existing benchmarks.
Q2: TaskIR v0.2 sufficient — no IR extension needed.
Q3: Yes — building runtime dataflow benchmark is NECESSARY.
