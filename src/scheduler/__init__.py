"""Scheduler interface placeholder.

The first working implementation lives in src/optimizer/scheduler/
(list_scheduler.py): resource-constrained list scheduling over TaskIR DAGs
with executor pools per resource class.

Still planned here (v0.2+):
  - executor *binding* decisions (choose among a skill's candidate
    executors by cost/availability), not just class-level pooling
  - expected-latency analysis with retry probabilities (profiling-driven)
  - effect-token chains as ordering edges once the effect system lands
    (docs/effect-system-proposal.md)
"""
