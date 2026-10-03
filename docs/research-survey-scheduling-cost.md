# Research Survey: Cost-Aware Scheduling & Resource Optimization for LLM-Based Agent Systems (2024–2026)

Compiled 2026-09-29. All arXiv IDs verified against arxiv.org abstract pages.

**Context for this survey.** Our Task Scheduler performs *resource-constrained list scheduling of TaskIR DAGs* across heterogeneous executors: a small LM, Python, external APIs, and a database. The papers below were selected for direct relevance to (1) DAG/workflow-level scheduling of LLM agent workloads over heterogeneous compute, (2) cost/latency/energy optimization in multi-step LLM systems, (3) small-model-first cascading and routing, and (4) hardware-aware deployment. Sections A–B are most load-bearing for our design; C supports our model-selection policy; D supports edge/cloud placement claims.

---

## A. Workflow-level scheduling of LLM agent workloads (closest to TaskIR DAG scheduling)

### A1. Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents
- **Authors:** J.G. Dai, Tianze Deng, Yueying Li, Tianyi Peng
- **arXiv:** https://arxiv.org/abs/2504.07347 (2025-04, rev. 2026-05)
- **Summary:** Develops queueing-theoretic foundations for LLM serving, proving that broad classes of "work-conserving" scheduling policies are throughput-optimal for AI-agent workloads with DAG and fork-join routing. Shows Orca and Sarathi-Serve are maximally stable while vanilla vLLM and FasterTransformer are not.
- **Relevance:** Theoretical backbone for scheduling LLM-agent DAGs. Gives us vocabulary (work-conserving policies, stability under DAG/fork-join arrival) to justify our list-scheduling policy's throughput properties; useful for the paper's analysis section rather than implementation.

### A2. Halo: Batch Query Processing and Optimization for Agentic Workflows
- **Authors:** Junyi Shen, Noppanat Wadlom, Yao Lu
- **arXiv:** https://arxiv.org/abs/2509.02121 (2025-09, rev. 2026-01)
- **Summary:** Treats agentic LLM workflows as structured query-plan DAGs and consolidates batched queries into a shared graph to eliminate redundant computation. Its cost model covers cache reuse, prefill/decode costs, and GPU placement, achieving up to 3.6x batch speedup and 2.6x throughput gains without quality loss.
- **Relevance:** *Highly relevant.* The closest published analogue to our Task Scheduler: agentic workflow as a query-plan DAG over heterogeneous executors with an explicit cost model (compute + cache + placement). Their cost-model structure is a good comparison point and baseline framing for a "list scheduling over TaskIR DAG" paper.

### A3. Helium: Efficient LLM Serving for Agentic Workflows — A Data Systems Perspective
- **Authors:** Noppanat Wadlom, Junyi Shen, Yao Lu
- **arXiv:** https://arxiv.org/abs/2603.16104 (2026-03)
- **Summary:** Workflow-aware serving framework that models agentic workloads as query plans with LLM calls as first-class operators, enabling cross-call optimization via proactive caching and cache-aware scheduling that reuses prompts, KV states, and workflow results. Reports up to 1.56x speedup over state-of-the-art agent serving.
- **Relevance:** Frames LLM calls as operators inside a data-system query plan — the same abstraction level as our TaskIR executors (LM/Python/API/DB). Their cache-aware scheduling ideas complement our resource-constrained list scheduling.

### A4. SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters
- **Authors:** Dongxin Guo, Jikun Wu, Siu Ming Yiu
- **arXiv:** https://arxiv.org/abs/2605.00528 (2026-05, rev. 2026-06)
- **Summary:** Argues that scheduling individual LLM inference calls as separate units wastes intermediate state and multiplies latency; schedules entire agent workflows atomically instead. On a 64-GPU cluster running agent benchmarks (e.g., SWE-bench), cuts task completion time by 1.64x vs vLLM, trading ~30% lower peak throughput for latency.
- **Relevance:** Direct evidence for the latency-vs-throughput tradeoff in workflow-granularity scheduling decisions — exactly the axis our scheduler tunes when deciding executor allocation per TaskIR node.

### A5. Latency-Aware Orchestration for Multi-Agent LLM Workflows on Heterogeneous GPUs
- **Authors:** Jinghao Wang, Yifeng Zhang, Xiao Zhou, Yao Lu, et al.
- **arXiv:** https://arxiv.org/abs/2609.03335 (2026-09)
- **Summary:** Prediction-guided runtime that forecasts device-specific activation latency, peak memory, and model-loading cost, then propagates these estimates through workflow dependencies to build and optimize a physical execution graph over heterogeneous GPU pools. Cuts makespan and p95 latency under bursts by up to 36.8% / 25.9% and saves up to 24.63 GPU-s per session.
- **Relevance:** *Highly relevant.* Dependency-propagated cost prediction + physical execution-graph construction over heterogeneous devices is structurally the same problem as our list scheduling over LM/Python/API/DB executors (they handle GPU heterogeneity; we handle executor-class heterogeneity). Strong related-work anchor.

### A6. Justitia: Fair and Efficient Scheduling of Task-parallel LLM Agents with Selective Pampering
- **Authors:** Mingyan Yang, Guanjie Wang, et al. (SJTU)
- **arXiv:** https://arxiv.org/abs/2510.17015 (2025-10, rev. 2026-03)
- **Summary:** Scheduler for task-parallel LLM agents on shared GPU servers that selectively prioritizes agents by expected completion order under idealized fair sharing. Quantifies agent cost in a memory-centric way, predicts costs with a lightweight method, and applies virtual-time fair queuing on vLLM.
- **Relevance:** Their lightweight agent-cost prediction (memory-centric) is a useful reference for our per-task cost estimator; relevant if we extend to concurrent sessions sharing one small-LM executor.

### A7. Parrot: Efficient Serving of LLM-based Applications with Semantic Variable
- **Authors:** Chaofan Lin, Zhenhua Han, et al. (Microsoft Research; OSDI 2024)
- **arXiv:** https://arxiv.org/abs/2405.19888 (2024-05)
- **Summary:** Introduces "Semantic Variable," an abstraction annotating input/output variables in prompts to expose dataflow between LLM requests of an application. This application-level dataflow knowledge enables cross-request optimizations with up to an order-of-magnitude end-to-end speedup.
- **Relevance:** Foundational argument that exposing the *dataflow graph* of an LLM application (vs. treating calls as an opaque stream) unlocks scheduling wins — the core premise of TaskIR's explicit DAG.

### A8. Teola: Towards End-to-End Optimization of LLM-based Applications
- **Authors:** Xin Tan, Yimin Jiang, Yitao Yang, Hong Xu (ASPLOS 2025)
- **arXiv:** https://arxiv.org/abs/2407.00326 (2024-06, rev. 2025-03)
- **Summary:** Orchestration framework using fine-grained "task primitives" and primitive-level dataflow graphs so optimization spans the whole LLM application rather than coarse task modules. Achieves up to 2.09x speedup over existing systems.
- **Relevance:** *Highly relevant.* Primitive-level dataflow graphs over heterogeneous stages are essentially TaskIR; validates that primitive-granularity (not agent-granularity) scheduling wins end-to-end.

### A9. Alto: Orchestrating Distributed Compound AI Systems with Nested Ancestry
- **Authors:** Deepti Raghavan, Matei Zaharia, et al. (Stanford)
- **arXiv:** https://arxiv.org/abs/2403.04311 (2024-03, rev. 2025-07)
- **Summary:** Automatically optimizes compound AI query execution via streaming and parallelism across heterogeneous components, using "nested ancestry" metadata to track partial outputs. Implementations of four applications match or beat LangGraph baselines with 10–30% latency improvements.
- **Relevance:** Compound AI over heterogeneous components (models, retrievers, tools) with dataflow-aware orchestration; explicit LangGraph comparison is useful since our compiler emits an agent-graph-style IR.

### A10. From Static Templates to Dynamic Runtime Graphs: A Survey of Workflow Optimization for LLM Agents
- **Authors:** Ling Yue, Kushal Raj Bhandari, Ching-Yun Ko, Dhaval Patel, et al. (IBM/VCU/RPI)
- **arXiv:** https://arxiv.org/abs/2603.22386 (2026-03)
- **Summary:** Survey treating LLM agent workflows as agentic computation graphs; organizes optimization methods by when structure is fixed (static pre-deployment templates vs. dynamic per-run adaptation) and by what is optimized and which feedback signals guide it. Companion repo: `IBM/awesome-agentic-workflow-optimization`.
- **Relevance:** Best single entry point for positioning our static TaskIR compilation + list scheduling within the landscape of workflow-optimization work.

---

## B. Request/cluster-level scheduling, latency SLOs, and GPU efficiency

### B1. Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference
- **Authors:** Amey Agrawal, Nitin Kedia, et al. (OSDI 2024)
- **arXiv:** https://arxiv.org/abs/2403.02310 (2024-03)
- **Summary:** Inference scheduler using chunked prefills and stall-free batching that coalesces decode and prefill compute; delivers capacity gains up to 2.6x (Mistral-7B) and 5.6x (Falcon-180B) vs vLLM while meeting tail-latency (TPOT) targets.
- **Relevance:** The standard reference for the throughput-latency tradeoff inside a single LM executor; our scheduler treats the small-LM executor as a latency/throughput resource, and this defines its internal scheduling behavior.

### B2. LLM Query Scheduling with Prefix Reuse and Latency Constraints
- **Authors:** Gregory Dexter, Shao Tang, A. Fatahi Baarzi, et al.
- **arXiv:** https://arxiv.org/abs/2502.04677 (2025-02, rev. 2026-01)
- **Summary:** Proves LLM query scheduling with prefix reuse (RadixAttention) under TTFT constraints is NP-hard; shows limits of FCFS and longest-prefix-match, and proposes k-LPM balancing prefix reuse against fairness with TTFT guarantees and large P99 TTFT reductions.
- **Relevance:** Formal complexity grounding for LLM request scheduling; supports our claim that DAG-level heuristics (list scheduling) are the right pragmatic choice once NP-hardness kicks in at even the single-queue level.

### B3. ServerlessLLM: Low-Latency Serverless Inference for Large Language Models
- **Authors:** Yao Fu, Dmitrii Ustiugov, et al. (OSDI 2024)
- **arXiv:** https://arxiv.org/abs/2406.09161 (2024-06)
- **Summary:** Addresses cold-start latency when loading model checkpoints to GPUs in serverless LLM inference via a storage-aware, locality-enhanced multi-tier loading design, beating serverless baselines on startup latency.
- **Relevance:** Hardware/deployment-aware placement — relevant when our small-LM executor is instantiated on demand (weights on local disk vs. network), and for arguments about local-first deployment economics.

---

## C. Model cascading and routing (small model first, escalate to large)

### C1. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance
- **Authors:** Lingjiao Chen, Matei Zaharia, James Zou (2023, foundational pre-2024 reference)
- **arXiv:** https://arxiv.org/abs/2305.05176
- **Summary:** The original LLM cascade: answer with a cheap model, escalate to progressively stronger/expensive models via confidence checks; matches GPT-4-level accuracy at up to ~98% lower cost on some tasks.
- **Relevance:** Canonical citation for "small model first, escalate on failure" — the same policy our scheduler applies at *task* granularity when a small-LM node's output fails validation and is re-queued or escalated.

### C2. RouteLLM: Learning to Route LLMs with Preference Data
- **Authors:** Isaac Ong, Amjad Almahairi, et al. (LMSYS, ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2406.18665 (2024-06)
- **Summary:** Trains lightweight router models on preference data to dynamically select between a stronger and weaker LLM per query at inference, cutting cost substantially (e.g., >2x on some benchmarks) while retaining most quality.
- **Relevance:** Learned per-query routing between small/large models; a candidate replacement for hand-tuned difficulty heuristics in our LM-node dispatch policy.

### C3. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing
- **Authors:** Dujian Ding, Bijaya Adhikari, et al. (Microsoft)
- **arXiv:** https://arxiv.org/abs/2404.14618 (2024-04, ICLR)
- **Summary:** Router assigns queries to small vs. large model based on jointly learned query-difficulty and model-capability representations under a target output quality, reducing large-model queries by up to 40% with no quality loss. Code: `microsoft/best-route-llm`.
- **Relevance:** Quality-aware (not just accuracy-aware) routing objective — matches our setting where task outputs must satisfy typed validators, i.e., route by *verifiability*, not raw difficulty.

### C4. A Unified Approach to Routing and Cascading for LLMs
- **Authors:** Jasper Dekoninck, Maximilian Baader, Martin Vechev (ETH; ICML 2025)
- **arXiv:** https://arxiv.org/abs/2410.10347 (2024-10, rev. 2025-05)
- **Summary:** Derives an optimal cascade strategy, proves optimality of an existing routing strategy, and unifies both as "cascade routing"; experiments show cascade routing dominates either alone, with quality-estimator fidelity identified as the critical factor.
- **Relevance:** Gives optimality theory for combine-router-with-fallback policies — directly applicable to our per-node "small LM + validator + escalate" strategy, and useful to cite when justifying it.

### C5. Cascadia: An Efficient Cascade Serving System for Large Language Models
- **Authors:** Youhe Jiang, Fangcheng Fu, Nicholas D. Lane, Binhang Yuan, et al.
- **arXiv:** https://arxiv.org/abs/2506.04203 (2025-06, rev. 2025-09)
- **Summary:** Cascade serving framework co-optimizing system deployment (model parallelism, batching — via MILP) and request routing (Chebyshev-guided); achieves up to 4x (2.3x avg) tighter latency SLOs at ~2.4x higher throughput at matched quality.
- **Relevance:** Shows cascading is a *systems* problem (deployment + routing co-design), not just a model-selection trick; strengthens our framing that scheduler and model-tier choice must be co-optimized.

---

## D. Energy efficiency and edge/cloud heterogeneous deployment

### D1. Towards Greener LLMs: Bringing Energy-Efficiency to the Forefront of LLM Inference
- **Authors:** Jovan Stojkovic, Esha Choukse, et al. (Microsoft Research; IISWC 2024)
- **arXiv:** https://arxiv.org/abs/2403.20306 (2024-03)
- **Summary:** Characterizes tuning knobs (input-dependent, model-dependent, SLA-dependent) for energy-efficient LLM serving under performance SLOs, analyzing their effect on latency, throughput, and energy on A100/H100-scale hardware.
- **Relevance:** Source for energy/cost knobs and tradeoff curves when we claim resource-constrained operation (energy as a budget dimension alongside latency and money).

### D2. EdgeShard: Efficient LLM Inference via Collaborative Edge Computing
- **Authors:** Mingjin Zhang, Jiannong Cao, et al.
- **arXiv:** https://arxiv.org/abs/2405.14371 (2024-05)
- **Summary:** Splits LLMs into shards distributed across heterogeneous edge devices and cloud servers, with a dynamic-programming algorithm for device selection and model partitioning; up to 50% lower latency and 2x throughput vs. baselines.
- **Relevance:** Classic formulation of "which piece of the model runs where" across an edge-cloud hierarchy via dynamic programming — analogous to our assignment of TaskIR nodes to executor classes under resource constraints.

---

## Synthesis: what this means for our Task Scheduler

1. **The DAG-scheduling framing is now established territory (2025–2026).** Halo (A2), Helium (A3), Teola (A8), Parrot (A7), and the 2026 survey (A10) all model LLM agent programs as graphs with heterogeneous operators, and SAGA (A4) and Wang et al. (A5) schedule whole workflows on shared/heterogeneous hardware. Our novelty must therefore be stated precisely: *list scheduling under explicit resource constraints over executor classes that include non-LLM executors (Python, API, DB) with a typed TaskIR*, rather than graph modeling per se.
2. **Cost models are the differentiator.** Halo's cost model (cache + prefill/decode + placement) and A5's predicted latency/memory/loading propagation are the two strongest baselines to compare our cost estimator against; Dexter et al. (B2) supplies NP-hardness arguments we can lean on for heuristic justification.
3. **Cascading has optimality theory.** C4's unified routing+cascading result plus C1–C3 give a clean lineage for per-node small-LM-first-then-escalate policies; we can cite optimality conditions rather than defending the heuristic ad hoc.
4. **Granularity tradeoffs are empirically settled.** SAGA (A4) quantifies the latency-for-throughput exchange of workflow-atomic scheduling; Sarathi-Serve (B1) does the same inside a single serving engine. Our scheduler sits between these levels.
5. **Gaps we can claim.** None of the surveyed systems treat Python/API/DB executors as first-class scheduled resources with their own concurrency limits and costs, and none combine resource-constrained *list* scheduling (classic operations-research style) with typed program IRs for agent compilation. Sections A–B optimize mostly within GPU serving stacks; C optimizes model choice without DAG placement; D optimizes placement without program structure.

## Additional leads (not fully verified — follow up before citing)

- Efficient LLM Scheduling under Demand Uncertainty — arXiv:2603.07917 (Time-to-Last-Token minimization).
- Future-State-Aware Scheduling for Heterogeneous LLM Serving — arXiv:2605.07238 (joint queue/placement/reuse signals for DAG workloads).
- TOPAS: Workflow-Aware Prefix-State Scheduling for Multi-Tenant LLM Serving — arXiv:2608.25523.
- Splitwise-style edge-cloud LLM partitioning — arXiv:2512.23310; sustainability-aware edge LLM serving — arXiv:2512.04088; networking-aware energy in agentic inference — arXiv:2604.07857.
- GATEKEEPER small-first cascades — arXiv:2502.19335.
- Batch scheduling of LLM requests on GPU clusters with latency SLOs — arXiv:2412.18169.
