# Research Survey: Neural Compilers and AI Agent Compilers (2024-2026)

> Survey compiled 2026-09-29. Scope: academic papers (2023-2026, emphasis on 2024-2026) on
> LLM-based compilation of natural language into structured executable representations
> (DAGs, IRs, workflows), parallel function calling, planning with intermediate
> representations, and compiler-inspired AI agent orchestration.
>
> Relevance note: this survey supports the "compile-anything" project — the common thread
> across all papers below is replacing the sequential, interpretive ReAct-style agent loop
> with a *compile-then-execute* pipeline: natural language goes in, an explicit structured
> program/graph/IR comes out, and a deterministic runtime schedules it.

---

## Theme 1: Compiling Natural Language into Task DAGs / Parallel Function Calling

The papers in this group are the closest ancestors of an "AI agent compiler": they treat a
user query as *source code*, emit a dataflow DAG of function calls as the *intermediate
representation*, and run a scheduler over it.

### 1.1 LLMCompiler: An LLM Compiler for Parallel Function Calling

- **Title:** An LLM Compiler for Parallel Function Calling
- **Authors:** Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W. Mahoney, Kurt Keutzer, Amir Gholami
- **Venue / Year:** ICML 2024 (preprint Dec 2023)
- **arXiv ID:** 2312.04511 — https://arxiv.org/abs/2312.04511
- **Core contribution:** Replaces the sequential ReAct-style agent loop with a compiler-style
  architecture: a *planner* LLM emits a DAG of function calls (with dependencies and
  placeholders for resolved values), a *task-fetching unit* schedules and executes
  independent tasks concurrently, and a *joiner* decides whether to finish or re-plan.
  Achieves up to ~3.7x latency speedup and lower cost versus sequential function calling,
  on both open-source and proprietary LLMs.
- **Relation to compiling NL tasks into programs:** The canonical reference for this project.
  Natural-language query is compiled to an explicit task DAG IR that a non-LLM scheduler
  executes — exactly the "compile-then-execute" separation. Directly inspires treating
  function calling as dataflow compilation rather than interpretation.

### 1.2 Plan-over-Graph: Towards Parallelable LLM Agent Schedule

- **Title:** Plan-over-Graph: Towards Parallelable LLM Agent Schedule
- **Authors:** Shiqi Zhang, Xinbei Ma, Zouying Cao, Zhuosheng Zhang, Hai Zhao
- **Venue / Year:** arXiv 2025 (Feb 2025)
- **arXiv ID:** 2502.14563 — https://arxiv.org/abs/2502.14563
- **Core contribution:** Proposes a "plan-over-graph" paradigm: decompose a textual task into
  executable subtasks, build an abstract task graph, then generate a plan *over the graph*
  for parallel execution. Includes an automated synthetic-graph generation pipeline and a
  two-stage training scheme so both API-based and open-source models learn graph-level
  planning.
- **Relation to compiling NL tasks into programs:** Explicitly trains the model to *emit a
  task graph as the IR* first and plan over it second — i.e., graph construction is a
  compilation step that makes parallelism discoverable by construction.

### 1.3 Parrot: Efficient Serving of LLM-based Applications with Semantic Variable

- **Title:** Parrot: Efficient Serving of LLM-based Applications with Semantic Variable
- **Authors:** Chaofan Lin, Zhenhua Han, Chengruidong Zhang, Yuqing Yang, Fan Yang, et al. (SJTU + Microsoft Research)
- **Venue / Year:** OSDI 2024 (USENIX Symposium on OS Design and Implementation)
- **arXiv ID:** 2405.19888 — https://arxiv.org/abs/2405.19888
- **Core contribution:** Introduces the *Semantic Variable* abstraction: LLM API requests are
  annotated so that input/output dataflow between LLM calls in an application is exposed to
  the serving system, enabling cross-request deduplication, better scheduling, and
  application-level optimizations that conventional serving stacks cannot see.
- **Relation to compiling NL tasks into programs:** Parrot is a "backend compiler" for
  multi-step LLM programs: once the natural-language steps of a pipeline are wired through
  semantic variables, the runtime optimizes the whole dataflow — the systems-level analogue
  of IR-level optimization passes.

---

## Theme 2: Programming Models and Compiler Infrastructure for LLM Programs

These works build the *toolchain* view: DSLs/front-ends for LLM pipelines, optimizers that
act like compiler passes, and OS-style runtimes that schedule agent "processes".

### 2.1 DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines

- **Title:** DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
- **Authors:** Omar Khattab, Arnav Singhvi, Moor Mugdha, Heather Miller, Matei Zaharia, Christopher Potts
- **Venue / Year:** ICLR 2024
- **arXiv ID:** 2310.03714 — https://arxiv.org/abs/2310.03714
- **Core contribution:** A programming model where LLM pipelines are written declaratively
  (signatures + modules like ChainOfThought/ReAct/ProgramOfThought) and a *compiler* with
  teleprompter-style optimizers (e.g., BootstrapFewShot, MIPRO) automatically tunes prompts
  and demonstrations against a metric — replacing hand-prompt engineering with program
  optimization.
- **Relation to compiling NL tasks into programs:** The strongest "compiler" framing in the
  LLM programming-model literature: prompt optimization is literally a compile step that
  lowers a declarative program into an executable, metric-optimized artifact.

### 2.2 SGLang: Efficient Execution of Structured Language Model Programs

- **Title:** SGLang: Efficient Execution of Structured Language Model Programs
- **Authors:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph Gonzalez, Clark Barrett
- **Venue / Year:** NeurIPS 2024
- **arXiv ID:** 2312.07104 — https://arxiv.org/abs/2312.07104
- **Core contribution:** A frontend DSL for programming LLM applications (control flow,
  prompt composition, constrained generation via regular/CFG grammars) plus a runtime with
  RadixAttention for automatic KV-cache reuse across requests and a compressed FSM
  interpreter for fast constrained decoding.
- **Relation to compiling NL tasks into programs:** Provides the front-end/middle-end/back-end
  stack for LLM programs: structured generation programs are compiled (FSM construction for
  grammar constraints) and executed with cache-aware scheduling — the systems substrate a
  neural compiler would target.

### 2.3 AIOS: LLM Agent Operating System

- **Title:** AIOS: LLM Agent Operating System
- **Authors:** Kai Mei, Zhiyu Wu, Yongfeng Zhang, et al. (Rutgers University)
- **Venue / Year:** arXiv 2024 (revised through 2025)
- **arXiv ID:** 2403.16971 — https://arxiv.org/abs/2403.16971
- **Core contribution:** An OS-style runtime for LLM agents with a three-layer architecture
  (application / agent-kernel / hardware), where the kernel provides an agent scheduler,
  context manager, memory manager, tool manager, and access manager, handling concurrent
  agent calls, context switching, and prioritized task queues.
- **Relation to compiling NL tasks into programs:** The *runtime systems* counterpart to a
  compiler: once tasks are compiled into schedulable units, AIOS-style kernels supply
  OS-level resource management, suspension/resumption, and concurrency for them.

---

## Theme 3: Workflow Synthesis and Program Search for Agents

These papers compile task descriptions into executable *agent programs* (code or workflow
graphs) via search, rather than hand-authoring pipelines.

### 3.1 AFlow: Automating Agentic Workflow Generation

- **Title:** AFlow: Automating Agentic Workflow Generation
- **Authors:** Jiayi Zhang, Xinyu Zheng, Ye Tian, Zicheng Sun, Kaijie Yang, Min Yang, Jie Tan (DeepGLM / SJTU)
- **Venue / Year:** ICLR 2025
- **arXiv ID:** 2410.10762 — https://arxiv.org/abs/2410.10762
- **Core contribution:** Represents agentic workflows as executable code-defined graphs over
  an operator set (LLM, Ensemble, Review, Retrieve, Code...) and uses Monte Carlo Tree
  Search over this workflow-graph space to automatically synthesize and iteratively refine
  workflows that outperform hand-designed baselines (GSM8K, HotpotQA, HumanEval, etc.).
- **Relation to compiling NL tasks into programs:** Automatic *program synthesis in the
  workflow IR*: the search over workflow graphs is a compiler that emits, executes, and
  refines a machine-readable plan per task domain.

### 3.2 Automated Design of Agentic Systems (ADAS / Meta Agent Search)

- **Title:** Automated Design of Agentic Systems
- **Authors:** Shengran Hu, Cong Lu, Jeff Clune, et al. (UBC)
- **Venue / Year:** ICLR 2025
- **arXiv ID:** 2408.08435 — https://arxiv.org/abs/2408.08435
- **Core contribution:** Defines the ADAS research area and proposes Meta Agent Search: a
  meta LLM agent *programs new agents in code*, archives discovered designs, and reuses
  them as building blocks. Representing agents in code makes the design space
  Turing-complete, enabling discovery of novel agent architectures beyond fixed workflow
  templates.
- **Relation to compiling NL tasks into programs:** The "agent as compiled artifact" thesis:
  agents are code objects produced by a search-based compiler, which can be linked,
  composed, and versioned like compiled modules.

### 3.3 GPTSwarm: Language Agents as Optimizable Graphs

- **Title:** GPTSwarm: Language Agents as Optimizable Graphs
- **Authors:** Mingchen Zhuge, Wenqi Wang, Luigi Kirsch, Francesco Faccio, Dmitrii Khizbullin, Jurgen Schmidhuber (KAUST / Meta)
- **Venue / Year:** ICML 2024 (Oral)
- **arXiv ID:** 2402.16823 — https://arxiv.org/abs/2402.16823
- **Core contribution:** Unifies single agents, multi-agent collaboration, and
  tool-integrated agents as computational graphs whose nodes are LLM invocations or code
  executions and whose edges control information flow; introduces joint node optimization
  (prompts) and edge optimization (graph connectivity via REINFORCE) so the agent
  *architecture itself* is learned.
- **Relation to compiling NL tasks into programs:** Treats the agent graph as a compilable
  and optimizable IR — connectivity rewiring is effectively a scheduling/optimization pass
  over the dataflow graph.

### 3.4 FlowMind: Automatic Workflow Generation with LLMs

- **Title:** FlowMind: Automatic Workflow Generation with LLMs
- **Authors:** Zhiyang Zeng, Jia Zhang, et al. (JPMorganChase AI Research)
- **Venue / Year:** CIKM 2023 (arXiv updated 2024)
- **arXiv ID:** 2404.13050 — https://arxiv.org/abs/2404.13050
- **Core contribution:** Uses LLMs to automatically generate executable workflows for
  robotic-process-automation-style tasks: it extracts "workflow ingredients" (entities,
  actions) from natural-language inputs, generates a structured workflow, and executes it
  with LLM-based agents while masking sensitive information.
- **Relation to compiling NL tasks into programs:** An early end-to-end demonstration of
  NL-to-workflow compilation for enterprise automation, with an explicit
  ingredient-extraction (semantic parsing) stage before workflow assembly.

---

## Theme 4: Planning with Intermediate Representations (Decoupling Planning from Execution)

This group establishes the *front-end* of agent compilation: turning language into a
plan/program representation that is then executed or fed to a symbolic engine.

### 4.1 ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models

- **Title:** ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models
- **Authors:** Binfeng Xu, et al.
- **Venue / Year:** arXiv 2023 (widely adopted baseline 2024+)
- **arXiv ID:** 2305.18323 — https://arxiv.org/abs/2305.18323
- **Core contribution:** Splits the agent into Planner → Worker(s) → Solver: the planner
  produces a complete list of interdependent reasoning steps (with variable placeholders)
  *before* any tool runs, workers fill the placeholders, and the solver synthesizes the
  answer. Cuts token consumption ~5x and enables parallel tool execution versus ReAct.
- **Relation to compiling NL tasks into programs:** The planner output is a
  variable-substitution program — a de-facto IR where dependencies between calls are
  explicit, enabling batched/parallel execution and reuse.

### 4.2 AdaPlanner: Adaptive Planning from Feedback with Language Models

- **Title:** AdaPlanner: Adaptive Planning from Feedback with Language Models
- **Authors:** Haotian Sun, Chi Zhang, Lingkai Kong, et al. (Georgia Tech)
- **Venue / Year:** NeurIPS 2023
- **arXiv ID:** 2305.16653 — https://arxiv.org/abs/2305.16653
- **Core contribution:** Closed-loop adaptive planning: the LLM first generates a complete
  natural-language plan, executes step-wise, and *refines the remaining plan* when execution
  feedback reveals a mismatch with the environment, avoiding full replanning from scratch.
- **Relation to compiling NL tasks into programs:** Introduces *incremental recompilation*:
  the emitted plan is a revisionable artifact (like a program with debug feedback) rather
  than a one-shot chain-of-thought, prefiguring repair/retry passes in agent compilers.

### 4.3 Language Models as Compilers: Simulating Pseudocode Execution (Think-and-Execute)

- **Title:** Language Models as Compilers: Simulating Pseudocode Execution Improves Algorithmic Reasoning in Language Models
- **Authors:** Hyungjoo Chae, Yeonghyeon Kim, Seungone Kim, Kai Tzu-iunn Ong, Jinyoung Yeo, et al.
- **Venue / Year:** arXiv Apr 2024 (ACL Anthology publication)
- **arXiv ID:** 2404.02575 — https://arxiv.org/abs/2404.02575
- **Core contribution:** Proposes Think-and-Execute: the LM "compiles" a task into
  task-level *pseudocode* capturing shared logic across instances, then "executes" it by
  step-by-step simulation per instance. Shows pseudocode guidance beats natural-language
  guidance for algorithmic reasoning across seven tasks.
- **Relation to compiling NL tasks into programs:** Directly names LLMs as compilers and
  empirically argues that a structured intermediate representation (pseudocode) is a better
  lowering target for reasoning than free-form language.

### 4.4 PROC2PDDL: Open-Domain Planning Representations from Texts

- **Title:** Proc2PDDL: Open-Domain Planning Representations from Texts
- **Authors:** Tianyi Zhang, et al.
- **Venue / Year:** NLRSE Workshop @ ACL 2024
- **arXiv ID:** 2403.00092 — https://arxiv.org/abs/2403.00092
- **Core contribution:** The first dataset pairing open-domain procedural texts (wikiHow)
  with expert-annotated PDDL — a formal planning language with domains, actions,
  preconditions, and effects. Benchmark results show even strong LLMs struggle to emit
  syntactically valid PDDL, especially action modeling.
- **Relation to compiling NL tasks into programs:** Grounds the "natural language → formal,
  machine-checkable IR" compilation problem for planning, and quantifies how hard the
  front-end translation step is for current LLMs — a key motivation for typed/validated
  intermediate representations.

---

## Theme 5: Agent-Program IRs, Static Analysis, and Compiler Back-Ends (2025-2026)

Emerging work that gives agent programs a *framework-independent IR* so classic compiler
analyses (dependence analysis, verification, optimization) become applicable.

### 5.1 AgentFlow: Agent Dependency Graphs for Static Analysis of Agent Programs

- **Title:** AgentFlow: Building Agent Dependency Graphs for Static Analysis of Diverse Agent Programs
- **Authors:** (see arXiv listing)
- **Venue / Year:** arXiv 2026 (July 2026)
- **arXiv ID:** 2607.01640 — https://arxiv.org/abs/2607.01640
- **Core contribution:** Defines the Agent Dependency Graph (ADG), a unified
  framework-agnostic IR capturing semantic dependencies between agents, tools, and data in
  agent programs, enabling static analysis across heterogeneous agent frameworks.
- **Relation to compiling NL tasks into programs:** The LLVM-for-agents direction: one IR
  many front-ends lower into, so dependence analysis, dead-code elimination, and
  verification passes can run on any agent stack.

### 5.2 IAL-Scan: Uncovering Infinite Agentic Loops in LLM Agents

- **Title:** Uncovering Infinite Agentic Loops in LLM Agents
- **Authors:** (see arXiv listing)
- **Venue / Year:** arXiv 2026 (July 2026)
- **arXiv ID:** 2607.01641 — https://arxiv.org/abs/2607.01641
- **Core contribution:** Abstracts heterogeneous agent code into a framework-independent
  Agent IR and builds an Agentic Loop Dependence Graph (ALDG) to statically detect infinite
  agent loops (a common failure mode of autonomous loops), without executing the agent.
- **Relation to compiling NL tasks into programs:** Applies classic compiler loop-analysis
  (termination/dependence analysis) to agent programs represented as IR — evidence that
  compile-time guarantees are becoming feasible for agent workloads.

---

## Cross-Cutting Observations

1. **Two-stage architecture converges everywhere.** LLMCompiler (planner/joiner), ReWOO
   (planner/solver), Plan-over-Graph (graph-then-plan), Think-and-Execute (think/execute)
   all converge on: an LLM front-end emits a structured plan (DAG/program/pseudocode), and
   a deterministic executor runs it. This is the compile-then-execute pattern.
2. **The IR is the differentiator.** Papers differ mainly in IR choice: task DAG
   (LLMCompiler), code-defined workflow graph (AFlow, ADAS), computational graph
   (GPTSwarm), semantic-variable dataflow (Parrot), pseudocode (Think-and-Execute), PDDL
   (Proc2PDDL), agent dependency graph (AgentFlow). Formal, checkable IRs (PDDL, code)
   enable verification but are harder for LLMs to emit correctly.
3. **Parallelism is the killer optimization.** The demonstrated wins of DAG-based
   compilation are latency (up to 3.7x in LLMCompiler) and token-cost reduction (ReWOO),
   both arising from exposing the dependency structure that sequential ReAct hides.
4. **Search-based compilation is rising.** AFlow (MCTS) and ADAS (Meta Agent Search)
   treat compilation as search over program space — the workflow itself becomes the
   optimization target, analogous to superoptimization.
5. **Systems support is maturing.** SGLang (runtime + constrained decoding), Parrot
   (dataflow-aware serving), and AIOS (agent OS kernel) supply the back-end: scheduling,
   caching, context management, and concurrency for compiled agent programs.
6. **Open problems:** framework-fragmented agent representations (motivates AgentFlow's
   ADG); LLMs' unreliable emission of formal IRs (Proc2PDDL findings); dynamic repair of
   compiled plans under execution feedback (AdaPlanner); static termination/correctness
   guarantees (IAL-Scan).

## Quick Reference Table

| # | Paper | arXiv | Venue/Year | IR / Target |
|---|-------|-------|-----------|-------------|
| 1 | LLMCompiler | [2312.04511](https://arxiv.org/abs/2312.04511) | ICML 2024 | Function-call DAG |
| 2 | Plan-over-Graph | [2502.14563](https://arxiv.org/abs/2502.14563) | arXiv 2025 | Task graph |
| 3 | Parrot | [2405.19888](https://arxiv.org/abs/2405.19888) | OSDI 2024 | Semantic-variable dataflow |
| 4 | DSPy | [2310.03714](https://arxiv.org/abs/2310.03714) | ICLR 2024 | Declarative pipeline + optimizer |
| 5 | SGLang | [2312.07104](https://arxiv.org/abs/2312.07104) | NeurIPS 2024 | Structured LM program DSL |
| 6 | AIOS | [2403.16971](https://arxiv.org/abs/2403.16971) | arXiv 2024 | Agent OS runtime |
| 7 | AFlow | [2410.10762](https://arxiv.org/abs/2410.10762) | ICLR 2025 | Code-defined workflow graph |
| 8 | ADAS | [2408.08435](https://arxiv.org/abs/2408.08435) | ICLR 2025 | Agent code (Turing-complete) |
| 9 | GPTSwarm | [2402.16823](https://arxiv.org/abs/2402.16823) | ICML 2024 | Optimizable compute graph |
| 10 | FlowMind | [2404.13050](https://arxiv.org/abs/2404.13050) | CIKM 2023 | NL-generated workflow |
| 11 | ReWOO | [2305.18323](https://arxiv.org/abs/2305.18323) | arXiv 2023 | Plan with variable placeholders |
| 12 | AdaPlanner | [2305.16653](https://arxiv.org/abs/2305.16653) | NeurIPS 2023 | Adaptive NL plan |
| 13 | LM-as-Compilers | [2404.02575](https://arxiv.org/abs/2404.02575) | arXiv/ACL 2024 | Task-level pseudocode |
| 14 | Proc2PDDL | [2403.00092](https://arxiv.org/abs/2403.00092) | NLRSE@ACL 2024 | PDDL |
| 15 | AgentFlow | [2607.01640](https://arxiv.org/abs/2607.01640) | arXiv 2026 | Agent Dependency Graph IR |
| 16 | IAL-Scan | [2607.01641](https://arxiv.org/abs/2607.01641) | arXiv 2026 | Agent IR + loop dependence graph |
