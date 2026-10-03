# Paper Outline — Compile Anything (target: ASPLOS / DAC)

> Working outline derived from: README.md, spec/taskir-spec.md, spec/skill-isa.md,
> docs/{effect-system-proposal, semantic-label-audit, dataset-audit, tau3-effect-audit,
> validator-audit, runtime-review, missing-skills, benchmark-schema-audit,
> external-benchmark-status}.md, and three competitor papers (LLMCompiler ICML'24;
> "Generating Workflow DAGs from NL with Non-Reasoning LLMs" Microsoft arXiv:2608.30250;
> "Can LLMs Understand IRs in Compilers?" ICML'25). Also on file in `_papers/`:
> AIOS Compiler/CoRE (arXiv:2405.06907) and an ISAC Agent Compiler vision paper
> (arXiv:2607.16269) — both useful for related work.
>
> Audience: **systems/architecture (ASPLOS) or EDA (DAC)**, not ML. The headline is
> the toolchain, not the model: *the front end is learned; everything past the front
> end is a spec, a validator, an optimizer, a scheduler, a runtime, and a cost model.*

---

## 1. Title candidates

**Primary (ASPLOS):**
> **Compile Anything: A Validated SSA Task IR, Effect System, and Cost-Aware Runtime for Executing AI Tasks on Heterogeneous Executors**

Alternates:
1. **TaskIR: An LLVM-Style Toolchain for Natural-Language Task Execution** — short, brandable; risks overclaiming "LLVM-style" (reviewers will probe).
2. **The Front End Is Learned: Static Validation, Effect Ordering, and Cost-Aware Scheduling for LLM-Compiled Task Programs** — makes the systems thesis explicit in the title.
3. **DAC variant:** *Compiling Natural-Language Design Intent into Verifiable Task Programs: An IR-Centric Toolchain for RTL Debug and EDA Workflows* — leans on the RTL/VerilogEval lifter, EDA skill classes (LINT/SYNTH/SIMULATE/FORMAL_CHECK), and RTL-Repo external benchmark. Weaker as the general paper; strong as a fallback if ASPLOS rejects.

Recommendation: use the primary title for ASPLOS; keep the DAC variant in reserve. The phrase "Compile Anything" already names the repo and frames the generality claim.

---

## 2. Abstract draft (~200 words) — Phase 5D frozen

Agent frameworks execute natural-language tasks by prompting a large model step by step: the plan is an ephemeral text artifact, side-effect ordering is implicit, cost is counted in tokens after the fact, and nothing about a plan can be checked before it runs. We argue that AI task execution needs a compiler toolchain, not a prompting stack. We present Compile Anything: (i) TaskIR, a versioned SSA-style intermediate representation with a written specification and a static validator enforcing six invariant classes; (ii) a Skill ISA of 31 typed instructions with a five-dimensional nominal cost model, separating task semantics from executor binding; (iii) a dependency-driven runtime in which verification and retry are IR-level constructs; and (iv) a learned front end — a 3B model fine-tuned on 31k audited examples lifted from 71,855 real benchmark samples. A 10,000-program fuzz campaign shows zero invalid programs accepted and zero undefined runtime failures. Experiments show that the 3B model learns TaskIR's structural compilation rules — 99% in-domain validator pass, with high structural validity retained across three external benchmark families never seen in training (92–99%) — while semantic grounding degrades sharply out of domain (BFCL end-to-end functional semantics 15.5%, tau3-bench grounded recall 0.6%), revealing a separation between structural compilation, semantic skill selection, and planning depth. Schema-conditioned training yields a small improvement in BFCL end-to-end functional semantics; naive depth reweighting increases plan length without improving semantic correctness. Explicit IR layering makes these failure modes separately measurable: the front end's skill selection is the bottleneck, while the back-end tool binder's oracle upper bound is 99.95%.

(~250 words; trim the fuzz sentence or the binder sentence if space is tight. Nominal cost numbers (3.4% energy / 37.9% makespan) are deliberately NOT headline — they stay in the scheduling section as nominal analytical results.)

---

## 3. Section structure

Total length: ASPLOS ~12 pages + references. Mapping of sections to repo artifacts is in §3.10.

### §1 Introduction (1.25 pp)
- The problem, stated in systems terms: today's "agent" is a prompting stack — plan is text, ordering is folklore, cost is an afterthought, nothing is checkable before execution. ReAct-style loops serialize LLM invocations; parallel-calling systems (LLMCompiler) fix latency but keep the plan as an untyped, unvalidated JSON DAG.
- The thesis: treat a natural-language task as a *compilation target*. Front end (learned) emits a typed SSA IR; everything downstream is deterministic: verify → schedule → execute → account.
- Punchline figure (Fig. 1): side-by-side of the LLVM stack (Clang front end → LLVM IR → verify → opt → llc → target) and our stack (neural compiler → TaskIR → V1–V6 validator → optimizer/scheduler → executors → trace + cost report).
- Contribution list (see §4 below, same numbering).

### §2 Background, Motivation, and Design Principles (1.25 pp)
- 2.1 Anatomy of an AI task: retrieval, transformation, computation, LM steps, and *actions with side effects* (irreversible world actions vs local state). Use tau3-bench numbers as the workload characterization: 14,834 ground-truth actions = 46.6% reversible-state, 37.8% irreversible-world, 3.2% read-only; 2,443 of 2,546 tasks are multi-action transactions. This is the section that tells an architecture audience "side effects and ordering are the memory-consistency problem of this domain."
- 2.2 Why prompt-time planning is not a compiler contract: no spec, no validation, no types, no cost model, no effect ordering; replanning on failure = full recompile by an LLM. Evidence: LLMCompiler replans via feedback loop; Workflow-DAG paper's "emission-density bottleneck" shows even strong models mis-emit dense structure; Jiang et al. show LLMs fail at CFG/loop/execution reasoning over LLVM IR.
- 2.3 Design principles (each traceable to a classical analogy): (P1) single canonical IR with a written spec and versioning; (P2) validation before execution (data-quality gate); (P3) branches by predication, not jumps (guard + SELECT, GPU/VLIW predication analogy — keeps IR flat, good for small-model emission); (P4) verification/retry as first-class IR structure (speculative execution + squash/replay analogy); (P5) ISA-level cost model from day one (latency/tokens/FLOPs/energy/memory per instruction); (P6) strict layering: semantic skills never mention tool names; executor binding is a separate lowering.

### §3 TaskIR: A Verifiable Intermediate Representation (2 pp)
- 3.1 Layering: TaskIR (task semantics, ~LLVM IR) / Skill ISA (capability, ~target ISA) / Executor (small LM, Python, API, DB — ~functional units). Hard rule: concrete tool names never enter the semantic program; they live only in `meta.provenance`.
- 3.2 Program structure: flat SSA node list in canonical topological order; `%id` values, `@input` globals; one node = one value.
- 3.3 Node schema: `inputs` (data deps), `after` (control-only), `params` (compile-time constants), `guard` (predication), `retry` (bounded, VERIFY-driven), `output_type`, `hints` (advisory scheduling).
- 3.4 Type system: atomic + parameterized + open domain types; conservative bidirectional-Any compatibility; no inference in v0.1 — types come from ISA signatures plus explicit narrowing.
- 3.5 Control without jumps: guard/SELECT as predicated execution with predicate propagation (skipped condition ⇒ guarded node skips); VERIFY produces Bool; `retry(on=%v)` invalidates the node's transitive dependents and re-executes (rollback-lite). Show the flight-verified example in text form.
- 3.6 Serialization duality: canonical JSON (machine) + text form (training target / debug) with printer↔parser roundtrip guarantee; forward-compatibility convention (open dicts, unknown-field warnings).

### §4 The Skill ISA and the Cost Model (1.25 pp)
- 4.1 Instruction table by class (io / retrieval / transform / compute / lm / action / control): 31 skills + VERIFY/SELECT. Present an excerpt table with signatures, resource class, and cost columns.
- 4.2 Two-level lowering: tool → semantic skill (lifter) → executor binding (scheduler/runtime). Benefit: provenance-preserving, benchmark-portable, and the optimizer sees semantics rather than vendor strings.
- 4.3 Cost model: five dimensions per instruction (latency_ms, tokens_in/out, FLOPs = 2·N_params·tokens, energy, resident memory); 70B single-shot baseline constant (~2500 ms, ~900 J, ~140 GB) as the reference point. Honest framing: nominal constants for workload comparison and scheduler objectives, with real profiling as the swap-in interface.
- 4.4 Evidence-driven ISA evolution: benchmark-lift rejections steer ISA additions (LOOP > STRING_OP > GROUP/TABLE_SCAN; JOIN not needed — relational JOIN stays inside QUERY_DB). This is ISA design by workload characterization, the argument a DAC/ASPLOS audience accepts.

### §5 Validation as a Compile-Time Gate (1 pp)
- 5.1 The six invariant classes V1–V6 (structure, def-use, DAG, types, skill availability, control-flow/retry legality) + warnings (dead nodes, guarded output).
- 5.2 Fuzzing as the correctness story: 10,000 random legal programs — validator accepts 100%, runtime produces only *defined* failures (consume-skipped / chosen-branch-skipped / retry-exhausted: 1,719 cases), zero invariant violations; runtime invariants IV1–IV6 (DAG order, attempts monotonicity, determinism, non-empty outputs, non-negative cost, Bool-produces-Bool).
- 5.3 The data-quality gate in practice: no record enters the corpus without passing V1–V6; three-gate evaluation for the learned front end (syntax → validation → execution) instead of exact match.
- 5.4 Known gaps (honesty subsection, feeds §11): effect ordering (M1 → §6), hints-vs-ISA resource conflicts (M2), open `params` schema (V5 limitation), DCE must not remove VERIFYs referenced by retry (M5).

### §6 Effects and Ordering: A MemorySSA for AI Tasks (1.25 pp)
- 6.1 The M1 problem, dramatized: two data-independent SENDs can be legally swapped by any parallel scheduler — "debit before confirmation email" is inexpressible in pure dataflow.
- 6.2 Effect tokens as constrained SSA values: `effect_in/out` chains per class (`world` / `state`), pure skills pay zero representation cost; chains are hard scheduling barriers; inter-class parallelism preserved. Design-space table (dedicated fields vs chain nodes vs implicit convention) and why option A wins.
- 6.3 Runtime semantics co-designed with the IR: guarded-skip token pass-through; no verify-retry across `world` actions (irreversible replay = double debit) unless `hints.idempotent`; rollback checkpoints must include effect-chain position; effect tokens banned from SELECT in v0.2 (conservative-but-correct, region effect-phi in v0.3).
- 6.4 Grounding in real workloads: the tau3-bench classification (§2.1) shows the class taxonomy is not invented — reversible/irreversible/read-only split matches 99.6% of observed actions (the 12.4% "other" is disclosed). Documented gap: cross-class all-or-nothing transactions (2,443 multi-action tasks) need transactional effect regions — stated as future work, not hidden.
- 6.5 Payoff analysis: effect serialization as a *parallelism-limitation* study — chain length vs critical path, the kind of quantitative structure analysis architecture reviewers recognize.

### §7 Scheduling and Runtime (1.25 pp)
- 7.1 Architecture: dependency-driven evaluation over the memo; memo = SSA value cache *up to* rollback, which versions values — the accurate analogy is out-of-order speculative execution with squash/replay (trace keeps `superseded` history; final state deterministic).
- 7.2 Rollback mechanics: invalidation over inputs+after+guard edges (retry edges excluded — they are temporal back-edges, like replay vs structural cycles); program-order re-evaluation of all invalidated nodes; three semantic defects found and fixed during review (predicate propagation, inline-verify false cycles, stale non-output dependents) with regression tests.
- 7.3 Critical path *excludes* retry edges by construction (static schedulability bound); expected-latency analysis with profiled failure rates is a separate probabilistic pass — stated.
- 7.4 List scheduler: critical-path-priority, ready-set dispatch, per-resource-class executor pools; `_edges()` reserves effect-edges as the §6 integration point. Known limits (eager SELECT branches, per-node peak memory, ~900-node recursion depth) disclosed.

### §8 The Learned Front End: Compiling NL into TaskIR (1.25 pp)
*(Keep deliberately the smallest "ML" section — it is one component of the toolchain.)*
- 8.1 Corpus construction as a lifter problem: 7 real datasets / 71,855 samples downloaded with sha256 provenance; adapters → lifters (toolbench/xLAM, humaneval, spider, rtl) → validator → simulator; zero synthetic masquerading (synthetic tagged `source=synthetic:*` only).
- 8.2 Quality gates that make supervision real: fallback-aware mapping (`mapping_kind`/confidence; fallback was 55.2% of calls before the fix, semantic-mapping coverage 36.2% — reported separately from lift/valid/exec coverage); two training views (`plan_target` vs `execution_target`); MinHash-LSH + union-find dedup and group-aware splits (cross-split exact = 0, near = 0); tiers A/B/C with C reserved for ablation.
- 8.3 Semantic-label audit: deterministic per-record replay against raw ground truth; v3 → v3.1 fix (parameter preservation 16.7% → 98.9%) took suspect rate 64.6% → 1.06% (Tier A 0.254%, Tier B 1.882%) by changing *only the lifter*, not the IR/ISA/splits. This is the "data quality is a gate, not a hope" result.
- 8.4 Training protocol and evaluation: Qwen 3B/7B, LoRA/QLoRA, frozen manifest (commits, SHA256s, seeds, pinned deps, preflight gates); token audit (p50 371 / p99 856, 0.17% > 2048); three-gate eval plus skill-F1, graph-edit similarity, generic-action rate, and **seen vs unseen composition** — the metric that separates compiling from template memorization.

### §9 Evaluation (2.5 pp) — see §7 of this outline for the experiment matrix.

### §10 Related Work (0.75 pp) — see §6 of this outline.

### §11 Discussion, Limitations, Future Work (0.5 pp)
- No LOOP yet (quantified cost: code-domain coverage ceiling ~5% today, ~70–80% projected with LOOP + assignment lifting + STRING_OP); nominal cost constants (profiling interface defined, measurements pending); mock executors (deterministic by design); single-program modules; cross-class transactional effect regions; text-parser coverage; effect system v0.2 currently a proposal with validator (V7) design.

### §12 Conclusion (0.25 pp)

### 3.10 Section → artifact map (for authors)

| Paper section | Repo evidence |
|---|---|
| §3 TaskIR | spec/taskir-spec.md; src/ir/ (JSON canonical + printer + parser) |
| §4 Skill ISA + cost | spec/skill-isa.md; src/isa/registry.py |
| §5 Validator + fuzz | src/validator/; scripts/fuzz_taskir.py; docs/validator-audit.md |
| §6 Effect system | docs/effect-system-proposal.md; docs/validator-v7-proposal.md; docs/tau3-effect-audit.md |
| §7 Runtime + scheduler | src/runtime/; docs/runtime-review.md; src/optimizer/scheduler/list_scheduler.py; tests/test_scheduler_cost.py |
| §8 Neural compiler | data/compiler_corpus_v3.1; docs/semantic-label-audit.md; docs/training.md; experiments/phase5b1/ |
| §9 Eval | data/reports/*; benchmark/; data/external_benchmarks/; docs/dataset-audit.md; docs/benchmark-schema-audit.md |

---

## 4. Key claims / contributions (numbered, specific)

1. **A specified, versioned IR for AI tasks (TaskIR).** Flat SSA with predicated branching (guard/SELECT), verification (VERIFY→Bool) and bounded retry with rollback as IR-level constructs, dual canonical serialization with printer↔parser roundtrip, and a forward-compatibility convention. Unlike prior plan formats (JSON DAGs, NL programs, YAML), TaskIR is a written semantics contract (spec/taskir-spec.md), not a serialization convention.
2. **Static validation as a data-quality gate.** Six invariant classes (V1–V6) enforced before any program enters a dataset or executes; validated by a 10,000-program fuzz campaign: zero invalid accepts, zero false rejects, and the runtime exhibits only *defined* failures under randomized guards and fault injection (1,719/10,000 defined-failure paths, 0 invariant violations).
3. **A typed Skill ISA with a first-class cost model.** 31 skills + 2 control instructions with signatures, resource classes, and per-instruction latency/token/FLOP/energy/memory costs; strict semantic/tool separation via two-level lowering. Enables static cost accounting and cost-aware scheduling without executing anything.
4. **An effect system for AI tasks (MemorySSA analogy).** Effect tokens as constrained SSA values; per-class linear chains as hard scheduling barriers; runtime rules (guarded-skip pass-through, no verify-retry across irreversible world actions, rollback checkpoints include effect position). Grounded in a 14,834-action audit of tau3-bench; cross-class transactional semantics identified and disclosed as a gap.
5. **Cost-aware scheduling and a runtime with speculative-execution semantics.** Critical-path-priority list scheduling over resource-class executor pools; memoization as versioned SSA storage with squash/replay rollback. Example benchmark: scheduled pipelines at nominal mean energy 3.4% and mean makespan 37.9% of a 70B single-shot baseline; cross-resource-class parallelism beats the sequential sum (rtl_debug: 1070 ms vs 1098 ms).
6. **An evidence-driven expressibility audit of the IR against real benchmarks.** 71,855 real samples from 7 datasets lifted end-to-end: 100% expressibility for tool-use (xLAM 60,000/60,000), SQL (Spider), and RTL (VerilogEval); 7.3% HumanEval / 3.3% MBPP, with machine-generated rejection histograms isolating LOOP/recursion (524 rejections) as the dominant, quantified ISA gap — driving v0.2 ISA priorities.
7. **A quality-audited corpus and a trained small-model front end.** 31k tiered training examples with zero cross-split leakage, deterministic semantic-label auditing (suspect 64.6% → 1.06% via a lifter-only fix), and a frozen, preflight-gated training package. Trained 3B QLoRA: 65.9→99.6% parse, 0→99.1% valid, OpSeq 88.5%, skill F1 0.92 in-domain; 7B zero-shot fails identically (0% valid), showing scale alone does not produce compiler semantics (trained-7B control NOT RUN). Structure/semantics separation across three unseen benchmark families + Phase 5C schema/depth 2×2 ablation are the headline evaluation results (paper-evaluation.md §3–§4).

Claims 1–7 all have frozen repo evidence (Phase 5D, `experiments/paper_snapshot_v1.json`). E3 (7B LoRA), official interactive AgentBoard metrics, real hardware cost profiling, and effect v0.2 implementation remain NOT RUN and are disclosed as such — never claimed.

---

## 5. Planned tables and figures

### Figures
- **Fig. 1 (headline):** Toolchain vs LLVM stack, side by side. NL → neural compiler (learned) → TaskIR → validator → optimizer/scheduler → executors → trace + cost. Annotate the deterministic region.
- **Fig. 2:** TaskIR text-form example (find_cheapest_flight_verified) annotated: SSA values, guard/SELECT predication, VERIFY + retry, hints.
- **Fig. 3:** Runtime semantics timeline: speculative execution, VERIFY failure, memo invalidation, squash/replay with `superseded` history (before/after trace).
- **Fig. 4:** Effect chains: world/state chains as barriers; world∥state parallelism; the debit/confirmation-email reordering bug a naive scheduler commits.
- **Fig. 5:** Corpus pipeline: download (sha256) → adapter → lifter → validator → simulator → quality tiers → splits, with the audit gates on top.
- **Fig. 6:** Coverage/rejection histogram across the 7 real datasets; LOOP highlighted as the dominant bar. (Data: data/reports/dataset_coverage.*)
- **Fig. 7:** Scheduling Gantt for rtl_debug showing EXTRACT∥SEARCH cross-class parallelism and makespan < sequential sum. (Data: benchmark_results.md)
- **Fig. 8:** Compile-quality vs program density for the trained front end (parse/valid/exec rates vs node count) — the TaskIR counterpart of the Workflow-DAG paper's emission-density curve; flat-SSA + guard predication is our design answer to it. (NOT RUN in this snapshot — E10 pending; do not include unless produced.)
- **Fig. 9 (optional, §6.5):** Parallelism-limitation curve: normalized critical path vs effect-chain length over tau3-lifted programs. (NOT RUN — depends on effect v0.2 implementation.)

### Tables
- **T1:** Representation comparison — TaskIR vs LLMCompiler DAG vs CoRE program vs Workflow-DAG IR (axes: written spec/versioning; static validator w/ invariants; types; predication; verify/retry in IR; effect ordering; per-instruction cost model; scheduler; trained front end; fuzzed semantics). This is the positioning table; every cell must be defensible from the cited papers.
- **T2:** Skill ISA excerpt (name, signature, resource class, latency, tokens, FLOPs, energy, memory) + full table in appendix.
- **T3:** Validator invariants V1–V6 + warnings, with fuzz outcome column.
- **T4:** Real-dataset expressibility audit (7 datasets × lift/valid/execute + rejection reason). (Data: docs/dataset-audit.md.)
- **T5:** Cost/scheduling results: sequential vs makespan vs critical path; energy, peak memory, LM/API calls, retries; % vs 70B baseline. Scale from the current 5-program table to the full corpus sweep.
- **T6:** tau3-bench effect classification (counts/shares per class and per domain) + expressibility verdict per class.
- **T7:** Neural compiler results: E0–E3 × {parse, valid, execute, op-seq exact, skill F1, GED, generic-action rate, seen, unseen}. (DONE for E0/E1/E2 — frozen numbers in paper-evaluation.md §2; E3 7B LoRA NOT RUN, disclosed.)
- **T8:** External evaluation suites A/B/C dual metrics. (DONE for BFCL / τ³ / AgentBoard-offline / Phase 5C 2×2 — frozen in paper-evaluation.md §3–§7; ToolBench full, BIRD, RTL-Repo official runs NOT RUN in this snapshot.)
- **T9 (appendix):** Corpus audit trail: tier counts, dedup stats, semantic-audit before/after, token-length distribution, freeze manifest hashes.

---

## 6. Related work positioning

**One-paragraph strategy:** we sit at the intersection of (a) LLM orchestration systems, (b) LLM-as-compiler/structured-emission work, and (c) IR understanding/foundation models. None of (a)–(c) supplies the *middle of the toolchain*: a spec'd IR + validator + effect system + cost-aware scheduler + runtime as one vertically integrated, fuzz-verified stack. T1 carries the argument.

- **LLMCompiler (Kim et al., ICML'24).** Closest systems ancestor: Planner LLM emits a task DAG with `$`-variables; Task Fetching Unit; parallel Executor; replanning via feedback. What it is not: no IR specification (the DAG is an ad-hoc JSON), no static validation, no types, no cost model beyond token counting, no effect ordering (nothing prevents reordering independent side-effecting calls), failure handling is *replanning* (LLM recompile) rather than IR-level verify/retry, and plans live in tool-name space (not semantic space). Our positioning: LLMCompiler optimizes *execution of a prompt-time plan*; we make the plan a *validated, typed, cost-annotated artifact* that admits static optimization — and our VERIFY/retry is bounded, checkable speculation where theirs is an unbounded LLM round-trip. Their measured wall-clock speedups (up to 3.7×, cost 6.7×) are what we must approximate with real-executor experiments (E4/E9) — cite as the bar, do not fight their numbers with nominal ones.
- **AIOS / AIOS Compiler / CoRE (Xu, Mei et al., Rutgers; arXiv:2405.06907).** AIOS is the OS analogy at the *system-services* level (agent scheduler, memory manager, tool manager, context manager). CoRE unifies NL/pseudo-code/flow programming with the **LLM as the interpreter**, executing "units" (sequential/parallel/call) with suspend/resume. Positioning: in CoRE the LLM stays on the hot path — every unit is re-interpreted at execution time, NL ambiguity persists at runtime, and there is no static contract (no validator, no types, no per-instruction costs, no effect classes). We compile *out of* the LLM: the front end emits a program the runtime can execute without further interpretation wherever the instruction is non-LM (FILTER, ARGMIN, QUERY_DB...). Complementary, and AIOS's scheduling observations motivate our resource-class executor pools.
- **Workflow DAG from NL with non-reasoning LLMs (Iyer et al., Microsoft, arXiv:2608.30250).** Philosophically closest on the front end: emission-density bottleneck → compact IR → deterministic compiler → validity by construction; learned registry selection for scaling; gap-bridging result (non-reasoning model matches a reasoning model). Differences: single domain (contact-center routing), target is vendor JSON *configuration* not an executable program semantics; validity means JSON-schema conformance, not semantic well-formedness (no types/def-use/effects); no runtime, no retry, no cost model, offline graph-match evaluation on synthetic templated rules (disclosed as proprietary). We adopt their diagnostic as motivation (dense emission is the failure mode; our flat SSA + predication is designed to minimize emission density) and our seen/unseen composition metric is the training-side counterpart of their density curve. Cite generously — they validate the approach's front end; we supply the middle and back end.
- **Jiang et al., "Can LLMs Understand IRs in Compilers?" (ICML'25).** Evidence base, not competitor: frontier LLMs parse LLVM IR syntax but fail at CFG reconstruction, loop handling, and instruction-level execution reasoning; they recommend IR-specific fine-tuning. We use this twice: (i) justifies *not* prompting a big model to emit dense branchy IR — TaskIR is flat SSA with predication precisely to shrink the emission/reasoning burden; (ii) justifies training a small specialized compiler (our §8) instead of prompting.
- **Also cite (one line each):** ReWOO/HuggingGPT/ViperGPT (plan-and-execute ancestors); grammar-constrained decoding (Outlines etc.) — syntactic validity at decode time vs our *semantic* validity at the program level; Meta LLM Compiler / IR2Vec / ProGraML (IR for *code compilation*, complementary — our IR is for *task execution*); WorfBench/agentic-workflow benchmarks (sequence-vs-graph gap); ISAC Agent Compiler (arXiv:2607.16269) — domain-specific evidence the "agent compiler" pattern is spreading, which strengthens generality but none of them ships a validated IR toolchain; SudoQ… no — skip; EDA angle for DAC: high-level synthesis flow comparisons (spec → IR → validate → schedule → map is the HLS shape; TaskIR is that shape for AI tasks).

**Framing sentence for §10:** *Prior systems use the compiler metaphor; we supply the compiler's actual obligations — a specification, a verifier, an ordering model for side effects, a cost-annotated instruction set, and a scheduler/runtime that consume them.*

---

## 7. Experiment matrix: claim → required experiments — Phase 5D status

Status legend: DONE (frozen artifact exists), NOT RUN (disclosed, never claimed). **No PENDING cells remain; completed experiments' numbers are frozen in `experiments/paper_snapshot_v1.json`.**

| # | Experiment | Protocol / metrics | Supports claim | Status |
|---|---|---|---|---|
| E1 | Validator + runtime fuzz | 10k random legal programs; validator accept-rate; runtime defined-failure-only; IV1–IV6 | C2 | DONE |
| E2 | Expressibility audit (real data) | 7 datasets, 71,855 samples; lift/valid/execute coverage; rejection histogram | C6 | DONE |
| E3 | Workload characterization of lifted corpus | graph stats over corpus programs | C3–C5 (motivation) | PARTIAL (1105-program stats; full-corpus sweep NOT RUN) |
| E4 | Cost & scheduling study | sequential vs list-scheduled vs critical-path; energy/latency vs 70B baseline | C5 | PARTIAL (5-program NOMINAL result: energy 3.4%, makespan 37.9%; full sweep + measured latencies NOT RUN) |
| E5 | Effect-system v0.2 implementation + study | tokens + V7; reordering bug; parallelism-loss curve | C4 | NOT RUN (proposal + τ³ classification DONE — classification supports workload motivation only) |
| E6 | Neural compiler training (E0–E3) | E0 3B zero-shot / E1 3B QLoRA / E2 7B zero-shot; three-gate eval, op-seq, skill F1, GES, seen/unseen | C7 | DONE for E0/E1/E2 (E0: 65.9/0/0; E1: 99.55/99.10/99.10, OpSeq 88.54, F1 0.92; E2: 68.2/0/0). **E3 7B LoRA NOT RUN** |
| E7 | Prompted-planner baselines | ReAct / LLMCompiler-style JSON plans through same gates | C7 | NOT RUN-in-this-snapshot (react_baseline.py implemented) |
| E8 | Data-quality ablation (v3 vs v3.1) | retrain on pre-fix corpus | C7 | NOT RUN as retrain; audit delta (suspect 64.6%→1.06%) frozen as evidence |
| E9 | External benchmark suites | A/B/C dual metrics | C1, C5 | **DONE for BFCL (valid 92.4 / E2E functional 15.50), τ³ (valid 98.86 / recall 0.62), Phase 5C 2×2 (schema +0.35pp E2E; depth negative), AgentBoard untouched offline confirm (351, valid 91.7)**. Official interactive metrics UNSCORED-OFFLINE |
| E10 | Emission-density analysis | compile-quality vs program size | C1 | NOT RUN |
| E11 | AgentBoard untouched confirmation | pre-registered, one-shot, offline compile protocol, 351 cases | C1 | DONE (docs/agentboard-preregistration.md; results final regardless of outcome) |

**Honesty rules carried into the paper:** nominal costs are labeled nominal analytical model (not measured hardware energy/latency); NOT RUN / UNSCORED-OFFLINE items are disclosed, never estimated; all dataset numbers cite the sha256'd manifests; synthetic data is tagged and never counted as real; AgentBoard evidence is scoped to offline structural confirmation, never task success.

---

## 8. Reviewer-risk register (pre-empt in writing)

1. **"Nominal cost model is not a measurement."** Answer in-text: costs are declared constants with a profiling interface; the scheduler/results are workload-relative; one measured-executor experiment (E4 extension) if at all possible. Otherwise scope claims as "architecture-level analysis."
2. **"Is this just an agent framework with extra steps?"** Fig. 1 + T1: no prior system has spec+validator+effects+cost-model+fuzzed runtime in one stack; the fuzz campaign is the falsifiable difference.
3. **"Effect system is only a proposal."** Either land v0.2 before submission (preferred — E5) or re-title §6 as design + workload-grounding analysis and demote C4 from a contribution bullet to a design section.
4. **"7.3%/3.3% code coverage looks weak."** Own it: the rejection histogram *is* the contribution (evidence-driven ISA gaps), and the v0.2 projection table (§5 of dataset-audit) turns weakness into a roadmap; LOOP must be implemented or the projection stays a projection.
5. **"Comparisons to LLMCompiler lack wall-clock."** E7(b) + E9 give compile-side and benchmark-side comparisons; state explicitly that runtime wall-clock with real executors is future work if not measured.
6. **"Where is the optimizer?"** Currently a pass-manager interface + list scheduler. Frame scheduling as the first optimization pass; DCE/code-motion constraints are already specified in the effect proposal (§6 "who benefits"). Do not claim unimplemented passes.
