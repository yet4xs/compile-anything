# Research Survey: Training LLMs to Generate Structured, Executable Outputs (2024–2026)

**Purpose:** Literature support for training a Qwen 3B/7B model to generate TaskIR (a structured, executable intermediate representation of tasks).
**Compiled:** 2026-09-29. Papers are ordered by relevance tier. All arXiv IDs were verified against arXiv directly (abs page or arXiv API) unless explicitly marked otherwise.
**Scope note:** A few foundational pre-2024 papers are included where they are the canonical reference for a technique; they are flagged with year 2023/2022.

---

## Executive summary (what the literature says for TaskIR)

1. **Fine-tuning small models to emit workflow/DAG/IR structures works.** WorkflowLLM (2411.05451) shows a data-centric SFT recipe (corpus → schema extension → reverse queries → pruning → self-contrastive training) is enough to make an 8B model orchestrate 70+ action types. OrchDAG (2510.24663) shows plan-DAG generation can be trained directly from synthetic DAG data with tunable complexity, and improved further with a graph-edit-distance-based RLVR reward.
2. **Representation choice matters: code/text serialization beats raw JSON prompts.** AFlow (2410.10762) and ADAS (2408.08435) represent workflows/agentic systems as *executable code* and search over that space; "Let Me Speak Freely?" (2408.02442) shows forcing JSON format on a model not trained for it degrades reasoning by 10–15%+. Implication: serialize TaskIR in a code-like DSL the model natively handles, and *train on that format* rather than coercing it at inference.
3. **Mixing an IR into pretraining transfers.** IRCoder (2403.03894) shows continued pretraining on a source→LLVM-IR parallel corpus improves multilingual generation, robustness, and instruction following — evidence that TaskIR-style artifacts as a pivot language teach transferable structure.
4. **Data quality beats data quantity for small models.** OSS-Instruct/Magicoder (2312.02120) and SelfCodeAlign (2405.17057) give concrete recipes for synthetic, execution-filtered instruction data that make 7B models competitive with much larger ones. Curriculum ordering adds a further 18–45% training-efficiency gain (2506.11300) and improves small code LMs (2407.10194).
5. **Structured rewards work when exact match is too sparse.** AST/graph-based reward components (PPOCoder 2305.00266; OrchDAG 2510.24663) and execution feedback (AFlow, RLVR-style) provide denser signals than exact match; grammar-aligned constrained decoding (2405.21047, 2410.07295) guarantees well-formedness at inference.
6. **Evaluation beyond exact match is an active area.** Graph edit distance over AST/workflow graphs (OrchDAG, SAGE-HLS, BOFLOW), per-node AST matching, execution success, and reliability metrics like pass^k / G-pass@k (2510.04265) and Max@k (2508.01174).

---

## Tier 1 — Training models to generate workflow / DAG / IR structures (most relevant to TaskIR)

### 1. WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models
- **arXiv:** https://arxiv.org/abs/2411.05451 (Fan et al., 2024; ICML 2025). Code: https://github.com/openbmb/workflowllm
- **Summary:** Data-centric framework for making LLMs create, execute, and adapt workflows. Builds WorkflowBench, a large-scale instruction–workflow fine-tuning corpus (~105K pairs covering ~1,503 APIs and 5,743 real-world workflows) from an existing workflow corpus via four strategies: workflow schema extension (WFEX), reverse-query reconstruction (turning a workflow back into natural-language queries), workflow pruning, and self-contrastive training to improve in-context workflow use. Fine-tuned open models orchestrate 70+ action types (~10x more than before) and beat GPT-4o on the accompanying benchmark, which scores syntax validity, execution success, and LLM-judged plan quality.
- **Relevance to TaskIR:** The closest published analog to this project. Its pipeline (harvest real executable artifacts → reconstruct NL queries from them → prune → SFT) is directly reusable for building a TaskIR corpus, and its three-axis evaluation (validity / executability / semantic quality) is a good benchmark template.

### 2. OrchDAG: Complex Tool Orchestration in Multi-Turn Interactions with Plan DAGs
- **arXiv:** https://arxiv.org/abs/2510.24663 (Lu, Liu, Dong, 2025)
- **Summary:** A synthetic-data pipeline that represents multi-turn tool execution as **plan DAGs with adjustable complexity** (controllable number of nodes, dependencies, and tools), used to train models for complex tool orchestration. Training combines SFT on the synthetic DAG data with RLVR (reinforcement learning with verifiable rewards) using a **graph-based reward**: per-node AST matching against the reference plan and a **graph edit distance (GED)** component over the whole DAG, which yields denser, more informative reward than binary correctness.
- **Relevance to TaskIR:** Literally "train an LLM to emit a DAG IR" — with the two missing pieces this project needs: tunable-complexity synthetic data generation and a GED/AST-based reward for RL beyond exact match.

### 3. AFlow: Automating Agentic Workflow Generation
- **arXiv:** https://arxiv.org/abs/2410.10762 (Zhang et al., 2024; ICLR 2025 oral). ~684 citations.
- **Summary:** Represents agentic workflows as **executable code** over an operator library and searches that space with Monte Carlo Tree Search, using execution feedback from both soft (LLM scores) and hard (code execution) signals. Outperforms hand-designed and few-shot baselines on GSM8K, HumanEval, MBPP, etc., at lower inference cost, and the searched workflows transfer to smaller models in-context.
- **Relevance to TaskIR:** Two lessons: (a) workflow-as-code is a search-friendly, LLM-friendly serialization (TaskIR should be a textual DSL, not raw JSON); (b) MCTS + execution feedback is a proven way to *generate high-quality TaskIR training data* automatically, which can later be distilled into the Qwen 3B/7B via SFT.

### 4. IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators
- **arXiv:** https://arxiv.org/abs/2403.03894 (Paul, Glavaš, Gurevych, 2024; ACL 2024). Code: https://github.com/acl2024-ircoder
- **Summary:** Builds SLTrans, a ~4M-file **parallel corpus of source code and its LLVM IR**, and shows that continued pretraining with source+IR mixing yields code LMs that are better at multilingual generation, more robust to prompt perturbations, and better at instruction following — even though IR is never the target output.
- **Relevance to TaskIR:** Empirical evidence that a compiler-style IR acts as a pivot that teaches transferable program structure. Directly supports a "continued pretraining on (natural task → TaskIR) pairs before SFT" stage for Qwen 3B/7B.

### 5. ADAS: Automated Design of Agentic Systems
- **arXiv:** https://arxiv.org/abs/2408.08435 (Hu, Lu, Clune, 2024)
- **Summary:** Introduces Meta Agent Search: a meta-agent (e.g., GPT-4) iteratively writes, evaluates, and archives **agents as code** in a growing design archive; discovered designs outperform hand-built ones and transfer across models/domains. Frames "automated design of agentic systems" as its own research direction.
- **Relevance to TaskIR:** Same recipe as AFlow from the discovery side; the archived (task, program) pairs are a candidate source of TaskIR training data via distillation.

### 6. GPTSwarm: Language Agents as Optimizable Graphs
- **arXiv:** https://arxiv.org/abs/2402.16823 (Zhuge et al., 2024; ICML 2024). ~540 citations. Code: https://github.com/metauto-ai/GPTSwarm
- **Summary:** Unifies LLM agent systems as **computational graphs** where nodes are LLM/programming operators and edges are information flow. Optimizes both node behavior (prompt/operator parameters) and, notably, **graph topology via REINFORCE over edges** — an early demonstration of learning structure at the graph level for agent programs.
- **Relevance to TaskIR:** Background for treating TaskIR topology (edges/dependencies) as a learnable, optimizable object rather than fixed scaffolding.

### 7. Large Language Models for Compiler Optimization
- **arXiv:** https://arxiv.org/abs/2309.07062 (Cummins et al., Meta, 2023 — *pre-2024, included as the canonical IR-emission result*)
- **Summary:** Trains a 7B decoder to take unoptimized LLVM IR and emit optimized IR as text. Key engineering: instruction tuning with identifier preservation (via debug metadata/card comments) so the model emits stable variable names; achieves a 3.0% instruction-count reduction over the compiler's default -Oz and shows the small model does genuine code reasoning.
- **Relevance to TaskIR:** Proof that a ~7B model can be instruction-tuned to emit a compiler IR losslessly and usefully — the "small model, formal target language" regime TaskIR lives in. Also a cautionary note on canonicalizing/normalizing IR syntax before training.

---

## Tier 2 — Data quality and curriculum for code-generation training

### 8. Magicoder: Source Code Is All You Need (OSS-Instruct)
- **arXiv:** https://arxiv.org/abs/2312.02120 (Wei et al., 2024; ICLR 2024 — *late 2023 preprint, 2024 venue*)
- **Summary:** OSS-Instruct seeds synthetic instruction generation with **randomly sampled open-source code snippets**, de-biasing and diversifying the synthetic data relative to pure self-instruct (Evol-Instruct); an evolutionary variant increases difficulty. Magicoder-7B (StarCoder-based) outperforms much larger models on HumanEval+/MBPP+ (EvalPlus), showing seed-grounded synthetic data is the lever, not scale.
- **Relevance to TaskIR:** The seeding idea transfers verbatim: seed TaskIR instruction generation from *real executable task fragments / tool traces* instead of asking the model to invent tasks from nothing.

### 9. SelfCodeAlign: Self-Alignment for Code Generation
- **arXiv:** https://arxiv.org/abs/2405.17057 (Wei et al., 2024)
- **Summary:** Aligns a code LLM **without distilling from bigger LLMs**: the base model (Qwen-Coder-7B class) generates concepts from seed snippets, poses problems, produces multiple candidate solutions, self-evaluates them, and curates the training set (nearest-neighbor selection for diversity) before SFT with the model's own filtered data. Matches or beats models trained on proprietary-model distillation and motivated the data pipelines behind later open coder releases.
- **Relevance to TaskIR:** Directly reusable for Qwen 3B/7B: generate TaskIR candidates with the base model, **filter by execution/simulation**, self-curate, then SFT — no GPT-4 dependency.

### 10. Curriculum Learning for Small Code Language Models
- **arXiv:** https://arxiv.org/abs/2407.10194 (Naïr et al., 2024; ACL 2024 SRW)
- **Summary:** Shows that ordering fine-tuning data **from easy to hard concepts** (simple syntactic tasks first, complex semantic ones later) significantly improves small decoder-only code LMs over random ordering and anti-curriculum. Follow-ups (e.g., "Should Code Models Learn Pedagogically?", arXiv:2502.03806, per search results — ID not independently verified) study the same question at pretraining scale.
- **Relevance to TaskIR:** A TaskIR curriculum is natural: single-node IR → linear chains → branching DAGs → loops/conditionals; or syntax-valid IR first, execution-correct IR later.

### 11. Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning
- **arXiv:** https://arxiv.org/abs/2506.11300 (Zhang, Mohamed, Abdine, Shang, Vazirgiannis, 2025)
- **Summary:** First systematic large-scale study of curriculum in LM pretraining: 200+ models across three scales and up to 100B tokens. Difficulty-ordered data reaches baseline performance with **18–45% fewer training steps**, and curriculum-as-warmup adds up to +3.5% final gain; effect is strongest with automated difficulty measures and at smaller compute budgets.
- **Relevance to TaskIR:** Small-budget (LoRA-class) training of a 3B/7B is exactly the regime where curriculum pays off most; use structural complexity of TaskIR samples as the difficulty signal.

---

## Tier 3 — Constrained decoding and format-aware training

### 12. Grammar-Aligned Decoding
- **arXiv:** https://arxiv.org/abs/2405.21047 (Park, Wang, Berg-Kirkpatrick, Polikarpova, D'Antoni; NeurIPS 2024)
- **Summary:** Proves that naive token-mask grammar-constrained decoding **distorts the model's conditional distribution**, producing grammatical but semantically degraded outputs, and proposes ASAp, a lookahead-based algorithm whose outputs match the model's true conditional probabilities under the grammar constraint.
- **Relevance to TaskIR:** If TaskIR ships with a CFG/JSON-schema and a constrained-decoding serving path, use grammar-aligned masking (not naive logit masks) — or better, make the model natively fluent in TaskIR so constraints are only a safety net.

### 13. IterGen: Iterative Semantic-aware Structured LLM Generation with Backtracking
- **arXiv:** https://arxiv.org/abs/2410.07295 (Ugare et al., 2024; ICLR 2025)
- **Summary:** Extends grammar-guided generation beyond syntax to **semantic constraints** (e.g., identifiers must be defined/used consistently) with a backtracking mechanism, improving SQL and Vega-Lite generation and preventing constraint-violating leakage.
- **Relevance to TaskIR:** TaskIR has semantic well-formedness rules beyond its grammar (e.g., dataflow: every edge references declared ports). IterGen-style semantic-aware decoding complements SFT at inference time.

### 14. "Let Me Speak Freely?" A Study on the Impact of Format Restrictions on LLM Performance
- **arXiv:** https://arxiv.org/abs/2408.02442 (Tam et al., 2024; EMNLP 2024)
- **Summary:** Shows strict format restrictions (JSON mode) during complex reasoning **degrade performance substantially** (e.g., Claude-3-Haiku on GSM8K: 86.51 → 23.44; GPT-3.5-Turbo: 76.60 → 49.25); format-following improves while reasoning quality drops.
- **Relevance to TaskIR:** The core argument for *training* TaskIR emission rather than prompting it: a model that has internalized the IR format pays no reasoning tax. Also argues for keeping the serialization model-friendly (whitespace, comments, code-like tokens) instead of dense JSON.

---

## Tier 4 — Structure-aware losses, rewards, and graph-structured generation

### 15. PPOCoder: Execution-based Code Generation using Deep Reinforcement Learning
- **arXiv:** https://arxiv.org/abs/2305.00266 (Shojaee et al., 2023 — *pre-2024, canonical AST-reward reference*)
- **Summary:** Fine-tunes code LMs with PPO using a **multi-component reward: unit-test execution, AST sub-tree matching against the reference, and error-type penalty**. The AST component gives a dense structural signal that improves sample efficiency over execution-only reward and avoids reward hacking on tests.
- **Relevance to TaskIR:** Template for a TaskIR reward: executability + structural similarity (AST/GED over the IR graph) + error typing. Related 2025 line: Graph-Reward-SQL (ACL 2025) uses AST-based structural rewards for execution-free RL (found via search; arXiv ID not independently verified).

### 16. DA-Transformer: Directed Acyclic Graph Transformer for Text Generation
- **arXiv:** https://arxiv.org/abs/2203.09148 (2022 — *pre-2024, canonical DAG-generation architecture reference*)
- **Summary:** Replaces the linear token chain with a **DAG of states**, each node generating tokens in parallel; paths through the DAG correspond to candidate outputs. Introduces the associated training objective (max over paths / expected loss over DAG hypotheses) and DAG-aware inference (Viterbi decoding over the DAG).
- **Relevance to TaskIR:** Background if TaskIR ever wants native non-autoregressive or parallel DAG emission; more immediately, its "loss over paths in a DAG" formalism inspires sequence-level losses over linearized IR (e.g., marginalizing over valid topological orders).

### 17. Execution-Aware / Structure-Aware Adaptation of Qwen for Code Generation
- **Source:** ACM Digital Library (Dec 2025), "Structure-Aware Multi-Stage Adaptation of Qwen for Code Generation" (found via search; **no independently verified arXiv ID**)
- **Summary:** Reports an explicit **"AST Alignment Loss"** in a multi-stage fine-tuning of Qwen-72B for code generation — structure-aware supervision beyond token-level cross-entropy.
- **Relevance to TaskIR:** Evidence that AST-level auxiliary losses are entering production-scale Qwen training; a TaskIR analogue would add a node/edge-consistency loss on the emitted graph.

---

## Tier 5 — Evaluation metrics beyond exact match

### 18. Don't Pass@k: A Bayesian Framework for LLM Evaluation
- **arXiv:** https://arxiv.org/abs/2510.04265 (2025)
- **Summary:** Critiques pass@k as a single-sample accuracy statistic and introduces **pass^k** (probability of passing in *all* k i.i.d. trials — reliability/consistency) plus a generalized **G-pass@k** continuum interpolating between the two, with Bayesian estimation.
- **Relevance to TaskIR:** TaskIR emission should be measured for *reliability* (every generation valid and executable), which pass^k captures and pass@k hides.

### 19. RSPO: Risk-Seeking Policy Optimization for Pass@k and Max@k
- **arXiv:** https://arxiv.org/abs/2508.01174 (2025)
- **Summary:** Aligns training objective with best-of-k selection: risk-seeking RL objective that optimizes **Max@k** (reward of the best of k samples) rather than the mean, improving pass@k at high k without reward models.
- **Relevance to TaskIR:** If the pipeline is "generate k TaskIR candidates, verify, pick one," RL objectives should target Max@k-style selection, not mean quality.

### 20. Graph-edit-distance-based evaluation in applied systems (cluster of recent work)
Found via search; individual IDs not independently verified — treat as leads:
- **SAGE-HLS** (DAC 2025; ACM DL): syntax-aware AST-guided LLM for high-level-synthesis code; evaluates with **graph edit distance between ASTs** of generated vs. reference code.
- **NB2P** (ACM): generating data-science pipelines, evaluated by **GED between workflow graphs/ASTs**.
- **BOFLOW** (OpenReview): Bayesian-optimization workflow generation; uses mean pairwise **GED over workflow ASTs as a diversity metric** for generated candidates.
- **DynSTEER** (arXiv, Sep 2026): trajectory evaluation combining AST checks and **GED for structural topology alignment**.
- **Relevance to TaskIR:** The emerging consensus metric set for structured generation: per-node exact/AST match + global GED + execution success + diversity (pairwise GED). OrchDAG (Tier 1) additionally uses GED as an RL *reward*, closing the loop between metric and training.

### 21. Property-based testing as functional-correctness evaluation
- **Source:** Virginia Tech thesis/repository 2025 (found via search; arXiv ID not verified)
- **Summary:** Argues for property-based testing (Hypothesis-style) instead of fixed unit-test suites for code-generation evaluation — checks invariants over generated inputs rather than memorized cases, complementary to pass@k.
- **Relevance to TaskIR:** Property-based checking of TaskIR (well-formedness invariants, dataflow soundness) is a natural verifier for RLVR.

---

## Tier 6 — Small-LM context: surveys and assessments for the 1B–8B regime

### 22. A Comprehensive Survey of Small Language Models in the Era of Large Language Models
- **arXiv:** https://arxiv.org/abs/2410.20011 (2024–2025). Companion practical survey of techniques, problem settings, and evaluation for SLMs.

### 23. Small Language Models Can Still Pack a Punch: A Survey
- **arXiv:** https://arxiv.org/abs/2501.05465 (2025). ~160 papers on 1–9B models, including fine-tuning techniques and on-device deployment.

### 24. Assessing Small Language Models for Code Generation
- **arXiv:** https://arxiv.org/abs/2507.03160 (2025). Empirical assessment of what SLMs can and cannot do for code generation; useful for calibrating expectations for a Qwen-3B code emitter.

### 25. Code Generation with Small Language Models: A Codeforces-Based Study
- **arXiv:** https://arxiv.org/abs/2504.07343 (2025). Competitive-programming evaluation of SLMs; documents reasoning limits of the 1–8B class that TaskIR training must design around.

---

## Additional leads (surfaced in search, not yet verified — check before citing)

- **FlowBench** — benchmark for workflow-guided LLM agents; **FlowReasoner** (query-level meta-agent, RL) — https://www.alphaxiv.org pages found in search.
- **EvoFlow**, **"When Does Multi-Agent RL Improve LLM Workflows"** — follow-ups to AFlow/ADAS on learned workflow structure.
- **Learning Context-Free Grammars for Grammar-Constrained Code Generation** — DSL generation in low-data regimes (learns the grammar, then constrains decoding).
- **Earley-driven dynamic pruning for efficient structured decoding** (Sun et al.) — efficient CFG-constrained logits masking; also **SynCode** (same group as Grammar-Aligned Decoding).
- **KODCODE**, **ACECoder**, **OpenCodeInstruct** — synthetic code-instruction datasets that improve on OSS-Instruct seed diversity; **HPC-Coder-v2/HPC-Instruct** applies the recipe to a low-resource DSL domain (closest domain analog to TaskIR data generation).
- **"Teaching According to Talents"** (ACL 2025) — difficulty-aware ordering of instruction-tuning data.
- **WorkArena** (arXiv:2403.07918) — enterprise web-agent benchmark; sometimes confused with WorkflowLLM's benchmark.

---

## Synthesis: a TaskIR training recipe assembled from this literature

1. **Corpus construction (WorkflowLLM + Magicoder + SelfCodeAlign):** harvest real executable tasks/tools; generate (natural task → TaskIR) pairs seeded from real fragments; use reverse-query reconstruction (TaskIR → NL query) to amplify the corpus; have the base Qwen model generate candidates and **filter by execution/simulation**, self-curate for diversity (nearest-neighbor de-dup).
2. **Stage 1 — continued pretraining (IRCoder):** mix TaskIR-annotated text into a light continued-pretraining run so the representation becomes familiar before SFT.
3. **Stage 2 — SFT with curriculum (2407.10194 + 2506.11300):** order samples by structural complexity (single node → chain → DAG → control flow); at 3B/7B scale with LoRA-class budgets this is where curriculum gains are largest. Serialize TaskIR as a code-like DSL (AFlow, Let-Me-Speak-Freely lesson), not dense JSON.
4. **Stage 3 — RLVR with structural reward (OrchDAG + PPOCoder + AFlow):** reward = execution success + per-node AST match + global graph edit distance + error-type penalty; if candidate selection is best-of-k, use a Max@k-aligned objective (RSPO).
5. **Inference:** grammar-constrained decoding with alignment (ASAp / IterGen) over the TaskIR CFG as a safety net; semantic constraints (dataflow/typing) checked via IterGen-style backtracking or a post-hoc verifier.
6. **Evaluation:** report (a) syntax validity rate, (b) execution success, (c) per-node accuracy, (d) global GED to reference IR, (e) pairwise-GED diversity, (f) reliability via pass^k rather than pass@k alone (2510.04265), (g) LLM-judged semantic adequacy as in WorkflowLLM's benchmark.
