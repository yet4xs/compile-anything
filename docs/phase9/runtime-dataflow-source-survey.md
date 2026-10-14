# Runtime Dataflow Source Survey

**Phase**: 9B-0 Task 5 | **Date**: 2026-10-15

## Survey: Public Workflow/Tool Trajectory Sources

### 1. τ³-bench (BitMind)

- **Domain**: airline, banking, retail, telecom
- **Tools**: 14-68 per domain (Python, @is_tool decorated)
- **License**: MIT
- **Multi-step**: Yes (multi-turn conversations with tool calls)
- **Runtime deps**: Likely but unverified (arguments not stored per-step)
- **Schema**: tools.py has Python type annotations; output fields readable from return types
- **Size**: ~2,546 test cases; many with 2+ reference actions
- **Feasibility**: HIGH — has actual runtime simulator; can replay trajectories to capture outputs
- **URL**: https://github.com/sierra-research/tau2-bench

### 2. ToolBench (OpenBMB)

- **Domain**: general (16,000+ RapidAPI endpoints)
- **Tools**: thousands
- **License**: Apache 2.0
- **Multi-step**: Yes (ReAct-style multi-turn)
- **Runtime deps**: NO (our audit: 77 multi-tool tasks, all independent)
- **Schema**: Partial (parameters in tool definition, no output schema)
- **Size**: ~126k instructions
- **Feasibility**: LOW — multi-tool calls are independent; no output schemas
- **URL**: https://github.com/OpenBMB/ToolBench

### 3. BFCL V4 (Berkeley)

- **Domain**: general function calling
- **Tools**: varies per category
- **License**: MIT
- **Multi-step**: NO (all single-tool)
- **Runtime deps**: N/A
- **Schema**: Good input schemas, no output
- **Size**: ~2,000 test + training
- **Feasibility**: N/A — designed for single-tool, not chain evaluation
- **URL**: https://gorilla.cs.berkeley.edu/leaderboard.html

### 4. xLAM (Salesforce)

- **Domain**: general function calling
- **License**: CC-BY-NC-ND
- **Multi-step**: Minimal
- **Runtime deps**: NO
- **Feasibility**: LOW
- **URL**: https://github.com/SalesforceAIResearch/xLAM

### 5. API-Bank (Microsoft)

- **Domain**: banking, 73 APIs
- **License**: Research use
- **Multi-step**: Yes (planned API calls)
- **Runtime deps**: Possible (banking workflows like transfer → confirm)
- **Schema**: Good (API specs with parameters)
- **Size**: ~314 dialogues
- **Feasibility**: MEDIUM — small but domain-focused
- **URL**: https://github.com/AlibabaResearch/DAMO-ConvAI

### 6. WorkArena (ServiceNow)

- **Domain**: enterprise web automation (700+ tasks)
- **License**: MIT
- **Multi-step**: Yes (complex UI workflows)
- **Runtime deps**: Likely (web form → extract value → use in next form)
- **Schema**: DOM-based (not API schemas)
- **Feasibility**: MEDIUM — different abstraction level (UI, not API)
- **URL**: https://github.com/ServiceNow/WorkArena

### 7. WebArena (CMU)

- **Domain**: web tasks (shopping, forum, git)
- **License**: MIT
- **Multi-step**: Yes
- **Runtime deps**: Likely (search → extract → navigate)
- **Schema**: No API schemas (web-based)
- **Feasibility**: LOW for our use case (web, not tool API)
- **URL**: https://webarena.dev

---

## Recommendation

**Primary source: τ³-bench** — has multi-step trajectories, real tool schemas, domain diversity, MIT license, and a runtime simulator that can capture actual output values. The main gap is that per-call arguments are not stored, requiring replay.

**Secondary source: API-Bank** — small but banking-domain focused with potential runtime deps in transfer/confirm workflows.

**Method: τ³ trajectory replay** — use τ³'s simulator to execute reference action sequences, capture actual tool outputs, then compare downstream inputs to identify runtime dependencies. This gives us REAL runtime values, not synthetic ones.

**Fallback: Synthetic-from-Real** — if τ³ replay yields < 50 L3 tasks, construct workflows using real API schemas (from ToolBench/τ³ tools) but human-curated dependency chains. This is NOT synthetic chain generation — it's realistic workflow construction with real tool interfaces.
