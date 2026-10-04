# Claim–Evidence Matrix — Phase 5D 冻结

> 每条论文 claim 对应的冻结证据、工件与**允许/禁止的表述**。Phase 6 起不得修改
> 本表所列结果文件（见 `experiments/paper_snapshot_v1.json`）；新实验进 `phase6/*`
> 或新 snapshot。

| # | Claim | Evidence（数字） | Commit / artifact | Status | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|---|
| 1 | TaskIR 结构编译可被小模型学会 | E0 valid 0% → E1 valid 99.10%、OpSeq 88.54%、skill F1 0.92（n=1,561） | `659061e` 前序；`experiments/phase5b1/`；`docs/results-phase5b1-complete.md` | DONE | "28k audited supervision teaches a 3B model near-complete structural TaskIR compilation (99.1% validator pass)" | "3B 优于 7B"（无训练后 7B 对照，E3 NOT RUN） |
| 2 | 跨域结构泛化（三个基准家族，零污染） | BFCL valid 92.4%（4,696）；τ³ 98.86%（2,546）；AgentBoard parse 97.7 / valid 91.7%（351，untouched 首测） | `da5c4fe`；`results/phase5c/`、`results/agentboard/first_test.json` | DONE | "structural validity transfers to three benchmark families never seen in training" | "AgentBoard task success / performance / beats baselines"（UNSCORED-OFFLINE） |
| 3 | BFCL 跨域语义接地弱 | E1-A Functional E2E 15.50%（4,541 分母）；条件口径 16.83% | `results/phase5c/bfcl_semantic.json`（冻结 audit 规则） | DONE | "end-to-end functional semantics 15.5%"；条件值必须标 Functional\|Valid diagnostic | 把 18.19% 当作 E2E 成功率；"schema gives +1.36pp E2E" |
| 4 | τ³ 规划深度/接地 | E5C-S pred/T 1.75（ref 5.83）；召回 0.58%（89–92/14,834） | `results/phase5c/tau3_semantic.json`（归一化协议） | DONE | "grounded semantic recall ~0.6%; plans remain 3.3× shallower than reference" | 用 τ³ 召回声称任何方法"验证了事务语义"；把 E1-A 旧协议 pred/T 与新协议混表 |
| 5 | Schema 条件化效应 | BFCL E2E 15.50→15.86（**+0.35pp**）；条件 16.83→18.19（诊断）；internal OpSeq 92.76→94.11；pred/T 1.00→1.75 | `experiments/phase5c/manifest.json` freeze_fix；`da5c4fe` | DONE | "small but consistent improvement in end-to-end semantic correctness; substantially changes planning-length prior" | "semantic breakthrough"；只引条件 +1.36pp 而不提 E2E +0.35pp |
| 6 | 深度课程负结果 | pred/T +0.65 但召回 0.62→0.55、internal OpSeq −4.77pp、τ³ valid −12pp、BFCL E2E −1.03pp；交互 −0.65（饱和） | 同上 | DONE（负结果保留） | "length is trainable; correctness is not obtained by naive depth reweighting" | 删除或弱化该负结果；把长度提升说成语义改善 |
| 7 | Binder 非瓶颈 | oracle 语义计划→binder→工具 P/R/F1 = 99.95% | `docs/results-tau3-offline*.md` | DONE | "front-end skill selection is the bottleneck; back-end binding oracle upper bound 99.95%" | — |
| 8 | EXEC_ACTION 边界为主导失败 | τ³ 96% 参考为 EXEC_ACTION、模型产出 ~0（训练有 1,649 例）；BFCL multi_turn 同构；AgentBoard tool-op 23.9% vs FETCH 压制 | `docs/results-phase5c.md` §5.1；`results/agentboard/` | DONE | "**dominant observed** cross-domain failure is semantic grounding at the retrieval–action boundary" | "EXEC_ACTION is the only semantic problem" |
| 9 | AgentBoard 确认 | 351 条、预注册（d59d00b 先于运行）、one-shot、离线编译协议 | `docs/agentboard-preregistration.md`；`results/agentboard/first_test.json` | DONE | "offline compiler confirmation: structural validity replicated on a third benchmark family" | "AgentBoard success improved"；与官方 SR 同图数值对比（只可文字引用为 non-comparable literature reference） |
| 10 | Effect system 为设计提案 | τ³ 14,834 动作分类（46.6/37.8/3.2%）；2,443 多动作任务 | `docs/effect-system-proposal.md`；`docs/tau3-effect-audit.md` | PROPOSED | "workload characterization motivating a proposed MemorySSA-style design; cross-class transactions identified as a gap" | "effect system implemented"；"τ³ validates transaction semantics"；V7 已实现 |
| 11 | 名义成本模型 | 5 程序示例：energy 3.4%、makespan 37.9%（vs 70B 单发） | `benchmark_results.md`；`src/cost/model.py` | NOMINAL | "nominal analytical model, not measured hardware energy/latency"；不作 headline | 任何未加 nominal 限定的引用；作为 abstract headline |
| 12 | 数据质量闸门 | suspect 64.6%→1.06%（参数保全 16.7%→98.9%，lifter-only 修复） | `docs/semantic-label-audit.md` | DONE | "deterministic semantic audit; lifter-only fix" | 声称做过 v3 重训对照（E8 重训 NOT RUN） |
| 13 | E3 / 官方交互 / 实测 profiling | — | `docs/paper-evaluation.md` §9 清单 | NOT RUN | "disclosed as not run" | 任何估计值填充 |

**核心结果压缩版（论文任何地方引用须与此一致）：**

```text
99% internal structural validity (E1, 99.10%)
high structural validity across BFCL / τ³ / AgentBoard (92.4 / 98.9 / 91.7%)
large external semantic gap (BFCL E2E 15.50%; τ³ recall ~0.6%)
BFCL E2E semantics: 15.50 → 15.86 with schema conditioning (+0.35pp)
τ³ semantic recall remains ~0.6% across all four 5C cells
planning length: 1.00 → 1.75 but correctness unchanged
binder oracle upper bound: 99.95%
EXEC_ACTION grounding is the dominant observed cross-domain failure
```


> **Phase 6A.5 修订**：第 5 行 Schema 条件化效应更名为 capability-marker conditioning —— xLAM 19,700 条的 capability 块为常量 `["EXEC_ACTION"]`（字符串迭代 bug）。数字不变；禁止把 E5C-S−E1-A 差值归因于 capability semantics。详见 docs/phase6/phase5c-capability-audit.md 与 paper_snapshot_v2。
