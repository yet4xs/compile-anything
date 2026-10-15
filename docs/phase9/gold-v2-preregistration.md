# Gold v2 预注册 — Phase 9B-2.5 T2

**状态**: 预注册。本文档在任何"严格任务计数"产出之前单独提交
(commit 见 git log)。看数后修改规则 = 新版本预注册 + 书面理由 +
新旧并列报告。

## 1. τ³ 版本锁定

- tau2-bench commit: **5ade54e**(2026-10-14, "Add Grok Voice 3.0
  custom voice submission (#614)")。CHANGELOG 显示 1.0.1 后仅
  banking_knowledge 变更。
- 文件 md5(**LF 归一化后**;Windows checkout 的 CRLF 会使原始字节
  md5 不同,归一化口径为本注册的规范口径):

| 文件 | md5 (LF-normalized) |
|------|---------------------|
| retail db.json | 69771c8c71d2cf8fa767574c3a65999a |
| retail tasks.json | c07e89e811d345bc062461aeff7de7d7 |
| airline db.json | 55b232f2101d058fab0deea60d306675 |
| airline tasks.json | 4588a63c1fa9844914c058b1d3dc107e |

以上四值与 advisor 方在干净克隆上的复核值逐一相符(本仓库已独立
复现)。不一致 → 停止并报告。

## 2. 候选池

官方 retail + airline 任务中**参考动作 ≥ 2** 的全部任务。本仓库独立
复现计数: **retail 92 + airline 25 = 117**(官方 test split: retail
33、airline 9)。旧 converter 的手写 DOMAIN_TOOLS 与启发式 dep 过滤
作废,统一使用 verified_registry_manifest.json(957e1ba)。
编译输入统一 compile_input_v2(a268cf6,全披露单轮协议)。

## 3. 参数类别 v2(可执行判定规则)

在 v1(9B-2.3)基础上扩展。判定顺序即规则序:

| 类别 | 判定规则 |
|------|----------|
| LITERAL | 值(逐元素)出现在 compile_input_v2 文本中 |
| DUAL_SOURCE | LITERAL 条件成立 **且** 值也出现在先前真实输出 → 不强制选边,equivalence set = {literal, ref} |
| runtime_ref | 值仅来自先前输出;canonical_source 按 v1 四规则(unique_candidate / canonical_field / canonical_minimal_depth / root_scalar_origin),不改 |
| runtime_ref_multi | 列表参数逐元素 runtime_ref |
| SAME_ENTITY_ECHO | 多候选但全部候选路径属于同一实体键(同生产者、同实体容器索引,如 orders[2].* 内部) → equivalence set = 全部候选 |
| COINCIDENTAL_VALUE | 多候选且跨实体键 → 值巧合相同,不入 equivalence set,**仅人工裁决可入** |
| LOW_ENTROPY_MATCH | 值为 bool / ≤2 位整数 / 枚举短词(economy/basic/…)/ ≤3 字母代码,且无实体键支撑 → 永不自动 strict,**仅人工裁决可入** |
| ENV_STATE | 值仅存在于初始 DB 且不属于行动用户自有实体 |
| USER_LATENT | 值仅存在于行动用户自有 DB 实体 |
| UNKNOWN | 以上皆非 |

实体键定义:候选路径 (producer_step, path) 的最近祖先实体容器
(List[Struct] 的 `[i]` 索引 + 其结构类型名),按 verified registry
的模型定义计算。

任务级标记:
- **EXPECTED_ERROR_STEP**: gold 动作序列含**预期的**失败步骤(先
  失败再回退的 gold 设计)。已知候选: retail_35/37(库外邮箱→回退)、
  retail_64(pending exchange 被拒→改 modify)、retail_2/3/4(先查
  不存在商品)。逐条以"失败步骤后 gold 继续,且后续步骤构成对失败
  的回退"判定;无法证明意图的标 REPLAY_ERROR。
- **POLICY_DEPENDENT**: gold 动作集合依赖 policy.md 而编译输入不含
  policy(例:用户请求的写操作在 gold 中缺失)。分诊信号:
  description.purpose(**仅此处允许读 description**,代码隔离于
  compile_input_v2 模块)+ "gold 缺用户明确请求的操作"检查。逐条
  附证据。
- **REPLAY_ERROR**: 非预期的重放异常,既无回退结构也无政策解释。

**勘误预告**: 9B-2.1 的"EXCLUDED 上报 τ³ 数据缺陷"建议撤回——
上述任务多为 guard/回退语义的真实用例(见 phase9b-2.5-review 勘误节)。

## 4. 等价类 Gold

每个参数存 canonical_source(v1 四规则)+ acceptable_sources[]:
- SAME_ENTITY_ECHO 与 DUAL_SOURCE 的集合自动生成;
- COINCIDENTAL_VALUE 与 LOW_ENTROPY_MATCH 只能经人工裁决入集合
  (κ ≥ 0.8 双标注,材料由 T5 生成,zcode 不代填)。

## 5. strict v2 准入(全部满足)

1. 重放无非预期异常(允许 EXPECTED_ERROR_STEP?**否**——strict v2
   排除 EXPECTED_ERROR_STEP 与 REPLAY_ERROR;它们单列为 guard/回退
   用例集);
2. 全部参数 ∈ {LITERAL, DUAL_SOURCE, runtime_ref,
   runtime_ref_multi, SAME_ENTITY_ECHO};
3. 非 POLICY_DEPENDENT;
4. 生产 E2E 通过(9B-2.4 同口径:lower→parse→validate→官方执行,
   逐步参数/工具/DB hash + 终态 hash 一致)。

## 6. 分组键

(domain, acting_user, 工具序列模板) — 沿用 9B-2.4,不改。

## 7. 测试条件(默认)

- **条件 A(域内)**: retail;test = strict v2 ∩ τ³ 官方 test split,
  按组闭包(组跨 split 整组归 train);永久排除 9B-2/9B-2.2 错误
  分析的 14 个任务。
- **条件 B(跨工具族)**: airline 全部 strict v2 整体作 held-out
  test;train 不得出现任何 airline 工具(工具 ID 加 domain 命名
  空间隔离)。
- 此条构成对 §2.1(工具族不重叠跨域条件)的协议修订。

## 8. 阈值(默认,所有者可在跑数前修改)

- 条件 A test **≥15 任务且 ≥8 组**才称"正式测试",否则只称"诊断"。
- 0 成功时 Clopper–Pearson 95% 上界: n=5→52.2%, n=15→21.8%,
  n=30→11.6%。

## 9. 指标(分列,不合并)

- strict_origin_exact(canonical source 精确,现行指标)
- equivalence_class_exact(acceptable_sources 集合内)
- value_match(值等)
- 终态 DB hash
- 各臂 Clopper–Pearson CI;臂间配对精确 McNemar。

## 10. 硬约束重申

不改写既有结果文件(新产物 _v2/_v3 后缀,JSON 内记录生成 commit、
输入 sha256、τ³ 版本锁定);人工裁决只由人完成;被推翻的旧数字进
勘误节;本注册与 advisor 方复核数字不一致时以本仓库独立复现为准
并列差异。
