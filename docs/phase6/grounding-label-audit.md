# Phase 6A Task 1 — Grounding 标签审计

> 冻结协议：`experiments/phase6/phase6a_manifest.json`（base `d6b42ef`）
> 数据：corpus v3.1 train.jsonl（28,093 条记录 → 30,886 个 tool call）
> 审计产物：`results/phase6/grounding_label_audit.json`

## 发现 1（结构）：corpus 的 `capabilities` 字段不是工具 schema

corpus 记录的 `capabilities` 是**语义技能名字符串**（如 `["SEND"]`，由
`semantic_capabilities()` 经 toolmap 从工具名导出），不是工具 schema。
Phase 5C 的 schema 条件化训练实际使用的是这种技能名列表（经
`format_capabilities` 渲染为 `[1] SEND`）。

真实工具 schema（name/description/parameters）保存在原始数据集中，需按
`lowering.tool` 回联：

| 来源 | schema 索引 | join 命中 |
|---|---|---:|
| xLAM raw 60k | 3,605 个唯一工具 | 30,875/30,886 = **100.0%** |
| toolbench_static | 2,128 个唯一工具 | （并入上数） |

**含义**：(i) Phase 6A 的 grounding 数据集完全可构建；(ii) Phase 5C 的
"schema conditioning" 实际条件是"语义技能名可见性"——这解释了为何它改变
长度先验（模型看到技能清单）而对工具级语义接地帮助有限；此发现须在论文
§8 如实补注（不改 5D 冻结数字，只改描述精度）。

## 发现 2（标签污染）：EXEC_ACTION 监督在边界上被子串正则污染

全部 1,649 个 EXEC_ACTION 标签都来自 `exact` 规则（fallback 为 0），由两条
子串正则产生：

| matched_rule | 数量 |
|---|---:|
| `calendar\|event\|meeting\|appointment\|schedule` | 923 |
| `book\|reserv\|order\|purchase\|buy\|checkout` | 726 |

**污染证据**：539/1,649（32.7%）的工具名带检索前缀（get_/list_/search_/
fetch_...）且描述不以突变动词开头——子串假阳性把数据查询端点标成了动作：

```
get_user_orders_for_demo_project   → EXEC_ACTION（"order" 子串命中；实为查订单）
get_team_schedule                  → EXEC_ACTION（"schedule" 子串命中；描述 "Fetches the schedule for an NBA team"）
getpastevents                      → EXEC_ACTION（"event" 子串命中；描述 "Fetches past Azure events"）
```

其余 1,110 个为 gold 候选（无检索前缀，或描述以明确突变动词开头）。
残留噪声仍可能存在（如 `ipo_calendar_for_twelve_data` 无检索前缀但语义上
是数据查询）——如实披露：过滤器是确定性启发式，不是完美清洗。

## 冻结的 gold 过滤器（写入 manifest，训练前冻结）

```text
EXEC_ACTION gold-eligible ⟺ mapping_kind=exact
  ∧ (¬retrieval_prefix(name) ∨ mutation_start(description))

contradictory_exact（539）→ 降级 G3（audit only，禁止进入 probe gold）
```

- `retrieval_prefix = ^（get|list|search|find|fetch|query|lookup|retrieve|show|view|check|is_|has_）`
- `mutation_start = 描述以 create|add|update|delete|cancel|book|reserve|place|make|toggle|grant|revoke|reset|activate|transfer|submit|publish|upload|subscribe|send|post|order 开头`
- toolmap.py 未做任何修改（协议禁止）。

## G-tier 分配（冻结）

| Tier | 判据 | 数量（call 级，含非边界技能） | 用途 |
|---|---|---:|---|
| G1 | mapping_kind=exact 且通过污染过滤 | 见 build 产物 | primary train/eval |
| G2 | mapping_kind=heuristic | ~17k | secondary ablation |
| G3 | fallback 或 contradictory_exact | 539+ | audit only |

## 边界技能监督规模（带 schema 的 G1/G2）

| skill | G1(exact) | G2(heuristic) |
|---|---:|---:|
| FETCH | 0 | 9,682 |
| SEARCH | 2,682 | 6,228 |
| QUERY_DB | 2,195 | 292 |
| EXEC_ACTION | 1,110（过滤后） | 0 |
| SEND | 0 | 1,428 |

注：EXEC_ACTION 与 SEND 全部经正则监督；FETCH 几乎全部来自 `get_|http|url`
前缀启发式。**边界两侧的监督都源自名字正则**——这正是 Task 14 name-masking
实验关键性的原因：若 grounder 只学名字正则，遮名后必然崩溃。
