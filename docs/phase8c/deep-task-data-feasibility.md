# Deep-Task Data Feasibility (Phase 8C Pre-study)

**日期**: 2026-10-14

## 现状
- v3 语料 4+ 步:389 条(2.7%),7+ 步:22 条 → 统计力不足
- 主因:corpus 来源(ToolBench-static/xLAM)以 1-2 步单工具任务为主

## 可用来源

| 来源 | 4+ 步规模 | 参数可追溯 | DATA/ORDER 可区分 | 隔离 |
|------|----------|-----------|-------------------|------|
| τ³ 剩余未用(~2,400 含部分多步) | ~600-800 | 部分(评估用 actions 含 arguments) | 可(oracle effect_class) | 需建独立 train/dev/test;不能与 eval 重叠 |
| ToolBench full conversations | 未审计 | 罕见(多为 thought-action 链) | 无 oracle | 与训练语料重叠需审计 |
| 合成参数传递链 | 任意 | 是(可编程) | 是(可控制) | 天然隔离 |

## 建议
1. **合成参数传递链**:在 v3 语料的工具家族内,构造"A 输出 → B 消费"的参数传递(读→算→写型),值可追溯,DAG 结构可控制;标记 synthetic;保持家族隔离
2. **τ³ 独立协议**:若用作训练,需从 2,400 未用中划出独立 train/dev/test(不触碰已用的 dev-30 + 120 diagnostic);同时保留其中 ~200 条作为最终 untouched 外部测试
3. **优先级**:合成链先行(无隔离风险,可即刻构建);τ³ 训练使用需 advisor 批准(涉及外部测试集重分类)

## 下一步
- 构建合成参数传递链 ~1,000 条(4-6 步,~200 条 7+ 步),验证 v3 lowering 兼容
- 之后再决定是否启用 τ³ 作为训练源
