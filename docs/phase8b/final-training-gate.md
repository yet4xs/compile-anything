# Phase 8B-1A.2: Final Training Gate — PASS

**日期**: 2026-10-13 | **基线**: `4ec3f08`

## Task 1 — 合成 Registry 修复
v3 合成从子任务**合并真实注册表**(description+required+properties);同工具不同 schema 的组合被拒;全部合成重跑 strict 链(注册表→schema→严格降级)。结果:**656 通过 / 744 拒绝**(unresolved 参数为主,全部可归因)。各深度段有效样本见 manifest(train 4-6 步 389、7+ 17——如实,不虚增)。

## Task 2 — EXEC_ACTION 数据依赖
数据依赖以**操作数**形式保留(真实值传递);顺序/副作用依赖走 after();无法表达 → `unresolved_data_dependency` 拒绝。回归测试:读订单→修改订单,`%p0` 作为 mutation 的操作数出现在降级 TaskIR 中;类型不可表达场景显式失败。**不再有任何静默数据边丢失或降级**。

## Task 3 — Schema 等级
四级:strict_verified 15,890 / schema_explicit_empty 51 / schema_unknown 1,034(不入正式 SFT)/ semantic_loss 4,847 / ambiguous 7,832。正式 SFT = strict + explicit_empty + synthetic(656)= **train 14,909**(真实 14,253 + 合成 656)。

## Task 4 — 泄漏完整检查
- 恒真断言(`or True`)已删除,代之以真实边界断言
- 近重复双通道(8-gram df≤25 + 12-token 前缀桶),高频 gram 无法隐藏配对;检出 23 对跨 split → 清除 19 → 复审 **0**
- 源谱系跨 split 组件 = 0;strict 切分全工具重叠 = 0;samefamily|train 共享 276 工具(设计内,manifest 明示非 unseen)
- manifest 数量与文件行数一致;SHA256 全文件冻结

## Task 5 — v3 冻结(data/semantic_skeleton_v3/)
train 20,382(SFT 14,909)/ dev 354(SFT 236)/ test 329(SFT 202)/ test_samefamily 1,132。深度:1 步 9,403、2-3 步 5,100、4-6 步 389、7+ 17。依赖保真 100%(有损样本已入 semantic_loss)。Strict Lowering:eval 切片 98.3-100%,train 90.4%(1,945 失败保存于 integrity_fails)。τ³ id = 0。

## Task 6 — 八项门槛
1. ✅ 最终训练样本全 strict-pass(train_sft 仅含三级通过者)
2. ✅ 合成经完整 Registry/Schema 检查(656/1400,拒绝可归因)
3. ✅ DATA/ORDER/EFFECT 不变量(测试 + 回归)
4. ✅ M0/M1 同任务 ID(见 train_pairing_audit.json)
5. ✅ CapabilityIR 完全一致(同一记录字段)
6. ✅ 训练/测试隔离(家族分量 + 近重复双通道清零)
7. ✅ 无 τ³ 混入
8. ✅ token 预算可对齐(M0 目标 85 tok vs M1 95 tok/条,按有效 token 调 epoch 对齐)

测试:69/69 通过(skeleton lowering 7 + 语义完整性 9 + 数据集 7 + τ³ grounding 12 + 约束装配 10 + 在线解码 24)。

**GATE: PASS → 启动 Phase 8B-1B**
