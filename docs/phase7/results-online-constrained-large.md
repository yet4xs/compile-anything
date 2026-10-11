# Phase 7C-2: Large-Scale Typed SSA Online Decoding Evaluation

**日期**: 2026-10-09 启动(运行中)
**基线**: `8e84561`(7C-1)→ 冻结代码 `3b3977c`
**清单**: `experiments/phase7/online_decoder_manifest.json`
**状态**: 确认集 150 × 3 种子 × 5 臂运行中(约 10 小时,断点续跑)

---

## 1. 评测口径修正(Task 1,已实现并有回归测试)

- **主指标 = Valid ∧ OpSeq(E2E)**,无效程序记零(7C-1 的 OpSeq% 不要求过 Validator,不能当端到端语义正确率;advisor 重算后 G0/G1/G2 = 53.3/70.0/73.3)
- 参数键质量:固定全集分母的 E2E 指标 + 条件指标分列;配对按**策略操作过滤后**的节点序列(消除位置对齐偏差)
- 新增 E2E 指标:Skill F1(过滤后操作多重集 F1)、EXEC_ACTION Recall(计数上限)
- 安全事件率正式入表:deadlock%、FSM violation%、forced 均值
- EOS `-1` 哨兵已从 `allowed_ids` 移除(旧实现会在合法 EOS 时把 -1 混入允许集,负索引误放行一个词表位)
- 真 tokenizer 一致性测试(服务器运行):字节级解码 vs `tok.decode` 抽样对拍、特殊 token 全状态禁用、非 ASCII token 全状态禁用、**tokenize(合法程序) 逐 token 过 FSM 全通过**(端到端一致性)

## 2. 实验臂(冻结于 `3b3977c`)

| 臂 | level | 内容 |
|----|-------|------|
| G0 | 0 | 无约束贪心基线 |
| G1 | 1 | 文法 FSM + EOS 门控 + 注释上限(800) |
| G2a | 2 | G1 + defined-before-use + arity 封顶 + 重复定义强制扩展 |
| G2 | 3 | G2a + Skill ISA 类型偏好(= 7C-1 G2,仅级别常数重构) |
| G3 | 4 | G2 + 规范输入类型字面量 trie(**仅名称规范化**,不声称语义兼容) |

G3 冻结类型表:ISA 签名 ∪ 语料输入类型(Str, List[Any])∪ 规范标量,共 20 个字面量。**诚实边界**:xlam-1107 类真实不兼容(`@page: Int` vs FETCH 要 Str)不会被 G3"修好"——trie 只强制规范拼写,不假装类型正确。

## 3. 数据集

- **dev-30**(13 tbs + 17 xlam):7C-1 全部调参发生于此;G0/G1/G2(s43)结果从 7C-1 导入并按新指标重算,G2a/G3 补跑
- **确认集 150**:964 池按 来源×参考深度 分层(random.seed=20261009)
- **已知局限(明示)**:964 池全部是 xlam——全部 30 个 tbs 任务恰好都在 dev-30 里。确认集无法覆盖 tbs 分布,tbs 证据仅存在于开发集。错误类型分层不可先验计算,以来源近似(来源携带了 dev-30 的错误画像)
- all-994:确认集无严重退化后才跑,且只作描述性报告,不称 untouched

## 4. 结果(待运行完成填充)

{{CONFIRMATION_TABLES}}

{{DEV30_TABLE}}

{{PAIRED_STATS}}

## 5. 论文裁决标准(Task 7,预冻结)

Typed SSA 升级为主要贡献,当且仅当:
1. G2/G2b 在确认集上优于 G1(配对,McNemar)
2. E2E 语义不退化(Skill F1 / EXEC_ACTION Recall / Param E2E)
3. 最差种子稳定受益,强种子无明显损害
4. 相对 G1 的增益不能完全被类型名称规范化(G3 的 str→Str)解释
5. 延迟开销与收益有意义权衡

否则:G1 = 可靠工程增强,Typed SSA = 探索性算法。

## 6. 复现

```bash
/opt/conda/bin/python tests/test_online_constrained_decoder.py      # 24 tests
/opt/conda/bin/python tests/test_token_mask_real_tokenizer.py       # 5 tests (server)
/opt/conda/bin/python scripts/sample_confirmation_set.py
bash scripts/run_7c2_driver.sh                                      # 全量驱动
/opt/conda/bin/python scripts/analyze_online_large.py               # 聚合 + 配对统计
```
