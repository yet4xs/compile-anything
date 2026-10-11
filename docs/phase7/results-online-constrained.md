# Phase 7C-1: Online Constraint-Guided TaskIR Decoding

**日期**: 2026-10-09
**基线**: Phase 7C-0.1(v2 事后装配)
**范围**: 仅 s43(最弱种子)n=30 机制验证;不训练新模型,不改 Composer 权重
**产物**: `src/compiler/online_constrained.py` 相关交付见文末

---

## 1. 实现概要

`src/compiler/online_constrained_decoder.py`:

- **TaskIRGenFSM** — 字符级增量文法自动机,追踪生成前缀;feed(ch) 返回 False 即文法违规
- **TokenMask** — 词表级掩码:token 合法当且仅当其**全部字符**逐个过 FSM 模拟无错(单 token 可跨状态,如 `"])\n`);自由区(注释/字符串/列表)内含终止符的 token 走模拟判定——终止符是合法出口
- **ConstrainedLogitsProcessor** — HuggingFace LogitsProcessor,每步:喂入新 token → 计算合法集 → 掩码;记录 banned 数、forced 步(模型 top-1 被禁)、deadlock 及回退、强制改名、类型偏好
- **EOS 门控** — 仅在 ≥1 节点且 return 引用完成后放行 EOS(return 后无需换行,`return %c4` + EOS 即完整程序)

三级:
- **G1 语法**: 头部/注释/inputs/节点行/参数/字面量/注解/return 全文法;注释 800 字符上限(仅掩码层强制换行,feed 保持宽容 → FSM 对合法语料无损)
- **G2 = G1 + 动态 SSA**: 操作数/return 引用限制在**已定义 id trie**(defined-before-use);重复定义通过禁止终止符强制模型改名(非事后替换);操作数计数封顶 spec.max_inputs;类型偏好(类型兼容 ∪ 未知类型,空集回退全部,全部记录)

保守策略(明示):注解体内引用只做文法检查;参数值类型不掩码;类型推断失败视为兼容。

## 2. Task 1: s43 十二个失败样本的解剖

| # | 样本 | 错误类 | 根因 | 可否在线预防 |
|---|------|--------|------|--------------|
| 1-3 | tbs-640/641/975 | parse: no nodes | 敌对 meta 指令 → 模型在 `; task:` 注释内提前 EOS(~350 token,远未到 512) | ✓ G1 EOS 门控 |
| 4 | xlam-978 | parse: node line | 裸行 `number: "55"`(非节点行) | ✓ G1 行首字符集 |
| 5-9 | xlam-203/205/344/1090/1107 | STRUCT bad input name | `inputs: is_id: Str` 忘写 `@` → 级联 UNDEFINED_REF | ✓ G1 强制 `@` |
| 10-11 | xlam-198/283 | ARITY | SEARCH 3 操作数 / CLASSIFY 2 操作数 | ✓ G2 arity 封顶 |
| 12 | xlam-899 | TYPE_MISMATCH | 输入类型声明错误(小写 `str` 等) | ✗ 当前不防(类型字面自由) |

**关键观察**: 8 个 validator 失败中 7 个 opseq 与参考**完全一致**(prefix_match = ref_len)——错误几乎全部发生在"把已决定的操作序列化为 TaskIR 文本"这一层,而非"决定生成什么操作"。这直接支持生成时文法约束的路线。

## 3. 调试过程中修掉的实现 bug(记录供复现)

1. 自由区把含 `\n`/`"` 的 token 全禁 → 注释内换行被禁,模型被逼成乱码(终止符必须是合法出口)
2. `allowed_args` 漏空格 → ` fields` 等参数键 token 被禁,每次参数被强制,连锁脱轨
3. **掩码缓存键缺 resume 栈** → 参数值字符串的禁集污染列表内字符串缓存,`"])\n` 被误禁 → 列表复读直到 token 耗尽
4. EOS 在 `return %c4`(无换行)时被禁 → 完整程序被逼续写
5. 关键词候选含首字母不一致(`return` 写成 `eturn`)→ 全部关键词匹配失败
6. `inputs:` 后空格处理、op 与 `(` 之间空格、操作数计数被 value 位置引用污染

## 4. 主结果(s43,n=30,G0/G1/G2 同提示同贪心,逐样本判定官方 parser+validator)

| 臂 | Parse% | Valid% | OpSeq% | 参数键精度 | 生成token均值 | P50/P95 延迟(s) |
|----|--------|--------|--------|-----------|--------------|-----------------|
| G0 基线 | 86.7 | 60.0 | 73.3 | 0.488 | 244.5 | 10.3 / 22.6 |
| G1 语法 | **100.0** | **80.0** | **86.7** | 0.529 | 238.0 | 19.7 / 30.1 |
| G2 语法+SSA | **100.0** | **83.3** | **86.7** | **0.577** | 238.3 | 17.7 / 28.4 |

对照:事后装配 D2 在同一批样本上为 63.3%(G2 = +20pp);强种子 s42/s44 无约束基线 90%。

逐样本转移(两个方向都报告):
- **G1: FIXED=6,BROKEN=0**(4 个 parse 失败全部修复 + xlam-344/1090 两个 invalid 修复;18 个 G0-valid 全部保持)
- **G2: FIXED=7,BROKEN=0**(+xlam-283 CLASSIFY arity)
- 约束开销:G1 forced 共 8 步、G2 13 步(平均每样本 <0.5 步);G2 死锁 11 次(回退放行,不截断)、fsm_violation 1 次、类型偏好应用 45 次

G2 仍失败的 5 个:
- 4× TYPE_MISMATCH:模型把输入类型写成小写 `str`/`int`(`@first_name: str`)——输入类型字面量在 FSM 中是自由文本,当前不防。**这是明确的下一杠杆:G2 加规范类型 trie(Str/Int/Float/Bool/List[...])可再收 4 个**
- 1× UNDEFINED_REF(xlam-227):死锁回退放行(保守策略,已记录)

## 5. Go/No-Go 判定(Task 5)

**GO**:
1. G1 修复了全部 4 个 Parse Fail(Task 5 第一判据)✓
2. G1/G2 对原本合法程序零破坏(BROKEN=0)✓
3. 语义正确性未受损反而提升:opseq +13.4pp,参数键精度 0.488→0.529→0.577 单调上升(排除"只提静态 Valid、降语义"的担忧)✓
4. G2 相对 G1 有独立增益(+3.3pp valid,+0.048 参数精度,类型偏好生效 45 次)——SSA 层约束不是纯工程模块 ✓
5. 延迟成本:批=1 贪心下 P50 约 1.7-1.9×(掩码缓存 amortize 后);规模化时的批内掩码向量化是工程优化空间

**对种子敏感性问题的意义**:s43(最弱种子)60% → 83.3%,逼近强种子无约束水平(90%)。机制上:种子差距的主导成分(序列化层错误)可被编译器文法+SSA 约束在线吸收,而操作选择层(opseq)本来就稳定(7/8 与参考一致)。

## 5b. 与事后装配(7C-0.1)的机制分界

| | 事后装配 D2 | 在线 G1/G2 |
|---|-----------|-----------|
| 介入时机 | 完整生成后 | 每 token |
| parse 失败 | 不可修(无文本可修) | 100% 预防 |
| 未定义引用 | 替换为兼容定义(改程序) | 解码时禁止,模型自选合法替代 |
| s43 valid | 63.3% | 83.3% |
| 语义风险 | 替换可能换错 | forced 步均值 <0.5,参数精度反升 |

## 6. 论文新颖性定位(Task 7,诚实)

- 文法约束解码本身不是原创(PICARD 的增量解析器掩码、SynCode 的 LALR/trie 掩码都是先例);G1 属于把标准技术接到 TaskIR 文法,工程贡献
- 候选主贡献是 **Typed SSA + Capability-aware constraint propagation**:G2 的 defined-before-use trie、arity 封顶、Skill ISA 类型偏好传播是神经编译器特有的约束(操作语义层,不是纯文法),且本轮显示相对 G1 的独立增益(valid +3.3pp、参数精度 +0.048、类型偏好 45 次)
- 本轮 n=30 单种子是机制验证;升级为主贡献需 994×3 种子全量对照 + G2-vs-G1 配对显著性,以及(若加类型 trie)TYPE_MISMATCH 类错误的进一步回收

## 7. 复现

```bash
/opt/conda/bin/python tests/test_online_constrained_decoder.py   # 19 项单测
/opt/conda/bin/python scripts/eval_online_constrained.py --n 30 --seed s43
```

交付物:
- `src/compiler/online_constrained_decoder.py`
- `tests/test_online_constrained_decoder.py`(语料无损 G1/G2、四类拒绝、掩码、处理器记账、注释换行/上限回归)
- `scripts/eval_online_constrained.py`
- `results/phase7/online_constrained.json`(含逐样本文本、判定、约束统计、转移表)
