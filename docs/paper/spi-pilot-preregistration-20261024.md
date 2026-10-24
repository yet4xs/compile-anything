# SPI_HOST 迁移 pilot 冻结协议 v6(调用前)

承 AES pilot 全套纪律。目的: 检验文档→契约→检查→修复管线是否
非 AES 特化。oracle = spi_contract_table_v1(9/9 机验,含 S8 冲突
保留条目)编译的 SPISubmitChecker。

## 任务族(4 变体,同族异参)

任务: 提交 3 个 TX segment 并以 polling 证据宣告全部发送完成。
变体仅场景参数不同(对模型可见): 就绪窗口 / 完成窗口 / 异常。

| # | ready 窗口(各命令后忙轮询数) | txempty_after | 特点 |
|---|------------------------------|---------------|------|
| U1 | (0,0,0) | 1 | 一切立即 |
| U2 | (0,1,2) | 3 | 递增忙窗口 |
| U3 | (2,3,4) | 5 | 深窗口 |
| U4 | (0,0,0) | 1 | never-ready(第 1 命令后永不就绪;正确=诚实停止/预算耗尽) |

## 信息权限: SPI_HOST 4 文件全文(pg/reg/theory/hjson,固定 commit);
中断分支整体排除(含 S8 冲突,任务陈述明示 polling-only)。

## 方法与预算: A(1)/B(1)/C(2: 抽取枚举
{CMD_READY_GATE, QUEUE_ONE_MORE, WAIT_AFTER_FULL, TXEMPTY_DONE,
CONFLICT_RETAINED} + 计划 + 结构自检) + v5 同款单轮修复(仅可修复
拒绝)。调用 ≤ 4×4 + 修复 ≤4。模型 glm-5.3-flash,thinking low,
temp 0,8192。逐调用账本 aes_pilot_v5 同目录新批 spi_v6。

## 评分(冻结): SUCCESS / 违反(S1/S3/S7/S2x) / INSUFFICIENT /
BUDGET(U4 正确={BUDGET, INSUFFICIENT} 无违反) / 拒绝。
结构自检(C): CMD_READY_GATE→cmd1 需先读 STATUS;WAIT_AFTER_FULL
→第 3 cmd 前需新读;TXEMPTY_DONE→终点前需读到 TXEMPTY=1 语义
(结构层以"最后动作序列含对 STATUS 的读"近似,深度由修复轮以
场景参数编译反馈)。B 引文回验同 AES 口径。
