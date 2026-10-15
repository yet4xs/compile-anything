# 裁决包 v2 — 使用说明

共 **61 条**待裁决项,覆盖 **27 个可解锁任务**
(解锁后 strict 池 34 个,超过 ≥15 阈值)。

## 每条要判断什么

消费方参数的值与多个候选来源(或低熵值)值相同。判断:**哪个候选是该参数
真实的语义来源**?

- `ACCEPT_<n>`: 第 n 个候选是真实来源(见 items.jsonl 的 candidates,
  含生产者工具、路径、实体上下文)
- `NONE`: 纯值巧合/泛值(如 economy、USA),不应建立绑定 → 该任务保持
  非 strict
- `UNSURE`: 拿不准 → 进仲裁表

## 判据参考(不是规则,由你判断)

- 候选路径的实体(如 orders[2] / payment_history[0])与消费语义同属
  一个业务实体 → 通常是真来源(回显)
- 值是泛词(economy/country=USA/短代码)且实体不同 → 通常 NONE
- compile_input 里用户给了等价描述但值本身不在文本 → 仍需选工具侧来源

## 流程

1. A、B 各自独立填 annotator_A.csv / annotator_B.csv(不许互相看)
2. 运行 `python scripts/compute_kappa.py` 计算 κ;κ ≥ 0.8 通过
3. 分歧条目进 arbitration.csv 由第三人裁决
4. 完成后把两个 csv + arbitration.csv 交回,zcode 按裁决结果重建 gold
   (此时代码只执行人的决定)

每条预计 20–40 秒(上下文都在 items.jsonl),总工作量约 30–45 分钟/人。
