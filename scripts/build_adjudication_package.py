"""Phase 9B-2.5 T5: adjudication package generator (materials ONLY —
no labels are filled; per preregistration §4 humans adjudicate).

Items are COINCIDENTAL_VALUE / LOW_ENTROPY_MATCH params on tasks whose
ONLY strict blockers are adjudicable classes. Each item shows the
compile input, the consuming call, every candidate source with entity
context, and the tasks it unlocks. Sorted by unlock impact.

Outputs:
  adjudication/v2/items.jsonl       full context (machine-readable)
  adjudication/v2/annotator_A.csv   blank judgment table
  adjudication/v2/annotator_B.csv   blank judgment table (identical)
  adjudication/v2/arbitration.csv   disagreements only (fill later)
  adjudication/v2/README.md         instructions
  scripts/compute_kappa.py is written separately

Run: python scripts/build_adjudication_package.py (plain python; reads
tau3-bench-src JSON directly)
"""

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAU2 = Path("D:/ccfa-2026/tau3-bench-src/data/tau2/domains")
OUT = REPO / "adjudication" / "v2"
ADJ = {"COINCIDENTAL_VALUE", "LOW_ENTROPY_MATCH"}
STRICT_OK = {"LITERAL", "DUAL_SOURCE", "runtime_ref", "runtime_ref_multi",
             "SAME_ENTITY_ECHO"}


def entity_prefix(path):
    m = re.match(r"^(.*\[\d+\])", path)
    return m.group(1) if m else ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = json.loads((REPO / "results" / "phase9" /
                    "tau3_gold_compiled_v2.json").read_text(encoding="utf-8"))

    # unlockable tasks + their adjudicable items
    task_items = {}
    for t in g["tasks"]:
        if t["strict_v2"] or not t["replay_ok"]:
            continue
        if "EXPECTED_ERROR_STEP" in t["flags"] or \
                "POLICY_DEPENDENT" in t["flags"]:
            continue
        if set(t["param_class_counts"]) - ADJ - STRICT_OK:
            continue
        its = []
        for node in t["program"]:
            for p, a in node["arguments"].items():
                if a["class"] in ADJ:
                    its.append((node["step"], p, a))
        if its:
            task_items[t["task_id"]] = (t, its)

    # per-task compile inputs from official data
    inputs = {}
    for dom in ("retail", "airline"):
        for raw in json.loads((TAU2 / dom / "tasks.json")
                              .read_text(encoding="utf-8")):
            inputs[f"tau3_{dom}_{raw['id']}"] = raw.get("user_scenario") or {}

    import sys
    sys.path.insert(0, str(REPO))
    from src.eval.compile_input_v2 import render_compile_input

    rows = []
    item_id = 0
    for tid, (t, its) in sorted(task_items.items()):
        text = render_compile_input(inputs.get(tid))
        for (step, param, a) in its:
            item_id += 1
            cands = []
            srcs = a.get("candidates") or []
            if not srcs and a.get("elements"):
                srcs = [e for e in a["elements"]
                        if isinstance(e, dict) and "path" in e]
            for c in srcs[:8]:
                ps = c.get("producer_step")
                ptool = t["program"][ps]["tool"] if ps is not None \
                    and ps < len(t["program"]) else "?"
                cands.append({
                    "n": len(cands) + 1,
                    "producer_step": ps, "producer_tool": ptool,
                    "path": c.get("path", ""),
                    "entity": entity_prefix(c.get("path", "")) or "(root)",
                })
            rows.append({
                "item_id": f"V2-{item_id:03d}",
                "task_id": tid,
                "unlocks_task": tid,
                "step": step,
                "consumer": f"{t['program'][step]['tool']}.{param}",
                "class": a["class"],
                "value": str(a.get("value"))[:60],
                "candidates": cands,
                "compile_input": text[:1500],
            })

    # sort by ... all items here unlock exactly their own task; order by
    # task with fewest items first (cheapest unlocks first)
    per_task = {}
    for r in rows:
        per_task.setdefault(r["task_id"], []).append(r)
    ordered = []
    for tid in sorted(per_task, key=lambda k: len(per_task[k])):
        ordered.extend(per_task[tid])

    with open(OUT / "items.jsonl", "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    header = ["item_id", "task_id", "step", "consumer", "class", "value",
              "n_candidates", "judgment", "notes"]
    for who in ("A", "B"):
        with open(OUT / f"annotator_{who}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in ordered:
                w.writerow([r["item_id"], r["task_id"], r["step"],
                            r["consumer"], r["class"], r["value"],
                            len(r["candidates"]), "", ""])
    with open(OUT / "arbitration.csv", "w", newline="",
              encoding="utf-8") as f:
        csv.writer(f).writerow(header + ["A", "B", "final", "reason"])

    (OUT / "README.md").write_text(f"""# 裁决包 v2 — 使用说明

共 **{len(ordered)} 条**待裁决项,覆盖 **{len(per_task)} 个可解锁任务**
(解锁后 strict 池 {7 + len(per_task)} 个,超过 ≥15 阈值)。

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
""", encoding="utf-8")

    print(f"items: {len(ordered)} | tasks: {len(per_task)}")
    print("cheapest tasks first:",
          [(tid, len(v)) for tid, v in
           sorted(per_task.items(), key=lambda kv: len(kv[1]))[:8]])


if __name__ == "__main__":
    main()
