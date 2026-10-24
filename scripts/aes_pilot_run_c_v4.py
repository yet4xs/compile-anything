"""v4 深度感知自检: 对 v3 已保存的 C2 计划重打分(零 API 调用)。

发现#4 的修复: 结构检查只验证"查过",不验证"查够"。v4 把任务陈述
中给定的场景窗口(初始忙/配置后忙/无限忙)机械编译进检查:

- initial_busy_polls=k  → 首笔 CTRL 写前需 ≥k+1 次 STATUS 读
  (k 次忙观测 + 1 次确认 idle=1;最后观测必须为就绪)
- ctrl_commit_busy_polls=k → CTRL 提交后、首笔 KEY 写前需 ≥k+1 次
  STATUS 读(同上;T2 教训:只查 k 次即继续=带着未就绪观测前进)
- never_idle_after_ctrl → 任何含 KEY 写的计划拒绝(无法在有限观测
  内建立就绪),有限轮询后停止 = 正确拒绝

输出: v4 判定表(对 v3 计划的三分: submit-pass / decline(可修复,
  窗口有限) / decline(不可行,无限窗口))。
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from aes_pilot_run import PKT, extract_json, plan_steps, oracle
from aes_pilot_run_c_v3 import content_of  # noqa: F401  (reused shape)


def parse_own_plan(raw_text):
    return plan_steps(extract_json(raw_text))


def depth_check(scenario, steps):
    """v4 深度检查。返回 (ok, declinable, reason)。
    declinable=True 表示窗口有限、计划可通过补轮询修复。"""
    init_busy = scenario.get("initial_busy_polls", 0) or (
        0 if scenario.get("initial_idle", True) else 10**9)
    ctrl_busy = scenario.get("ctrl_commit_busy_polls", 0)
    never = scenario.get("never_idle_after_ctrl", False)

    reads_before_ctrl = 0
    ctrl_committed_at = None
    first_key_at = None
    reads_after_ctrl_before_key = 0
    shadow_vals = []
    for i, st in enumerate(steps):
        if st["op"] == "read" and st["reg"] == "STATUS":
            if ctrl_committed_at is None:
                reads_before_ctrl += 1
            elif first_key_at is None:
                reads_after_ctrl_before_key += 1
        elif st["op"] == "write" and st["reg"] == "CTRL_SHADOWED":
            shadow_vals.append(st.get("value"))
            if len(shadow_vals) >= 2 and \
                    shadow_vals[-1] == shadow_vals[-2] and \
                    ctrl_committed_at is None:
                ctrl_committed_at = i
                shadow_vals = shadow_vals[-2:]
        elif st["op"] == "write" and st["reg"].startswith("KEY_") and \
                first_key_at is None:
            first_key_at = i
    # 未提交/未写 key 的计划交给结构检查路径(此处只做深度)
    if ctrl_committed_at is None or first_key_at is None:
        return True, False, "n/a (structure handles)"
    if never:
        return False, False, "unbounded busy window: readiness cannot " \
                             "be established with finite observations"
    if init_busy and reads_before_ctrl < init_busy + 1:
        return False, True, (f"initial window {init_busy}: only "
                             f"{reads_before_ctrl} pre-ctrl reads "
                             f"(need {init_busy + 1}: busy x{init_busy} "
                             f"+ 1 confirming idle)")
    if reads_after_ctrl_before_key < ctrl_busy + 1:
        return False, True, (f"post-ctrl window {ctrl_busy}: only "
                             f"{reads_after_ctrl_before_key} reads "
                             f"before first key write "
                             f"(need {ctrl_busy + 1})")
    return True, False, "pass"


def main():
    raws = json.loads(
        (REPO / "results/aes_pilot_20261024/raw_outputs_c_v3.json")
        .read_text(encoding="utf-8"))
    c2 = {r["tag"].split("/")[1]: r["content"] for r in raws
          if r["tag"].startswith("C2v3/")}
    out = []
    for v in PKT["variants"]:
        plan = parse_own_plan(c2.get(v["id"], ""))
        rec = {"task": v["id"]}
        if plan is None:
            rec.update({"v4": "UNPARSEABLE"})
        else:
            ok, fixable, why = depth_check(v["scenario"], plan)
            rec.update({"v4_depth": "pass" if ok else "decline",
                        "fixable": fixable, "reason": why})
            if ok:
                o = oracle(v, plan)
                rec["oracle_if_submitted"] = o.judgment
            else:
                rec["oracle_if_submitted"] = "NOT_SUBMITTED(v4)"
        out.append(rec)
        print(v["id"], json.dumps(rec, ensure_ascii=False)[:130])
    (REPO / "results/aes_pilot_20261024/results_c_v4_rescore.json") \
        .write_text(json.dumps(out, indent=1, ensure_ascii=False),
                    encoding="utf-8")


if __name__ == "__main__":
    main()
