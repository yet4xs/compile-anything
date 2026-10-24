"""AES pilot 单次执行器(协议冻结于 aab80db5,本脚本执行后不改)。

方法 A/B/C 各 6 变体;C 两段(抽契约→按自抽契约生成计划→机械编译
自检→通过提交/否则 no_valid_plan)。oracle = 人工契约检查器。
用量逐调用登记 raw_api_usage_ledger_v5 格式。

Run: python scripts/aes_pilot_run.py   (exactly once)
"""
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.contract.aes_init_checker import (
    AESInitChecker, Scenario, KEY_REGS, CTRL, STATUS)

sys.path.insert(0, str(REPO / "experiments/serverless_eval_v1"))
from glm_client import complete

PKT = json.loads((REPO / "results/aes_pilot_20261024/tasks.json")
                 .read_text(encoding="utf-8"))
OUTD = REPO / "results/aes_pilot_20261024"

# ── 冻结的机械映射: C 自抽契约条目 → 自检检查器参数(执行前固定) ──
# 只允许从条目的 kind 与目标寄存器做存在性检测;不做语义解释。
C_ENUM = {"INIT_IDLE": "initial_idle_gate",
          "CTRL_BEFORE_KEY": "ctrl_before_key",
          "RECHECK_AFTER_CTRL": "recheck_after_ctrl",
          "CTRL_TWO_WRITE": "ctrl_two_write",
          "KEY_ALL_REGS": "key_all_registers",
          "READ_RESETS_PHASE": "read_resets_phase"}


def docs_block():
    parts = []
    for k, meta in PKT["documents"].items():
        txt = Path(meta["path"]).read_text(encoding="utf-8", errors="replace")
        parts.append(f"===== 文件 {k} ({meta['sha256'][:12]}) =====\n{txt}")
    return "\n\n".join(parts)


def task_stmt(v):
    kl = {"aes128": "128", "aes192": "192", "aes256": "256"}[v["key_len"]]
    return f"""任务: 你是嵌入式软件工程师。AES 单元处于复位后状态。请给出从当前状态到
"初始密钥全部写入完成"的完整操作计划(只到密钥写完,不含 IV 与数据)。
选定 key 长度 {kl}(KEY_LEN 字段)。场景: {v['given']}
输出严格 JSON: {{"plan": [{{"op":"read","reg":"STATUS","field":"IDLE"}} 或
{{"op":"write","reg":"...","value":"0x..."}} ...]}}。
只输出 JSON,无其他文字。寄存器名必须与文档一致。"""


def call(model_msgs, tag):
    payload = {"model": "glm-5.3-flash", "messages": model_msgs,
               "temperature": 0, "max_tokens": 8192,
               "thinking": {"type": "enabled"}, "reasoning_effort": "low",
               "stream": False}
    r = complete(payload, timeout=180)
    u = (r.get("response") or {}).get("usage", {})
    rec = {"tag": tag, "http_status": r["http_status"],
           "seconds": round(r["seconds"], 2), "usage": u,
           "error": (r["error"] or None)}
    return r, rec


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def plan_steps(j):
    if not isinstance(j, dict) or not isinstance(j.get("plan"), list):
        return None
    steps = []
    for st in j["plan"]:
        if not isinstance(st, dict) or "op" not in st:
            return None
        steps.append({"op": st.get("op"), "reg": st.get("reg", ""),
                      "field": st.get("field"),
                      "value": st.get("value")})
    return steps


def oracle(v, steps):
    sc = Scenario(**{k: w for k, w in v["scenario"].items()
                     if k in ("initial_idle", "ctrl_commit_busy_polls",
                              "never_idle_after_ctrl",
                              "initial_busy_polls")})
    return AESInitChecker(sc, max_steps=300).check(steps)


def compile_own_checker(entries):
    """v2 机械映射:模型自报枚举 id 的集合存在性,无语义解释。"""
    ids = {e.get("id") for e in entries if isinstance(e, dict)}
    return {f: (eid in ids) for eid, f in C_ENUM.items()}


def own_check(flags, steps):
    """用 C 自抽开关做结构检查(独立于 oracle)。"""
    idle_seen = ctrl_done = recheck = False
    ctrl_writes = []
    keyregs = set()
    polls = 0
    for i, st in enumerate(steps):
        if st["op"] == "read" and st["reg"] == STATUS:
            polls += 1
            idle_seen = True        # 保守:读过即算(结构层不解释返回值)
            if ctrl_done:
                recheck = True
        elif st["op"] == "write" and st["reg"] == CTRL:
            ctrl_writes.append((i, st.get("value")))
        elif st["op"] == "write" and st["reg"] in KEY_REGS:
            if flags.get("ctrl_before_key") and not ctrl_done:
                return False, "key before ctrl"
            if flags.get("recheck_after_ctrl") and not recheck:
                return False, "no recheck after ctrl"
            keyregs.add(st["reg"])
        if len(ctrl_writes) >= 2 and \
                ctrl_writes[-1][1] == ctrl_writes[-2][1]:
            ctrl_done = True
    if flags.get("initial_idle_gate") and not idle_seen and steps:
        return False, "no status observation"
    if flags.get("ctrl_two_write") and not ctrl_done:
        return False, "ctrl not committed"
    if flags.get("key_all_registers") and len(keyregs) != 16:
        return False, f"key regs {len(keyregs)}/16"
    return True, "pass"


def main():
    doc = docs_block()
    results, usage_recs = [], []
    t0 = time.time()
    for v in PKT["variants"]:
        # ── A 直接生成 ──
        rA, uA = call([{"role": "system", "content":
                        "你是硬件软件接口专家,按文档输出操作计划。"},
                       {"role": "user", "content": doc + "\n\n" +
                        task_stmt(v)}], f"A/{v['id']}")
        usage_recs.append(uA)
        planA = plan_steps(extract_json(
            (rA.get("response") or {}).get("choices", [{}])[0]
            .get("message", {}).get("content", "")))
        vdA = {"task": v["id"], "method": "A",
               "plan_ok": planA is not None}
        if planA is not None:
            o = oracle(v, planA)
            vdA.update({"judgment": o.judgment, "contract": o.contract})
        else:
            vdA.update({"judgment": "UNPARSEABLE", "contract": None})
        results.append(vdA)

        # ── B 检索+出处 ──
        rB, uB = call([{"role": "system", "content":
                        "你是硬件软件接口专家。计划外,每步必须给出处 "
                        "{'file','line','quote'}。输出 JSON: "
                        "{'plan':[...], 'sources':[{'step':i,...}]}"},
                       {"role": "user", "content": doc + "\n\n" +
                        task_stmt(v)}], f"B/{v['id']}")
        usage_recs.append(uB)
        jB = extract_json((rB.get("response") or {})
                          .get("choices", [{}])[0]
                          .get("message", {}).get("content", ""))
        planB = plan_steps(jB)
        vdB = {"task": v["id"], "method": "B",
               "plan_ok": planB is not None}
        if planB is not None:
            o = oracle(v, planB)
            vdB.update({"judgment": o.judgment, "contract": o.contract})
        else:
            vdB.update({"judgment": "UNPARSEABLE", "contract": None})
        # 出处回验
        cites = []
        for s in (jB or {}).get("sources", []) if isinstance(jB, dict) else []:
            f = s.get("file")
            if f in PKT["documents"]:
                lines = Path(PKT["documents"][f]["path"]).read_text(
                    encoding="utf-8", errors="replace").splitlines()
                ln = s.get("line")
                q = str(s.get("quote", ""))[:40]
                ok = isinstance(ln, int) and 1 <= ln <= len(lines) and \
                    q[:20] in lines[ln - 1]
                cites.append(ok)
        vdB["cites_total"] = len(cites)
        vdB["cites_valid"] = sum(cites)
        results.append(vdB)

        # ── C 两段 ──
        rC1, uC1 = call([{"role": "system", "content":
                          "从文档抽取机器可检查的契约条目。输出 JSON 数组,"
                          "每条 {id,kind,target,quote,file,line};kind 取 "
                          "precondition/side_effect/ordering/protocol/"
                          "completion;id 必须取自枚举 {INIT_IDLE, "
                          "CTRL_BEFORE_KEY, RECHECK_AFTER_CTRL, "
                          "CTRL_TWO_WRITE, KEY_ALL_REGS, "
                          "READ_RESETS_PHASE} 之一。没有依据的不要写。"},
                         {"role": "user", "content": doc}],
                        f"C1/{v['id']}")
        usage_recs.append(uC1)
        entries = extract_json((rC1.get("response") or {})
                               .get("choices", [{}])[0]
                               .get("message", {}).get("content", ""))
        flags = compile_own_checker(entries if isinstance(entries, list)
                                    else [])
        rC2, uC2 = call([{"role": "system", "content":
                          "你是硬件软件接口专家。严格遵守你拥有的契约条目,"
                          "输出操作计划 JSON {'plan':[...]}。"},
                         {"role": "user", "content": doc + "\n\n" +
                          "抽取到的契约条目:\n" +
                          json.dumps(entries, ensure_ascii=False)[:4000] +
                          "\n\n" + task_stmt(v)}], f"C2/{v['id']}")
        usage_recs.append(uC2)
        planC = plan_steps(extract_json(
            (rC2.get("response") or {}).get("choices", [{}])[0]
            .get("message", {}).get("content", "")))
        vdC = {"task": v["id"], "method": "C",
               "flags": flags,
               "entries_n": len(entries) if isinstance(entries, list) else 0}
        if planC is None:
            vdC.update({"judgment": "UNPARSEABLE", "submitted": False})
        else:
            ok, why = own_check(flags, planC)
            vdC["self_check"] = why
            if not ok:
                vdC.update({"judgment": "NO_VALID_PLAN",
                            "submitted": False})
            else:
                o = oracle(v, planC)
                vdC.update({"judgment": o.judgment,
                            "contract": o.contract, "submitted": True})
        results.append(vdC)
        print(v["id"], "done", flush=True)

    # ledger
    tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage_recs:
        for k in tot:
            tot[k] += (u.get("usage") or {}).get(k, 0)
    ledger = {
        "schema": "raw_api_usage_ledger_v5",
        "prior_ledger_path":
            "results/usage_ledger_v5/through_shadow24_20261024/usage.json",
        "batch": {"planned": 24, "attempts": len(usage_recs),
                  "http200": sum(1 for u in usage_recs
                                 if u["http_status"] == 200),
                  "reported_tokens": tot},
        "calls": usage_recs,
    }
    (REPO / "results/usage_ledger_v5/aes_pilot_20261024").mkdir(
        parents=True, exist_ok=True)
    (REPO / "results/usage_ledger_v5/aes_pilot_20261024/usage.json") \
        .write_text(json.dumps(ledger, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    (OUTD / "results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(tot), "wall_min", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
