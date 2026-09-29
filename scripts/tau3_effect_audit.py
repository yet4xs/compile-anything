"""τ³-bench effect semantics audit (Task 14) — evidence for the Phase 3
effect proposal against a real stateful benchmark. Analysis only; no IR
changes.

    python scripts/tau3_effect_audit.py   -> docs/tau3-effect-audit.md
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.adapters import load_tau3                    # noqa: E402

DOM = ROOT / "data" / "external_benchmarks" / "tau3_bench" / "domains"

# effect classification by action-name semantics (deterministic rules)
READ_ONLY = re.compile(r"^(get|find|search|query|list|retrieve|lookup|"
                       r"check|show|view|read|calculate|compare)", re.I)
IRREVERSIBLE = re.compile(r"(cancel|refund|purchase|pay|book|reserv|send|"
                          r"delete|remove|grant|revoke|reboot|reset|"
                          r"refuel|unlock|purchase)", re.I)
REVERSIBLE = re.compile(r"(toggle|enable|disable|set|update|change|modify|"
                        r"edit|switch|turn_on|turn_off|add|create|insert)",
                        re.I)
VERIFY = re.compile(r"(verify|confirm|validate|check)", re.I)


def classify_action(name: str) -> str:
    n = name or ""
    if READ_ONLY.match(n):
        return "read_only"
    if IRREVERSIBLE.search(n):
        return "irreversible_world"
    if REVERSIBLE.search(n):
        return "reversible_state"
    return "other"


def main() -> int:
    samples = load_tau3()
    per_domain: dict = {}
    action_class = Counter()
    tasks_multi_action = 0
    tasks_with_verify_assert = 0
    action_name_class: dict = {}

    for s in samples:
        dom = s.metadata["domain"]
        d = per_domain.setdefault(dom, {
            "tasks": 0, "with_expected_actions": 0, "multi_action": 0,
            "nl_assertions": 0, "classes": Counter(), "examples": {}})
        d["tasks"] += 1
        acts = s.reference_actions or []
        if acts:
            d["with_expected_actions"] += 1
        if len(acts) > 1:
            d["multi_action"] += 1
            tasks_multi_action += 1
        if s.metadata.get("reward_basis") and "NL_ASSERTION" in (
                s.metadata["reward_basis"] or []):
            d["nl_assertions"] += 1
            tasks_with_verify_assert += 1
        for a in acts:
            name = a.get("name") or a.get("action_id") or ""
            cls = classify_action(name)
            action_class[cls] += 1
            d["classes"][cls] += 1
            action_name_class.setdefault(cls, Counter())[name] += 1

    total_actions = sum(action_class.values())
    lines = ["# τ³-bench Effect Semantics Audit", "",
             f"- tasks: **{len(samples)}** across 4 domains; "
             f"**{tasks_multi_action}** expect multi-action sequences "
             f"(transactions), **{tasks_with_verify_assert}** carry NL "
             f"assertions (verification criteria)",
             f"- classified ground-truth actions: **{total_actions}**", "",
             "## Action effect classification", "",
             "| class | count | share |", "|---|---:|---:|"]
    for cls, n in action_class.most_common():
        lines.append(f"| {cls} | {n} | "
                     f"{100*n/total_actions:.1f}% |" if total_actions else
                     f"| {cls} | {n} | - |")
    lines += ["", "Top actions per class:", ""]
    for cls in action_class:
        tops = ", ".join(f"`{k}`({v})" for k, v in
                         action_name_class[cls].most_common(5))
        lines.append(f"- **{cls}**: {tops}")
    lines += ["", "## Per domain", "",
              "| domain | tasks | with actions | multi-action | NL-assert |"
              " read-only | reversible | irreversible |", "|---|---:|---:|"
              "---:|---:|---:|---:|---:|"]
    for dom, d in per_domain.items():
        c = d["classes"]
        lines.append(f"| {dom} | {d['tasks']} | {d['with_expected_actions']} "
                     f"| {d['multi_action']} | {d['nl_assertions']} "
                     f"| {c.get('read_only', 0)} "
                     f"| {c.get('reversible_state', 0)} "
                     f"| {c.get('irreversible_world', 0)} |")

    # expressibility check vs our effect proposal
    lines += ["", "## Expressibility vs the Phase 3 effect proposal", ""]
    ro = action_class.get("read_only", 0)
    rv = action_class.get("reversible_state", 0)
    ir = action_class.get("irreversible_world", 0)
    lines += [
        "- read-only actions: pure skills — expressible today "
        f"({ro}/{total_actions} = {100*ro/total_actions:.1f}%)",
        "- reversible state changes: `state` effect class + rollback "
        "semantics — expressible per proposal "
        f"({rv}/{total_actions} = {100*rv/total_actions:.1f}%)",
        "- irreversible world actions: `world` effect class, and the "
        "proposal's rule 'no verify-retry across a world action' is "
        f"exactly the safety property τ³ rewards ({ir}/{total_actions} = "
        f"{100*ir/total_actions:.1f}%)",
        f"- multi-action transactions ({tasks_multi_action} tasks): the "
        "proposal's per-class linear effect chain serializes same-class "
        "actions — sufficient for ordering, but τ³ transactions may need "
        "all-or-nothing rollback ACROSS classes (world+state in one "
        "transaction), which the current single-class chains do NOT "
        "express -> **documented gap: cross-class transactional effect "
        "regions**",
        "- verification: NL assertions map to VERIFY nodes; tasks with "
        "assertions need verify-after-effect ordering, expressible via "
        "after-edges + effect chain",
        "",
        "**Verdict**: the proposal covers read/reversible/irreversible "
        "classes and retry safety; the real gap is cross-class "
        "transactional semantics — recorded for effect-system v0.2 design "
        "(not implemented this phase)."]
    out = ROOT / "docs" / "tau3-effect-audit.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"actions={total_actions} classes={dict(action_class)} "
          f"multi_action_tasks={tasks_multi_action}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
