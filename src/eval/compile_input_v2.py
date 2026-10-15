"""Phase 9B-2.5 T1: the ONE compile-input renderer shared by training,
evaluation and gold reconstruction.

v1 bug being fixed: convert_tau3_l3.py preferred ``description.purpose``
(evaluator metadata like "Test agent's capacity to handle a flight
change.") over ``user_scenario.instructions``. Official airline tasks are
50/50 annotated with purpose, so all airline benchmark instructions were
metadata — the user's reason_for_call / known_info never reached the
model, and every airline LITERAL/USER_LATENT verdict was made against a
missing input.

Protocol (see docs/phase9/runtime-benchmark-design.md revision): the
FULL-DISCLOSURE single-shot compile protocol renders ONLY
user_scenario.instructions, in a fixed field order, with field labels.
description.* (purpose/notes/relevant_policies) NEVER enters model input
— it contains grader-side intent hints and is label leakage; it may be
used only inside policy-dependence triage, with code-level separation
enforced by this module having no accessor for it.
"""

from typing import Optional

# fixed rendering order; absent fields are omitted
FIELD_ORDER = ("reason_for_call", "known_info", "unknown_info",
               "task_instructions")


def render_compile_input(user_scenario: Optional[dict]) -> str:
    """user_scenario: the tau3 task's user_scenario dict (raw JSON)."""
    ins = (user_scenario or {}).get("instructions")
    if isinstance(ins, str):
        ins = {"task_instructions": ins}
    ins = ins or {}
    lines = []
    for f in FIELD_ORDER:
        v = ins.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            lines.append(f"{f}: {s}")
    return "\n".join(lines)


def compile_input_from_task(task: dict) -> str:
    return render_compile_input(task.get("user_scenario"))
