"""Prompt formats for the Neural Compiler (Qwen2B Compiler) SFT/inference.

The compiler model maps:  NL task  ->  TaskIR (text form).

Two formats are provided:
  - chat messages ({"messages": [...]}, Qwen chat-template compatible)
  - plain text (single string, for tokenizer-level control)
"""
from __future__ import annotations

from typing import Dict, List, Optional

SYSTEM_PROMPT = (
    "You are a neural compiler. Compile the user's natural-language task into "
    "a TaskIR program (v0.1, text form).\n"
    "Rules:\n"
    "- One node per line: `%id = OP(input, ..., key=value, ...)`, in "
    "topological order (define before use).\n"
    "- Inputs may reference earlier `%id` values or declared `@name` globals; "
    "- Use only Skill ISA ops (SEARCH, FILTER, ARGMIN, GENERATE, VERIFY, "
    "SELECT, ...) — never concrete tool or API names.\n"
    "- End with `return %id` naming the node that produces the final answer.\n"
    "- Add a final VERIFY (and optionally retry on it) when correctness "
    "matters.\n"
    "Output only the TaskIR text, no explanations."
)


def build_messages(task: str, taskir_text: Optional[str] = None) -> Dict:
    """SFT example (assistant given) or inference payload (assistant absent)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task}]
    if taskir_text is not None:
        msgs.append({"role": "assistant", "content": taskir_text})
    return {"messages": msgs}


def build_plain_text(task: str, taskir_text: Optional[str] = None) -> str:
    """Single-string format: system/user/assistant with explicit markers."""
    parts = [f"<|system|>\n{SYSTEM_PROMPT}", f"<|user|>\n{task}"]
    if taskir_text is not None:
        parts.append(f"<|assistant|>\n{taskir_text}")
    return "\n".join(parts) + "\n"


def extract_taskir_text(completion: str) -> str:
    """Post-process a model completion: strip fences/markers, keep the IR.

    The generated text is validated afterwards by the TaskIR validator —
    this is purely cosmetic normalization."""
    text = completion.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1:] if first_nl != -1 else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip() + "\n"
