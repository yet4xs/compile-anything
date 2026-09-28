"""Neural Compiler training-side components (prompt formats, SFT data).

This package is what makes the repo a *compiler* project rather than a
TaskIR runtime: it produces the (NL task -> TaskIR) supervision that the
Qwen2B Compiler is trained on. Model/training code lands here in Phase 2
(tokenizer adapter, LoRA config, trainer entry).
"""
from .prompt_format import (  # noqa: F401
    SYSTEM_PROMPT, build_messages, build_plain_text, extract_taskir_text,
)
from .train_data import load_pairs, dedup, split, write_sft  # noqa: F401
