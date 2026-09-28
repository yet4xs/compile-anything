"""SFT dataset for the Neural Compiler (Qwen + LoRA).

Reads compiler corpus v3 records and yields chat-format samples:

    system: compiler prompt (src/compiler/prompt_format.SYSTEM_PROMPT)
    user:   instruction [+ optional capability context]
    assistant: plan_target (default) or execution_target

Pure-python record handling is dependency-free (unit-testable); the torch
Dataset/DatasetDict wrappers import transformers lazily so this module can
be used on the training server only.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Iterator, List, Optional

from ..prompt_format import SYSTEM_PROMPT


def build_user_text(instruction: str,
                    capabilities: Optional[List[str]] = None) -> str:
    """Input view A (instruction only) or B (+ normalized capability
    context — semantic skills only, never concrete API names)."""
    if capabilities:
        caps = "\n".join(f"- {c}" for c in capabilities)
        return (f"{instruction}\n\nAvailable capabilities:\n{caps}")
    return instruction


@dataclass
class SFTRecord:
    instruction: str
    target: str
    source: str
    capabilities: List[str]
    quality_tier: str
    id: str


def iter_corpus_records(paths, target_field: str = "plan_target",
                        tiers: Optional[List[str]] = None,
                        capability_context: bool = False
                        ) -> Iterator[SFTRecord]:
    for p in paths:
        p = pathlib.Path(p)
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if tiers and r.get("quality_tier") not in tiers:
                    continue
                target = r.get(target_field) or r.get("plan_target")
                if not target:
                    continue
                yield SFTRecord(
                    instruction=build_user_text(
                        r.get("instruction", ""),
                        r.get("capabilities") if capability_context else None),
                    target=target,
                    source=r.get("source", ""),
                    capabilities=r.get("capabilities") or [],
                    quality_tier=r.get("quality_tier", ""),
                    id=r.get("id", ""))


def records_to_chat(records: Iterator[SFTRecord]) -> Iterator[dict]:
    for r in records:
        yield {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r.instruction},
            {"role": "assistant", "content": r.target}],
            "source": r.source, "id": r.id, "quality_tier": r.quality_tier}


def load_hf_dataset(train_path, val_path, **kwargs):
    """Lazy transformers Dataset construction (server-side)."""
    from datasets import Dataset                       # noqa: PLC0415

    def gen(path):
        yield from records_to_chat(iter_corpus_records([path], **kwargs))

    return {"train": Dataset.from_generator(gen, train_path),
            "val": Dataset.from_generator(gen, val_path)}
