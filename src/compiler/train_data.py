"""SFT dataset construction for the Qwen2B Compiler.

Consumes (task, taskir_text, taskir_json) pair files produced by the dataset
pipeline (data/train/*_pairs.jsonl), dedups by task text, splits train/val,
and writes chat-format records ready for `transformers` / `ms-swift` style
SFT. Every pair comes from validator-passing TaskIR (enforced upstream).
"""
from __future__ import annotations

import json
import random
from typing import Dict, List


def load_pairs(paths) -> List[Dict]:
    items: List[Dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("task") and rec.get("taskir_text"):
                        items.append(rec)
    return items


def dedup(items: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for it in items:
        key = " ".join(it["task"].lower().split())
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def split(items: List[Dict], val_ratio: float = 0.05,
          seed: int = 0) -> (List[Dict], List[Dict]):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
    val_ids = set(idx[:n_val])
    return ([it for i, it in enumerate(items) if i not in val_ids],
            [it for i, it in enumerate(items) if i in val_ids])


def write_sft(items: List[Dict], out_path, val_ratio: float = 0.05,
              seed: int = 0) -> Dict:
    from .prompt_format import build_messages
    train, val = split(dedup(items), val_ratio=val_ratio, seed=seed)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("val", val)):
        with open(out_path / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for it in subset:
                f.write(json.dumps(build_messages(it["task"], it["taskir_text"]),
                                   ensure_ascii=False) + "\n")
    return {"train": len(train), "val": len(val)}
