"""Neural Compiler LoRA training entry (server-side; NOT run in this repo
environment — no GPU/deps here by design).

    python -m src.compiler.train.train_lora --config src/compiler/train/config/qwen3b.yaml \
        --model /path/to/Qwen2.5-3B-Instruct \
        --train data/compiler_corpus_v3/train.jsonl --val data/compiler_corpus_v3/val.jsonl \
        --out runs/qwen3b-lora

Supports LoRA and QLoRA (4-bit) via bitsandbytes. Everything is
parameterized; model paths are never hardcoded.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_config(path: str) -> dict:
    import yaml                                        # noqa: PLC0415
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="yaml under "
                    "src/compiler/train/config/")
    ap.add_argument("--model", required=True,
                    help="local model path (e.g. downloaded Qwen weights)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="runs/neural-compiler")
    ap.add_argument("--target-field", default="plan_target",
                    help="plan_target (first round) or execution_target")
    ap.add_argument("--capability-context", action="store_true",
                    help="input view B: append normalized capabilities")
    ap.add_argument("--tiers", default="A,B",
                    help="quality tiers to train on (comma separated)")
    ap.add_argument("--quantize", choices=["none", "4bit", "8bit"],
                    default=None, help="overrides config")
    args = ap.parse_args()

    import torch                                       # noqa: PLC0415
    from transformers import (AutoModelForCausalLM, AutoTokenizer,   # noqa
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer                         # noqa: PLC0415

    cfg = load_config(args.config)
    quant = args.quantize or cfg.get("quantization", "none")
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig   # noqa: PLC0415
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    elif quant == "8bit":
        from transformers import BitsAndBytesConfig   # noqa: PLC0415
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if quant != "none":
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules",
                               ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    from src.compiler.train.dataset import (records_to_chat,     # noqa
                                            iter_corpus_records)
    train_rows = list(records_to_chat(iter_corpus_records(
        [args.train], target_field=args.target_field, tiers=tiers,
        capability_context=args.capability_context)))
    val_rows = list(records_to_chat(iter_corpus_records(
        [args.val], target_field=args.target_field, tiers=tiers,
        capability_context=args.capability_context)))
    print(f"train={len(train_rows)} val={len(val_rows)} "
          f"(tiers={tiers}, target={args.target_field})")

    def to_text(example):
        return {"text": tok.apply_chat_template(
            example["messages"], tokenize=False)}

    from datasets import Dataset                      # noqa: PLC0415
    train_ds = Dataset.from_list(train_rows).map(to_text)
    val_ds = Dataset.from_list(val_rows).map(to_text)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=cfg.get("epochs", 2),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=cfg.get("lr", 1e-4),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("scheduler", "cosine"),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 20),
        eval_strategy="steps" if val_rows else "no",
        eval_steps=cfg.get("eval_steps", 200),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=2,
        report_to=[],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )
    trainer = SFTTrainer(
        model=model, args=targs, train_dataset=train_ds,
        eval_dataset=val_ds if val_rows else None,
        processing_class=tok, dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_length", 2048),
    )
    trainer.train()
    trainer.save_model(f"{args.out}/final")
    tok.save_pretrained(f"{args.out}/final")
    print(f"saved -> {args.out}/final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
