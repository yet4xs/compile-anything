"""Neural Compiler LoRA training entry (server-side; NOT run in this repo
environment — no GPU/deps here by design).

    python -m src.compiler.train.train_lora --config src/compiler/train/config/qwen3b.yaml \
        --model weights/Qwen2.5-3B-Instruct \
        --train data/compiler_corpus_v3/train.jsonl --val data/compiler_corpus_v3/val.jsonl \
        --out runs/phase5b1/e1_3b_qlora

API note: pinned to trl==0.12.2 (requirements-training.txt), where data
parameters live in SFTConfig (max_seq_length / dataset_text_field), NOT in
SFTTrainer kwargs. A small compat shim tolerates the 0.13+ rename
(max_seq_length -> max_length). scripts/training_preflight.py verifies the
installed versions before training.
"""
from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_config(path: str) -> dict:
    import yaml                                        # noqa: PLC0415
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_sft_args_class():
    """Return (SFTConfig class, dataset kwargs) adapted to the installed
    TRL version."""
    from trl import SFTConfig                          # noqa: PLC0415
    params = inspect.signature(SFTConfig.__init__).parameters
    kw = {}
    if "dataset_text_field" in params:
        kw["dataset_text_field"] = "text"
    if "max_seq_length" in params:
        kw["max_seq_length_key"] = "max_seq_length"
    elif "max_length" in params:
        kw["max_seq_length_key"] = "max_length"
    else:
        kw["max_seq_length_key"] = None
    return SFTConfig, kw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True,
                    help="local model path (e.g. weights/Qwen2.5-3B-Instruct)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-field", default="plan_target")
    ap.add_argument("--capability-context", action="store_true")
    ap.add_argument("--tiers", default="A,B")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="sanity mode (e.g. 50); -1 = full epochs")
    ap.add_argument("--resume", default="auto",
                    help="auto|none|path to checkpoint dir")
    ap.add_argument("--quantize", choices=["none", "4bit", "8bit"],
                    default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch                                       # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa
    from peft import (LoraConfig, get_peft_model,       # noqa
                      prepare_model_for_kbit_training)

    cfg = load_config(args.config)
    quant = args.quantize or cfg.get("quantization", "none")
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig   # noqa: PLC0415
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    elif quant == "8bit":
        from transformers import BitsAndBytesConfig   # noqa: PLC0415
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if quant != "none":
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules",
                               ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]),
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    from src.compiler.train.dataset import (records_to_chat,     # noqa
                                            iter_corpus_records)

    def rows(path):
        return list(records_to_chat(iter_corpus_records(
            [path], target_field=args.target_field, tiers=tiers,
            capability_context=args.capability_context)))

    train_rows, val_rows = rows(args.train), rows(args.val)
    print(f"train={len(train_rows)} val={len(val_rows)} tiers={tiers} "
          f"target={args.target_field} quant={quant} seed={args.seed}")

    def to_text(example):
        return {"text": tok.apply_chat_template(
            example["messages"], tokenize=False)}

    from datasets import Dataset                      # noqa: PLC0415
    train_ds = Dataset.from_list(train_rows).map(to_text)
    val_ds = Dataset.from_list(val_rows).map(to_text) if val_rows else None

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    SFTConfig, sftkw = make_sft_args_class()
    sft_extra = {}
    if sftkw.get("dataset_text_field"):
        sft_extra["dataset_text_field"] = sftkw["dataset_text_field"]
    if sftkw.get("max_seq_length_key"):
        sft_extra[sftkw["max_seq_length_key"]] = cfg.get("max_seq_length",
                                                         2048)
    sft_args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=cfg.get("epochs", 2),
        max_steps=args.max_steps,
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=cfg.get("lr", 1e-4),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("scheduler", "cosine"),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 20),
        eval_strategy="steps" if val_ds else "no",
        eval_steps=cfg.get("eval_steps", 200),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=2,
        report_to=[],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim="paged_adamw_8bit" if quant != "none" else "adamw_torch",
        seed=args.seed,
        **sft_extra,
    )

    from trl import SFTTrainer                         # noqa: PLC0415
    trainer_kwargs = dict(model=model, args=sft_args, train_dataset=train_ds,
                          eval_dataset=val_ds)
    try:
        trainer = SFTTrainer(processing_class=tok, **trainer_kwargs)
    except TypeError:                                  # older TRL
        trainer = SFTTrainer(tokenizer=tok, **trainer_kwargs)

    resume_from = None
    if args.resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint  # noqa
        if out.exists() and get_last_checkpoint(str(out)):
            resume_from = get_last_checkpoint(str(out))
            print(f"resuming from {resume_from}")
    elif args.resume not in ("none", ""):
        resume_from = args.resume

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    (out / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "argv": vars(args), "yaml": cfg}, indent=2, default=str),
        encoding="utf-8")
    print(f"saved -> {out / 'final'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
