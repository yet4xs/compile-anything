"""Training preflight — run on the SERVER before any training.

    python scripts/training_preflight.py --model weights/Qwen2.5-3B-Instruct

Checks (in order): dependency versions vs requirements-training.txt,
CUDA + bf16, bitsandbytes 4-bit load, tokenizer chat template, model load,
LoRA target modules present, corpus load, 10-sample tokenization + length
stats, forward pass, 1 optimizer step, checkpoint save/reload.

Exit codes: 0 ok | 1 failure | 2 environment not trainable (no CUDA etc.)
--static mode runs only the dependency-free checks (usable on this
GPU-less dev machine).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQ = ROOT / "requirements-training.txt"


def _req_pins() -> dict:
    pins = {}
    for line in REQ.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            k, v = line.split("==")
            pins[k.strip()] = v.strip()
    return pins


def check(name, fn) -> bool:
    try:
        detail = fn()
        print(f"[PASS] {name}" + (f" ({detail})" if detail else ""))
        return True
    except Exception as e:                             # noqa: BLE001
        print(f"[FAIL] {name}: {e}")
        return False


def static_checks(args) -> int:
    ok = True
    ok &= check("requirements file parseable", lambda: f"{len(_req_pins())} pins")
    ok &= check("corpus v3 train loads",
                lambda: (n := sum(1 for _ in open(args.train, encoding="utf-8")))
                and f"{n} lines" if pathlib.Path(args.train).exists()
                else (_ for _ in ()).throw(FileNotFoundError(args.train)))
    ok &= check("tiers present in corpus",
                lambda: len({json.loads(l).get("quality_tier")
                             for l in open(args.train, encoding="utf-8")
                             .readlines()[:200]}))
    import yaml                                         # noqa: PLC0415
    for cfgf in sorted((ROOT / "src/compiler/train/config").glob("*.yaml")):
        ok &= check(f"config {cfgf.name}",
                    lambda p=cfgf: str(sorted(yaml.safe_load(
                        open(p, encoding="utf-8")).keys())))
    from src.compiler.train.dataset import iter_corpus_records
    ok &= check("SFT records selectable (plan_target, A+B)",
                lambda: str(len(list(iter_corpus_records(
                    [args.train], tiers=["A", "B"])))) + " records")
    print("[INFO] static checks done — full preflight requires the server "
          "(CUDA + pinned deps)")
    return 0 if ok else 1


def full_preflight(args) -> int:
    pins = _req_pins()
    ok = True

    def dep_versions():
        import torch, transformers, peft, trl, datasets  # noqa
        got = {"torch": torch.__version__, "transformers": transformers.__version__,
               "peft": peft.__version__, "trl": trl.__version__,
               "datasets": datasets.__version__}
        mismatch = {k: (v, pins.get(k)) for k, v in got.items()
                    if pins.get(k) and not v.startswith(pins[k])}
        if mismatch:
            raise RuntimeError(f"version mismatch vs pin: {mismatch}")
        return got

    ok &= check("dependency versions match requirements-training.txt",
                dep_versions)

    def cuda_check():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — cannot train")
        cap = torch.cuda.get_device_capability(0)
        bf16 = torch.cuda.is_bf16_supported()
        return f"{torch.cuda.get_device_name(0)}, cc={cap}, bf16={bf16}"

    ok &= check("CUDA visible + bf16 capability", cuda_check)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = None

    def tok_template():
        nonlocal tok
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tok.chat_template is None:
            raise RuntimeError("tokenizer has no chat template")
        s = tok.apply_chat_template(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "u"}], tokenize=False)
        return f"template ok, {len(s)} chars"

    ok &= check("tokenizer chat template", tok_template)

    def bnb_4bit():
        import bitsandbytes as bnb
        return f"bnb {bnb.__version__}"

    if args.quantize == "4bit":
        ok &= check("bitsandbytes importable", bnb_4bit)

    model = None

    def load_model():
        nonlocal model
        kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                  device_map="auto")
        if args.quantize == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
        return f"{model.config.num_hidden_layers} layers, {model.device}"

    ok &= check("model loads" + (" (4-bit NF4)" if args.quantize == "4bit"
                                 else ""), load_model)

    def lora_targets():
        from peft import LoraConfig, get_peft_model
        import yaml
        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
        targets = cfg.get("target_modules", [])
        missing = []
        for t in targets:
            found = any(t in n for n, _ in model.named_modules())
            if not found:
                missing.append(t)
        if missing:
            raise RuntimeError(f"missing LoRA target modules: {missing}")
        return f"{len(targets)} target modules present"

    ok &= check("LoRA target modules exist", lora_targets)

    def corpus_tokenize():
        from src.compiler.train.dataset import iter_corpus_records
        recs = list(iter_corpus_records([args.train], tiers=["A", "B"]))
        if len(recs) < 10:
            raise RuntimeError(f"only {len(recs)} records")
        import random
        sample = random.Random(0).sample(recs, 10)
        lens = []
        for r in sample:
            prompt = tok.apply_chat_template(
                [{"role": "system",
                  "content": __import__("src.compiler.prompt_format",
                                        fromlist=["SYSTEM_PROMPT"]
                                        ).SYSTEM_PROMPT},
                 {"role": "user", "content": r.instruction},
                 {"role": "assistant", "content": r.target}],
                tokenize=True)
            lens.append(len(prompt))
        if max(lens) > 4096:
            raise RuntimeError(f"sample max tokens {max(lens)} > 4096")
        return f"10 samples tokenized, len {min(lens)}..{max(lens)}"

    ok &= check("corpus loads + 10-sample tokenize", corpus_tokenize)

    def forward_pass():
        ids = tok("def f(x):", return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**ids)
        return f"logits {tuple(out.logits.shape)}"

    ok &= check("forward pass", forward_pass)

    def one_step():
        from peft import LoraConfig, get_peft_model
        import yaml
        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
        peft_model = get_peft_model(model, LoraConfig(
            r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
            target_modules=cfg.get("target_modules", []),
            task_type="CAUSAL_LM"))
        peft_model.train()
        ids = tok("%1 = SEARCH(@task)\nreturn %1\n", return_tensors="pt"
                  ).to(model.device)
        out = peft_model(**ids, labels=ids["input_ids"])
        out.loss.backward()
        for p in [p for p in peft_model.parameters()
                  if p.requires_grad][:1]:
            if p.grad is None:
                raise RuntimeError("no gradient on LoRA param")
        peft_model.zero_grad()
        return f"loss={out.loss.item():.4f}, backward + grads ok"

    ok &= check("1 optimizer step (LoRA fwd/bwd)", one_step)

    def ckpt_roundtrip(tmp: pathlib.Path = ROOT / "runs" / "_preflight_ckpt"):
        from peft import PeftModel
        peft_model = model
        if not hasattr(model, "save_pretrained") or \
                not isinstance(peft_model, PeftModel):
            from peft import LoraConfig, get_peft_model
            import yaml
            cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
            peft_model = get_peft_model(model, LoraConfig(
                r=8, target_modules=cfg.get("target_modules", [])[:2],
                task_type="CAUSAL_LM"))
        peft_model.save_pretrained(str(tmp))
        from peft import PeftModel as PM
        reloaded = PM.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True), str(tmp))
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        return f"adapter save + reload ok ({len(reloaded.peft_config)} cfg)"

    ok &= check("checkpoint save/reload", ckpt_roundtrip)

    print("\nPREFLIGHT " + ("OK — safe to train" if ok else
                            "FAILED — do NOT start training"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--train", default=str(ROOT / "data" / "compiler_corpus_v3"
                                           / "train.jsonl"))
    ap.add_argument("--config", default=str(ROOT / "src/compiler/train"
                                            / "config/qwen3b.yaml"))
    ap.add_argument("--quantize", default="4bit")
    ap.add_argument("--static", action="store_true",
                    help="dependency-free checks only (no torch)")
    args = ap.parse_args()
    if args.static:
        return static_checks(args)
    if not args.model:
        print("--model required for full preflight")
        return 2
    try:
        return full_preflight(args)
    except Exception as e:                             # noqa: BLE001
        print(f"[ERROR] preflight could not run: {e}")
        print("        environment not trainable — fix the above first")
        return 2


if __name__ == "__main__":
    sys.exit(main())
