"""Phase 2: V-DPO (Validator-Guided Direct Preference Optimization).

Uses V1-V6 / schema validation results as preference pairs:
  - Generate multiple TaskIR/function call candidates per task
  - Validate each → rank by validation quality
  - Best candidate = "chosen", worst = "rejected"
  - Train with DPO on these pairs

This is novel: deterministic compiler validation as DPO reward signal.
"""
import json, os, sys, re, random, gc
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

SYSTEM_FC = (
    "You are a function-calling assistant. Given a task and available function "
    "schemas, output the correct function call(s). If no function is relevant, "
    "say 'No function should be called.' Otherwise output function call(s) in "
    "the format: function_name(param1=value1, param2=value2)"
)


def validate_against_schema(calls, schemas):
    """Returns (num_errors, error_list)."""
    errors = []
    schema_map = {s["name"]: s for s in schemas if isinstance(s, dict) and "name" in s}
    for call in calls:
        fname = call.get("name", "")
        if fname not in schema_map:
            errors.append(f"unknown_function:{fname}")
            continue
        schema = schema_map[fname]
        params = schema.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        args = call.get("arguments", {})
        for req in required:
            if req not in args:
                errors.append(f"missing_required:{fname}.{req}")
        for pname, pvalue in args.items():
            if pname not in props:
                errors.append(f"unexpected_param:{fname}.{pname}")
                continue
            et = props[pname].get("type", "any")
            at = type(pvalue).__name__
            if et == "string" and at != "str": errors.append(f"type_err:{fname}.{pname}")
            elif et == "integer" and at != "int": errors.append(f"type_err:{fname}.{pname}")
            elif et == "boolean" and at != "bool": errors.append(f"type_err:{fname}.{pname}")
            elif et == "number" and at not in ("int", "float"): errors.append(f"type_err:{fname}.{pname}")
            elif et == "array" and at != "list": errors.append(f"type_err:{fname}.{pname}")
    return len(errors), errors


def parse_calls(text):
    if "no function" in text.lower():
        return []
    calls = []
    for m in re.finditer(r"(\w+)\((.*?)\)", text, re.DOTALL):
        name, ps = m.group(1), m.group(2).strip()
        args = {}
        if ps:
            depth = 0; parts = []; cur = ""
            for ch in ps:
                if ch in "([{": depth += 1
                elif ch in ")]}": depth -= 1
                elif ch == "," and depth == 0: parts.append(cur); cur = ""; continue
                cur += ch
            if cur.strip(): parts.append(cur)
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k, v = k.strip(), v.strip()
                    try: args[k] = json.loads(v)
                    except:
                        if v.lower() in ("true", "false"): args[k] = v.lower() == "true"
                        else: args[k] = v.strip("'\"")
        calls.append({"name": name, "arguments": args})
    return calls


def score_output(output, schemas, target):
    """Score a model output: higher is better."""
    calls = parse_calls(output)
    if not calls:
        if "no function" in target.lower():
            return 100  # correctly refused
        return -100  # failed to generate

    n_errors, _ = validate_against_schema(calls, schemas)
    if n_errors > 0:
        return -n_errors  # negative score proportional to errors

    # Check name match with target
    target_calls = parse_calls(target)
    if not target_calls:
        return 50  # generated something when no function needed (partial credit)

    if len(calls) == len(target_calls):
        names_match = all(c["name"] == t["name"] for c, t in zip(calls, target_calls))
        if names_match:
            return 100  # perfect function selection
    return 50  # partial match


def generate_candidates(model, tok, data, n_candidates=4, batch=8):
    """Generate multiple candidates per task using sampling."""
    all_candidates = defaultdict(list)

    prompts = []
    for s in data:
        user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_FC},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

    for cand_idx in range(n_candidates):
        outputs = []
        for bs in range(0, len(prompts), batch):
            if bs % 80 == 0:
                print(f"    candidate {cand_idx+1}/{n_candidates}, {bs}/{len(prompts)}", flush=True)
            chunk = prompts[bs:bs+batch]
            inputs = tok(chunk, return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=256,
                    do_sample=True, temperature=0.8, top_p=0.95,
                    pad_token_id=tok.pad_token_id)
            outputs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True) for o in out)
        for i, output in enumerate(outputs):
            all_candidates[i].append(output)

    return all_candidates


def build_dpo_dataset(data, candidates):
    """Build preference pairs: (prompt, chosen, rejected)."""
    pairs = []
    for i, s in enumerate(data):
        cands = candidates.get(i, [])
        if len(cands) < 2:
            continue

        scored = []
        for output in cands:
            score = score_output(output, s.get("schemas", []), s.get("target", ""))
            scored.append((score, output))

        scored.sort(key=lambda x: -x[0])
        best_score, best_output = scored[0]
        worst_score, worst_output = scored[-1]

        # Only create pair if there's meaningful difference
        if best_score - worst_score < 5:
            continue

        user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
        pairs.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_FC},
                {"role": "user", "content": user},
            ],
            "chosen": best_output,
            "rejected": worst_output,
        })
    return pairs


def main():
    random.seed(42)

    # Load training data
    data = [json.loads(l) for l in open("data/bfcl_training/bfcl_train.jsonl",
                                         encoding="utf-8") if l.strip()]
    random.shuffle(data)
    data = data[:5000]  # Use 5k for candidate generation (time constraint)
    print(f"Candidate generation on {len(data)} samples")

    # Load model
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-7B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, "runs/bfcl_track/fc_7b_max_s42/final")
    model.eval()

    # Generate candidates
    print("\nGenerating candidates (4 per task)...")
    candidates = generate_candidates(model, tok, data, n_candidates=4)

    # Build DPO dataset
    print("\nBuilding DPO preference pairs...")
    pairs = build_dpo_dataset(data, candidates)
    print(f"Created {len(pairs)} preference pairs")

    if len(pairs) < 100:
        print("Too few pairs, skipping DPO training")
        return

    del model
    gc.collect(); torch.cuda.empty_cache()

    # Load fresh model for DPO training
    base2 = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="auto")
    base2 = prepare_model_for_kbit_training(base2)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"],
                       task_type="CAUSAL_LM")
    train_model = get_peft_model(base2, lora)
    train_model.print_trainable_parameters()

    # Create dataset
    ds = Dataset.from_list(pairs)

    # DPO training
    dpo_config = DPOConfig(
        output_dir="runs/bfcl_track/vdpo_7b",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        max_length=2048,
        max_prompt_length=1024,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        seed=42,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=train_model,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()

    train_model.save_pretrained("runs/bfcl_track/vdpo_7b/final")
    tok.save_pretrained("runs/bfcl_track/vdpo_7b/final")
    print("Saved runs/bfcl_track/vdpo_7b/final")

    # Save pair statistics
    stats = {
        "total_samples": len(data),
        "preference_pairs": len(pairs),
        "pairs_ratio": len(pairs) / len(data),
    }
    with open("results/bfcl_track/vdpo_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("V-DPO-DONE")


if __name__ == "__main__":
    from peft import PeftModel  # import here to avoid circular
    main()
