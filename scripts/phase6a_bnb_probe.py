"""Minimal repro: does 4-bit load segfault with/without expandable_segments?"""
import os, sys
print("alloc_conf =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"), flush=True)
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
base = AutoModelForCausalLM.from_pretrained(
    "/ccfa2026/compile-anything/weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto")
print("4-bit load OK", flush=True)
x = torch.tensor([[1]], device="cuda")
y = base.get_input_embeddings()(x)
print("embed forward OK", y.shape, flush=True)
print("PROBE-OK", flush=True)
