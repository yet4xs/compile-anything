"""Generate the Phase 5B-1 frozen experiment manifest.

    python scripts/make_phase5b1_manifest.py                 # local fields
    python scripts/make_phase5b1_manifest.py --runtime       # + CUDA/torch
                                                             # (server)

Records git commit, corpus SHA256s, tier/split counts, TaskIR + prompt
versions/hashes, model, seed, pinned dependency versions. Runtime-only
fields (CUDA/torch) stay null until --runtime runs on the server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.taskir import TASKIR_VERSION                    # noqa: E402
from src.compiler.prompt_format import SYSTEM_PROMPT        # noqa: E402

FROZEN_FILES = [
    "src/lifter/toolmap.py",
    "spec/taskir-spec.md",
    "spec/skill-isa.md",
    "src/compiler/prompt_format.py",
    "data/compiler_corpus_v3_1/train.jsonl",
    "data/compiler_corpus_v3_1/val.jsonl",
    "data/compiler_corpus_v3_1/test.jsonl",
]


def sha256_file(p: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_lines(p: pathlib.Path) -> int:
    return sum(1 for _ in open(p, encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments" / "phase5b1"
                                         / "manifest.json"))
    ap.add_argument("--runtime", action="store_true",
                    help="fill runtime-only fields (run on the server)")
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
        text=True).stdout.strip()

    manifest = {
        "experiment": "phase5b1",
        "git_commit": commit,
        "taskir_version": TASKIR_VERSION,
        "prompt": {
            "version": "v1",
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode()).hexdigest(),
            "capability_context": "OFF (frozen for 5B-1)",
        },
        "frozen_inputs": {
            f: {"sha256": sha256_file(ROOT / f),
                "bytes": (ROOT / f).stat().st_size}
            for f in FROZEN_FILES if (ROOT / f).exists()
        },
        "corpus": {},
        "seed": 42,
        "dependencies": {},
        "runtime": {"cuda": None, "gpu": None, "torch": None,
                    "transformers": None, "trl": None,
                    "note": "fill via --runtime on the server"},
        "models": {
            "e0_e1": "weights/Qwen2.5-3B-Instruct",
            "e2_e3": "weights/Qwen2.5-7B-Instruct",
        },
        "prohibited_changes": [
            "src/lifter/toolmap.py", "data/compiler_corpus_v3_1/* (supersedes v3 per semantic audit)",
            "spec/taskir-spec.md", "train/test split",
            "(record issues -> Phase 5B-2)"],
    }

    for split in ("train", "val", "test"):
        p = ROOT / "data" / "compiler_corpus_v3_1" / f"{split}.jsonl"
        if p.exists():
            manifest["corpus"][split] = {"lines": count_lines(p)}

    stats_p = ROOT / "data" / "compiler_corpus_v3_1" / "stats.json"
    if stats_p.exists():
        s = json.loads(stats_p.read_text(encoding="utf-8"))
        manifest["corpus"]["tiers"] = s.get("tiers")
        manifest["corpus"]["leakage"] = s.get("leakage")

    for line in (ROOT / "requirements-training.txt").read_text(
            encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            k, v = line.split("==")
            manifest["dependencies"][k.strip()] = v.strip()

    if args.runtime:
        import torch                                       # noqa: PLC0415
        import transformers, trl                           # noqa: PLC0415
        manifest["runtime"].update({
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "note": "filled at server runtime",
        })

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"manifest -> {out} "
          f"(runtime fields {'filled' if args.runtime else 'null'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
