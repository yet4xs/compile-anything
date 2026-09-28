"""Verify downloaded weights and write weights/MANIFEST.json.

    python scripts/verify_weights.py                     # all dirs present
    python scripts/verify_weights.py --dir weights/Qwen2.5-3B-Instruct
    python scripts/verify_weights.py --full-hash         # sha256 safetensors

Records model, source (ModelScope), file list, sizes, and sha256 of small
key files always (config/tokenizer); --full-hash adds shard hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

SMALL_KEY_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "vocab.json", "merges.txt", "generation_config.json",
                   "model.safetensors.index.json"]


def sha256_file(p: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(d: pathlib.Path, full_hash: bool) -> dict:
    files = sorted(p for p in d.rglob("*") if p.is_file())
    entry = {
        "model": d.name,
        "source": "modelscope (Qwen/Qwen2.5-*-Instruct)",
        "files": [str(p.relative_to(d)) for p in files],
        "total_bytes": sum(p.stat().st_size for p in files),
        "sha256": {},
    }
    cfg = d / "config.json"
    if cfg.exists():
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
            entry["revision"] = c.get("_name_or_path", "")
            entry["params_b"] = round(c.get("num_hidden_layers", 0), 1)
        except json.JSONDecodeError:
            pass
    for p in files:
        if p.name in SMALL_KEY_FILES or (full_hash
                                         and p.suffix == ".safetensors"):
            entry["sha256"][str(p.relative_to(d))] = sha256_file(p)
    has_weights = any(p.suffix == ".safetensors" or p.suffix == ".bin"
                      for p in files)
    entry["weights_present"] = has_weights
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=[])
    ap.add_argument("--full-hash", action="store_true")
    args = ap.parse_args()

    dirs = [pathlib.Path(d) for d in args.dir] or \
        sorted((ROOT / "weights").glob("Qwen*")) if (ROOT / "weights").exists() \
        else []
    manifest_path = ROOT / "weights" / "MANIFEST.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    ok = True
    for d in dirs:
        if not d.is_dir():
            print(f"[SKIP] {d} not downloaded yet")
            continue
        e = verify(d, args.full_hash)
        manifest[d.name] = e
        status = "OK" if e["weights_present"] else "INCOMPLETE (no weights)"
        print(f"[{status}] {d.name}: {len(e['files'])} files, "
              f"{e['total_bytes'] / 1e9:.2f} GB, "
              f"{len(e['sha256'])} hashed")
        ok &= e["weights_present"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2,
                                        ensure_ascii=False),
                             encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
