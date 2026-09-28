"""Dataset downloader — fetches real benchmarks per registry plan.

Writes into data/raw/<name>/..., records sha256 + provenance in
data/raw/metadata.json. Raw data is gitignored (only metadata is
committed). Uses stdlib urllib only.
"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import pathlib
import shutil
import time
import urllib.request
from typing import Dict, List, Optional

from .registry import REGISTRY, DatasetSpec

USER_AGENT = "compile-anything/0.1 (research; dataset acquisition)"
RAW_ROOT = pathlib.Path("data") / "raw"


def _fetch(url: str, dest: pathlib.Path, timeout: int = 120,
           retries: int = 2) -> Optional[str]:
    """Download url -> dest. Returns sha256 or None on failure."""
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r, \
                    open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return hashlib.sha256(dest.read_bytes()).hexdigest()
        except Exception as e:                        # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    print(f"    FAILED {url}: {last}")
    return None


def _github_tree(repo: str, ref: str, cache_dir: pathlib.Path) -> List[str]:
    cache = cache_dir / f"{repo.replace('/', '_')}_{ref}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        tree = json.load(r).get("tree", [])
    paths = sorted(t["path"] for t in tree if t.get("type") == "blob")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(paths), encoding="utf-8")
    return paths


def download_dataset(name: str, raw_root: pathlib.Path = RAW_ROOT,
                     sleep_s: float = 0.05) -> Dict:
    spec: DatasetSpec = REGISTRY[name]
    root = pathlib.Path(raw_root)
    dest_dir = root / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    (root / ".cache").mkdir(parents=True, exist_ok=True)

    entry: Dict = {"dataset": name, "version": "", "url": spec.source_url,
                   "license": spec.license,
                   "download_time": datetime.datetime.now().isoformat(
                       timespec="seconds"),
                   "sha256": "", "files": [], "status": "ok"}
    n_ok = n_fail = 0
    for kind, p in spec.plan:
        if kind == "file":
            dest = dest_dir / p["dest"]
            sha = _fetch(p["url"], dest)
            if sha:
                n_ok += 1
                entry["files"].append({"path": str(dest.relative_to(root)),
                                       "url": p["url"], "sha256": sha})
            else:
                n_fail += 1
        elif kind == "modelscope":
            # domestic mirror for HF-gated/unreachable datasets (no auth for
            # public repos): resolve-style download
            base = ("https://www.modelscope.cn/datasets/"
                    f"{p['ns']}/{p['name']}/resolve/{p.get('rev', 'master')}/")
            for rel in p["files"]:
                dest = dest_dir / rel.replace("/", "_")
                sha = _fetch(base + rel, dest, timeout=600)
                if sha:
                    n_ok += 1
                    entry["files"].append(
                        {"path": str(dest.relative_to(root)), "url": base + rel,
                         "sha256": sha, "mirror": "modelscope"})
                else:
                    n_fail += 1
        elif kind == "github-dir":
            paths = _github_tree(p["repo"], p["ref"], root / ".cache")
            wanted = [q for q in paths
                      if q.startswith(p["prefix"])
                      and any(q.endswith(s) for s in p["suffixes"])]
            sub = dest_dir / p["dest_dir"]
            for q in wanted:
                rel = q[len(p["prefix"]):]
                out = sub / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                url = (f"https://raw.githubusercontent.com/{p['repo']}/"
                       f"{p['ref']}/{q}")
                sha = _fetch(url, out)
                if sha:
                    n_ok += 1
                    entry["files"].append(
                        {"path": str(out.relative_to(root)), "url": url,
                         "sha256": sha})
                else:
                    n_fail += 1
                time.sleep(sleep_s)

    if n_fail or not n_ok:
        entry["status"] = "partial" if n_ok else "failed"
    entry["files_ok"] = n_ok
    entry["files_failed"] = n_fail

    meta_path = root / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    meta[name] = entry
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"{name:12s} {entry['status']:8s} files ok={n_ok} failed={n_fail}")
    return entry


def gunzip_file(path: pathlib.Path) -> pathlib.Path:
    """Decompress <path>.gz next to it; returns the plain file path."""
    out = path.with_suffix("")
    with gzip.open(path, "rb") as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out
