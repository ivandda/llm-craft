from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def timestamp_run_id(model_name: str, loss_type: str, run_name: str | None) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    if run_name:
        suffix = slugify(run_name)
    else:
        suffix = f"{slugify(short_model_name(model_name))}_{slugify(loss_type)}"
    return f"{timestamp}_{suffix}"


def short_model_name(model_name: str) -> str:
    tail = model_name.rstrip("/").split("/")[-1]
    return tail.replace("Qwen3-4B-Instruct-2507", "qwen4b")


def slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        cleaned.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(cleaned).split("_") if part)


def make_run_dir(output_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True, allow_unicode=True)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    digest = hashlib.sha256()
    size = 0
    line_count = 0
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            line_count += chunk.count(b"\n")
            digest.update(chunk)
    return {
        "path": str(file_path),
        "sha256": digest.hexdigest(),
        "bytes": size,
        "lines": line_count,
    }


def git_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        info.update({"available": True, "commit": commit, "branch": branch, "dirty": dirty})
    except Exception as exc:
        info["error"] = str(exc)
    return info


def package_versions() -> dict[str, str | None]:
    packages = ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]
    versions: dict[str, str | None] = {"python": sys.version.replace("\n", " "), "platform": platform.platform()}
    for package in packages:
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", None)
        except Exception:
            versions[package] = None
    return versions


def save_command(path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(" ".join([sys.executable, *sys.argv]) + "\n")


def save_rng_state(path: str | Path) -> None:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
        "torch_cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(state, path)


def latest_checkpoint_file(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "latest"


def rotate_checkpoints(checkpoints_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    checkpoints = sorted(
        [path for path in checkpoints_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: path.name,
    )
    for checkpoint in checkpoints[:-save_total_limit]:
        for child in sorted(checkpoint.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        checkpoint.rmdir()
