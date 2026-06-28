from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = "configs/sft/default.yaml"


@dataclass
class SFTConfig:
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    train_path: str = "datasets/final-small-dataset/train.jsonl"
    dev_path: str = "datasets/final-small-dataset/dev.jsonl"
    output_dir: str = "runs/sft"
    run_name: str | None = None
    loss_type: str = "concept_set"
    ce_target: str = "rank1"
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_seq_length: int = 256
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "auto"
    resume_from_checkpoint: str | None = None
    max_train_examples: int | None = None
    max_dev_examples: int | None = None
    length_normalize_concept_logprob: bool = False
    weight_field: str = "weight"
    weight_fallback: str = "inverse_rank"
    trust_remote_code: bool = False
    dataloader_num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _str_to_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return loaded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal LM with SFT QLoRA on recipe candidates.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config with SFT defaults.")
    optional_int_fields = {"max_train_examples", "max_dev_examples"}
    for field in dataclasses.fields(SFTConfig):
        name = field.name
        default = field.default
        arg_name = "--" + name
        kwargs: dict[str, Any] = {"default": None}
        if name in optional_int_fields:
            kwargs["type"] = int
        elif isinstance(default, bool):
            kwargs["type"] = _str_to_bool
        elif isinstance(default, int) and not isinstance(default, bool):
            kwargs["type"] = int
        elif isinstance(default, float):
            kwargs["type"] = float
        else:
            kwargs["type"] = str
        parser.add_argument(arg_name, **kwargs)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SFTConfig:
    values = SFTConfig().to_dict()
    values.update(load_yaml_config(args.config))
    for field in dataclasses.fields(SFTConfig):
        override = getattr(args, field.name)
        if override is not None:
            values[field.name] = override
    config = SFTConfig(**values)
    validate_config(config)
    return config


def validate_config(config: SFTConfig) -> None:
    if config.loss_type not in {"concept_set", "ce", "soft_ce"}:
        raise ValueError("loss_type must be one of: concept_set, ce, soft_ce")
    if config.ce_target not in {"rank1", "observed", "first"}:
        raise ValueError("ce_target must be one of: rank1, observed, first")
    if config.weight_fallback not in {"uniform", "inverse_rank"}:
        raise ValueError("weight_fallback must be one of: uniform, inverse_rank")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if config.per_device_train_batch_size < 1 or config.per_device_eval_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
