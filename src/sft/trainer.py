from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

from src.sft.collator import SFTDataCollator
from src.sft.config import SFTConfig
from src.sft.dataset import RecipeSFTDataset
from src.sft.losses import SFTLossComponents, compute_sft_loss_components
from src.sft.plotting import plot_losses
from src.sft.utils import (
    append_jsonl,
    format_duration,
    latest_checkpoint_file,
    rotate_checkpoints,
    save_rng_state,
    write_json,
)


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def infer_lora_target_modules(model: torch.nn.Module) -> list[str]:
    common_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "fc1",
        "fc2",
    }
    found = set()
    for name, module in model.named_modules():
        if name.split(".")[-1] in common_suffixes and hasattr(module, "weight"):
            found.add(name.split(".")[-1])
    if found:
        return sorted(found)
    return ["q_proj", "v_proj"]


def build_model_and_tokenizer(
    config: SFTConfig,
    *,
    apply_lora: bool = True,
) -> tuple[torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("The SFT pipeline requires a fast tokenizer for concept span offsets.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    if config.load_in_4bit:
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch_dtype_from_name(config.bnb_4bit_compute_dtype),
        )
    elif config.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif config.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)

    if not apply_lora:
        return model, tokenizer

    target_modules = (
        infer_lora_target_modules(model)
        if config.lora_target_modules == "auto"
        else [item.strip() for item in config.lora_target_modules.split(",") if item.strip()]
    )
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config), tokenizer


def build_dataloaders(config: SFTConfig, tokenizer: Any) -> tuple[DataLoader, DataLoader]:
    train_dataset = RecipeSFTDataset(
        config.train_path,
        weight_field=config.weight_field,
        weight_fallback=config.weight_fallback,
        max_examples=config.max_train_examples,
        merge_duplicate_recipes=config.merge_duplicate_recipes,
    )
    dev_dataset = RecipeSFTDataset(
        config.dev_path,
        weight_field=config.weight_field,
        weight_fallback=config.weight_fallback,
        max_examples=config.max_dev_examples,
        merge_duplicate_recipes=config.merge_duplicate_recipes,
    )
    train_collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        prompt_format=config.prompt_format,
        system_prompt=config.system_prompt,
        include_rationale=config.rationale_loss_weight > 0,
        rationale_position=config.rationale_position,
    )
    eval_collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        prompt_format=config.prompt_format,
        system_prompt=config.system_prompt,
        include_rationale=config.rationale_loss_weight > 0,
        rationale_position=config.rationale_position,
    )
    # `per_device_*_batch_size` counts recipes. Each collated batch can expand to
    # more tokenized rows because every candidate for a recipe stays in the batch.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=config.dataloader_num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=config.dataloader_num_workers,
    )
    return train_loader, dev_loader


def batch_loss_components(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: SFTConfig,
) -> SFTLossComponents:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return compute_sft_loss_components(
        outputs.logits,
        batch["input_ids"],
        batch["concept_mask"],
        batch["group_ids"],
        batch["candidate_weights"],
        candidate_weighting=config.candidate_weighting,
        candidate_aggregation=config.candidate_aggregation,
        loss_type=config.loss_type,
        length_normalize=config.length_normalize_concept_logprob,
        rationale_mask=batch.get("rationale_mask"),
        rationale_loss_weight=config.rationale_loss_weight,
        length_normalize_rationale=config.length_normalize_rationale_logprob,
    )


def batch_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor], config: SFTConfig) -> torch.Tensor:
    return batch_loss_components(model, batch, config).total_loss


@torch.no_grad()
def evaluate_components(
    model: torch.nn.Module,
    dev_loader: DataLoader,
    config: SFTConfig,
    accelerator: Accelerator,
) -> dict[str, float]:
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_concept_loss = torch.tensor(0.0, device=accelerator.device)
    total_rationale_loss = torch.tensor(0.0, device=accelerator.device)
    total_groups = torch.tensor(0.0, device=accelerator.device)
    for batch in dev_loader:
        components = batch_loss_components(model, batch, config)
        groups = torch.unique(batch["group_ids"]).numel()
        total_loss += accelerator.gather_for_metrics(components.total_loss.detach() * groups).sum()
        total_concept_loss += accelerator.gather_for_metrics(components.concept_loss.detach() * groups).sum()
        total_rationale_loss += accelerator.gather_for_metrics(components.rationale_loss.detach() * groups).sum()
        total_groups += accelerator.gather_for_metrics(torch.tensor(float(groups), device=accelerator.device)).sum()
    model.train()
    denom = total_groups.clamp_min(1.0)
    return {
        "loss": (total_loss / denom).item(),
        "concept_loss": (total_concept_loss / denom).item(),
        "rationale_loss": (total_rationale_loss / denom).item(),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, dev_loader: DataLoader, config: SFTConfig, accelerator: Accelerator) -> float:
    return evaluate_components(model, dev_loader, config, accelerator)["loss"]


def trainer_state(global_step: int, epoch: int, best_dev_loss: float | None, latest_checkpoint: str | None) -> dict[str, Any]:
    return {
        "global_step": global_step,
        "epoch": epoch,
        "best_dev_loss": best_dev_loss,
        "latest_checkpoint": latest_checkpoint,
    }


def save_adapter(model: torch.nn.Module, tokenizer: Any, output_dir: Path, accelerator: Accelerator) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    run_dir: Path,
    accelerator: Accelerator,
    *,
    global_step: int,
    epoch: int,
    best_dev_loss: float | None,
    save_total_limit: int,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / f"checkpoint-{global_step:06d}"
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint_dir / "state"))
    save_adapter(model, tokenizer, checkpoint_dir / "adapter", accelerator)
    if accelerator.is_main_process:
        state = trainer_state(global_step, epoch, best_dev_loss, str(checkpoint_dir))
        write_json(checkpoint_dir / "trainer_state.json", state)
        write_json(run_dir / "trainer_state.json", state)
        save_rng_state(checkpoint_dir / "rng_state.pt")
        latest_checkpoint_file(run_dir).write_text(str(checkpoint_dir), encoding="utf-8")
        rotate_checkpoints(run_dir / "checkpoints", save_total_limit)
    accelerator.wait_for_everyone()
    return checkpoint_dir


def load_checkpoint_if_needed(accelerator: Accelerator, config: SFTConfig) -> tuple[int, int, float | None]:
    if not config.resume_from_checkpoint:
        return 0, 0, None
    checkpoint = Path(config.resume_from_checkpoint)
    state_dir = checkpoint / "state"
    accelerator.load_state(str(state_dir if state_dir.exists() else checkpoint))
    trainer_state_path = checkpoint / "trainer_state.json"
    if not trainer_state_path.exists():
        return 0, 0, None
    import json

    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    return int(state.get("global_step", 0)), int(state.get("epoch", 0)), state.get("best_dev_loss")


def train(config: SFTConfig, run_dir: Path) -> dict[str, Any]:
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.bf16 else "fp16" if config.fp16 else "no",
    )
    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, dev_loader = build_dataloaders(config, tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / config.gradient_accumulation_steps))
    if config.max_steps > 0:
        total_steps = config.max_steps
        total_epochs = max(1, math.ceil(total_steps / updates_per_epoch))
    else:
        total_epochs = max(1, math.ceil(config.num_train_epochs))
        total_steps = total_epochs * updates_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if accelerator.is_main_process:
        print(
            "[sft] plan "
            f"train_recipes={len(train_loader.dataset)} "
            f"dev_recipes={len(dev_loader.dataset)} "
            f"updates_per_epoch={updates_per_epoch} "
            f"total_epochs={total_epochs} "
            f"total_steps={total_steps}",
            flush=True,
        )

    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        dev_loader,
        scheduler,
    )
    global_step, start_epoch, best_dev_loss = load_checkpoint_if_needed(accelerator, config)
    latest_checkpoint: str | None = config.resume_from_checkpoint
    running_loss = 0.0
    running_concept_loss = 0.0
    running_rationale_loss = 0.0
    running_count = 0
    train_start_time = time.perf_counter()

    model.train()
    for epoch in range(start_epoch, total_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss_components = batch_loss_components(model, batch, config)
                loss = loss_components.total_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().float().item()
            running_concept_loss += loss_components.concept_loss.detach().float().item()
            running_rationale_loss += loss_components.rationale_loss.detach().float().item()
            running_count += 1
            if accelerator.sync_gradients:
                global_step += 1
                if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                    avg_loss = running_loss / max(1, running_count)
                    avg_concept_loss = running_concept_loss / max(1, running_count)
                    avg_rationale_loss = running_rationale_loss / max(1, running_count)
                    running_loss = 0.0
                    running_concept_loss = 0.0
                    running_rationale_loss = 0.0
                    running_count = 0
                    elapsed_seconds = max(0.0, time.perf_counter() - train_start_time)
                    avg_step_seconds = elapsed_seconds / max(1, global_step)
                    remaining_steps = max(0, total_steps - global_step)
                    eta_seconds = avg_step_seconds * remaining_steps
                    record = {"step": global_step, "epoch": epoch, "loss": avg_loss}
                    if config.rationale_loss_weight > 0:
                        record.update(
                            {
                                "concept_loss": avg_concept_loss,
                                "rationale_loss": avg_rationale_loss,
                                "rationale_loss_weight": config.rationale_loss_weight,
                            }
                        )
                    if accelerator.is_main_process:
                        loss_parts = f"train_loss={avg_loss:.4f}"
                        if config.rationale_loss_weight > 0:
                            loss_parts += (
                                f" concept_loss={avg_concept_loss:.4f}"
                                f" rationale_loss={avg_rationale_loss:.4f}"
                            )
                        print(
                            f"[sft] step={global_step} epoch={epoch} {loss_parts} "
                            f"elapsed={format_duration(elapsed_seconds)} "
                            f"eta={format_duration(eta_seconds)}",
                            flush=True,
                        )
                        append_jsonl(run_dir / "train_losses.jsonl", record)
                        append_jsonl(
                            run_dir / "metrics.jsonl",
                            {
                                "split": "train",
                                "elapsed_seconds": elapsed_seconds,
                                "eta_seconds": eta_seconds,
                                **record,
                            },
                        )

                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    dev_components = evaluate_components(model, dev_loader, config, accelerator)
                    dev_loss = dev_components["loss"]
                    if accelerator.is_main_process:
                        print(f"[sft] step={global_step} epoch={epoch} dev_loss={dev_loss:.4f}", flush=True)
                        eval_record = {"step": global_step, "epoch": epoch, "loss": dev_loss}
                        if config.rationale_loss_weight > 0:
                            eval_record.update(
                                {
                                    "concept_loss": dev_components["concept_loss"],
                                    "rationale_loss": dev_components["rationale_loss"],
                                    "rationale_loss_weight": config.rationale_loss_weight,
                                }
                            )
                        append_jsonl(run_dir / "eval_losses.jsonl", eval_record)
                        append_jsonl(run_dir / "metrics.jsonl", {"split": "dev", **eval_record})
                    if best_dev_loss is None or dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        save_adapter(model, tokenizer, run_dir / "best_adapter", accelerator)

                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    checkpoint = save_checkpoint(
                        model,
                        tokenizer,
                        run_dir,
                        accelerator,
                        global_step=global_step,
                        epoch=epoch,
                        best_dev_loss=best_dev_loss,
                        save_total_limit=config.save_total_limit,
                    )
                    latest_checkpoint = str(checkpoint)

                if config.max_steps > 0 and global_step >= config.max_steps:
                    break
        if config.max_steps > 0 and global_step >= config.max_steps:
            break

    final_dev_components = evaluate_components(model, dev_loader, config, accelerator) if len(dev_loader) > 0 else None
    final_dev_loss = final_dev_components["loss"] if final_dev_components is not None else None
    if final_dev_loss is not None and accelerator.is_main_process:
        final_eval_record = {"step": global_step, "epoch": total_epochs, "loss": final_dev_loss}
        if config.rationale_loss_weight > 0 and final_dev_components is not None:
            final_eval_record.update(
                {
                    "concept_loss": final_dev_components["concept_loss"],
                    "rationale_loss": final_dev_components["rationale_loss"],
                    "rationale_loss_weight": config.rationale_loss_weight,
                }
            )
        append_jsonl(run_dir / "eval_losses.jsonl", final_eval_record)
        append_jsonl(run_dir / "metrics.jsonl", {"split": "dev", **final_eval_record})
    if final_dev_loss is not None and (best_dev_loss is None or final_dev_loss < best_dev_loss):
        best_dev_loss = final_dev_loss
        save_adapter(model, tokenizer, run_dir / "best_adapter", accelerator)

    expected_final_checkpoint = f"checkpoint-{global_step:06d}"
    if global_step > 0 and (latest_checkpoint is None or not latest_checkpoint.endswith(expected_final_checkpoint)):
        checkpoint = save_checkpoint(
            model,
            tokenizer,
            run_dir,
            accelerator,
            global_step=global_step,
            epoch=total_epochs,
            best_dev_loss=best_dev_loss,
            save_total_limit=config.save_total_limit,
        )
        latest_checkpoint = str(checkpoint)

    save_adapter(model, tokenizer, run_dir / "final_adapter", accelerator)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(run_dir / "tokenizer")
        write_json(
            run_dir / "trainer_state.json",
            trainer_state(global_step, total_epochs, best_dev_loss, latest_checkpoint),
        )
        plot_losses(run_dir)
    accelerator.wait_for_everyone()
    return trainer_state(global_step, total_epochs, best_dev_loss, latest_checkpoint)
