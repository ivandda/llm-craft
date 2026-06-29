from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.sft.dataset import RecipeExample


OUTPUT_SENTINEL = "<<LLM_CRAFT_OUTPUT_SENTINEL>>"


def render_prefix(input_a: str, input_b: str) -> str:
    return f"Input A: {input_a}\nInput B: {input_b}\nFinal concept:"


def render_user_prompt(input_a: str, input_b: str) -> str:
    return (
        "Given two concepts, combine them into one resulting concept.\n\n"
        f"Concept A: {input_a}\n"
        f"Concept B: {input_b}\n\n"
        "Return only the resulting concept."
    )


def render_candidate_text(input_a: str, input_b: str, output: str) -> tuple[str, int, int]:
    prefix = render_prefix(input_a, input_b)
    text = f"{prefix} {output}"
    start = len(prefix) + 1
    end = start + len(output)
    return text, start, end


def _sentinel_for_output(output: str) -> str:
    sentinel = OUTPUT_SENTINEL
    suffix = 0
    while sentinel in output:
        suffix += 1
        sentinel = f"{OUTPUT_SENTINEL}_{suffix}"
    return sentinel


def render_qwen_chat_candidate_text(
    tokenizer: Any,
    input_a: str,
    input_b: str,
    output: str,
    *,
    system_prompt: str | None = None,
) -> tuple[str, int, int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("prompt_format='qwen_chat' requires a tokenizer with apply_chat_template().")

    sentinel = _sentinel_for_output(output)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": render_user_prompt(input_a, input_b)})
    messages.append({"role": "assistant", "content": sentinel})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or sentinel not in rendered:
        raise ValueError("The chat template did not preserve the assistant sentinel span.")

    concept_start = rendered.index(sentinel)
    text = rendered.replace(sentinel, output, 1)
    concept_end = concept_start + len(output)
    return text, concept_start, concept_end


def concept_mask_from_offsets(
    offsets: list[tuple[int, int]] | list[list[int]],
    concept_start: int,
    concept_end: int,
) -> list[bool]:
    mask: list[bool] = []
    for start, end in offsets:
        mask.append(end > concept_start and start < concept_end and end > start)
    return mask


@dataclass
class SFTDataCollator:
    tokenizer: Any
    max_seq_length: int = 256
    prompt_format: str = "plain"
    system_prompt: str | None = None
    loss_type: str = "concept_set"
    ce_target: str = "rank1"

    def __post_init__(self) -> None:
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("SFTDataCollator requires a fast tokenizer with offset mappings.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _rows_for_example(self, example: RecipeExample, group_id: int) -> list[dict[str, Any]]:
        rows = []
        for candidate in example.candidates:
            if self.prompt_format == "plain":
                text, concept_start, concept_end = render_candidate_text(
                    example.input_a, example.input_b, candidate.output
                )
            elif self.prompt_format == "qwen_chat":
                text, concept_start, concept_end = render_qwen_chat_candidate_text(
                    self.tokenizer,
                    example.input_a,
                    example.input_b,
                    candidate.output,
                    system_prompt=self.system_prompt,
                )
            else:
                raise ValueError(f"Unsupported prompt_format: {self.prompt_format}")
            rows.append(
                {
                    "text": text,
                    "concept_start": concept_start,
                    "concept_end": concept_end,
                    "group_id": group_id,
                    "weight": float(candidate.weight if candidate.weight is not None else 1.0),
                }
            )
        return rows

    def __call__(self, examples: list[RecipeExample]) -> dict[str, torch.Tensor]:
        # The dataloader batches recipes. The collator then expands each recipe into
        # one tokenized row per acceptable candidate while preserving shared group_ids.
        rows: list[dict[str, Any]] = []
        for group_id, example in enumerate(examples):
            rows.extend(self._rows_for_example(example, group_id))
        if not rows:
            raise ValueError("Cannot collate an empty SFT batch.")

        tokenized = self.tokenizer(
            [row["text"] for row in rows],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = tokenized.pop("offset_mapping").tolist()
        concept_masks = [
            concept_mask_from_offsets(example_offsets, row["concept_start"], row["concept_end"])
            for example_offsets, row in zip(offsets, rows, strict=True)
        ]
        concept_mask = torch.tensor(concept_masks, dtype=torch.bool)
        if not concept_mask.any(dim=1).all():
            raise ValueError("At least one candidate concept was fully truncated. Increase max_seq_length.")

        tokenized["concept_mask"] = concept_mask
        tokenized["group_ids"] = torch.tensor([row["group_id"] for row in rows], dtype=torch.long)
        tokenized["candidate_weights"] = torch.tensor([row["weight"] for row in rows], dtype=torch.float)
        return tokenized
