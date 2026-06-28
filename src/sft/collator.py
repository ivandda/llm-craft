from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.sft.dataset import RecipeExample, select_ce_candidate


def render_prefix(input_a: str, input_b: str) -> str:
    return f"Input A: {input_a}\nInput B: {input_b}\nFinal concept:"


def render_candidate_text(input_a: str, input_b: str, output: str) -> tuple[str, int, int]:
    prefix = render_prefix(input_a, input_b)
    text = f"{prefix} {output}"
    start = len(prefix) + 1
    end = start + len(output)
    return text, start, end


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
    loss_type: str = "concept_set"
    ce_target: str = "rank1"

    def __post_init__(self) -> None:
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("SFTDataCollator requires a fast tokenizer with offset mappings.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _rows_for_example(self, example: RecipeExample, group_id: int) -> list[dict[str, Any]]:
        candidates = example.candidates
        if self.loss_type == "ce":
            candidates = [select_ce_candidate(candidates, self.ce_target)]
        rows = []
        for candidate in candidates:
            text, concept_start, concept_end = render_candidate_text(example.input_a, example.input_b, candidate.output)
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
