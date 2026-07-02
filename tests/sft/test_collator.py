import torch
import pytest

from src.sft.collator import (
    SFTDataCollator,
    concept_mask_from_offsets,
    render_candidate_with_rationale_text,
    render_candidate_text,
    render_qwen_chat_candidate_with_rationale_text,
    render_qwen_chat_candidate_text,
)
from src.sft.dataset import Candidate, RecipeExample


class TokenizedDict(dict):
    def tolist(self):
        return self


class DummyFastTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        truncation=True,
        max_length=256,
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    ):
        encoded = []
        offsets = []
        max_len = 0
        for text in texts:
            ids = [ord(char) % 97 + 1 for char in text][:max_length]
            row_offsets = [(index, index + 1) for index in range(min(len(text), max_length))]
            encoded.append(ids)
            offsets.append(row_offsets)
            max_len = max(max_len, len(ids))
        for ids, row_offsets in zip(encoded, offsets, strict=True):
            pad = max_len - len(ids)
            ids.extend([self.pad_token_id] * pad)
            row_offsets.extend([(0, 0)] * pad)
        return TokenizedDict(
            {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor([[int(token != 0) for token in row] for row in encoded]),
                "offset_mapping": torch.tensor(offsets, dtype=torch.long),
            }
        )

    def save_pretrained(self, path):
        return None


class DummyChatTokenizer(DummyFastTokenizer):
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        chunks = []
        for message in messages:
            chunks.append(f"<|{message['role']}|>\n{message['content']}\n")
        if add_generation_prompt:
            chunks.append("<|assistant|>\n")
        else:
            chunks.append("<|end|>")
        return "".join(chunks)


def test_concept_mask_from_offsets_marks_final_concept_span():
    text, start, end = render_candidate_text("fire", "water", "steam")
    offsets = [(index, index + 1) for index in range(len(text))]

    mask = concept_mask_from_offsets(offsets, start, end)

    assert sum(mask) == len("steam")
    assert "".join(char for char, keep in zip(text, mask, strict=True) if keep) == "steam"


def test_collator_flattens_variable_candidates_by_recipe():
    examples = [
        RecipeExample(
            input_a="fire",
            input_b="water",
            candidates=[
                Candidate(output="steam", rank=1, weight=2 / 3),
                Candidate(output="vapor", rank=2, weight=1 / 3),
            ],
        ),
        RecipeExample(
            input_a="earth",
            input_b="water",
            candidates=[Candidate(output="mud", rank=1, weight=1.0)],
        ),
    ]

    batch = SFTDataCollator(DummyFastTokenizer(), max_seq_length=128)(examples)

    assert batch["input_ids"].shape[0] == 3
    assert batch["group_ids"].tolist() == [0, 0, 1]
    assert batch["candidate_weights"].tolist() == pytest.approx([2 / 3, 1 / 3, 1.0])
    assert batch["concept_mask"].shape == batch["input_ids"].shape
    assert batch["concept_mask"].any(dim=1).all()


def test_collator_keeps_all_candidates_for_ce_alias():
    examples = [
        RecipeExample(
            input_a="fire",
            input_b="water",
            candidates=[
                Candidate(output="teacher", source="teacher", rank=1, weight=0.5),
                Candidate(output="observed", source="observed", rank=2, weight=0.5),
            ],
        )
    ]

    batch = SFTDataCollator(DummyFastTokenizer(), loss_type="ce", ce_target="observed")(examples)

    assert batch["input_ids"].shape[0] == 2
    assert batch["group_ids"].tolist() == [0, 0]
    assert batch["candidate_weights"].tolist() == pytest.approx([0.5, 0.5])


def test_render_qwen_chat_candidate_text_marks_only_assistant_output():
    text, start, end = render_qwen_chat_candidate_text(
        DummyChatTokenizer(),
        "fire",
        "water",
        "steam",
        system_prompt="You are a careful crafting assistant.",
    )

    assert text[start:end] == "steam"
    assert text[end:].startswith("\n<|end|>")
    assert "Given two concepts, combine them into one resulting concept." in text
    assert "Concept A: fire" in text
    assert "Concept B: water" in text
    assert "Return only the resulting concept." in text


def test_collator_supports_qwen_chat_prompt_format():
    examples = [
        RecipeExample(
            input_a="fire",
            input_b="water",
            candidates=[Candidate(output="steam", rank=1, weight=1.0)],
        )
    ]

    batch = SFTDataCollator(
        DummyChatTokenizer(),
        max_seq_length=256,
        prompt_format="qwen_chat",
        system_prompt="Return the final concept only.",
    )(examples)

    assert batch["input_ids"].shape[0] == 1
    assert batch["concept_mask"].sum().item() == len("steam")
    assert batch["concept_mask"].any(dim=1).all()


def test_collator_adds_rationale_mask_when_enabled():
    examples = [
        RecipeExample(
            input_a="fire",
            input_b="water",
            candidates=[
                Candidate(
                    output="steam",
                    rationale="Fire heats water until it becomes steam.",
                    rank=1,
                    weight=1.0,
                )
            ],
        )
    ]

    batch = SFTDataCollator(
        DummyFastTokenizer(),
        max_seq_length=256,
        include_rationale=True,
    )(examples)

    assert batch["concept_mask"].sum().item() == len("steam")
    assert batch["rationale_mask"].sum().item() == len("Fire heats water until it becomes steam.")


def test_collator_leaves_empty_rationale_mask_for_missing_rationale():
    examples = [
        RecipeExample(
            input_a="fire",
            input_b="water",
            candidates=[Candidate(output="steam", rank=1, weight=1.0)],
        )
    ]

    batch = SFTDataCollator(
        DummyFastTokenizer(),
        max_seq_length=256,
        include_rationale=True,
    )(examples)

    assert batch["concept_mask"].any(dim=1).all()
    assert not batch["rationale_mask"].any()


def test_collator_supports_qwen_chat_rationale_mask():
    examples = [
        RecipeExample(
            input_a="fire",
            input_b="water",
            candidates=[
                Candidate(
                    output="steam",
                    rationale="Fire heats water until it becomes steam.",
                    rank=1,
                    weight=1.0,
                )
            ],
        )
    ]

    batch = SFTDataCollator(
        DummyChatTokenizer(),
        max_seq_length=256,
        prompt_format="qwen_chat",
        system_prompt="Explain the recipe briefly.",
        include_rationale=True,
    )(examples)

    assert batch["concept_mask"].sum().item() == len("steam")
    assert batch["rationale_mask"].sum().item() == len("Fire heats water until it becomes steam.")


def test_render_candidate_with_rationale_can_place_rationale_before_output():
    text, concept_start, concept_end, rationale_start, rationale_end = render_candidate_with_rationale_text(
        "fire",
        "water",
        "steam",
        "Fire heats water until it becomes steam.",
        rationale_position="output_after_rationale",
    )

    assert text == (
        "Input A: fire\n"
        "Input B: water\n"
        "Rationale: Fire heats water until it becomes steam.\n"
        "Final concept: steam"
    )
    assert text[rationale_start:rationale_end] == "Fire heats water until it becomes steam."
    assert text[concept_start:concept_end] == "steam"
    assert rationale_start < concept_start


def test_render_qwen_chat_rationale_can_place_rationale_before_output():
    text, concept_start, concept_end, rationale_start, rationale_end = render_qwen_chat_candidate_with_rationale_text(
        DummyChatTokenizer(),
        "fire",
        "water",
        "steam",
        "Fire heats water until it becomes steam.",
        system_prompt="Explain first.",
        rationale_position="output_after_rationale",
    )

    assert "Rationale: Fire heats water until it becomes steam.\nFinal concept: steam" in text
    assert text[rationale_start:rationale_end] == "Fire heats water until it becomes steam."
    assert text[concept_start:concept_end] == "steam"
    assert rationale_start < concept_start
