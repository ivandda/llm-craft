"""Offline tests for the DPO collator/dataset (CPU, char-level dummy tokenizers)."""

from __future__ import annotations

import json

import pytest
import torch

from src.sft.dpo import DPODataCollator, PreferencePair, PreferencePairDataset, _render_answer_text

from tests.sft.test_collator import DummyChatTokenizer, DummyFastTokenizer


def _pairs():
    return [
        PreferencePair(0, "fire", "water", chosen="steam", rejected="a hot cloud of water vapor rising"),
        PreferencePair(1, "earth", "water", chosen="mud", rejected="wet brown squishy soil mixture"),
    ]


def _masked_substring(text, mask):
    return "".join(char for char, keep in zip(text, mask, strict=True) if keep)


def test_collator_emits_2n_rows_chosen_first():
    collator = DPODataCollator(DummyChatTokenizer(), max_seq_length=256, prompt_format="qwen_chat", system_prompt="S")
    batch = collator(_pairs())
    assert batch["input_ids"].shape[0] == 4  # 2 pairs -> chosen(2) + rejected(2)
    assert batch["completion_mask"].shape == batch["input_ids"].shape
    assert batch["pair_index"].tolist() == [0, 1]


def test_completion_mask_covers_answer_through_eos():
    tok = DummyChatTokenizer()
    collator = DPODataCollator(tok, max_seq_length=256, prompt_format="qwen_chat", system_prompt="S")
    pairs = _pairs()
    batch = collator(pairs)
    # row 0 = chosen of pair 0; reconstruct its text to know the expected masked span.
    text, _ = _render_answer_text("qwen_chat", tok, "fire", "water", "steam", "S")
    mask = batch["completion_mask"][0].tolist()[: len(text)]
    masked = _masked_substring(text, mask)
    assert masked.startswith("steam")           # answer included
    assert masked.endswith("<|end|>")           # EOS/terminator included (Correction C)
    assert len(masked) > len("steam")           # extends past the answer to EOS


def test_padding_tokens_are_not_masked():
    collator = DPODataCollator(DummyChatTokenizer(), max_seq_length=256, prompt_format="qwen_chat")
    batch = collator(_pairs())
    # chosen "steam"/"mud" are short -> their rows get padded; padded positions must be False.
    input_ids = batch["input_ids"]
    completion_mask = batch["completion_mask"]
    pad_positions = input_ids == 0
    assert not (completion_mask & pad_positions).any()


def test_truncation_guard_raises_when_completion_cut():
    # tiny max_seq_length truncates the long rejected completion entirely.
    collator = DPODataCollator(DummyChatTokenizer(), max_seq_length=8, prompt_format="qwen_chat", system_prompt="S")
    with pytest.raises(ValueError, match="fully truncated"):
        collator(_pairs())


def test_plain_format_mask_matches_output():
    collator = DPODataCollator(DummyFastTokenizer(), max_seq_length=256, prompt_format="plain")
    pairs = [PreferencePair(0, "fire", "water", chosen="steam", rejected="hot vapor cloud")]
    batch = collator(pairs)
    text, _ = _render_answer_text("plain", DummyFastTokenizer(), "fire", "water", "steam", None)
    mask = batch["completion_mask"][0].tolist()[: len(text)]
    assert _masked_substring(text, mask) == "steam"  # plain has no trailing structure


def test_preference_pair_dataset_reads_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"pair_id": "fire+water", "input_a": "fire", "input_b": "water", "chosen": "steam", "rejected": "hot vapor cloud"},
        {"pair_id": "earth+water", "input_a": "earth", "input_b": "water", "chosen": "mud", "rejected": "wet soil"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ds = PreferencePairDataset(path)
    assert len(ds) == 2
    assert ds[0].pair_index == 0 and ds[0].chosen == "steam"
    assert ds[1].input_a == "earth"


def test_preference_pair_dataset_rejects_empty(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"input_a": "a", "input_b": "b", "chosen": "", "rejected": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        PreferencePairDataset(path)
