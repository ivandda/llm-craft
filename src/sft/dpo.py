"""DPO (Direct Preference Optimization) for the recipe student.

The SFT winner `soft_ce` already *knows* the answers (any@k validity ~97%) but its
top-1 is too verbose or garbled. DPO adds the missing negative/ranking signal: push
a short valid answer above the model's own verbose/garbage sample.

This module holds the DPO-specific pieces; the training loop (`trainer.py`) reuses
everything else via a `objective: sft|dpo` branch. Design points:

- **Reference = the SFT policy itself** (`soft_ce`), not the base model. The policy is
  initialized from the `soft_ce` adapter (`init_adapter_path`) and the reference
  log-probs are precomputed once from that same model before training (they are
  constant because the reference is frozen). We never use `disable_adapter()` — that
  would make the reference the raw base and pull the policy away from `soft_ce`.
- **Completion mask covers the whole answer through EOS** (not just the concept span),
  so DPO can penalise the verbose tail and reward stopping. Built in `DPODataCollator`.
- Sequence log-probs reuse `causal_masked_logprobs` from `losses.py` (summed, not
  length-normalized) — the same primitive the SFT loss uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.sft.collator import (
    concept_mask_from_offsets,
    render_candidate_text,
    render_qwen_chat_candidate_text,
)
from src.sft.losses import causal_masked_logprobs


@dataclass(frozen=True)
class DPOLossComponents:
    """DPO loss + diagnostics. Mirrors `SFTLossComponents` so the training loop can
    treat both uniformly: it always reads `.total_loss` and `.log_metrics()`."""

    total_loss: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    reward_margin: torch.Tensor
    reward_accuracy: torch.Tensor
    chosen_logp: torch.Tensor
    rejected_logp: torch.Tensor

    def log_metrics(self) -> dict[str, float]:
        return {
            "reward_margin": float(self.reward_margin.item()),
            "reward_accuracy": float(self.reward_accuracy.item()),
            "chosen_reward": float(self.chosen_reward.item()),
            "rejected_reward": float(self.rejected_reward.item()),
            "chosen_logp": float(self.chosen_logp.item()),
            "rejected_logp": float(self.rejected_logp.item()),
        }


def compute_dpo_loss_components(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> DPOLossComponents:
    """Standard DPO loss over a batch of preference pairs.

    All four inputs are per-sequence summed completion log-probs, shape ``[B]``. Only
    the ``policy_*`` tensors should carry gradients; the ``reference_*`` tensors are
    frozen constants (precomputed) and must be detached by the caller.

        loss = -[ (1-eps)·logσ(z) + eps·logσ(-z) ],   z = beta·((Δpolicy) - (Δreference))

    where ``Δpolicy = logπθ(y_w) - logπθ(y_l)`` and likewise for the reference. With
    ``label_smoothing=0`` this is the plain DPO loss ``-logσ(z)``. ``reference_free``
    drops the reference anchor (``Δreference = 0``) — not recommended here, it removes
    the pull toward the SFT policy.

    Uses ``F.logsigmoid`` for numerical stability (safe for large |z|).
    """
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}.")

    policy_delta = policy_chosen_logps - policy_rejected_logps
    if reference_free:
        reference_delta = torch.zeros_like(policy_delta)
    else:
        reference_delta = reference_chosen_logps - reference_rejected_logps

    logits = beta * (policy_delta - reference_delta)
    # cDPO / robust label smoothing: mix in the flipped-preference term.
    losses = -(
        (1.0 - label_smoothing) * F.logsigmoid(logits)
        + label_smoothing * F.logsigmoid(-logits)
    )
    total_loss = losses.mean()

    # Rewards are the (detached-reference) implicit rewards β·(logπθ - logπref).
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps).mean()
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps).mean()
    reward_margin = chosen_reward - rejected_reward
    # Per-pair accuracy: did the policy rank chosen above rejected (vs the reference)?
    per_pair_margin = beta * (policy_delta - reference_delta)
    reward_accuracy = (per_pair_margin > 0).float().mean()

    return DPOLossComponents(
        total_loss=total_loss,
        chosen_reward=chosen_reward.detach(),
        rejected_reward=rejected_reward.detach(),
        reward_margin=reward_margin.detach(),
        reward_accuracy=reward_accuracy.detach(),
        chosen_logp=policy_chosen_logps.mean().detach(),
        rejected_logp=policy_rejected_logps.mean().detach(),
    )


# --------------------------------------------------------------------------- #
# Preference dataset + collator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreferencePair:
    pair_index: int
    input_a: str
    input_b: str
    chosen: str
    rejected: str


class PreferencePairDataset:
    """Reads a DPO `pairs.jsonl` (one preference pair per line). Each pair keeps a
    stable `pair_index` (line order) used to look up precomputed reference log-probs."""

    def __init__(self, path: str | Path, *, max_examples: int | None = None) -> None:
        self.pairs: list[PreferencePair] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                chosen = str(row["chosen"]).strip()
                rejected = str(row["rejected"]).strip()
                if not chosen or not rejected:
                    raise ValueError(f"Pair {index} has an empty chosen/rejected.")
                self.pairs.append(
                    PreferencePair(
                        pair_index=index,
                        input_a=row["input_a"],
                        input_b=row["input_b"],
                        chosen=chosen,
                        rejected=rejected,
                    )
                )
                if max_examples is not None and len(self.pairs) >= max_examples:
                    break

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PreferencePair:
        return self.pairs[idx]


def _render_answer_text(
    prompt_format: str,
    tokenizer: Any,
    input_a: str,
    input_b: str,
    output: str,
    system_prompt: str | None,
) -> tuple[str, int]:
    """Render prompt+answer with the SAME templates SFT/eval use, and return the char
    offset where the answer begins. The DPO completion mask runs from there to EOS."""
    if prompt_format == "qwen_chat":
        text, concept_start, _ = render_qwen_chat_candidate_text(
            tokenizer, input_a, input_b, output, system_prompt=system_prompt
        )
    elif prompt_format == "plain":
        text, concept_start, _ = render_candidate_text(input_a, input_b, output)
    else:
        raise ValueError(f"Unsupported prompt_format: {prompt_format}")
    return text, concept_start


@dataclass
class DPODataCollator:
    """Builds a single tokenized batch holding chosen rows then rejected rows (2N rows
    for N pairs) so both sides are scored in one forward pass. The completion mask spans
    the answer through end-of-sequence (Correction C) — not just the concept span — so
    DPO can penalise the verbose tail and reward stopping."""

    tokenizer: Any
    max_seq_length: int = 512
    prompt_format: str = "qwen_chat"
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("DPODataCollator requires a fast tokenizer with offset mappings.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, pairs: list[PreferencePair]) -> dict[str, torch.Tensor]:
        if not pairs:
            raise ValueError("Cannot collate an empty DPO batch.")
        texts: list[str] = []
        answer_starts: list[int] = []
        # chosen rows first, then rejected rows (rows 0..N-1 chosen, N..2N-1 rejected).
        for pair in pairs:
            text, start = _render_answer_text(
                self.prompt_format, self.tokenizer, pair.input_a, pair.input_b, pair.chosen, self.system_prompt
            )
            texts.append(text)
            answer_starts.append(start)
        for pair in pairs:
            text, start = _render_answer_text(
                self.prompt_format, self.tokenizer, pair.input_a, pair.input_b, pair.rejected, self.system_prompt
            )
            texts.append(text)
            answer_starts.append(start)

        tokenized = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = tokenized.pop("offset_mapping").tolist()
        # completion mask: answer_start .. end-of-text (covers answer + EOS/im_end,
        # excludes prompt and padding whose offsets are (0,0) or end<=answer_start).
        masks = [
            concept_mask_from_offsets(row_offsets, start, len(text))
            for row_offsets, start, text in zip(offsets, answer_starts, texts, strict=True)
        ]
        completion_mask = torch.tensor(masks, dtype=torch.bool)
        if not completion_mask.any(dim=1).all():
            raise ValueError(
                "A chosen/rejected completion was fully truncated. Increase max_seq_length "
                "(verbose 'rejected' sequences are long by construction)."
            )
        tokenized["completion_mask"] = completion_mask
        tokenized["pair_index"] = torch.tensor([p.pair_index for p in pairs], dtype=torch.long)
        return tokenized


# --------------------------------------------------------------------------- #
# Forward / split / reference precompute
# --------------------------------------------------------------------------- #
def policy_pair_logps(
    model: Any, batch: dict[str, torch.Tensor], *, length_normalize: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward over the concatenated chosen+rejected rows (2N), split into
    (chosen_logps, rejected_logps), each ``[N]``. Rows 0..N-1 are chosen, N..2N-1
    rejected (the `DPODataCollator` order)."""
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logps = causal_masked_logprobs(
        outputs.logits, batch["input_ids"], batch["completion_mask"], length_normalize=length_normalize
    )
    n = logps.shape[0] // 2
    return logps[:n], logps[n:]


@torch.no_grad()
def precompute_reference_logprobs(
    model: Any, loader: Any, *, length_normalize: bool = False
) -> dict[int, tuple[float, float]]:
    """Run the frozen policy (== `soft_ce` at init) once over the DPO data and cache the
    reference (chosen, rejected) log-probs per `pair_index`. Constant during training, so
    a single pass suffices and no `disable_adapter()` / second model copy is needed."""
    was_training = model.training
    model.eval()
    reference: dict[int, tuple[float, float]] = {}
    for batch in loader:
        chosen, rejected = policy_pair_logps(model, batch, length_normalize=length_normalize)
        for i, pair_index in enumerate(batch["pair_index"].tolist()):
            reference[int(pair_index)] = (float(chosen[i].item()), float(rejected[i].item()))
    if was_training:
        model.train()
    return reference
