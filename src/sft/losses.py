from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_concept_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    concept_mask: torch.Tensor,
    *,
    length_normalize: bool = False,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_concept_mask = concept_mask[:, 1:].to(shift_logits.dtype)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    concept_logprob = (token_log_probs * shift_concept_mask).sum(dim=-1)

    if length_normalize:
        lengths = shift_concept_mask.sum(dim=-1).clamp_min(1.0)
        concept_logprob = concept_logprob / lengths
    return concept_logprob


def concept_set_loss_from_logprobs(
    concept_logprobs: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for group_id in torch.unique(group_ids, sorted=True):
        group_mask = group_ids == group_id
        group_logprobs = concept_logprobs[group_mask]
        group_weights = candidate_weights[group_mask].to(group_logprobs.dtype)
        group_weights = group_weights / group_weights.sum().clamp_min(torch.finfo(group_weights.dtype).tiny)
        losses.append(-torch.logsumexp(torch.log(group_weights.clamp_min(1e-30)) + group_logprobs, dim=0))
    return torch.stack(losses).mean()


def concept_set_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    concept_mask: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
    *,
    length_normalize: bool = False,
) -> torch.Tensor:
    concept_logprobs = causal_concept_logprobs(
        logits,
        input_ids,
        concept_mask,
        length_normalize=length_normalize,
    )
    return concept_set_loss_from_logprobs(concept_logprobs, group_ids, candidate_weights)


def ce_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    concept_mask: torch.Tensor,
    *,
    length_normalize: bool = False,
) -> torch.Tensor:
    concept_logprobs = causal_concept_logprobs(
        logits,
        input_ids,
        concept_mask,
        length_normalize=length_normalize,
    )
    return -concept_logprobs.mean()


def compute_sft_loss(
    loss_type: str,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    concept_mask: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
    *,
    length_normalize: bool = False,
) -> torch.Tensor:
    if loss_type == "concept_set":
        return concept_set_loss(
            logits,
            input_ids,
            concept_mask,
            group_ids,
            candidate_weights,
            length_normalize=length_normalize,
        )
    if loss_type == "ce":
        return ce_loss(logits, input_ids, concept_mask, length_normalize=length_normalize)
    raise NotImplementedError(f"Unsupported loss_type: {loss_type}")
