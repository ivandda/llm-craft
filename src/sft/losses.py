from __future__ import annotations

import torch
import torch.nn.functional as F

LOSS_TYPE_ALIASES: dict[str, tuple[str, str]] = {
    "ce": ("uniform", "expected_logprob"),
    "soft_ce": ("dataset", "expected_logprob"),
    "concept_set": ("dataset", "logsumexp_prob"),
    "concept_set_uniform": ("uniform", "logsumexp_prob"),
}
VALID_CANDIDATE_WEIGHTINGS = {"uniform", "dataset"}
VALID_CANDIDATE_AGGREGATIONS = {"expected_logprob", "logsumexp_prob"}


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


def resolve_candidate_loss_config(
    *,
    loss_type: str | None = None,
    candidate_weighting: str | None = None,
    candidate_aggregation: str | None = None,
) -> tuple[str, str]:
    if candidate_weighting is not None and candidate_weighting not in VALID_CANDIDATE_WEIGHTINGS:
        raise ValueError(
            f"candidate_weighting must be one of: {', '.join(sorted(VALID_CANDIDATE_WEIGHTINGS))}"
        )
    if candidate_aggregation is not None and candidate_aggregation not in VALID_CANDIDATE_AGGREGATIONS:
        raise ValueError(
            f"candidate_aggregation must be one of: {', '.join(sorted(VALID_CANDIDATE_AGGREGATIONS))}"
        )
    if candidate_weighting is not None and candidate_aggregation is not None:
        return candidate_weighting, candidate_aggregation
    if loss_type is None:
        raise ValueError("Either loss_type or both candidate_weighting and candidate_aggregation must be provided.")
    if loss_type not in LOSS_TYPE_ALIASES:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    alias_weighting, alias_aggregation = LOSS_TYPE_ALIASES[loss_type]
    return candidate_weighting or alias_weighting, candidate_aggregation or alias_aggregation


def _normalized_group_weights(
    group_weights: torch.Tensor,
    *,
    candidate_weighting: str,
) -> torch.Tensor:
    if candidate_weighting == "uniform":
        effective_weights = torch.ones_like(group_weights)
    elif candidate_weighting == "dataset":
        effective_weights = group_weights
    else:
        raise ValueError(f"Unsupported candidate_weighting: {candidate_weighting}")
    denom = effective_weights.sum().clamp_min(torch.finfo(effective_weights.dtype).tiny)
    return effective_weights / denom


def sft_loss_from_logprobs(
    concept_logprobs: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
    *,
    candidate_weighting: str,
    candidate_aggregation: str,
) -> torch.Tensor:
    losses = []
    for group_id in torch.unique(group_ids, sorted=True):
        group_mask = group_ids == group_id
        group_logprobs = concept_logprobs[group_mask]
        group_weights = candidate_weights[group_mask].to(group_logprobs.dtype)
        normalized_weights = _normalized_group_weights(
            group_weights,
            candidate_weighting=candidate_weighting,
        )
        if candidate_aggregation == "expected_logprob":
            losses.append(-(normalized_weights * group_logprobs).sum())
            continue
        if candidate_aggregation == "logsumexp_prob":
            losses.append(
                -torch.logsumexp(torch.log(normalized_weights.clamp_min(1e-30)) + group_logprobs, dim=0)
            )
            continue
        raise ValueError(f"Unsupported candidate_aggregation: {candidate_aggregation}")
    return torch.stack(losses).mean()


def compute_sft_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    concept_mask: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
    *,
    candidate_weighting: str | None = None,
    candidate_aggregation: str | None = None,
    loss_type: str | None = None,
    length_normalize: bool = False,
) -> torch.Tensor:
    candidate_weighting, candidate_aggregation = resolve_candidate_loss_config(
        loss_type=loss_type,
        candidate_weighting=candidate_weighting,
        candidate_aggregation=candidate_aggregation,
    )
    concept_logprobs = causal_concept_logprobs(
        logits,
        input_ids,
        concept_mask,
        length_normalize=length_normalize,
    )
    return sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        candidate_weights,
        candidate_weighting=candidate_weighting,
        candidate_aggregation=candidate_aggregation,
    )


def concept_set_loss_from_logprobs(
    concept_logprobs: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> torch.Tensor:
    return sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        candidate_weights,
        candidate_weighting="dataset",
        candidate_aggregation="logsumexp_prob",
    )


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


def soft_ce_loss_from_logprobs(
    concept_logprobs: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> torch.Tensor:
    return sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        candidate_weights,
        candidate_weighting="dataset",
        candidate_aggregation="expected_logprob",
    )


def soft_ce_loss(
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
    return soft_ce_loss_from_logprobs(concept_logprobs, group_ids, candidate_weights)


def ce_loss_from_logprobs(
    concept_logprobs: torch.Tensor,
    group_ids: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> torch.Tensor:
    return sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        candidate_weights,
        candidate_weighting="uniform",
        candidate_aggregation="expected_logprob",
    )


def ce_loss(
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
    return ce_loss_from_logprobs(concept_logprobs, group_ids, candidate_weights)
