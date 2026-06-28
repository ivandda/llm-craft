import torch

from src.sft.losses import (
    compute_sft_loss,
    resolve_candidate_loss_config,
    sft_loss_from_logprobs,
)


def test_grouped_ce_matches_uniform_expected_logprob():
    concept_logprobs = torch.tensor([-1.0, -2.0, -0.5])
    group_ids = torch.tensor([0, 0, 1])
    weights = torch.tensor([0.9, 0.1, 1.0])

    got = sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        weights,
        candidate_weighting="uniform",
        candidate_aggregation="expected_logprob",
    )

    expected = torch.tensor(((1.0 + 2.0) / 2.0 + 0.5) / 2.0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_soft_ce_matches_dataset_expected_logprob():
    concept_logprobs = torch.tensor([-1.0, -2.0, -0.5])
    group_ids = torch.tensor([0, 0, 1])
    weights = torch.tensor([3.0, 1.0, 1.0])

    got = sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        weights,
        candidate_weighting="dataset",
        candidate_aggregation="expected_logprob",
    )

    expected = torch.tensor((1.25 + 0.5) / 2.0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_concept_set_uniform_matches_logsumexp_prob():
    concept_logprobs = torch.tensor([-1.0, -2.0, -0.5])
    group_ids = torch.tensor([0, 0, 1])
    weights = torch.tensor([5.0, 1.0, 1.0])

    got = sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        weights,
        candidate_weighting="uniform",
        candidate_aggregation="logsumexp_prob",
    )

    expected_group0 = -torch.log(0.5 * torch.exp(torch.tensor(-1.0)) + 0.5 * torch.exp(torch.tensor(-2.0)))
    expected = (expected_group0 + torch.tensor(0.5)) / 2.0
    assert torch.allclose(got, expected, atol=1e-6)


def test_concept_set_weighted_matches_logsumexp_prob():
    concept_logprobs = torch.tensor([-1.0, -2.0, -0.5])
    group_ids = torch.tensor([0, 0, 1])
    weights = torch.tensor([3.0, 1.0, 1.0])

    got = sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        weights,
        candidate_weighting="dataset",
        candidate_aggregation="logsumexp_prob",
    )

    expected_group0 = -torch.log(0.75 * torch.exp(torch.tensor(-1.0)) + 0.25 * torch.exp(torch.tensor(-2.0)))
    expected = (expected_group0 + torch.tensor(0.5)) / 2.0
    assert torch.allclose(got, expected, atol=1e-6)


def test_candidate_weights_are_normalized_per_group():
    concept_logprobs = torch.tensor([-1.0, -2.0, -0.5])
    group_ids = torch.tensor([0, 0, 1])

    first = sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        torch.tensor([3.0, 1.0, 7.0]),
        candidate_weighting="dataset",
        candidate_aggregation="expected_logprob",
    )
    second = sft_loss_from_logprobs(
        concept_logprobs,
        group_ids,
        torch.tensor([30.0, 10.0, 70.0]),
        candidate_weighting="dataset",
        candidate_aggregation="expected_logprob",
    )

    assert torch.allclose(first, second, atol=1e-6)


def test_compute_sft_loss_alias_ce_matches_grouped_ce():
    batch_size = 3
    seq_len = 4
    vocab_size = 6
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 2, 5],
        ],
        dtype=torch.long,
    )
    concept_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
        ]
    )
    logits = torch.full((batch_size, seq_len, vocab_size), -20.0)
    logits[0, 2, 3] = 0.0
    logits[1, 2, 4] = 0.0
    logits[2, 2, 5] = 0.0
    logits.requires_grad_()
    group_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    weights = torch.tensor([0.8, 0.2, 1.0], dtype=torch.float)

    alias_loss = compute_sft_loss(
        logits,
        input_ids,
        concept_mask,
        group_ids,
        weights,
        loss_type="ce",
    )
    explicit_loss = compute_sft_loss(
        logits,
        input_ids,
        concept_mask,
        group_ids,
        weights,
        candidate_weighting="uniform",
        candidate_aggregation="expected_logprob",
    )
    alias_loss.backward()

    assert torch.allclose(alias_loss, explicit_loss, atol=1e-6)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loss_aliases_include_uniform_concept_set():
    assert resolve_candidate_loss_config(loss_type="concept_set_uniform") == ("uniform", "logsumexp_prob")
