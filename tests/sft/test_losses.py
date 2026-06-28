import torch

from src.sft.losses import (
    concept_set_loss,
    concept_set_loss_from_logprobs,
    soft_ce_loss,
    soft_ce_loss_from_logprobs,
)


def test_concept_set_loss_is_scalar_finite_and_differentiable():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11, requires_grad=True)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 6, 7, 5],
            [1, 8, 9, 10, 5],
        ],
        dtype=torch.long,
    )
    concept_mask = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
            [False, True, True, False, False],
        ]
    )
    group_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    weights = torch.tensor([0.75, 0.25, 1.0], dtype=torch.float)

    loss = concept_set_loss(logits, input_ids, concept_mask, group_ids, weights)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_soft_ce_loss_is_scalar_finite_and_differentiable():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11, requires_grad=True)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 6, 7, 5],
            [1, 8, 9, 10, 5],
        ],
        dtype=torch.long,
    )
    concept_mask = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
            [False, True, True, False, False],
        ]
    )
    group_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    weights = torch.tensor([0.75, 0.25, 1.0], dtype=torch.float)

    loss = soft_ce_loss(logits, input_ids, concept_mask, group_ids, weights)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_soft_ce_matches_manual_formula():
    """Debe dar exactamente mean_grupos( -Sum_k w_k * log p_k ) con pesos normalizados."""
    concept_logprobs = torch.tensor([-1.0, -2.0, -0.5])
    group_ids = torch.tensor([0, 0, 1])
    weights = torch.tensor([3.0, 1.0, 1.0])  # sin normalizar

    got = soft_ce_loss_from_logprobs(concept_logprobs, group_ids, weights)

    # grupo 0: pesos [3,1] -> [0.75,0.25]; -(0.75*-1.0 + 0.25*-2.0) = 1.25
    # grupo 1: peso [1] -> [1.0];          -(1.0*-0.5)             = 0.5
    # promedio = (1.25 + 0.5) / 2 = 0.875
    assert torch.allclose(got, torch.tensor(0.875), atol=1e-6)


def test_soft_ce_is_upper_bound_of_concept_set():
    """Por Jensen: Sum_k w_k(-log p_k) >= -log Sum_k w_k p_k."""
    concept_logprobs = torch.tensor([-1.0, -2.0, -4.0])
    group_ids = torch.tensor([0, 0, 0])
    weights = torch.tensor([3.0, 2.0, 1.0])

    soft = soft_ce_loss_from_logprobs(concept_logprobs, group_ids, weights)
    cset = concept_set_loss_from_logprobs(concept_logprobs, group_ids, weights)

    assert soft >= cset


def test_concept_set_matches_manual_formula():
    """Debe dar mean_grupos( -log Sum_k w_k * p_k ) normalizando los pesos adentro."""
    concept_logprobs = torch.tensor([-1.0, -2.0, -0.5])
    group_ids = torch.tensor([0, 0, 1])
    weights = torch.tensor([3.0, 1.0, 1.0])  # crudos, sin normalizar

    got = concept_set_loss_from_logprobs(concept_logprobs, group_ids, weights)

    # grupo 0: pesos [3,1] -> [0.75,0.25]; -log(0.75*e^-1 + 0.25*e^-2)
    g0 = -torch.log(
        torch.tensor(0.75) * torch.exp(torch.tensor(-1.0))
        + torch.tensor(0.25) * torch.exp(torch.tensor(-2.0))
    )
    # grupo 1: peso [1] -> [1.0]; -log(1.0*e^-0.5) = 0.5
    g1 = torch.tensor(0.5)
    expected = (g0 + g1) / 2

    assert torch.allclose(got, expected, atol=1e-6)
