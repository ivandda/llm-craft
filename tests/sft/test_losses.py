import torch

from src.sft.losses import concept_set_loss


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
