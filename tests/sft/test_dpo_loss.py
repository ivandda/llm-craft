"""Offline unit tests for the DPO loss (no model, CPU, deterministic)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.sft.dpo import compute_dpo_loss_components


def _logps(chosen_pol, rej_pol, chosen_ref, rej_ref):
    return (
        torch.tensor(chosen_pol, dtype=torch.float32),
        torch.tensor(rej_pol, dtype=torch.float32),
        torch.tensor(chosen_ref, dtype=torch.float32),
        torch.tensor(rej_ref, dtype=torch.float32),
    )


def test_loss_matches_neg_logsigmoid_formula():
    pc, pr, rc, rr = _logps([-2.0], [-5.0], [-3.0], [-3.0])  # policy Δ=3, ref Δ=0
    beta = 0.1
    out = compute_dpo_loss_components(pc, pr, rc, rr, beta=beta)
    z = beta * ((-2.0 - -5.0) - (-3.0 - -3.0))  # 0.1 * 3 = 0.3
    expected = -math.log(1.0 / (1.0 + math.exp(-z)))
    assert math.isclose(out.total_loss.item(), expected, rel_tol=1e-6)


def test_reward_margin_and_accuracy_positive_when_policy_prefers_chosen():
    # policy prefers chosen (Δ=4) more than reference (Δ=1) -> positive margin.
    pc, pr, rc, rr = _logps([-1.0], [-5.0], [-2.0], [-3.0])
    out = compute_dpo_loss_components(pc, pr, rc, rr, beta=0.1)
    assert out.reward_margin.item() > 0
    assert out.reward_accuracy.item() == 1.0


def test_reward_accuracy_zero_when_policy_prefers_rejected_relative_to_ref():
    # policy Δ=1 but reference Δ=3 -> relative preference is negative.
    pc, pr, rc, rr = _logps([-2.0], [-3.0], [-1.0], [-4.0])
    out = compute_dpo_loss_components(pc, pr, rc, rr, beta=0.1)
    assert out.reward_accuracy.item() == 0.0
    assert out.reward_margin.item() < 0


def test_loss_decreases_as_chosen_logp_increases():
    _, pr, rc, rr = _logps([0.0], [-5.0], [-3.0], [-3.0])
    low = compute_dpo_loss_components(torch.tensor([-4.0]), pr, rc, rr, beta=0.1).total_loss
    high = compute_dpo_loss_components(torch.tensor([-1.0]), pr, rc, rr, beta=0.1).total_loss
    assert high.item() < low.item()  # higher chosen logp -> lower loss


def test_reference_gets_no_grad_policy_does():
    pc = torch.tensor([-2.0], requires_grad=True)
    pr = torch.tensor([-5.0], requires_grad=True)
    rc = torch.tensor([-3.0])  # reference: no grad (detached constant)
    rr = torch.tensor([-3.0])
    out = compute_dpo_loss_components(pc, pr, rc, rr, beta=0.1)
    out.total_loss.backward()
    assert pc.grad is not None and pr.grad is not None
    assert rc.grad is None and rr.grad is None


def test_numerically_stable_for_extreme_margins():
    pc, pr, rc, rr = _logps([100.0], [-100.0], [0.0], [0.0])
    out = compute_dpo_loss_components(pc, pr, rc, rr, beta=1.0)
    assert torch.isfinite(out.total_loss)
    assert out.total_loss.item() >= 0.0
    # huge negative margin should also stay finite
    out2 = compute_dpo_loss_components(pr, pc, rc, rr, beta=1.0)
    assert torch.isfinite(out2.total_loss)


def test_label_smoothing_pulls_loss_toward_symmetric():
    pc, pr, rc, rr = _logps([-1.0], [-5.0], [-3.0], [-3.0])
    plain = compute_dpo_loss_components(pc, pr, rc, rr, beta=0.5, label_smoothing=0.0).total_loss
    smoothed = compute_dpo_loss_components(pc, pr, rc, rr, beta=0.5, label_smoothing=0.2).total_loss
    # with a confidently-correct pair, smoothing raises the loss (hedges the label)
    assert smoothed.item() > plain.item()


def test_reference_free_ignores_reference():
    pc, pr = torch.tensor([-1.0]), torch.tensor([-5.0])
    a = compute_dpo_loss_components(pc, pr, torch.tensor([-9.0]), torch.tensor([0.0]), beta=0.1, reference_free=True)
    b = compute_dpo_loss_components(pc, pr, torch.tensor([1.0]), torch.tensor([2.0]), beta=0.1, reference_free=True)
    assert math.isclose(a.total_loss.item(), b.total_loss.item(), rel_tol=1e-6)


def test_batch_mean_over_pairs():
    pc, pr, rc, rr = _logps([-1.0, -2.0], [-5.0, -6.0], [-3.0, -3.0], [-3.0, -3.0])
    out = compute_dpo_loss_components(pc, pr, rc, rr, beta=0.1)
    beta = 0.1
    z0 = beta * (4.0)
    z1 = beta * (4.0)
    expected = (-F.logsigmoid(torch.tensor(z0)) - F.logsigmoid(torch.tensor(z1))) / 2
    assert math.isclose(out.total_loss.item(), expected.item(), rel_tol=1e-6)
