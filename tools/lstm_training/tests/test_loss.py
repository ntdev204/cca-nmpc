"""Tests for the weighted trajectory loss (Eq. 13.2)."""
import torch

from tools.lstm_training.loss import weighted_trajectory_loss


def test_zero_loss_on_exact_match():
    x = torch.randn(3, 12, 4)
    assert float(weighted_trajectory_loss(x, x)) == 0.0


def test_lambda_scales_velocity_term_only():
    pred = torch.zeros(1, 1, 4)
    target = torch.zeros(1, 1, 4)
    target[..., 2] = 2.0  # velocity error only
    l_half = float(weighted_trajectory_loss(pred, target, lambda_vel=0.5))
    l_one = float(weighted_trajectory_loss(pred, target, lambda_vel=1.0))
    assert l_one > l_half
    assert abs(l_one - 2.0 * l_half) < 1e-6


def test_position_term_independent_of_lambda():
    pred = torch.zeros(1, 1, 4)
    target = torch.zeros(1, 1, 4)
    target[..., 0] = 3.0  # position error only
    l_a = float(weighted_trajectory_loss(pred, target, lambda_vel=0.1))
    l_b = float(weighted_trajectory_loss(pred, target, lambda_vel=5.0))
    assert abs(l_a - l_b) < 1e-6  # velocity term is zero, lambda irrelevant


def test_known_value():
    pred = torch.zeros(1, 1, 4)
    target = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # pos x error = 1
    # MSE_pos = (1^2 + 0)/2 = 0.5 ; MSE_vel = 0 ; loss = 0.5
    assert abs(float(weighted_trajectory_loss(pred, target, 0.5)) - 0.5) < 1e-6


def test_shape_mismatch_raises():
    try:
        weighted_trajectory_loss(torch.zeros(1, 2, 4), torch.zeros(1, 3, 4))
        assert False
    except ValueError:
        pass
