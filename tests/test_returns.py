import numpy as np
import pytest
import torch

from dreamer_core.returns import ReturnNormalizer, lambda_return


def test_lambda_return_hand_computed_example() -> None:
    # gamma = 0.9, lambda = 0.5, three steps, the episode terminates after the second step.
    reward = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    cont = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    value = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float64)
    # R_3 = v_3 = 40
    # R_2 = 3 + 0.9 * 1 * (0.5 * 40 + 0.5 * 40) = 39
    # R_1 = 2 + 0.9 * 0 * (...)                 = 2
    # R_0 = 1 + 0.9 * 1 * (0.5 * 20 + 0.5 * 2)  = 10.9
    expected = torch.tensor([[10.9, 2.0, 39.0]], dtype=torch.float64)
    got = lambda_return(reward, cont, value, gamma=0.9, lam=0.5)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)


def _gae_reference(reward, cont, value, gamma, lam):
    """Independent formulation via GAE (Schulman et al. 2015): R_t = A_t + V_t."""
    batch, horizon = reward.shape
    out = np.zeros((batch, horizon))
    for b in range(batch):
        adv = 0.0
        for t in reversed(range(horizon)):
            delta = reward[b, t] + gamma * cont[b, t] * value[b, t + 1] - value[b, t]
            adv = delta + gamma * lam * cont[b, t] * adv
            out[b, t] = adv + value[b, t]
    return out


@pytest.mark.parametrize(("gamma", "lam"), [(0.997, 0.95), (0.9, 0.0), (0.99, 1.0), (0.5, 0.7)])
def test_lambda_return_matches_gae_reference(gamma: float, lam: float) -> None:
    rng = np.random.default_rng(0)
    reward = rng.normal(size=(4, 15))
    cont = (rng.random((4, 15)) > 0.2).astype(np.float64)
    value = rng.normal(size=(4, 16))
    got = lambda_return(
        torch.from_numpy(reward), torch.from_numpy(cont), torch.from_numpy(value), gamma, lam
    )
    np.testing.assert_allclose(got.numpy(), _gae_reference(reward, cont, value, gamma, lam))


def test_lambda_zero_is_one_step_td_and_lambda_one_is_monte_carlo() -> None:
    rng = np.random.default_rng(1)
    reward = torch.from_numpy(rng.normal(size=(2, 5)))
    cont = torch.ones(2, 5, dtype=torch.float64)
    value = torch.from_numpy(rng.normal(size=(2, 6)))
    gamma = 0.9
    td = lambda_return(reward, cont, value, gamma, lam=0.0)
    torch.testing.assert_close(td, reward + gamma * value[:, 1:])
    mc = lambda_return(reward, cont, value, gamma, lam=1.0)
    discounts = gamma ** torch.arange(5, dtype=torch.float64)
    expected_first = (reward * discounts).sum(-1) + gamma**5 * value[:, -1]
    torch.testing.assert_close(mc[:, 0], expected_first)


def test_lambda_return_rejects_missing_bootstrap() -> None:
    with pytest.raises(ValueError, match="bootstrap"):
        lambda_return(torch.zeros(1, 3), torch.ones(1, 3), torch.zeros(1, 3))


def test_normalizer_ema_tracks_percentiles_exactly() -> None:
    norm = ReturnNormalizer(decay=0.9)
    returns = torch.arange(101, dtype=torch.float32)  # P5 = 5, P95 = 95
    for _ in range(3):
        norm.update(returns)
    weight = 1 - 0.9**3
    torch.testing.assert_close(norm.low, torch.tensor(5.0 * weight))
    torch.testing.assert_close(norm.high, torch.tensor(95.0 * weight))
    torch.testing.assert_close(norm.scale, torch.tensor(90.0 * weight))


def test_small_returns_are_not_amplified() -> None:
    norm = ReturnNormalizer(decay=0.0)
    norm.update(torch.linspace(0.0, 0.1, 50))
    assert norm.scale == 1.0
    adv = norm.advantage(torch.tensor([0.05]), torch.tensor([0.0]))
    torch.testing.assert_close(adv, torch.tensor([0.05]))


def test_large_returns_are_divided_by_their_percentile_range() -> None:
    norm = ReturnNormalizer(decay=0.0)
    returns = torch.arange(101, dtype=torch.float32) * 1000.0
    norm.update(returns)
    torch.testing.assert_close(norm.scale, torch.tensor(90_000.0))
    adv = norm.advantage(returns, torch.zeros_like(returns))
    assert adv.max() < 1.2  # 100000 / 90000


def test_scale_is_robust_to_a_few_outliers() -> None:
    norm_clean = ReturnNormalizer(decay=0.0)
    norm_outlier = ReturnNormalizer(decay=0.0)
    returns = torch.linspace(0.0, 10.0, 1000)
    with_outliers = returns.clone()
    with_outliers[:5] = 1e6
    norm_clean.update(returns)
    norm_outlier.update(with_outliers)
    torch.testing.assert_close(norm_outlier.scale, norm_clean.scale, rtol=0.02, atol=0.0)
