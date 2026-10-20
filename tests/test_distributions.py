import pytest
import torch
import torch.nn.functional as F

from dreamer_core.distributions import (
    categorical_entropy,
    categorical_kl,
    mode_one_hot,
    sample_one_hot,
    unimix_logits,
)


@pytest.mark.parametrize("unimix", [0.01, 0.1])
def test_unimix_lower_bounds_every_probability(unimix: float) -> None:
    classes = 32
    logits = torch.tensor([[1e4] + [-1e4] * (classes - 1)], dtype=torch.float64)
    probs = unimix_logits(logits, unimix).exp()
    assert (probs >= unimix / classes - 1e-15).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(1, dtype=torch.float64))


def test_unimix_bounds_the_kl_between_extreme_distributions() -> None:
    # Two maximally confident, disagreeing distributions: without unimix the KL is ~2e4 nats,
    # with 1% unimix it is bounded by log(K / unimix).
    classes = 32
    p = torch.full((classes,), -1e4, dtype=torch.float64)
    q = p.clone()
    p[0] = q[1] = 1e4
    raw = categorical_kl(p, q)
    mixed = categorical_kl(unimix_logits(p, 0.01), unimix_logits(q, 0.01))
    assert raw > 1e3
    assert mixed <= torch.log(torch.tensor(classes / 0.01, dtype=torch.float64))


def test_unimix_zero_is_plain_softmax() -> None:
    logits = torch.randn(4, 7, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(unimix_logits(logits, 0.0), F.log_softmax(logits, -1))


def test_sample_forward_value_is_exactly_one_hot() -> None:
    logits = torch.randn(64, 32, 16, generator=torch.Generator().manual_seed(0))
    sample = sample_one_hot(logits, torch.Generator().manual_seed(1))
    assert torch.equal(sample.sum(-1), torch.ones(64, 32))
    assert torch.equal(sample, sample.round())


def test_straight_through_gradient_equals_softmax_jacobian() -> None:
    # d sample / d logits must be the Jacobian of softmax, evaluated at the same logits,
    # even though the forward value is a non-differentiable one-hot draw.
    logits = torch.randn(5, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    weights = torch.randn(5, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(1))

    leaf = logits.clone().requires_grad_(True)
    (sample_one_hot(leaf, torch.Generator().manual_seed(2)) * weights).sum().backward()
    assert leaf.grad is not None

    ref = logits.clone().requires_grad_(True)
    (torch.softmax(ref, -1) * weights).sum().backward()
    assert ref.grad is not None
    torch.testing.assert_close(leaf.grad, ref.grad, rtol=0, atol=1e-14)

    # And explicitly against the closed form J = diag(p) - p p^T.
    p = torch.softmax(logits, -1)
    jac = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    torch.testing.assert_close(leaf.grad, (jac @ weights.unsqueeze(-1)).squeeze(-1))


def test_mode_has_straight_through_gradient() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0]], requires_grad=True)
    mode = mode_one_hot(logits)
    assert torch.equal(mode.detach(), torch.tensor([[0.0, 1.0, 0.0]]))
    mode[0, 1].backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_sample_frequencies_match_probabilities() -> None:
    probs = torch.tensor([0.05, 0.15, 0.3, 0.5])
    logits = probs.log().expand(200_000, 4)
    counts = sample_one_hot(logits, torch.Generator().manual_seed(0)).sum(0)
    freq = counts / counts.sum()
    # Four standard deviations of a binomial proportion with n = 2e5.
    tol = 4 * torch.sqrt(probs * (1 - probs) / 200_000)
    assert ((freq - probs).abs() < tol).all()


def test_kl_and_entropy_match_torch_distributions() -> None:
    gen = torch.Generator().manual_seed(0)
    p = torch.randn(10, 6, dtype=torch.float64, generator=gen)
    q = torch.randn(10, 6, dtype=torch.float64, generator=gen)
    dist_p = torch.distributions.Categorical(logits=p)
    dist_q = torch.distributions.Categorical(logits=q)
    torch.testing.assert_close(
        categorical_kl(p, q), torch.distributions.kl_divergence(dist_p, dist_q)
    )
    torch.testing.assert_close(categorical_entropy(p), dist_p.entropy())
