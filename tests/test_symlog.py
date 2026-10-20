import numpy as np
import pytest
import torch

from dreamer_core.symlog import TwoHot, symexp, symlog

# Values spanning many orders of magnitude, both signs, and exact zero.
VALUES = torch.tensor(
    [0.0, 1e-8, -1e-8, 1e-3, -0.5, 1.0, -1.0, 3.7, -42.0, 1e3, -2.5e4, 1e6, -3e7],
    dtype=torch.float64,
)


def test_symlog_symexp_are_exact_inverses() -> None:
    torch.testing.assert_close(symexp(symlog(VALUES)), VALUES, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(symlog(symexp(VALUES / 1e6)), VALUES / 1e6, rtol=1e-12, atol=0.0)


def test_symlog_is_identity_near_zero_and_odd() -> None:
    small = torch.tensor([1e-6, -1e-6, 1e-4], dtype=torch.float64)
    torch.testing.assert_close(symlog(small), small, rtol=1e-4, atol=0.0)
    torch.testing.assert_close(symlog(-VALUES), -symlog(VALUES))


def test_symlog_matches_closed_form() -> None:
    expected = np.sign(VALUES.numpy()) * np.log1p(np.abs(VALUES.numpy()))
    np.testing.assert_allclose(symlog(VALUES).numpy(), expected, rtol=1e-12)


def _reference_twohot(value: float, bins: np.ndarray) -> np.ndarray:
    """Independent loop-based two-hot in symlog space, written without vectorization."""
    x = float(np.sign(value) * np.log1p(abs(value)))
    x = min(max(x, bins[0]), bins[-1])
    out = np.zeros_like(bins)
    for k in range(len(bins) - 1):
        if bins[k] <= x < bins[k + 1]:
            w_above = (x - bins[k]) / (bins[k + 1] - bins[k])
            out[k] = 1.0 - w_above
            out[k + 1] = w_above
            return out
    out[-1] = 1.0  # x sits exactly on the last bin
    return out


@pytest.mark.parametrize("num_bins", [255, 41, 2])
def test_twohot_matches_independent_reference(num_bins: int) -> None:
    head = TwoHot(num_bins=num_bins)
    weights = head.encode(VALUES)
    bins = head.bins.numpy()
    for value, row in zip(VALUES.tolist(), weights.numpy(), strict=True):
        np.testing.assert_allclose(row, _reference_twohot(value, bins), rtol=0, atol=1e-12)


def test_twohot_weights_sum_to_one_on_adjacent_bins() -> None:
    head = TwoHot()
    gen = torch.Generator().manual_seed(0)
    targets = symexp(torch.empty(2000, dtype=torch.float64).uniform_(-19.9, 19.9, generator=gen))
    weights = head.encode(targets)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2000, dtype=torch.float64))
    assert (weights >= 0).all()
    nonzero = weights > 0
    assert (nonzero.sum(-1) <= 2).all()
    for row in nonzero:
        idx = torch.nonzero(row).flatten()
        if len(idx) == 2:
            assert idx[1] - idx[0] == 1


def test_twohot_decode_of_encode_is_identity() -> None:
    head = TwoHot()
    gen = torch.Generator().manual_seed(1)
    targets = symexp(torch.empty(2000, dtype=torch.float64).uniform_(-19.9, 19.9, generator=gen))
    targets = torch.cat([targets, VALUES])
    torch.testing.assert_close(head.decode(head.encode(targets)), targets, rtol=1e-9, atol=1e-12)


def test_twohot_float32_roundtrip_for_typical_rewards() -> None:
    head = TwoHot()
    targets = torch.linspace(-1000.0, 1000.0, 501)
    torch.testing.assert_close(head.decode(head.encode(targets)), targets, rtol=1e-4, atol=1e-4)


def test_target_on_a_bin_is_single_hot() -> None:
    head = TwoHot(num_bins=41, low=-20.0, high=20.0)  # bins at every integer in symlog space
    target = symexp(torch.tensor([3.0, -20.0, 20.0], dtype=torch.float64))
    weights = head.encode(target)
    assert torch.equal((weights > 0).sum(-1), torch.ones(3, dtype=torch.long))
    assert weights.max(-1).values.eq(1.0).all()


def test_out_of_range_targets_clip_to_edge_bins() -> None:
    head = TwoHot(num_bins=11, low=-2.0, high=2.0)
    weights = head.encode(torch.tensor([1e6, -1e6], dtype=torch.float64))
    assert weights[0, -1] == 1.0
    assert weights[1, 0] == 1.0


def test_loss_gradient_is_softmax_minus_two_hot() -> None:
    # Cross-entropy against a fixed target distribution w has gradient softmax(l) - w, so the
    # loss is minimized exactly when the predicted distribution equals the two-hot target.
    head = TwoHot(num_bins=21, low=-5.0, high=5.0)
    target = torch.tensor([2.3, -0.7, 40.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 21, dtype=torch.float64, generator=gen).requires_grad_(True)
    head.loss(logits, target).sum().backward()
    expected = torch.softmax(logits.detach(), -1) - head.encode(target)
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, expected)


def test_two_hot_gradient_stays_bounded_where_a_squared_error_grows_with_the_target() -> None:
    # The reason for the two-hot head. Its gradient w.r.t. the logits is p - twohot(y): the
    # target part is a distribution on at most two adjacent bins whatever the scale of y, so
    # the gradient has the same structure and size for y = 1 and y = 1e6. A squared error on
    # the raw target has gradient 2 (pred - y), which grows linearly with y.
    head = TwoHot()
    targets = [1.0, 1e3, 1e6]
    twohot_norms, mse_norms = [], []
    for y in targets:
        logits = torch.zeros(1, 255, dtype=torch.float64, requires_grad=True)
        head.loss(logits, torch.tensor([y], dtype=torch.float64)).sum().backward()
        assert logits.grad is not None
        target_part = torch.softmax(logits.detach(), -1) - logits.grad
        support = (target_part[0].abs() > 1e-12).nonzero().flatten()
        assert len(support) <= 2
        assert support.max() - support.min() <= 1
        torch.testing.assert_close(target_part.sum(), torch.tensor(1.0, dtype=torch.float64))
        twohot_norms.append(logits.grad.abs().sum().item())

        pred = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        (pred - y).pow(2).sum().backward()
        assert pred.grad is not None
        mse_norms.append(pred.grad.abs().sum().item())
    assert max(twohot_norms) / min(twohot_norms) < 1.01
    assert mse_norms[-1] / mse_norms[0] > 1e5
