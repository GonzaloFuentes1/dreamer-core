import math

import torch

from dreamer_core.nets import MLP, LayerNormGRUCell


def _reference_gru(cell: LayerNormGRUCell, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Hand-written LayerNorm GRU: ``LN(W [x, h])`` split into reset, candidate and update."""
    z = torch.cat([x, h], -1) @ cell.linear.weight.T
    mean = z.mean(-1, keepdim=True)
    var = ((z - mean) ** 2).mean(-1, keepdim=True)
    z = (z - mean) / torch.sqrt(var + cell.norm.eps) * cell.norm.weight + cell.norm.bias
    n = cell.hidden
    reset, cand, update = z[..., :n], z[..., n : 2 * n], z[..., 2 * n :]
    cand = torch.tanh(torch.sigmoid(reset) * cand)
    update = torch.sigmoid(update - 1.0)
    return update * cand + (1.0 - update) * h


def test_layernorm_gru_matches_hand_written_formula() -> None:
    torch.manual_seed(0)
    cell = LayerNormGRUCell(in_dim=5, hidden=7).double()
    with torch.no_grad():  # non-trivial LayerNorm affine parameters
        cell.norm.weight.copy_(1.0 + 0.5 * torch.randn(21, dtype=torch.float64))
        cell.norm.bias.copy_(0.5 * torch.randn(21, dtype=torch.float64))
    x = torch.randn(4, 5, dtype=torch.float64)
    h = torch.randn(4, 7, dtype=torch.float64)
    torch.testing.assert_close(cell(x, h), _reference_gru(cell, x, h), rtol=0, atol=1e-14)


def test_gru_keeps_most_of_the_state_when_pre_activations_are_zero() -> None:
    # With zero gate pre-activations the update gate is sigmoid(0 - 1) and the candidate is
    # tanh(0) = 0, so the output keeps a fraction sigmoid(1) ~ 0.73 of the previous state.
    cell = LayerNormGRUCell(in_dim=3, hidden=6).double()
    with torch.no_grad():
        cell.linear.weight.zero_()
    h = torch.randn(2, 6, dtype=torch.float64)
    out = cell(torch.randn(2, 3, dtype=torch.float64), h)
    keep = 1.0 / (1.0 + math.exp(-1.0))
    torch.testing.assert_close(out, keep * h, rtol=0, atol=1e-15)
    assert 0.73 < keep < 0.74


def test_mlp_zero_init_output_predicts_exact_zeros() -> None:
    mlp = MLP(4, 8, layers=2, out_dim=3, zero_init_output=True)
    assert torch.equal(mlp(torch.randn(5, 4) * 100), torch.zeros(5, 3))
    assert mlp.out_dim == 3
    assert MLP(4, 8, layers=2).out_dim == 8
