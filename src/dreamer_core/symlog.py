"""Symlog transform and two-hot encoding over symexp-spaced bins.

DreamerV3 predicts rewards and values whose scale is unknown in advance and can differ by
orders of magnitude across environments. Regressing such targets with a squared error lets
large targets dominate the gradient. The two mechanisms here remove that sensitivity:

* ``symlog`` compresses large magnitudes while staying the identity around zero and
  preserving sign, so a network output in symlog space has a bounded, well-conditioned range.
* The two-hot head turns regression into classification over fixed bins. The cross-entropy
  gradient does not scale with the magnitude of the target, which decouples the size of the
  update from the scale of the rewards.

Bins are evenly spaced in symlog space, which makes them exponentially spaced (symexp-spaced)
in the raw space. Interpolation between the two neighboring bins happens in symlog space and
decoding applies ``symexp`` to the expected bin position, following the convention of the
reference implementation (``DiscDist`` in danijar/dreamerv3).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def symlog(x: Tensor) -> Tensor:
    """``sign(x) * log(1 + |x|)``.

    ``log1p`` keeps the transform accurate for tiny inputs, where ``log(1 + x)`` would round
    ``1 + x`` to one and lose the signal entirely.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """``sign(x) * (exp(|x|) - 1)``, the exact inverse of :func:`symlog`."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHot(nn.Module):
    """Two-hot encoder/decoder over ``num_bins`` bins evenly spaced in symlog space.

    The bins are stored in float64 and cast to the input dtype on use, so the same module can
    be used for float32 training and for float64 property tests without losing precision in
    the bin positions themselves.
    """

    bins: Tensor

    def __init__(self, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("two-hot encoding needs at least two bins")
        if not low < high:
            raise ValueError("low must be smaller than high")
        self.num_bins = num_bins
        self.low = low
        self.high = high
        self.register_buffer(
            "bins", torch.linspace(low, high, num_bins, dtype=torch.float64), persistent=False
        )

    def encode(self, target: Tensor) -> Tensor:
        """Return two-hot weights of shape ``(*target.shape, num_bins)``.

        Targets outside the representable range are clipped to the edge bins: a single-hot
        on the edge is the closest distribution the head can express.
        """
        bins = self.bins.to(dtype=target.dtype, device=target.device)
        x = symlog(target).clamp(self.low, self.high)
        below = (bins <= x.unsqueeze(-1)).sum(-1) - 1
        below = below.clamp(0, self.num_bins - 1)
        above = (below + 1).clamp(max=self.num_bins - 1)
        bin_below = bins[below]
        bin_above = bins[above]
        span = bin_above - bin_below
        # When below == above (x on the top edge) the span is zero and all mass goes below.
        safe_span = torch.where(span > 0, span, torch.ones_like(span))
        weight_above = torch.where(span > 0, (x - bin_below) / safe_span, torch.zeros_like(x))
        weight_below = 1.0 - weight_above
        weights = torch.zeros(*x.shape, self.num_bins, dtype=target.dtype, device=target.device)
        weights.scatter_add_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
        weights.scatter_add_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
        return weights

    def decode(self, probs: Tensor) -> Tensor:
        """Map a distribution over bins back to a scalar: ``symexp(sum_k p_k b_k)``."""
        bins = self.bins.to(dtype=probs.dtype, device=probs.device)
        return symexp((probs * bins).sum(-1))

    def mean(self, logits: Tensor) -> Tensor:
        """Decode the prediction of a head that outputs unnormalized logits."""
        return self.decode(torch.softmax(logits, dim=-1))

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """Cross-entropy between the two-hot encoding of ``target`` and ``softmax(logits)``.

        The target is detached: the loss trains the head, it never moves the target.
        """
        weights = self.encode(target.detach().to(logits.dtype))
        return -(weights * F.log_softmax(logits, dim=-1)).sum(-1)
