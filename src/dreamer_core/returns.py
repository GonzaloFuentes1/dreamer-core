"""Lambda-returns and percentile return normalization.

Lambda-returns trade off the bias of bootstrapping from a learned critic against the variance
of long Monte-Carlo sums of predicted rewards; imagined trajectories are short (15 steps in
DreamerV3), so bootstrapping at the horizon is required anyway.

Return normalization makes a single entropy coefficient work across reward scales. Dividing
advantages by the spread of returns keeps the policy-gradient magnitude roughly constant, and
the ``max(1, S)`` floor avoids *amplifying* noise when returns are small, which is what a
plain standard-deviation normalization would do in sparse-reward tasks. Percentiles instead of
min/max make the scale robust to a few outlier returns.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def lambda_return(
    reward: Tensor,
    cont: Tensor,
    value: Tensor,
    gamma: float = 0.997,
    lam: float = 0.95,
) -> Tensor:
    """Compute ``R_t = r_t + gamma * c_t * ((1 - lam) * v_{t+1} + lam * R_{t+1})``.

    Shapes: ``reward`` and ``cont`` are ``(batch, T)`` and describe the transition from step
    ``t`` to ``t + 1``; ``value`` is ``(batch, T + 1)`` with the bootstrap ``v_T`` in the last
    column, which also initializes the recursion (``R_T = v_T``). ``cont`` is the probability
    (or indicator) that the episode continues, so a terminal transition cuts the bootstrap.
    """
    if value.shape[1] != reward.shape[1] + 1:
        raise ValueError("value needs one more time step than reward (the bootstrap)")
    returns = []
    last = value[:, -1]
    for t in reversed(range(reward.shape[1])):
        last = reward[:, t] + gamma * cont[:, t] * ((1.0 - lam) * value[:, t + 1] + lam * last)
        returns.append(last)
    return torch.stack(returns[::-1], dim=1)


class ReturnNormalizer(nn.Module):
    """EMA of the 5th and 95th return percentiles, used as ``S = max(1, P95 - P5)``.

    The two percentiles are tracked separately with an exponential moving average, as in the
    reference implementation, so a single batch with unusual returns moves the scale only by
    ``1 - decay``.
    """

    low: Tensor
    high: Tensor

    def __init__(
        self,
        decay: float = 0.99,
        limit: float = 1.0,
        low_quantile: float = 0.05,
        high_quantile: float = 0.95,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.limit = limit
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.register_buffer("low", torch.zeros(()))
        self.register_buffer("high", torch.zeros(()))

    @torch.no_grad()
    def update(self, returns: Tensor) -> None:
        flat = returns.detach().flatten().to(self.low.dtype)
        low = torch.quantile(flat, self.low_quantile)
        high = torch.quantile(flat, self.high_quantile)
        self.low.mul_(self.decay).add_((1.0 - self.decay) * low)
        self.high.mul_(self.decay).add_((1.0 - self.decay) * high)

    @property
    def scale(self) -> Tensor:
        return torch.clamp(self.high - self.low, min=self.limit)

    def advantage(self, returns: Tensor, baseline: Tensor) -> Tensor:
        """Normalized advantage ``(R - v) / max(1, S)`` (no offset: it cancels in R - v)."""
        return (returns - baseline) / self.scale.to(returns.dtype)
