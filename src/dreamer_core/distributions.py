"""Categorical helpers shared by the latent state and the policy.

All functions take *normalized log-probabilities* (the output of :func:`unimix_logits`) or raw
logits; both work because every function re-normalizes with ``log_softmax``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def unimix_logits(logits: Tensor, unimix: float = 0.01) -> Tensor:
    """Mix ``softmax(logits)`` with a uniform distribution and return log-probabilities.

    ``p = (1 - unimix) * softmax(logits) + unimix / K``. Every class keeps probability at
    least ``unimix / K``, so the KL divergence between two such distributions is bounded and
    a confident but wrong prior cannot produce an exploding KL spike. For the policy it keeps
    a floor on exploration without an extra entropy schedule.
    """
    if not 0.0 <= unimix < 1.0:
        raise ValueError("unimix must lie in [0, 1)")
    if unimix == 0.0:
        return F.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    probs = (1.0 - unimix) * probs + unimix / logits.shape[-1]
    return torch.log(probs)


def sample_one_hot(logits: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Draw a one-hot sample with straight-through gradients.

    The forward value is an exact one-hot vector, so downstream networks see the same discrete
    input at training and imagination time. The backward pass uses the gradient of the
    probabilities, ``d sample / d logits := d softmax(logits) / d logits``, which is the
    straight-through estimator: biased, but low variance and cheap, and it lets the
    reconstruction loss train the encoder through a discrete bottleneck.
    """
    probs = torch.softmax(logits, dim=-1)
    flat = probs.detach().reshape(-1, probs.shape[-1])
    index = torch.multinomial(flat, 1, generator=generator).reshape(probs.shape[:-1])
    one_hot = F.one_hot(index, probs.shape[-1]).to(probs.dtype)
    # The parentheses matter: (p - sg(p)) is exactly zero, so the forward value stays an
    # exact one-hot; one_hot + p - sg(p) would round to 1 - eps in floating point.
    return one_hot + (probs - probs.detach())


def mode_one_hot(logits: Tensor) -> Tensor:
    """Most likely class as a one-hot vector, with the same straight-through gradient."""
    probs = torch.softmax(logits, dim=-1)
    one_hot = F.one_hot(probs.argmax(-1), probs.shape[-1]).to(probs.dtype)
    return one_hot + (probs - probs.detach())


def categorical_kl(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """``KL(p || q)`` over the last dimension, from (possibly unnormalized) logits."""
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def categorical_entropy(logits: Tensor) -> Tensor:
    """Entropy over the last dimension, from (possibly unnormalized) logits."""
    log_p = F.log_softmax(logits, dim=-1)
    return -(log_p.exp() * log_p).sum(-1)
