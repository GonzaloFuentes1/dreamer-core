"""KL balancing with free bits.

A single ``KL(q || p)`` term trains the posterior ``q`` and the prior ``p`` towards each other
at the same rate. That is undesirable: the posterior can collapse towards an uninformative
prior faster than the prior learns to predict, and the representation stops carrying
information about the observation. DreamerV3 splits the KL into two terms with stop-gradients:

* dynamics loss ``L_dyn = max(free, KL(sg(q) || p))`` trains only the prior to predict the
  posterior, with a larger weight (0.5);
* representation loss ``L_rep = max(free, KL(q || sg(p)))`` trains only the posterior (and the
  encoder behind it) to be predictable, with a smaller weight (0.1).

Free bits clip each term below ``free`` nats (1 by default). Once the KL is already small,
pushing it further only removes information from the latent; below the threshold the
gradient is exactly zero and the reconstruction loss is free to shape the representation.
The clip is applied to the KL summed over all latent variables, per batch and time step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from dreamer_core.distributions import categorical_kl


@dataclass(frozen=True)
class KLConfig:
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    free_nats: float = 1.0


@dataclass
class KLTerms:
    """Per-element terms of shape ``(batch, time)``; reduce them in the caller."""

    loss: Tensor  # dyn_scale * dyn + rep_scale * rep
    dyn: Tensor  # clipped dynamics term (trains the prior)
    rep: Tensor  # clipped representation term (trains the posterior)
    kl: Tensor  # unclipped KL(q || p), for logging


def kl_balance(post_logits: Tensor, prior_logits: Tensor, cfg: KLConfig | None = None) -> KLTerms:
    """Balanced KL between posterior and prior logits of shape ``(..., stoch, classes)``."""
    cfg = cfg or KLConfig()
    dyn_kl = categorical_kl(post_logits.detach(), prior_logits).sum(-1)
    rep_kl = categorical_kl(post_logits, prior_logits.detach()).sum(-1)
    free = torch.as_tensor(cfg.free_nats, dtype=dyn_kl.dtype, device=dyn_kl.device)
    dyn = torch.maximum(dyn_kl, free)
    rep = torch.maximum(rep_kl, free)
    return KLTerms(
        loss=cfg.dyn_scale * dyn + cfg.rep_scale * rep,
        dyn=dyn,
        rep=rep,
        kl=dyn_kl.detach(),
    )
