"""Recurrent state-space model with discrete latents (DreamerV2/V3).

The latent state has two parts:

* a deterministic recurrent state ``h_t`` (GRU) that carries information across many steps
  without sampling noise, and
* a stochastic state ``z_t`` made of ``stoch`` categorical variables with ``classes`` classes
  each, which represents what the deterministic path cannot predict.

Two networks produce a distribution over ``z_t``: the *prior* ``p(z_t | h_t)`` used during
imagination, and the *posterior* ``q(z_t | h_t, e_t)`` that also sees the embedding ``e_t`` of
the current observation. Training pulls the two together with the balanced KL in
:mod:`dreamer_core.losses`; imagination then only needs the prior.

Time convention: ``prev_action[:, t]`` is the action that led *into* observation ``t``. At the
first step of an episode it is ignored, together with the previous latent state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dreamer_core.distributions import sample_one_hot, unimix_logits
from dreamer_core.nets import MLP, LayerNormGRUCell


@dataclass(frozen=True)
class RSSMConfig:
    """Sizes of the RSSM. Defaults follow the DreamerV3 latent layout (32 x 32 categoricals)."""

    action_dim: int
    embed_dim: int
    deter: int = 512
    stoch: int = 32
    classes: int = 32
    hidden: int = 512
    unimix: float = 0.01


@dataclass
class State:
    """Latent state. Leading dimensions are batch (and optionally time).

    ``logits`` are the unimix-ed log-probabilities of the distribution ``stoch`` was sampled
    from: the prior during imagination, the posterior during observation.
    """

    deter: Tensor  # (..., deter)
    stoch: Tensor  # (..., stoch, classes), one-hot with straight-through gradients
    logits: Tensor  # (..., stoch, classes)

    def feature(self) -> Tensor:
        """Concatenation ``[h, flatten(z)]`` consumed by decoders, heads and the policy."""
        return torch.cat([self.deter, self.stoch.flatten(-2)], dim=-1)

    def detach(self) -> State:
        return State(self.deter.detach(), self.stoch.detach(), self.logits.detach())

    def flatten_time(self) -> State:
        """Merge batch and time so that every posterior can start an imagined rollout."""
        return State(self.deter.flatten(0, 1), self.stoch.flatten(0, 1), self.logits.flatten(0, 1))

    def __getitem__(self, index: int | slice) -> State:
        """Index the time dimension of a stacked ``(batch, time, ...)`` state."""
        return State(self.deter[:, index], self.stoch[:, index], self.logits[:, index])

    @staticmethod
    def stack(states: Sequence[State], dim: int = 1) -> State:
        return State(
            torch.stack([s.deter for s in states], dim),
            torch.stack([s.stoch for s in states], dim),
            torch.stack([s.logits for s in states], dim),
        )


class RSSM(nn.Module):
    """Recurrent state-space model with ``stoch x classes`` categorical latents."""

    def __init__(self, cfg: RSSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        latent = cfg.stoch * cfg.classes
        self.img_in = MLP(latent + cfg.action_dim, cfg.hidden, layers=1)
        self.gru = LayerNormGRUCell(cfg.hidden, cfg.deter)
        self.prior_net = MLP(cfg.deter, cfg.hidden, layers=1, out_dim=latent)
        self.post_net = MLP(cfg.deter + cfg.embed_dim, cfg.hidden, layers=1, out_dim=latent)

    @property
    def feature_dim(self) -> int:
        return self.cfg.deter + self.cfg.stoch * self.cfg.classes

    def initial(self, batch: int, device: torch.device | str = "cpu") -> State:
        """All-zero state. ``logits`` of zero correspond to a uniform distribution."""
        cfg = self.cfg
        return State(
            torch.zeros(batch, cfg.deter, device=device),
            torch.zeros(batch, cfg.stoch, cfg.classes, device=device),
            torch.zeros(batch, cfg.stoch, cfg.classes, device=device),
        )

    def _logits(self, raw: Tensor) -> Tensor:
        return unimix_logits(
            raw.reshape(*raw.shape[:-1], self.cfg.stoch, self.cfg.classes), self.cfg.unimix
        )

    def img_step(
        self, prev: State, prev_action: Tensor, generator: torch.Generator | None = None
    ) -> State:
        """One step of the prior: ``h_t = f(h_{t-1}, z_{t-1}, a_{t-1})``, ``z_t ~ p(z | h_t)``."""
        x = self.img_in(torch.cat([prev.stoch.flatten(-2), prev_action], dim=-1))
        deter = self.gru(x, prev.deter)
        logits = self._logits(self.prior_net(deter))
        return State(deter, sample_one_hot(logits, generator), logits)

    def obs_step(
        self,
        prev: State,
        prev_action: Tensor,
        embed: Tensor,
        is_first: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[State, State]:
        """Return ``(posterior, prior)`` for one observed step.

        Where ``is_first`` is set the previous state and action are replaced by zeros, so an
        episode boundary inside a training sequence does not leak memory across episodes.
        """
        if is_first is not None:
            keep = (1.0 - is_first.to(prev.deter.dtype)).unsqueeze(-1)
            prev = State(prev.deter * keep, prev.stoch * keep.unsqueeze(-1), prev.logits)
            prev_action = prev_action * keep
        prior = self.img_step(prev, prev_action, generator)
        logits = self._logits(self.post_net(torch.cat([prior.deter, embed], dim=-1)))
        post = State(prior.deter, sample_one_hot(logits, generator), logits)
        return post, prior

    def observe(
        self,
        embed: Tensor,
        prev_action: Tensor,
        is_first: Tensor,
        state: State | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[State, State]:
        """Filter a sequence. Inputs are ``(batch, time, ...)``; outputs are stacked states."""
        if state is None:
            state = self.initial(embed.shape[0], embed.device)
        posts, priors = [], []
        for t in range(embed.shape[1]):
            state, prior = self.obs_step(
                state, prev_action[:, t], embed[:, t], is_first[:, t], generator
            )
            posts.append(state)
            priors.append(prior)
        return State.stack(posts), State.stack(priors)

    def imagine(
        self, state: State, actions: Tensor, generator: torch.Generator | None = None
    ) -> State:
        """Open-loop rollout of the prior under a given ``(batch, time, action_dim)`` sequence.

        ``actions[:, t]`` is applied to the state reached after ``t`` steps, so the output at
        index ``t`` is the prior state one step after ``actions[:, t]``.
        """
        states = []
        for t in range(actions.shape[1]):
            state = self.img_step(state, actions[:, t], generator)
            states.append(state)
        return State.stack(states)
