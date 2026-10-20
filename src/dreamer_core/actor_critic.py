"""Actor-critic trained purely in imagination (discrete actions).

The world model is frozen from the point of view of these losses: trajectories are imagined
under ``torch.no_grad`` and only the actor and critic parameters receive gradients.

* Actor: REINFORCE with normalized advantages and an entropy bonus. For discrete actions the
  score-function estimator is unbiased and does not require differentiating through the
  learned dynamics, which is why DreamerV3 uses it instead of backpropagating returns.
* Critic: a two-hot head over symlog bins (the same head as the reward predictor), trained on
  lambda-returns, and regularized toward the predictions of a slow EMA copy of itself so the
  value targets do not chase their own fast-moving estimates.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dreamer_core.distributions import categorical_entropy, sample_one_hot, unimix_logits
from dreamer_core.nets import MLP
from dreamer_core.returns import ReturnNormalizer, lambda_return
from dreamer_core.rssm import RSSM, State
from dreamer_core.symlog import TwoHot


@dataclass(frozen=True)
class ActorCriticConfig:
    horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    entropy_coef: float = 3e-4
    slow_decay: float = 0.98
    slow_reg: float = 1.0
    unimix: float = 0.01
    hidden: int = 256
    layers: int = 2


class Actor(nn.Module):
    """Categorical policy over discrete actions with a unimix floor."""

    def __init__(self, feature_dim: int, action_dim: int, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.net = MLP(feature_dim, cfg.hidden, cfg.layers, out_dim=action_dim)
        self.unimix = cfg.unimix

    def forward(self, feature: Tensor) -> Tensor:
        """Return normalized action log-probabilities."""
        return unimix_logits(self.net(feature), self.unimix)


class Critic(nn.Module):
    """Two-hot value head. The output layer is zero-initialized so ``V = 0`` at the start."""

    def __init__(self, feature_dim: int, cfg: ActorCriticConfig, num_bins: int = 255) -> None:
        super().__init__()
        self.net = MLP(feature_dim, cfg.hidden, cfg.layers, out_dim=num_bins, zero_init_output=True)
        self.twohot = TwoHot(num_bins)

    def forward(self, feature: Tensor) -> Tensor:
        return self.net(feature)

    def value(self, feature: Tensor) -> Tensor:
        return self.twohot.mean(self.net(feature))


def reinforce_loss(
    logits: Tensor, actions: Tensor, advantage: Tensor, entropy_coef: float
) -> Tensor:
    """Per-step ``-sg(A) * log pi(a|s) - eta * H(pi(.|s))``.

    The advantage is detached inside the function so a caller cannot accidentally train the
    critic or the world model through the policy-gradient term.
    """
    log_prob = (actions * F.log_softmax(logits, dim=-1)).sum(-1)
    return -advantage.detach() * log_prob - entropy_coef * categorical_entropy(logits)


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    """``theta_target <- decay * theta_target + (1 - decay) * theta_source``."""
    for t, s in zip(target.parameters(), source.parameters(), strict=True):
        t.lerp_(s, 1.0 - decay)


@dataclass
class Imagination:
    """An imagined rollout: features for ``H + 1`` states and the ``H`` actions between them."""

    features: Tensor  # (batch, H + 1, feature_dim)
    actions: Tensor  # (batch, H, action_dim)
    reward: Tensor  # (batch, H): reward of transition t -> t + 1
    cont: Tensor  # (batch, H): continuation probability after transition t -> t + 1


class ImaginationActorCritic(nn.Module):
    """Actor, critic, slow critic and return normalizer, trained from imagined rollouts."""

    def __init__(self, feature_dim: int, action_dim: int, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.actor = Actor(feature_dim, action_dim, cfg)
        self.critic = Critic(feature_dim, cfg)
        self.slow_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.normalizer = ReturnNormalizer()

    @torch.no_grad()
    def imagine(
        self,
        rssm: RSSM,
        start: State,
        reward_fn: Callable[[Tensor], Tensor],
        cont_fn: Callable[[Tensor], Tensor],
        generator: torch.Generator | None = None,
    ) -> Imagination:
        """Roll the prior forward ``horizon`` steps with actions sampled from the actor."""
        state = start.detach()
        features = [state.feature()]
        actions = []
        for _ in range(self.cfg.horizon):
            action = sample_one_hot(self.actor(features[-1]), generator)
            state = rssm.img_step(state, action, generator)
            features.append(state.feature())
            actions.append(action)
        feats = torch.stack(features, 1)
        return Imagination(
            features=feats,
            actions=torch.stack(actions, 1),
            reward=reward_fn(feats[:, 1:]),
            cont=cont_fn(feats[:, 1:]),
        )

    def loss(self, traj: Imagination) -> tuple[Tensor, dict[str, Tensor]]:
        """Actor and critic losses for one imagined batch. Updates the return normalizer."""
        cfg = self.cfg
        feats = traj.features.detach()
        with torch.no_grad():
            value = self.critic.value(feats)
            returns = lambda_return(traj.reward, traj.cont, value, cfg.gamma, cfg.lam)
            # Weight of step t: probability that the imagined episode is still running.
            alive = torch.cumprod(torch.cat([torch.ones_like(traj.cont[:, :1]), traj.cont], 1), 1)
            weight = alive[:, :-1]
            self.normalizer.update(returns)
            advantage = self.normalizer.advantage(returns, value[:, :-1])
            slow_value = self.slow_critic.value(feats[:, :-1])

        logits = self.actor(feats[:, :-1])
        actor_loss = (
            weight * reinforce_loss(logits, traj.actions, advantage, cfg.entropy_coef)
        ).mean()

        critic_logits = self.critic(feats[:, :-1])
        twohot = self.critic.twohot
        critic_loss = twohot.loss(critic_logits, returns) + cfg.slow_reg * twohot.loss(
            critic_logits, slow_value
        )
        critic_loss = (weight * critic_loss).mean()

        metrics = {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "return_mean": returns.mean(),
            "return_scale": self.normalizer.scale.detach(),
            "policy_entropy": categorical_entropy(logits).mean().detach(),
        }
        return actor_loss + critic_loss, metrics

    def update_slow_critic(self) -> None:
        ema_update(self.slow_critic, self.critic, self.cfg.slow_decay)
