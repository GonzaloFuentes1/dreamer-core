import torch
import torch.nn.functional as F
from torch import nn

from dreamer_core.actor_critic import (
    Actor,
    ActorCriticConfig,
    Critic,
    ImaginationActorCritic,
    ema_update,
    reinforce_loss,
)
from dreamer_core.distributions import categorical_entropy, sample_one_hot
from dreamer_core.returns import ReturnNormalizer
from dreamer_core.rssm import RSSM, RSSMConfig

SMALL = ActorCriticConfig(horizon=5, hidden=16, layers=1)


def test_reinforce_gradient_matches_score_function_closed_form() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 4, dtype=torch.float64, generator=gen).requires_grad_(True)
    actions = F.one_hot(torch.randint(0, 4, (6,), generator=gen), 4).double()
    advantage = torch.randn(6, dtype=torch.float64, generator=gen)
    reinforce_loss(logits, actions, advantage, entropy_coef=0.0).sum().backward()
    probs = torch.softmax(logits.detach(), -1)
    expected = -advantage.unsqueeze(-1) * (actions - probs)  # d log pi(a) / d logits = a - p
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, expected)


def test_reinforce_does_not_backpropagate_into_the_advantage() -> None:
    logits = torch.zeros(3, 2, requires_grad=True)
    advantage = torch.ones(3, requires_grad=True)
    actions = F.one_hot(torch.tensor([0, 1, 0]), 2).float()
    reinforce_loss(logits, actions, advantage, entropy_coef=0.1).sum().backward()
    assert advantage.grad is None


def test_entropy_bonus_increases_entropy_when_advantage_is_zero() -> None:
    logits = torch.tensor([[3.0, 0.0, -1.0]], requires_grad=True)
    actions = torch.tensor([[1.0, 0.0, 0.0]])
    before = categorical_entropy(logits).item()
    opt = torch.optim.SGD([logits], lr=1.0)
    for _ in range(20):
        opt.zero_grad()
        reinforce_loss(logits, actions, torch.zeros(1), entropy_coef=0.1).sum().backward()
        opt.step()
    assert categorical_entropy(logits).item() > before


def test_ema_update_matches_formula() -> None:
    torch.manual_seed(0)
    source, target = nn.Linear(3, 2), nn.Linear(3, 2)
    expected = [
        0.9 * t.detach() + 0.1 * s.detach()
        for t, s in zip(target.parameters(), source.parameters(), strict=True)
    ]
    ema_update(target, source, decay=0.9)
    for got, want in zip(target.parameters(), expected, strict=True):
        torch.testing.assert_close(got.detach(), want)


def test_critic_predicts_zero_at_initialisation() -> None:
    # Zero output weights give uniform bin probabilities, whose mean over the symmetric bins
    # is zero up to float32 rounding, for any input.
    critic = Critic(feature_dim=10, cfg=SMALL)
    logits = critic(torch.randn(8, 10) * 100)
    assert torch.equal(logits, torch.zeros(8, 255))
    torch.testing.assert_close(critic.value(torch.randn(8, 10)), torch.zeros(8), rtol=0, atol=1e-6)


def test_reinforce_with_normalised_advantages_solves_a_bandit() -> None:
    # Arm 0 pays 1, arm 1 pays 0: the policy must concentrate on arm 0.
    torch.manual_seed(0)
    actor = Actor(feature_dim=1, action_dim=2, cfg=SMALL)
    norm = ReturnNormalizer(decay=0.9)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    feature = torch.ones(256, 1)
    for _ in range(150):
        logits = actor(feature)
        actions = sample_one_hot(logits, gen).detach()
        reward = actions[:, 0]
        norm.update(reward)
        advantage = norm.advantage(reward, reward.mean().expand_as(reward))
        loss = reinforce_loss(logits, actions, advantage, entropy_coef=3e-4).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    prob_best = actor(feature[:1]).exp()[0, 0].item()
    assert prob_best > 0.95


def _setup() -> tuple[RSSM, nn.Module, nn.Module, ImaginationActorCritic]:
    torch.manual_seed(0)
    rssm = RSSM(RSSMConfig(action_dim=3, embed_dim=4, deter=8, stoch=3, classes=4, hidden=8))
    reward_head = nn.Linear(rssm.feature_dim, 1)
    cont_head = nn.Linear(rssm.feature_dim, 1)
    behavior = ImaginationActorCritic(rssm.feature_dim, 3, SMALL)
    return rssm, reward_head, cont_head, behavior


def test_imagination_losses_train_only_actor_and_critic() -> None:
    rssm, reward_head, cont_head, behavior = _setup()
    start = rssm.initial(6)
    traj = behavior.imagine(
        rssm,
        start,
        reward_fn=lambda f: reward_head(f).squeeze(-1),
        cont_fn=lambda f: torch.sigmoid(cont_head(f)).squeeze(-1),
        generator=torch.Generator().manual_seed(0),
    )
    assert traj.features.shape == (6, SMALL.horizon + 1, rssm.feature_dim)
    assert traj.actions.shape == (6, SMALL.horizon, 3)
    loss, metrics = behavior.loss(traj)
    assert torch.isfinite(loss)
    loss.backward()
    for module in (rssm, reward_head, cont_head, behavior.slow_critic):
        assert all(p.grad is None for p in module.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in behavior.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in behavior.critic.parameters())
    assert metrics["return_scale"] >= 1.0


def test_slow_critic_follows_critic_with_ema() -> None:
    _, _, _, behavior = _setup()
    with torch.no_grad():
        for p in behavior.critic.parameters():
            p.add_(1.0)
    before = [p.clone() for p in behavior.slow_critic.parameters()]
    behavior.update_slow_critic()
    decay = SMALL.slow_decay
    for slow, old, fast in zip(
        behavior.slow_critic.parameters(), before, behavior.critic.parameters(), strict=True
    ):
        torch.testing.assert_close(slow, decay * old + (1 - decay) * fast.detach())
