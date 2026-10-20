import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dreamer_core.actor_critic import (
    Actor,
    ActorCriticConfig,
    Critic,
    Imagination,
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


def test_critic_predicts_zero_at_initialization() -> None:
    # Zero output weights give uniform bin probabilities, whose mean over the symmetric bins
    # is zero up to float32 rounding, for any input.
    critic = Critic(feature_dim=10, cfg=SMALL)
    logits = critic(torch.randn(8, 10) * 100)
    assert torch.equal(logits, torch.zeros(8, 255))
    torch.testing.assert_close(critic.value(torch.randn(8, 10)), torch.zeros(8), rtol=0, atol=1e-6)


def test_reinforce_with_normalized_advantages_solves_a_bandit() -> None:
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
    assert set(metrics) >= {"actor_loss", "critic_loss", "return_scale"}


def test_imagine_wires_actions_rewards_and_continuation_to_the_right_states() -> None:
    rssm, reward_head, cont_head, behavior = _setup()
    calls: list[torch.Tensor] = []

    def reward_fn(feature: torch.Tensor) -> torch.Tensor:
        calls.append(feature)
        return reward_head(feature).squeeze(-1)

    start = rssm.initial(4)
    start.stoch[:, :, 0] = 1.0
    traj = behavior.imagine(
        rssm,
        start,
        reward_fn=reward_fn,
        cont_fn=lambda f: torch.sigmoid(cont_head(f)).squeeze(-1),
        generator=torch.Generator().manual_seed(0),
    )
    deter = rssm.cfg.deter
    torch.testing.assert_close(traj.features[:, 0], start.feature(), rtol=0, atol=0)
    with torch.no_grad():
        # The deterministic part of state t + 1 is a function of state t and action t only.
        for t in range(SMALL.horizon):
            prev = traj.features[:, t]
            stoch = prev[:, deter:]
            x = rssm.img_in(torch.cat([stoch, traj.actions[:, t]], -1))
            torch.testing.assert_close(
                rssm.gru(x, prev[:, :deter]), traj.features[:, t + 1, :deter]
            )
        # Reward and continuation of transition t -> t + 1 come from the features of t + 1.
        torch.testing.assert_close(traj.reward, reward_head(traj.features[:, 1:]).squeeze(-1))
        expected_cont = torch.sigmoid(cont_head(traj.features[:, 1:])).squeeze(-1)
        torch.testing.assert_close(traj.cont, expected_cont)
    assert len(calls) == 1
    assert torch.equal(traj.actions.sum(-1), torch.ones(4, SMALL.horizon))


def _symlog_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _twohot_ce_np(logits: np.ndarray, target: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Loop-based cross-entropy between a two-hot target and ``softmax(logits)``."""
    log_p = logits - logits.max(-1, keepdims=True)
    log_p = log_p - np.log(np.exp(log_p).sum(-1, keepdims=True))
    out = np.zeros(target.shape)
    for idx in np.ndindex(*target.shape):
        x = float(np.clip(_symlog_np(target[idx]), bins[0], bins[-1]))
        k = min(int(np.searchsorted(bins, x, side="right")) - 1, len(bins) - 2)
        frac = (x - bins[k]) / (bins[k + 1] - bins[k])
        out[idx] = -((1.0 - frac) * log_p[idx][k] + frac * log_p[idx][k + 1])
    return out


def test_imagination_loss_matches_loop_based_reference() -> None:
    # Fixed features, actions, rewards and continuations; random critic and slow critic heads
    # so that values are non-zero and differ between the two. The reference below rebuilds
    # lambda-returns, the percentile normalizer, advantages, step weights and both losses with
    # explicit loops from the raw network outputs.
    torch.manual_seed(0)
    cfg = ActorCriticConfig(horizon=6, hidden=16, layers=1)
    feature_dim, action_dim, batch, horizon = 5, 3, 4, cfg.horizon
    behavior = ImaginationActorCritic(feature_dim, action_dim, cfg).double()
    with torch.no_grad():
        for module, std in ((behavior.critic, 0.3), (behavior.slow_critic, 0.5)):
            for p in module.parameters():
                p.add_(torch.randn_like(p) * std)
    gen = torch.Generator().manual_seed(1)
    feats = torch.randn(batch, horizon + 1, feature_dim, dtype=torch.float64, generator=gen)
    actions = F.one_hot(torch.randint(0, action_dim, (batch, horizon), generator=gen), action_dim)
    # Large rewards so that the first normalizer update already gives a scale above 1.
    reward = 300.0 * torch.randn(batch, horizon, dtype=torch.float64, generator=gen)
    cont = 0.3 + 0.7 * torch.rand(batch, horizon, dtype=torch.float64, generator=gen)
    traj = Imagination(feats, actions.double(), reward, cont)

    with torch.no_grad():
        v = behavior.critic.value(feats).numpy()
        log_pi = behavior.actor(feats[:, :-1]).numpy()
        critic_logits = behavior.critic(feats[:, :-1]).numpy()
        slow_v = behavior.slow_critic.value(feats[:, :-1]).numpy()
    bins = behavior.critic.twohot.bins.numpy()
    r, c, a = reward.numpy(), cont.numpy(), actions.numpy()

    ret = np.zeros((batch, horizon))
    for b in range(batch):
        last = v[b, horizon]
        for t in reversed(range(horizon)):
            boot = (1 - cfg.lam) * v[b, t + 1] + cfg.lam * last
            last = r[b, t] + cfg.gamma * c[b, t] * boot
            ret[b, t] = last
    low = 0.01 * np.quantile(ret, 0.05)  # EMA with decay 0.99 starting from zero
    high = 0.01 * np.quantile(ret, 0.95)
    scale = max(1.0, high - low)
    assert scale > 1.0  # otherwise the test could not see a missing normalizer update
    adv = (ret - v[:, :-1]) / scale
    weight = np.ones((batch, horizon))
    for t in range(1, horizon):
        weight[:, t] = weight[:, t - 1] * c[:, t - 1]
    entropy = -(np.exp(log_pi) * log_pi).sum(-1)
    log_prob = (a * log_pi).sum(-1)
    actor_ref = (weight * (-adv * log_prob - cfg.entropy_coef * entropy)).mean()
    ce = _twohot_ce_np(critic_logits, ret, bins)
    ce_slow = _twohot_ce_np(critic_logits, slow_v, bins)
    critic_ref = (weight * (ce + cfg.slow_reg * ce_slow)).mean()

    loss, metrics = behavior.loss(traj)
    np.testing.assert_allclose(metrics["actor_loss"].item(), actor_ref, rtol=1e-10)
    np.testing.assert_allclose(metrics["critic_loss"].item(), critic_ref, rtol=1e-10)
    np.testing.assert_allclose(loss.item(), actor_ref + critic_ref, rtol=1e-10)
    np.testing.assert_allclose(behavior.normalizer.low.item(), low, rtol=1e-12)
    np.testing.assert_allclose(behavior.normalizer.high.item(), high, rtol=1e-12)
    np.testing.assert_allclose(metrics["return_scale"].item(), scale, rtol=1e-12)


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
