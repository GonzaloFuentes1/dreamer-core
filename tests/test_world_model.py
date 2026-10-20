import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dreamer_core.losses import KLConfig, kl_balance
from dreamer_core.toy.env import NUM_ACTIONS, DotGridConfig, collect, sample_windows
from dreamer_core.world_model import WorldModel, WorldModelConfig


def make_model(kl: KLConfig | None = None) -> WorldModel:
    torch.manual_seed(0)
    cfg = WorldModelConfig(
        action_dim=NUM_ACTIONS,
        deter=16,
        stoch=4,
        classes=4,
        hidden=16,
        embed_dim=16,
        cnn_depth=4,
        kl=kl or KLConfig(),
    )
    return WorldModel(cfg)


def make_batch(batch: int = 3, time: int = 6, seed: int = 0) -> dict[str, torch.Tensor]:
    data = collect(DotGridConfig(), 400, np.random.default_rng(seed))
    windows = sample_windows(data, batch, time, np.random.default_rng(seed + 1))
    return {k: torch.from_numpy(v) for k, v in windows.items() if k != "pos"}


def _loss(model: WorldModel, b: dict[str, torch.Tensor], seed: int = 0):
    return model.loss(
        b["image"],
        b["prev_action"],
        b["reward"],
        b["cont"],
        b["is_first"],
        generator=torch.Generator().manual_seed(seed),
    )


def test_loss_terms_match_recomputation_from_components() -> None:
    # No free bits, so the dyn and rep metrics are the raw KL values and depend on the order.
    model = make_model(KLConfig(free_nats=0.0))
    with torch.no_grad():  # the reward head is zero-initialized; make its input matter
        assert model.reward_head.out is not None
        model.reward_head.out.weight.normal_(0.0, 0.1)
    b = make_batch()
    out = _loss(model, b)
    embed = model.encoder(b["image"])
    post, prior = model.rssm.observe(
        embed, b["prev_action"], b["is_first"], generator=torch.Generator().manual_seed(0)
    )
    feat = post.feature()
    image = (model.decoder(feat) - b["image"]).pow(2).sum((-3, -2, -1))
    reward = model.twohot.loss(model.reward_head(feat), b["reward"])
    cont = F.binary_cross_entropy_with_logits(
        model.cont_head(feat).squeeze(-1), b["cont"], reduction="none"
    )
    # Posterior first, prior second: the dynamics term is KL(sg(post) || prior).
    kl = kl_balance(post.logits, prior.logits, model.cfg.kl)
    expected = {
        "loss": (image + reward + cont + kl.loss).mean(),
        "image": image.mean(),
        "reward": reward.mean(),
        "cont": cont.mean(),
        "dyn": kl.dyn.mean(),
        "rep": kl.rep.mean(),
        "kl": kl.kl.mean(),
    }
    for name, value in expected.items():
        torch.testing.assert_close(out.metrics[name], value.detach(), msg=name)
    torch.testing.assert_close(out.loss, expected["loss"])
    assert torch.equal(out.post.stoch, post.stoch)

    # The comparison is sensitive to the KL argument order and to the decoder input.
    swapped = kl_balance(prior.logits, post.logits, model.cfg.kl)
    assert not torch.allclose(swapped.kl.mean(), kl.kl.mean())
    no_stoch = torch.cat([post.deter, torch.zeros_like(post.stoch.flatten(-2))], -1)
    no_stoch_image = (model.decoder(no_stoch) - b["image"]).pow(2).sum((-3, -2, -1))
    assert not torch.allclose(no_stoch_image.mean(), image.mean())


def _post_net_grad(kl: KLConfig, b: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    model = make_model(kl)
    model.zero_grad()
    _loss(model, b).loss.backward()
    return [p.grad.clone() for p in model.rssm.post_net.parameters()]


def test_dynamics_term_does_not_train_the_posterior_inside_the_world_model() -> None:
    # Single-step windows: there is no next recurrent state through which the posterior sample
    # could reach a later prior, so the dynamics term must leave the posterior head untouched.
    b = make_batch(time=1)
    without_kl = _post_net_grad(KLConfig(dyn_scale=0.0, rep_scale=0.0, free_nats=0.0), b)
    with_dyn = _post_net_grad(KLConfig(dyn_scale=1.0, rep_scale=0.0, free_nats=0.0), b)
    with_rep = _post_net_grad(KLConfig(dyn_scale=0.0, rep_scale=1.0, free_nats=0.0), b)
    for g0, g_dyn in zip(without_kl, with_dyn, strict=True):
        torch.testing.assert_close(g_dyn, g0, rtol=0, atol=1e-7)
    assert any(
        not torch.allclose(g0, g_rep) for g0, g_rep in zip(without_kl, with_rep, strict=True)
    )


def test_is_first_resets_the_state_inside_the_world_model_loss() -> None:
    model = make_model()
    a, b = make_batch(seed=0), make_batch(seed=5)
    k = 3
    # Same observations and labels from step k on, different history before it.
    for key in ("image", "reward", "cont"):
        b[key][:, k:] = a[key][:, k:]
    b["prev_action"][:, k + 1 :] = a["prev_action"][:, k + 1 :]
    a["is_first"][:, k] = 1.0
    b["is_first"] = a["is_first"].clone()
    assert not torch.equal(a["image"][:, :k], b["image"][:, :k])
    out_a, out_b = _loss(model, a), _loss(model, b)
    assert not torch.allclose(out_a.post.deter[:, k - 1], out_b.post.deter[:, k - 1])
    assert torch.equal(out_a.post.deter[:, k:], out_b.post.deter[:, k:])
    assert torch.equal(out_a.post.logits[:, k:], out_b.post.logits[:, k:])


@pytest.mark.parametrize("context", [2, 4])
def test_open_loop_decodes_imagination_from_the_last_context_posterior(context: int) -> None:
    model = make_model()
    b = make_batch(time=7)
    pred = model.open_loop(
        b["image"], b["prev_action"], b["is_first"], context, torch.Generator().manual_seed(3)
    )
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        post, _ = model.rssm.observe(
            model.encoder(b["image"][:, :context]),
            b["prev_action"][:, :context],
            b["is_first"][:, :context],
            generator=gen,
        )
        imagined = model.rssm.imagine(post[-1], b["prev_action"][:, context:], gen)
        expected = model.decoder(imagined.feature()).clamp(0.0, 1.0)
    assert pred.shape == (3, 7 - context, 1, 16, 16)
    assert torch.equal(pred, expected)


def test_decoder_output_depends_on_the_stochastic_latent() -> None:
    model = make_model()
    b = make_batch()
    with torch.no_grad():
        post = _loss(model, b).post
        flipped = post.stoch.roll(1, dims=-1)
        feat = torch.cat([post.deter, flipped.flatten(-2)], -1)
        assert not torch.allclose(model.decoder(post.feature()), model.decoder(feat))
