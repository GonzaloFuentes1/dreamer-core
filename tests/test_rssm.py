import pytest
import torch
import torch.nn.functional as F

from dreamer_core.rssm import RSSM, RSSMConfig, State


def make_rssm(stoch: int = 4, classes: int = 5, seed: int = 0) -> RSSM:
    torch.manual_seed(seed)
    cfg = RSSMConfig(action_dim=3, embed_dim=7, deter=16, stoch=stoch, classes=classes, hidden=12)
    return RSSM(cfg)


def make_inputs(batch: int = 2, time: int = 6, seed: int = 1) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)
    embed = torch.randn(batch, time, 7, generator=gen)
    actions = F.one_hot(torch.randint(0, 3, (batch, time), generator=gen), 3).float()
    is_first = torch.zeros(batch, time)
    is_first[:, 0] = 1.0
    return embed, actions, is_first


@pytest.mark.parametrize(("stoch", "classes"), [(4, 5), (32, 32), (1, 2)])
def test_observe_and_imagine_shapes(stoch: int, classes: int) -> None:
    rssm = make_rssm(stoch, classes)
    embed, actions, is_first = make_inputs()
    post, prior = rssm.observe(embed, actions, is_first)
    for state in (post, prior):
        assert state.deter.shape == (2, 6, 16)
        assert state.stoch.shape == (2, 6, stoch, classes)
        assert state.logits.shape == (2, 6, stoch, classes)
        assert state.feature().shape == (2, 6, rssm.feature_dim)
    imagined = rssm.imagine(post[-1], actions)
    assert imagined.stoch.shape == (2, 6, stoch, classes)
    assert post.flatten_time().deter.shape == (12, 16)


def test_latent_distributions_respect_unimix_floor() -> None:
    rssm = make_rssm()
    # Push the prior and posterior heads to extreme logits.
    with torch.no_grad():
        for head in (rssm.prior_net, rssm.post_net):
            assert head.out is not None
            head.out.bias.copy_(torch.linspace(-1e3, 1e3, head.out.bias.numel()))
    embed, actions, is_first = make_inputs()
    post, prior = rssm.observe(embed, actions, is_first)
    floor = rssm.cfg.unimix / rssm.cfg.classes
    assert post.logits.exp().min() >= floor * (1 - 1e-5)
    assert prior.logits.exp().min() >= floor * (1 - 1e-5)


def test_observe_priors_equal_one_step_imagination_from_previous_posteriors() -> None:
    rssm = make_rssm()
    embed, actions, is_first = make_inputs()
    post, prior = rssm.observe(embed, actions, is_first, generator=torch.Generator().manual_seed(0))
    for t in range(1, embed.shape[1]):
        step = rssm.img_step(post[t - 1], actions[:, t])
        torch.testing.assert_close(step.deter, prior.deter[:, t], rtol=0, atol=0)
        torch.testing.assert_close(step.logits, prior.logits[:, t], rtol=0, atol=0)
    # The posterior shares the deterministic state of its prior.
    assert torch.equal(post.deter, prior.deter)


def test_imagine_is_a_loop_of_img_step() -> None:
    rssm = make_rssm()
    embed, actions, is_first = make_inputs()
    post, _ = rssm.observe(embed, actions, is_first, generator=torch.Generator().manual_seed(0))
    start = post[2]
    imagined = rssm.imagine(start, actions, generator=torch.Generator().manual_seed(5))
    gen = torch.Generator().manual_seed(5)
    state = start
    for t in range(actions.shape[1]):
        state = rssm.img_step(state, actions[:, t], gen)
        assert torch.equal(state.deter, imagined.deter[:, t])
        assert torch.equal(state.stoch, imagined.stoch[:, t])


def test_is_first_resets_memory_inside_a_sequence() -> None:
    rssm = make_rssm()
    embed_a, actions_a, is_first = make_inputs(seed=1)
    embed_b, actions_b, _ = make_inputs(seed=2)
    # Same suffix from step 3 on, different history before it, episode boundary at step 3.
    embed_b[:, 3:] = embed_a[:, 3:]
    actions_b[:, 3:] = actions_a[:, 3:]
    is_first[:, 3] = 1.0
    post_a, _ = rssm.observe(
        embed_a, actions_a, is_first, generator=torch.Generator().manual_seed(0)
    )
    post_b, _ = rssm.observe(
        embed_b, actions_b, is_first, generator=torch.Generator().manual_seed(0)
    )
    assert not torch.allclose(post_a.deter[:, 2], post_b.deter[:, 2])
    torch.testing.assert_close(post_a.deter[:, 3], post_b.deter[:, 3], rtol=0, atol=0)
    torch.testing.assert_close(post_a.logits[:, 3], post_b.logits[:, 3], rtol=0, atol=0)


def test_reconstruction_gradient_reaches_encoder_through_discrete_latents() -> None:
    # Without the straight-through estimator the sampled one-hot would cut the graph and a
    # loss on the latent sample could not train the encoder that produced the embedding.
    rssm = make_rssm()
    embed, actions, is_first = make_inputs()
    embed.requires_grad_(True)
    post, _ = rssm.observe(embed, actions, is_first, generator=torch.Generator().manual_seed(0))
    target = torch.randn_like(post.stoch)
    (post.stoch * target).sum().backward()
    assert embed.grad is not None
    assert embed.grad.abs().sum() > 0


def test_state_stack_and_index_roundtrip() -> None:
    rssm = make_rssm()
    states = [rssm.initial(3) for _ in range(4)]
    stacked = State.stack(states)
    assert stacked.deter.shape == (3, 4, 16)
    assert torch.equal(stacked[1].stoch, states[1].stoch)
