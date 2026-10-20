import torch
import torch.nn.functional as F
from torch import nn

from dreamer_core.distributions import categorical_kl
from dreamer_core.losses import KLConfig, kl_balance
from dreamer_core.rssm import RSSM, RSSMConfig

NO_FREE = KLConfig(free_nats=0.0)


def _logits(seed: int, scale: float = 2.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return scale * torch.randn(3, 4, 8, 5, dtype=torch.float64, generator=gen)


def test_default_scales_follow_dreamerv3() -> None:
    cfg = KLConfig()
    assert (cfg.dyn_scale, cfg.rep_scale, cfg.free_nats) == (0.5, 0.1, 1.0)


def test_loss_value_combines_both_terms() -> None:
    post, prior = _logits(0), _logits(1)
    terms = kl_balance(post, prior)
    kl = categorical_kl(post, prior).sum(-1)
    torch.testing.assert_close(terms.kl, kl)
    # Both terms have the same forward value; only their gradients differ.
    torch.testing.assert_close(terms.dyn, kl.clamp(min=1.0))
    torch.testing.assert_close(terms.rep, kl.clamp(min=1.0))
    torch.testing.assert_close(terms.loss, 0.5 * kl.clamp(min=1.0) + 0.1 * kl.clamp(min=1.0))


def test_dyn_term_only_moves_prior_and_rep_term_only_moves_posterior() -> None:
    post = _logits(0).requires_grad_(True)
    prior = _logits(1).requires_grad_(True)

    terms = kl_balance(post, prior, NO_FREE)
    d_post, d_prior = torch.autograd.grad(terms.dyn.sum(), [post, prior], allow_unused=True)
    assert d_post is None or d_post.abs().max() == 0
    assert d_prior is not None
    assert d_prior.abs().max() > 0

    terms = kl_balance(post, prior, NO_FREE)
    r_post, r_prior = torch.autograd.grad(terms.rep.sum(), [post, prior], allow_unused=True)
    assert r_prior is None or r_prior.abs().max() == 0
    assert r_post is not None
    assert r_post.abs().max() > 0


def test_split_gradients_match_the_unsplit_kl() -> None:
    # With equal scales and no free bits, the balanced loss must reproduce the gradient of a
    # plain KL(q || p) exactly: the split only re-weights the two directions.
    post = _logits(2).requires_grad_(True)
    prior = _logits(3).requires_grad_(True)
    cfg = KLConfig(dyn_scale=1.0, rep_scale=1.0, free_nats=0.0)
    balanced = torch.autograd.grad(kl_balance(post, prior, cfg).loss.sum(), [post, prior])
    plain = torch.autograd.grad(categorical_kl(post, prior).sum(), [post, prior])
    for got, want in zip(balanced, plain, strict=True):
        torch.testing.assert_close(got, want)


def test_free_bits_zero_the_gradient_below_the_threshold() -> None:
    post = _logits(0, scale=0.01).requires_grad_(True)
    prior = _logits(1, scale=0.01).requires_grad_(True)
    terms = kl_balance(post, prior)
    assert (terms.kl < 1.0).all()
    torch.testing.assert_close(terms.loss, torch.full_like(terms.loss, 0.6))
    grads = torch.autograd.grad(terms.loss.sum(), [post, prior], allow_unused=True)
    for grad in grads:
        assert grad is None or grad.abs().max() == 0


def test_free_bits_are_applied_per_element_not_on_the_batch_mean() -> None:
    post = torch.cat([_logits(0, scale=0.01), _logits(4, scale=3.0)]).requires_grad_(True)
    prior = torch.cat([_logits(1, scale=0.01), _logits(5, scale=3.0)])
    terms = kl_balance(post, prior)
    (grad,) = torch.autograd.grad(terms.rep.sum(), [post])
    assert grad[:3].abs().max() == 0  # below threshold: clipped
    assert grad[3:].abs().max() > 0  # above threshold: trained


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(6, 7)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _rssm_terms(time: int) -> tuple[RSSM, _Encoder, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    rssm = RSSM(RSSMConfig(action_dim=3, embed_dim=7, deter=16, stoch=4, classes=5, hidden=12))
    encoder = _Encoder()
    gen = torch.Generator().manual_seed(1)
    obs = torch.randn(2, time, 6, generator=gen)
    actions = F.one_hot(torch.randint(0, 3, (2, time), generator=gen), 3).float()
    is_first = torch.zeros(2, time)
    is_first[:, 0] = 1.0
    post, prior = rssm.observe(encoder(obs), actions, is_first, generator=gen)
    terms = kl_balance(post.logits, prior.logits, NO_FREE)
    return rssm, encoder, terms.dyn, terms.rep


def _grad_norm(loss: torch.Tensor, module: nn.Module) -> float:
    params = list(module.parameters())
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return sum(0.0 if g is None else g.abs().sum().item() for g in grads)


def test_gradient_routing_inside_the_rssm() -> None:
    rssm, encoder, dyn, rep = _rssm_terms(time=1)
    dyn, rep = dyn.sum(), rep.sum()
    # Dynamics term: trains the prior head, never the posterior head or the encoder.
    assert _grad_norm(dyn, rssm.prior_net) > 0
    assert _grad_norm(dyn, rssm.post_net) == 0
    assert _grad_norm(dyn, encoder) == 0
    # Representation term: trains the posterior head and the encoder, never the prior head.
    assert _grad_norm(rep, rssm.post_net) > 0
    assert _grad_norm(rep, encoder) > 0
    assert _grad_norm(rep, rssm.prior_net) == 0


def test_dyn_term_reaches_encoder_only_through_the_recurrent_sample_path() -> None:
    # The stop-gradient acts on the KL inputs only. Over several steps the posterior *sample*
    # z_{t-1} feeds h_t through the straight-through estimator, so the dynamics loss at step t
    # does reach the encoder through earlier samples. This matches DreamerV3 and is documented
    # rather than hidden: the routing guarantee above is per time step.
    _, encoder, dyn, _ = _rssm_terms(time=3)
    assert _grad_norm(dyn[:, 0].sum(), encoder) == 0
    assert _grad_norm(dyn[:, 2].sum(), encoder) > 0
