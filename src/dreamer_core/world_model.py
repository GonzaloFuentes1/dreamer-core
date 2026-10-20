"""A compact image world model assembled from the building blocks.

It wires a convolutional encoder and decoder around the :class:`~dreamer_core.rssm.RSSM`, adds
a two-hot reward head and a Bernoulli continuation head, and combines their losses with the
balanced KL. Sizes are deliberately small so that it trains on a CPU in minutes; the
structure, not the scale, is what this module demonstrates.

Batch convention (DreamerV3 replay format), all tensors ``(batch, time, ...)``:
``image[t]`` is observation ``t``; ``prev_action[t]`` the action that led to it; ``reward[t]``
the reward received on arriving at it; ``cont[t]`` is 0 if it is terminal; ``is_first[t]`` marks
the first observation of an episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dreamer_core.losses import KLConfig, kl_balance
from dreamer_core.nets import MLP
from dreamer_core.rssm import RSSM, RSSMConfig, State
from dreamer_core.symlog import TwoHot


@dataclass(frozen=True)
class WorldModelConfig:
    action_dim: int
    image_size: int = 16
    channels: int = 1
    cnn_depth: int = 16
    embed_dim: int = 128
    deter: int = 128
    stoch: int = 16
    classes: int = 16
    hidden: int = 128
    head_layers: int = 1
    unimix: float = 0.01
    kl: KLConfig = field(default_factory=KLConfig)

    def rssm_config(self) -> RSSMConfig:
        return RSSMConfig(
            action_dim=self.action_dim,
            embed_dim=self.embed_dim,
            deter=self.deter,
            stoch=self.stoch,
            classes=self.classes,
            hidden=self.hidden,
            unimix=self.unimix,
        )


def _num_downsamples(image_size: int) -> int:
    stages = int(math.log2(image_size // 4))
    if image_size < 8 or 4 * 2**stages != image_size:
        raise ValueError("image_size must be a power of two >= 8")
    return stages


class ConvEncoder(nn.Module):
    """Strided convolutions down to 4x4, then a linear projection to the embedding."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels, depth = cfg.channels, cfg.cnn_depth
        for i in range(_num_downsamples(cfg.image_size)):
            out = depth * 2**i
            layers += [nn.Conv2d(channels, out, 4, 2, 1), nn.GroupNorm(1, out), nn.SiLU()]
            channels = out
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Linear(channels * 16, cfg.embed_dim)

    def forward(self, image: Tensor) -> Tensor:
        lead = image.shape[:-3]
        x = self.convs(image.reshape(-1, *image.shape[-3:]) - 0.5)
        return self.proj(x.flatten(1)).reshape(*lead, -1)


class ConvDecoder(nn.Module):
    """Mirror of the encoder; predicts the mean image (trained with a squared error)."""

    def __init__(self, cfg: WorldModelConfig, feature_dim: int) -> None:
        super().__init__()
        stages = _num_downsamples(cfg.image_size)
        self.top = cfg.cnn_depth * 2 ** (stages - 1)
        self.proj = nn.Linear(feature_dim, self.top * 16)
        layers: list[nn.Module] = []
        channels = self.top
        for i in reversed(range(stages)):
            last = i == 0
            out = cfg.channels if last else cfg.cnn_depth * 2 ** (i - 1)
            layers.append(nn.ConvTranspose2d(channels, out, 4, 2, 1))
            if not last:
                layers += [nn.GroupNorm(1, out), nn.SiLU()]
            channels = out
        self.deconvs = nn.Sequential(*layers)
        self.shape = (cfg.channels, cfg.image_size, cfg.image_size)

    def forward(self, feature: Tensor) -> Tensor:
        lead = feature.shape[:-1]
        x = self.proj(feature.reshape(-1, feature.shape[-1])).reshape(-1, self.top, 4, 4)
        return (self.deconvs(x) + 0.5).reshape(*lead, *self.shape)


@dataclass
class WorldModelOutput:
    loss: Tensor
    metrics: dict[str, Tensor]
    post: State


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg)
        self.rssm = RSSM(cfg.rssm_config())
        feat = self.rssm.feature_dim
        self.decoder = ConvDecoder(cfg, feat)
        self.twohot = TwoHot()
        self.reward_head = MLP(
            feat, cfg.hidden, cfg.head_layers, out_dim=self.twohot.num_bins, zero_init_output=True
        )
        self.cont_head = MLP(feat, cfg.hidden, cfg.head_layers, out_dim=1)

    def reward(self, feature: Tensor) -> Tensor:
        return self.twohot.mean(self.reward_head(feature))

    def cont(self, feature: Tensor) -> Tensor:
        return torch.sigmoid(self.cont_head(feature).squeeze(-1))

    def loss(
        self,
        image: Tensor,
        prev_action: Tensor,
        reward: Tensor,
        cont: Tensor,
        is_first: Tensor,
        generator: torch.Generator | None = None,
    ) -> WorldModelOutput:
        embed = self.encoder(image)
        post, prior = self.rssm.observe(embed, prev_action, is_first, generator=generator)
        feat = post.feature()
        image_loss = (self.decoder(feat) - image).pow(2).sum((-3, -2, -1))
        reward_loss = self.twohot.loss(self.reward_head(feat), reward)
        cont_loss = F.binary_cross_entropy_with_logits(
            self.cont_head(feat).squeeze(-1), cont, reduction="none"
        )
        kl = kl_balance(post.logits, prior.logits, self.cfg.kl)
        total = (image_loss + reward_loss + cont_loss + kl.loss).mean()
        metrics = {
            "loss": total.detach(),
            "image": image_loss.mean().detach(),
            "reward": reward_loss.mean().detach(),
            "cont": cont_loss.mean().detach(),
            "dyn": kl.dyn.mean().detach(),
            "rep": kl.rep.mean().detach(),
            "kl": kl.kl.mean(),
        }
        return WorldModelOutput(total, metrics, post)

    @torch.no_grad()
    def open_loop(
        self,
        image: Tensor,
        prev_action: Tensor,
        is_first: Tensor,
        context: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Observe ``context`` frames, then imagine the rest from the actions alone.

        Returns decoded frames for the imagined steps, ``(batch, time - context, C, H, W)``.
        """
        embed = self.encoder(image[:, :context])
        post, _ = self.rssm.observe(
            embed, prev_action[:, :context], is_first[:, :context], generator=generator
        )
        imagined = self.rssm.imagine(post[-1], prev_action[:, context:], generator)
        return self.decoder(imagined.feature()).clamp(0.0, 1.0)
