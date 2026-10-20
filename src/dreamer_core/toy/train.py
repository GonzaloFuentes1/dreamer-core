"""Seeded CPU training and open-loop evaluation of the world model on DotGrid."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from dreamer_core.toy.env import NUM_ACTIONS, DotGridConfig, collect, sample_windows
from dreamer_core.utils import seed_everything, torch_generator
from dreamer_core.world_model import WorldModel, WorldModelConfig


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    steps: int = 3000
    batch: int = 16
    seq_len: int = 32
    lr: float = 1e-3
    grad_clip: float = 1000.0
    data_steps: int = 50_000
    log_every: int = 25
    env: DotGridConfig = field(default_factory=DotGridConfig)


def make_world_model(cfg: TrainConfig) -> WorldModel:
    return WorldModel(WorldModelConfig(action_dim=NUM_ACTIONS, image_size=cfg.env.image_size))


def _to_torch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in batch.items() if k != "pos"}


def train_world_model(
    cfg: TrainConfig, max_seconds: float | None = None
) -> tuple[WorldModel, list[dict[str, float]], dict[str, Any]]:
    """Train from a fixed random-policy dataset. Returns ``(model, history, info)``.

    Everything random (dataset, window sampling, initialisation, latent samples) derives from
    ``cfg.seed``, so two calls with the same config produce bitwise-identical results on the
    same CPU and torch build.
    """
    rng = seed_everything(cfg.seed)
    data = collect(cfg.env, cfg.data_steps, rng)
    model = make_world_model(cfg)
    gen = torch_generator(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-8)
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    step = 0
    for step in range(1, cfg.steps + 1):
        batch = _to_torch(sample_windows(data, cfg.batch, cfg.seq_len, rng))
        out = model.loss(
            batch["image"],
            batch["prev_action"],
            batch["reward"],
            batch["cont"],
            batch["is_first"],
            generator=gen,
        )
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.log_every == 0 or step == 1:
            history.append({"step": step, **{k: float(v) for k, v in out.metrics.items()}})
        if max_seconds is not None and time.perf_counter() - start > max_seconds:
            break
    info = {
        "steps_completed": step,
        "train_seconds": time.perf_counter() - start,
        "dataset_rewards": float(data["reward"].sum()),
        "dataset_episodes": int(data["is_first"].sum()),
    }
    return model, history, info


def _valid_windows(data: dict[str, np.ndarray], length: int, count: int) -> np.ndarray:
    """Start indices of windows without an episode boundary or terminal after the start."""
    starts = []
    for s in range(0, len(data["image"]) - length, length):
        inside = slice(s + 1, s + length)
        if data["is_first"][inside].any() or (data["cont"][s : s + length] == 0).any():
            continue
        starts.append(s)
        if len(starts) == count:
            break
    return np.array(starts)


def _dot_cell(images: np.ndarray, cells: int) -> np.ndarray:
    """Cell with the brightest 2x2 block, for ``(..., 1, 2 * cells, 2 * cells)`` images."""
    blocks = images[..., 0, :, :].reshape(*images.shape[:-3], cells, 2, cells, 2).mean((-3, -1))
    flat = blocks.reshape(*blocks.shape[:-2], -1).argmax(-1)
    return np.stack([flat // cells, flat % cells], -1)


@torch.no_grad()
def evaluate_open_loop(
    model: WorldModel,
    cfg: TrainConfig,
    seed: int,
    context: int = 5,
    horizon: int = 20,
    episodes: int = 64,
) -> dict[str, Any]:
    """Observe ``context`` frames, imagine ``horizon`` frames from actions only, and compare.

    Evaluation data comes from a separate seed. ``copy_last`` is the trivial baseline that
    predicts the last observed frame for every future step; the model must beat it to show
    that it uses the actions.
    """
    rng = np.random.default_rng(seed)
    length = context + horizon
    data = collect(cfg.env, 40 * length * episodes, rng)
    starts = _valid_windows(data, length, episodes)
    index = starts[:, None] + np.arange(length)[None]
    batch = {k: v[index] for k, v in data.items()}
    batch["is_first"][:, 0] = 1.0
    tb = _to_torch(batch)
    pred = model.open_loop(
        tb["image"], tb["prev_action"], tb["is_first"], context, torch_generator(seed)
    ).numpy()
    truth = batch["image"][:, context:]
    copy_last = np.repeat(batch["image"][:, context - 1 : context], horizon, axis=1)
    mse = ((pred - truth) ** 2).mean((0, 2, 3, 4))
    mse_copy = ((copy_last - truth) ** 2).mean((0, 2, 3, 4))
    true_cell = batch["pos"][:, context:]
    acc = (_dot_cell(pred, cfg.env.cells) == true_cell).all(-1).mean(0)
    acc_copy = (_dot_cell(copy_last, cfg.env.cells) == true_cell).all(-1).mean(0)
    return {
        "context": context,
        "horizon": horizon,
        "sequences": len(starts),
        "mse_per_step": mse.tolist(),
        "copy_last_mse_per_step": mse_copy.tolist(),
        "dot_cell_accuracy_per_step": acc.tolist(),
        "copy_last_dot_cell_accuracy_per_step": acc_copy.tolist(),
        "mse_mean": float(mse.mean()),
        "copy_last_mse_mean": float(mse_copy.mean()),
        "dot_cell_accuracy_mean": float(acc.mean()),
        "copy_last_dot_cell_accuracy_mean": float(acc_copy.mean()),
        "_frames": {"truth": truth, "pred": pred, "context": batch["image"][:, :context]},
    }


def config_dict(cfg: TrainConfig) -> dict[str, Any]:
    return asdict(cfg)
