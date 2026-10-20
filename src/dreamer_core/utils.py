"""Seeding helpers.

World-model training draws randomness from three places (Python, NumPy for the
environment, and torch for initialization and latent sampling). Seeding only one of them
is the most common reason two "identical" runs diverge, so they are seeded together.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and torch, and return a fresh NumPy generator.

    The returned generator is what environment code should use: passing it explicitly keeps
    the environment stream independent of any library that consumes the global NumPy state.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code may still use the legacy global state
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Return a dedicated torch generator for latent and action sampling.

    Sampling from a dedicated generator means that adding an unrelated random call elsewhere
    (for example a dropout layer or a data augmentation) does not shift the latent samples.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen
