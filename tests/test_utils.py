import random

import numpy as np
import torch

from dreamer_core.utils import seed_everything, torch_generator


def _draw(seed: int) -> tuple[float, np.ndarray, np.ndarray, torch.Tensor]:
    rng = seed_everything(seed)
    legacy = np.random.rand(3)  # noqa: NPY002 - the legacy global stream is what is tested
    return random.random(), legacy, rng.random(3), torch.rand(3)


def test_seed_everything_reproduces_all_streams() -> None:
    a = _draw(123)
    b = _draw(123)
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])
    assert torch.equal(a[3], b[3])


def test_different_seeds_give_different_streams() -> None:
    a = _draw(1)
    b = _draw(2)
    assert not torch.equal(a[3], b[3])
    assert not np.array_equal(a[2], b[2])


def test_dedicated_generator_is_isolated_from_global_state() -> None:
    gen_a = torch_generator(7)
    first = torch.rand(4, generator=gen_a)
    gen_b = torch_generator(7)
    torch.rand(100)  # consuming the global stream must not affect the dedicated one
    second = torch.rand(4, generator=gen_b)
    assert torch.equal(first, second)
