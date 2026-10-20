"""DotGrid: an action-conditioned image environment small enough for CPU world-model tests.

A 2x2-pixel dot moves on an ``cells x cells`` grid rendered as a single-channel image with
``2 * cells`` pixels per side. A static goal cell is drawn at half intensity. Reaching the goal
gives reward 1 and ends the episode, so the reward and continuation heads have something to
predict, and training sequences contain episode boundaries that exercise ``is_first``.

The dynamics are deterministic given the action, which makes open-loop prediction error a
direct measure of how well the latent state tracks position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# noop, up, down, left, right
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
NUM_ACTIONS = len(MOVES)


@dataclass(frozen=True)
class DotGridConfig:
    cells: int = 8
    goal: tuple[int, int] = (1, 6)
    goal_intensity: float = 0.5
    max_steps: int = 200

    @property
    def image_size(self) -> int:
        return 2 * self.cells


class DotGrid:
    def __init__(self, cfg: DotGridConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.pos = np.zeros(2, dtype=np.int64)
        self.steps = 0
        goal = np.zeros((cfg.image_size, cfg.image_size), dtype=np.float32)
        gr, gc = cfg.goal
        goal[2 * gr : 2 * gr + 2, 2 * gc : 2 * gc + 2] = cfg.goal_intensity
        self._background = goal

    def reset(self) -> np.ndarray:
        goal = np.array(self.cfg.goal)
        while True:
            self.pos = self.rng.integers(0, self.cfg.cells, size=2)
            if not np.array_equal(self.pos, goal):
                break
        self.steps = 0
        return self.render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        """Return ``(image, reward, terminal, truncated)``."""
        self.pos = np.clip(self.pos + MOVES[action], 0, self.cfg.cells - 1)
        self.steps += 1
        at_goal = bool(np.array_equal(self.pos, np.array(self.cfg.goal)))
        truncated = self.steps >= self.cfg.max_steps and not at_goal
        return self.render(), float(at_goal), at_goal, truncated

    def render(self) -> np.ndarray:
        image = self._background.copy()
        r, c = self.pos
        image[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = 1.0
        return image[None]


def collect(
    cfg: DotGridConfig, steps: int, rng: np.random.Generator, repeat_prob: float = 0.75
) -> dict[str, np.ndarray]:
    """Collect a continuous stream of ``steps`` frames with a sticky random policy.

    Sticky actions (repeat the previous action with probability ``repeat_prob``) produce longer
    straight moves than uniform noise, so the data covers the grid and the goal is reached
    often enough to train the reward and continuation heads.
    """
    env = DotGrid(cfg, rng)
    size = cfg.image_size
    out = {
        "image": np.zeros((steps, 1, size, size), dtype=np.float32),
        "prev_action": np.zeros((steps, NUM_ACTIONS), dtype=np.float32),
        "reward": np.zeros(steps, dtype=np.float32),
        "cont": np.ones(steps, dtype=np.float32),
        "is_first": np.zeros(steps, dtype=np.float32),
        "pos": np.zeros((steps, 2), dtype=np.int64),
    }
    image, done, action = env.reset(), False, 0
    out["image"][0], out["is_first"][0], out["pos"][0] = image, 1.0, env.pos
    for t in range(1, steps):
        if done:
            image, done = env.reset(), False
            out["image"][t], out["is_first"][t], out["pos"][t] = image, 1.0, env.pos
            continue
        if rng.random() >= repeat_prob:
            action = int(rng.integers(NUM_ACTIONS))
        image, reward, terminal, truncated = env.step(action)
        out["image"][t], out["reward"][t], out["pos"][t] = image, reward, env.pos
        out["prev_action"][t, action] = 1.0
        out["cont"][t] = 0.0 if terminal else 1.0
        done = terminal or truncated
    return out


def sample_windows(
    data: dict[str, np.ndarray], batch: int, length: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Sample ``batch`` random windows of ``length`` steps from a stream.

    The first step of every window is marked ``is_first``: the model starts each window from
    the zero state, so it must not condition on an action whose preceding state it never saw.
    """
    starts = rng.integers(0, len(data["image"]) - length + 1, size=batch)
    index = starts[:, None] + np.arange(length)[None]
    out = {key: value[index] for key, value in data.items()}
    out["is_first"][:, 0] = 1.0
    return out
