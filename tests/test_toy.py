import numpy as np
import torch

from dreamer_core.toy.env import MOVES, NUM_ACTIONS, DotGrid, DotGridConfig, collect, sample_windows
from dreamer_core.toy.train import TrainConfig, train_world_model
from dreamer_core.world_model import WorldModel, WorldModelConfig

TINY = TrainConfig(seed=3, steps=15, batch=4, seq_len=8, data_steps=500, log_every=1)


def test_env_moves_dot_and_clips_at_walls() -> None:
    cfg = DotGridConfig(cells=4, goal=(0, 3))
    env = DotGrid(cfg, np.random.default_rng(0))
    env.reset()
    env.pos = np.array([0, 0])
    image, reward, terminal, _ = env.step(1)  # up at the top wall: no move
    assert np.array_equal(env.pos, [0, 0])
    assert image.shape == (1, 8, 8)
    assert image[0, 0:2, 0:2].min() == 1.0
    assert image[0, 0:2, 6:8].max() == cfg.goal_intensity
    env.step(4)
    env.step(4)
    _, reward, terminal, _ = env.step(4)
    assert np.array_equal(env.pos, [0, 3])
    assert (reward, terminal) == (1.0, True)


def test_collected_stream_is_consistent_with_actions() -> None:
    cfg = DotGridConfig()
    data = collect(cfg, 2000, np.random.default_rng(0))
    for t in range(1, 2000):
        if data["is_first"][t]:
            assert data["prev_action"][t].sum() == 0
            assert (data["reward"][t], data["cont"][t]) == (0.0, 1.0)
            continue
        move = MOVES[data["prev_action"][t].argmax()]
        expected = np.clip(data["pos"][t - 1] + move, 0, cfg.cells - 1)
        assert np.array_equal(data["pos"][t], expected)
        at_goal = data["pos"][t].tolist() == list(cfg.goal)
        assert data["reward"][t] == float(at_goal)
        assert data["cont"][t] == float(not at_goal)
    assert data["reward"].sum() > 0  # the goal is actually reached


def test_windows_start_with_is_first() -> None:
    data = collect(DotGridConfig(), 300, np.random.default_rng(0))
    batch = sample_windows(data, 8, 10, np.random.default_rng(1))
    assert batch["image"].shape == (8, 10, 1, 16, 16)
    assert (batch["is_first"][:, 0] == 1).all()


def test_world_model_loss_backpropagates_to_all_modules() -> None:
    torch.manual_seed(0)
    model = WorldModel(
        WorldModelConfig(
            action_dim=NUM_ACTIONS,
            deter=16,
            stoch=4,
            classes=4,
            hidden=16,
            embed_dim=16,
            cnn_depth=4,
        )
    )
    data = collect(DotGridConfig(), 200, np.random.default_rng(0))
    batch = {
        k: torch.from_numpy(v)
        for k, v in sample_windows(data, 2, 6, np.random.default_rng(0)).items()
    }
    out = model.loss(
        batch["image"], batch["prev_action"], batch["reward"], batch["cont"], batch["is_first"]
    )
    out.loss.backward()
    for name in ("encoder", "decoder", "rssm", "reward_head", "cont_head"):
        grads = [p.grad for p in getattr(model, name).parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), name
    pred = model.open_loop(batch["image"], batch["prev_action"], batch["is_first"], context=2)
    assert pred.shape == (2, 4, 1, 16, 16)


def test_same_seed_gives_bitwise_identical_training() -> None:
    model_a, hist_a, _ = train_world_model(TINY)
    model_b, hist_b, _ = train_world_model(TINY)
    assert hist_a == hist_b
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(p_a, p_b)


def test_different_seed_changes_training() -> None:
    _, hist_a, _ = train_world_model(TINY)
    _, hist_b, _ = train_world_model(TrainConfig(**{**TINY.__dict__, "seed": 4}))
    assert hist_a != hist_b
