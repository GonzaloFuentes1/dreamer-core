"""Train the world model on DotGrid on CPU and record what was measured.

Usage:
    python scripts/train_toy.py --seed 0 --steps 1500

Writes ``results/wm_toy.json`` (config, seed, per-step losses, wall time, hardware, open-loop
evaluation) and renders ``assets/wm_toy_openloop.png`` and ``assets/wm_toy_openloop.gif``
from the evaluation rollouts. Requires the ``toy`` extra (Pillow) for the images.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamer_core.toy.train import (
    TrainConfig,
    config_dict,
    evaluate_open_loop,
    train_world_model,
)

ROOT = Path(__file__).resolve().parents[1]


def hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _upscale(frame: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(frame[0], np.ones((factor, factor), dtype=frame.dtype))


def render_assets(frames: dict[str, np.ndarray], out_dir: Path, rows: int = 4) -> list[str]:
    from PIL import Image  # optional dependency, only needed for rendering

    out_dir.mkdir(parents=True, exist_ok=True)
    truth, pred = frames["truth"][:rows], frames["pred"][:rows]
    factor, gap = 6, 2
    size = truth.shape[-1] * factor
    horizon = truth.shape[1]
    width = horizon * (size + gap) - gap
    height = rows * 2 * (size + gap) + (rows - 1) * 3 * gap
    canvas = np.full((height, width), 0.25, dtype=np.float32)
    y = 0
    for r in range(rows):
        for series in (truth[r], pred[r]):
            for t in range(horizon):
                x = t * (size + gap)
                canvas[y : y + size, x : x + size] = _upscale(series[t], factor)
            y += size + gap
        y += 3 * gap
    png = out_dir / "wm_toy_openloop.png"
    Image.fromarray((canvas * 255).astype(np.uint8)).save(png)

    gif_frames = []
    for t in range(horizon):
        tile = np.full((rows * (size + gap) - gap, 2 * size + 3 * gap), 0.25, dtype=np.float32)
        for r in range(rows):
            yy = r * (size + gap)
            tile[yy : yy + size, :size] = _upscale(truth[r, t], factor)
            tile[yy : yy + size, size + 3 * gap :] = _upscale(pred[r, t], factor)
        gif_frames.append(Image.fromarray((tile * 255).astype(np.uint8)))
    gif = out_dir / "wm_toy_openloop.gif"
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:], duration=250, loop=0)
    return [Path(os.path.relpath(f.resolve(), ROOT)).as_posix() for f in (png, gif)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "wm_toy.json")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(seed=args.seed, steps=args.steps)
    wall_start = time.perf_counter()
    model, history, info = train_world_model(cfg, max_seconds=args.max_seconds)
    evaluation = evaluate_open_loop(model, cfg, seed=args.seed + 1000)
    frames = evaluation.pop("_frames")
    assets = [] if args.no_render else render_assets(frames, args.assets)
    result = {
        "command": "python scripts/train_toy.py "
        + f"--seed {args.seed} --steps {args.steps}"
        + (f" --max-seconds {args.max_seconds}" if args.max_seconds else ""),
        "seed": args.seed,
        "config": config_dict(cfg),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train": info,
        "wall_seconds": time.perf_counter() - wall_start,
        "hardware": hardware(),
        "losses": history,
        "eval_open_loop": evaluation,
        "assets": assets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    last = history[-1]
    print(
        f"steps={info['steps_completed']} train_s={info['train_seconds']:.1f} "
        f"loss={last['loss']:.3f} image={last['image']:.3f} kl={last['kl']:.3f} | "
        f"open-loop mse={evaluation['mse_mean']:.4f} "
        f"(copy-last {evaluation['copy_last_mse_mean']:.4f}) "
        f"dot-cell acc={evaluation['dot_cell_accuracy_mean']:.3f} "
        f"(copy-last {evaluation['copy_last_dot_cell_accuracy_mean']:.3f})"
    )


if __name__ == "__main__":
    main()
