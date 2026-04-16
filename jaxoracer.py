import math
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import typer
import yaml


class Map:
    def __init__(self, path: Path, lidar_range: float) -> None:
        with open(path, "r") as f:
            self.meta = yaml.safe_load(f)

        raw = cv2.imread(str(path.parent / self.meta["image"]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Error reading {path.parent / self.meta['image']}")

        self.occupied = jnp.asarray(raw < 128)
        self.resolution = self.meta["resolution"]
        self.origin = self.meta["origin"]

        self.lookup = self._build_lookup(lidar_range / self.resolution)

    def _build_lookup(self, max_steps: float):
        h, w = self.occupied.shape

        rows = jnp.arange(h)
        cols = jnp.arange(w)
        angles = jnp.linspace(0, 2 * jnp.pi, 360)

        dcs = jnp.cos(angles)
        drs = -jnp.sin(angles)

        def cast(row, col, dc, dr):
            return self._cast_ray(self.occupied, row, col, dc, dr, max_steps)

        batched = jax.vmap(cast, (None, 0, None, None))  # over col
        batched = jax.vmap(batched, (0, None, None, None))  # over row
        batched = jax.vmap(batched, (None, None, 0, 0))  # over angle

        # (360, H, W) → (H, W, 360)
        return batched(rows, cols, dcs, drs).transpose(1, 2, 0) * self.resolution

    @staticmethod
    def _cast_ray(occ, row, col, dc, dr, max_steps):
        h, w = occ.shape
        steps = jnp.arange(1, max_steps + 1, dtype=jnp.float32)

        def step(best, t):
            r = jnp.int32(jnp.round(row + t * dr))
            c = jnp.int32(jnp.round(col + t * dc))

            in_bounds = (0 <= r) & (r < h) & (0 <= c) & (c < w)
            hit = ~in_bounds | occ[jnp.clip(r, 0, h - 1), jnp.clip(c, 0, w - 1)]

            return jnp.where(hit, jnp.minimum(best, t), best), None

        dist, _ = jax.lax.scan(step, jnp.float32(max_steps), steps)
        return dist


def main(
    yaml_path: Path,
    num_envs: int = 1024,
    seed: int = 42,
    lidar_range: float = 20,  # m
):
    # wandb.init(
    #     project="f1tenth-ppo-jax",
    #     name=f"jaxoracer__{seed}",
    #     save_code=True,
    # )
    map = Map(yaml_path, lidar_range)


if __name__ == "__main__":
    typer.run(main)
