from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import typer
import yaml

import wandb


class Map:
    def __init__(self, path: Path) -> None:
        with open(path, "r") as f:
            self.meta = yaml.safe_load(f)
        raw = cv2.imread(str(path.parent / self.meta["image"]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Error reading {path.parent / self.meta['image']}")
        self.occupied = jnp.asarray(raw < 128)

        cast_all = jax.vmap(jax.vmap(jax.vmap(     # fmt: skip
             Map.cast_ray, (None, None, None, 0)), # fmt: skip
                           (None, None, 0, None)), # fmt: skip
                           (None, 0, None, None))  # fmt: skip
        self.lookup_table = cast_all(
            self.occupied,
            jnp.arange(self.occupied.shape[0]),
            jnp.arange(self.occupied.shape[1]),
            jnp.linspace(0, 2 * jnp.pi, 360),
        )

    @staticmethod
    @jax.jit
    def cast_ray(
        occupied: jax.Array,
        start_row: jax.Array,
        start_col: jax.Array,
        theta: jax.Array,
    ):
        c, s = jnp.cos(theta), -jnp.sin(theta)

        @jax.jit
        def is_occ(t):
            row = jnp.int32(jnp.round(start_row + t * c))
            col = jnp.int32(jnp.round(start_col + t * s))
            return (
                col
                < 0 | col
                >= occupied.shape[1] | row
                < 0 | row
                >= occupied.shape[0] | occupied[row, col]
            )

        @jax.jit
        def step(t):
            return t + 1.0

        # def step(carry, _):
        #     t, hit = carry
        #     row = jnp.int32(jnp.round(start_row + t * c))
        #     col = jnp.int32(jnp.round(start_col + t * s))

        #     hit = hit | (0 <= col) & (col < occupied.shape[1]) & (0 <= row) & (
        #         row < occupied.shape[0]
        #     )
        #     t = jnp.where(hit, t, t + 1.0)
        #     return (t, hit), None

        return jax.lax.while_loop(is_occ, step, 0.0)


def main(yaml_path: Path, num_envs: int = 1024, seed: int = 42):
    # wandb.init(
    #     project="f1tenth-ppo-jax",
    #     name=f"jaxoracer__{seed}",
    #     save_code=True,
    # )
    map = Map(yaml_path)


if __name__ == "__main__":
    typer.run(main)
