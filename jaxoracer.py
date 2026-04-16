from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import typer
import yaml

import wandb


class Map:
    @staticmethod
    def cast_ray(
        image: jax.Array,
        x: jax.Array,
        y: jax.Array,
        theta: jax.Array,
    ):
        c, s = jnp.cos(theta), -jnp.sin(theta)

        def step(carry, _):
            t, hit = carry
            col = jnp.round(x + t * c).astype(jnp.int32)
            row = jnp.round(y + t * s).astype(jnp.int32)
            hit = hit | (0 <= col) & (col < image.shape[1]) & (0 <= row) & (
                row < image.shape[0]
            )
            t = jnp.where(hit, t, t + 1.0)
            return (t, hit), None

        w, h = image.shape
        _, hit = jax.lax.scan(step, (0.0, False), None, length=int(jnp.hypot(w, h)))
        return hit

    def __init__(self, yaml_path: Path) -> None:
        self.yaml_path = yaml_path
        with open(yaml_path, "r") as f:
            self.meta = yaml.safe_load(f)
        self.image_path: Path = yaml_path.parent / self.meta["image"]
        self.image = jnp.asarray(cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE))
        self.resolution = self.meta["resolution"]
        self.origin_x, self.origin_y, self.origin_theta = self.meta["origin"]

        vmap_theta = jax.vmap(Map.cast_ray, in_axes=(None, None, None, 0))
        vmap_y = jax.vmap(vmap_theta, in_axes=(None, None, 0, None))
        vmap_x = jax.vmap(vmap_y, in_axes=(None, 0, None, None))
        self.lookup_table = vmap_x(
            self.image,
            jnp.arange(self.image.shape[0]),
            jnp.arange(self.image.shape[1]),
            jnp.linspace(0, 2 * jnp.pi, 360),
        )


def main(yaml_path: Path, num_envs: int = 1024, seed: int = 42):
    # wandb.init(
    #     project="f1tenth-ppo-jax",
    #     name=f"jaxoracer__{seed}",
    #     save_code=True,
    # )
    map = Map(yaml_path)


if __name__ == "__main__":
    typer.run(main)
