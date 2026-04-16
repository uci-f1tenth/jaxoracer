from collections import deque
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from cv2 import IMREAD_GRAYSCALE, imread
from rerun import Image, LineStrips2D, init, log
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from typer import run
from yaml import safe_load


class Map:
    def __init__(self, path: Path, lidar_range: float) -> None:
        with open(path, "r") as f:
            self.meta = safe_load(f)

        raw = imread(str(path.parent / self.meta["image"]), IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Error reading {path.parent / self.meta['image']}")

        occ = raw < 230
        self.occupied = jnp.asarray(occ)
        self.resolution = self.meta["resolution"]
        self.origin = self.meta["origin"]
        self.dt = jnp.asarray(distance_transform_edt(~occ))
        log("map", Image(raw))
        log("dt", Image(self.dt))
        self.compute_centerline(raw)
        self.lookup = self.build_lookup(lidar_range / self.resolution) * self.resolution

    @partial(jax.jit, static_argnums=(0,))
    def build_lookup(self, max_steps: float):
        h, w = self.occupied.shape
        angles = jnp.linspace(0, 2 * jnp.pi, 360)

        def cast(row, col, dc, dr):
            return Map._cast_ray(self.dt, row, col, dc, dr, max_steps)

        batched = jax.vmap(cast, (None, 0, None, None))
        batched = jax.vmap(batched, (0, None, None, None))
        batched = jax.vmap(batched, (None, None, 0, 0))
        return batched(
            jnp.arange(h), jnp.arange(w), jnp.cos(angles), -jnp.sin(angles)
        ).transpose(1, 2, 0)

    def compute_centerline(self, drivable_area, smooth_window=51):
        skel = skeletonize(drivable_area > 230)
        log("skeleton", Image((skel * 255).astype(jnp.uint8)))

        fg = set(map(tuple, jnp.argwhere(skel).tolist()))  # {(row, col), ...}

        def adj(p):
            r, c = p
            return [
                (r + dr, c + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in fg
            ]

        h = skel.shape[0]
        ox, oy, _ = self.origin
        orow, ocol = h - 1 + oy / self.resolution, -ox / self.resolution
        start = min(fg, key=lambda p: (p[0] - orow) ** 2 + (p[1] - ocol) ** 2)

        nbrs = adj(start)
        assert len(nbrs) >= 2, f"start {start} has < 2 neighbors"

        src, targets = nbrs[0], set(nbrs[1:])
        parent = {src: src}
        q = deque([src])
        found = None
        while q and found is None:
            cur = q.popleft()
            for n in adj(cur):
                if n == start or n in parent:
                    continue
                parent[n] = cur
                if n in targets:
                    found = n
                    break
                q.append(n)

        assert found is not None, "no loop found in skeleton"

        path = [start]
        p = found
        while p != src:
            path.append(p)
            p = parent[p]
        path.append(src)
        path.reverse()

        res = self.resolution
        world = jnp.array([[c * res + ox, (h - 1 - r) * res + oy] for r, c in path])

        w = min(smooth_window, len(world) // 2 * 2 - 1)
        if w >= 5:
            world = jnp.column_stack(
                [
                    savgol_filter(world[:, 0], w, 3, mode="wrap"),
                    savgol_filter(world[:, 1], w, 3, mode="wrap"),
                ]
            )

        self.centerline_world = world
        self.centerline_px = jnp.column_stack(
            [
                (world[:, 0] - ox) / res,
                h - 1 - (world[:, 1] - oy) / res,
            ]
        )
        log(
            "map/centerline",
            LineStrips2D([self.centerline_px], colors=[[255, 0, 0]]),
        )

    @staticmethod
    def _cast_ray(dt, row, col, dc, dr, max_steps):
        h, w = dt.shape

        def step(state, _):
            t, done = state
            r = jnp.int32(jnp.round(row + t * dr))
            c = jnp.int32(jnp.round(col + t * dc))
            in_bounds = (0 <= r) & (r < h) & (0 <= c) & (c < w)
            d = jnp.where(
                in_bounds, dt[jnp.clip(r, 0, h - 1), jnp.clip(c, 0, w - 1)], 0.0
            )
            hit = (d < 1.0) | ~in_bounds
            new_t = jnp.minimum(t + jnp.maximum(d, 1.0), max_steps)
            new_done = done | hit
            return (jnp.where(done, t, jnp.where(new_done, t, new_t)), new_done), None

        (dist, _), _ = jax.lax.scan(step, (jnp.float32(0.0), False), None, length=32)
        return dist


class Environment:
    def __init__(self, path: Path, lidar_range: float, seed=42):
        self.map = Map(path, lidar_range)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        state, key = state

    def _get_obsv(self, state):
        pass  # TODO

    def _maybe_reset(self, state, done):
        key = state[1]
        return jax.lax.cond(done, self._reset, lambda key: state, key)

    def _reset(self, key):
        new_state = jax.random.choice(key, self.map.centerline_px)
        # TODO: add rotation
        new_key = jax.random.split(key)[0]
        return new_state, new_key

    def reset(self, key):
        state, key = self._reset(key)
        return state, self._get_obsv(state)


def main(
    yaml_path: Path, num_envs: int = 1024, seed: int = 42, lidar_range: float = 20.0
):
    init("jaxoracer", spawn=True)
    map = Map(yaml_path, lidar_range)

    # h, w = map_.occupied.shape
    # ox, oy, _ = map_.origin
    # origin_col = int(-ox / map_.resolution)
    # origin_row = int(h - 1 + oy / map_.resolution)

    # distances = np.asarray(map_.lookup[origin_row, origin_col])
    # angles = np.linspace(0, 2 * np.pi, 360)

    # endpoints_x = origin_col + (distances / map_.resolution) * np.cos(angles)
    # endpoints_y = origin_row - (distances / map_.resolution) * np.sin(angles)

    # rr.log(
    #     "map/origin",
    #     rr.Points2D([[origin_col, origin_row]], radii=3, colors=[[255, 0, 0]]),
    # )
    # rr.log(
    #     "map/lidar",
    #     rr.Points2D(
    #         np.column_stack([endpoints_x, endpoints_y]), radii=1, colors=[[0, 255, 0]]
    #     ),
    # )


if __name__ == "__main__":
    run(main)
