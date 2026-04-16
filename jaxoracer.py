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


def round_int(x):
    return jnp.int32(jnp.round(x))


def wrap_angle(a):
    """Wrap angle to [-π, π)."""
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


class Map:
    def __init__(self, path: Path, lidar_range: float, n_lookup_angles: int) -> None:
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
        self.n_lookup_angles = n_lookup_angles

        log("map", Image(raw))
        log("dt", Image(self.dt))
        self.compute_centerline(raw)

        self.lookup = (
            self.build_lookup(lidar_range / self.resolution, n_lookup_angles)
            * self.resolution
        )

    @partial(jax.jit, static_argnums=(0,))
    def build_lookup(self, max_steps: float, n_lookup_angles: int):
        h, w = self.occupied.shape
        angles = jnp.linspace(0, 2 * jnp.pi, n_lookup_angles, endpoint=False)

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

        fg = set(map(tuple, jnp.argwhere(skel).tolist()))

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
        orow = h - 1 + oy / self.resolution
        ocol = -ox / self.resolution
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
                [  # pyright: ignore[reportArgumentType]
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

    # ray marching is unchanged -----------------------------------------
    @staticmethod
    def _cast_ray(dt, row, col, dc, dr, max_steps):
        h, w = dt.shape

        def step(state, _):
            t, done = state
            r = round_int(row + t * dr)
            c = round_int(col + t * dc)
            in_bounds = (0 <= r) & (r < h) & (0 <= c) & (c < w)
            d = jnp.where(
                in_bounds,
                dt[jnp.clip(r, 0, h - 1), jnp.clip(c, 0, w - 1)],
                0.0,
            )
            hit = (d < 1.0) | ~in_bounds  # pyright: ignore[reportOperatorIssue]
            new_t = jnp.minimum(t + jnp.maximum(d, 1.0), max_steps)  # pyright: ignore[reportArgumentType]
            new_done = done | hit
            return (
                jnp.where(done, t, jnp.where(new_done, t, new_t)),  # pyright: ignore[reportCallIssue, reportArgumentType]
                new_done,
            ), None

        (dist, _), _ = jax.lax.scan(step, (jnp.float32(0.0), False), None, length=32)
        return dist


class Environment:
    def __init__(
        self,
        path: Path,
        lidar_range: float,
        n_beams: int,
        fov_deg: float,
        n_lookup_angles: int,
    ):
        self.n_beams = n_beams
        self.fov = jnp.radians(fov_deg)
        self.n_lookup_angles = n_lookup_angles
        self.map = Map(path, lidar_range, n_lookup_angles)

        self.beam_offsets = jnp.linspace(-self.fov / 2, self.fov / 2, n_beams)

        cl = self.map.centerline_world  # (N, 2)
        diffs = jnp.diff(cl, axis=0, append=cl[:1])
        self.angles = jnp.arctan2(diffs[:, 1], diffs[:, 0])  # (N,)
        seg_lens = jnp.sqrt((diffs**2).sum(axis=1))
        self.cum_dist = jnp.cumsum(seg_lens)  # (N,)
        self.track_length = self.cum_dist[-1]
        self.n_waypoints = cl.shape[0]

    def _world_to_px(self, x, y):
        h = self.map.occupied.shape[0]
        ox, oy, _ = self.map.origin
        res = self.map.resolution
        col = (x - ox) / res
        row = h - 1 - (y - oy) / res
        return row, col

    def _get_obsv(self, state):
        x, y, theta, speed, prog_idx = state
        prog_idx = jnp.int32(prog_idx)
        row, col = self._world_to_px(x, y)
        h, w = self.map.occupied.shape

        r_idx = jnp.clip(round_int(row), 0, h - 1)
        c_idx = jnp.clip(round_int(col), 0, w - 1)

        abs_angles = theta + self.beam_offsets

        lookup_idx = (
            round_int(abs_angles / (2 * jnp.pi) * self.n_lookup_angles)
            % self.n_lookup_angles
        )

        lidar = self.map.lookup[r_idx, c_idx, lookup_idx]

        cl = self.map.centerline_world
        track_angle = self.angles[prog_idx]
        rel_heading = wrap_angle(theta - track_angle)

        to_car = jnp.array([x, y]) - cl[prog_idx]
        normal = jnp.array([-jnp.sin(track_angle), jnp.cos(track_angle)])
        lat_offset = jnp.dot(to_car, normal)

        return jnp.concatenate([lidar, jnp.array([speed, lat_offset, rel_heading])])

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        state, key = state

    def _reset(self, key):
        cl = self.map.centerline_world
        idx = jax.random.randint(key, (), 0, self.n_waypoints)
        pos = cl[idx]
        angle = self.angles[idx]
        return jnp.array([pos[0], pos[1], angle, 0.0, jnp.float32(idx)])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        key, subkey = jax.random.split(key)
        state = self._reset(subkey)
        obs = self._get_obsv(state)
        return state, obs, key


def main(
    yaml_path: Path,
    num_envs: int = 1024,
    seed: int = 42,
    lidar_range: float = 20.0,
    n_beams: int = 108,
    fov_deg: float = 270.0,
    n_lookup_angles: int = 108 * 3,
):
    init("jaxoracer", spawn=True)
    env = Environment(yaml_path, lidar_range, n_beams, fov_deg, n_lookup_angles)

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
