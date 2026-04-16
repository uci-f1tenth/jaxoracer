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


def _wrap_angle(a):
    """Wrap angle to [-π, π)."""
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


# ---------------------------------------------------------------------------
# Map — owns the occupancy grid, distance transform, centerline, and the
#        precomputed ray-distance lookup table at *uniform absolute angles*
#        covering the full 360°.
# ---------------------------------------------------------------------------
class Map:
    def __init__(
        self, path: Path, lidar_range: float, n_lookup_angles: int = 360
    ) -> None:
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

        # lookup shape: (H, W, n_lookup_angles) — distances in metres
        self.lookup = (
            self.build_lookup(lidar_range / self.resolution, n_lookup_angles)
            * self.resolution
        )

    @partial(jax.jit, static_argnums=(0, 2))
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

    # centerline extraction is unchanged --------------------------------
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

    # ray marching is unchanged -----------------------------------------
    @staticmethod
    def _cast_ray(dt, row, col, dc, dr, max_steps):
        h, w = dt.shape

        def step(state, _):
            t, done = state
            r = jnp.int32(jnp.round(row + t * dr))
            c = jnp.int32(jnp.round(col + t * dc))
            in_bounds = (0 <= r) & (r < h) & (0 <= c) & (c < w)
            d = jnp.where(
                in_bounds,
                dt[jnp.clip(r, 0, h - 1), jnp.clip(c, 0, w - 1)],
                0.0,
            )
            hit = (d < 1.0) | ~in_bounds
            new_t = jnp.minimum(t + jnp.maximum(d, 1.0), max_steps)
            new_done = done | hit
            return (
                jnp.where(done, t, jnp.where(new_done, t, new_t)),
                new_done,
            ), None

        (dist, _), _ = jax.lax.scan(step, (jnp.float32(0.0), False), None, length=32)
        return dist


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Environment:
    DT = 0.1
    WHEELBASE = 0.32
    MAX_SPEED = 8.0
    MIN_SPEED = 0.0
    MAX_STEER = 0.4
    MAX_ACCEL = 7.0
    MAX_DECEL = -7.0
    COLLISION_PENALTY = -10.0
    MAX_STEPS = 2000

    def __init__(
        self,
        path: Path,
        lidar_range: float,
        n_beams: int = 108,
        fov_deg: float = 270.0,
        n_lookup_angles: int = 360,
        seed: int = 42,
    ):
        self.n_beams = n_beams
        self.fov = jnp.radians(fov_deg)
        self.n_lookup_angles = n_lookup_angles
        self.map = Map(path, lidar_range, n_lookup_angles)

        # ego-centric beam offsets: 0 = dead ahead,
        # negative = right, positive = left
        self.beam_offsets = jnp.linspace(-self.fov / 2, self.fov / 2, n_beams)

        # cumulative arc-length along centerline (for progress reward)
        cl = self.map.centerline_world
        diffs = jnp.diff(cl, axis=0, append=cl[:1])
        seg_lens = jnp.sqrt((diffs**2).sum(axis=1))
        self.cum_dist = jnp.cumsum(seg_lens)
        self.track_length = self.cum_dist[-1]
        self.n_waypoints = cl.shape[0]

    # ---- coordinate helpers -------------------------------------------

    def _world_to_px(self, x, y):
        h = self.map.occupied.shape[0]
        ox, oy, _ = self.map.origin
        res = self.map.resolution
        col = (x - ox) / res
        row = h - 1 - (y - oy) / res
        return row, col

    # ---- observation --------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _get_obsv(self, state):
        x, y, theta, speed, prog_idx = state[:5]
        row, col = self._world_to_px(x, y)
        h, w = self.map.occupied.shape

        r_idx = jnp.clip(jnp.int32(jnp.round(row)), 0, h - 1)
        c_idx = jnp.clip(jnp.int32(jnp.round(col)), 0, w - 1)

        # absolute angle of every beam
        abs_angles = theta + self.beam_offsets  # (n_beams,)

        # nearest index into the [0, 2π) lookup ring
        lookup_idx = (
            jnp.int32(jnp.round(abs_angles / (2 * jnp.pi) * self.n_lookup_angles))
            % self.n_lookup_angles
        )

        ego_lidar = self.map.lookup[r_idx, c_idx, lookup_idx]  # (n_beams,)

        # relative heading to centerline tangent
        cl = self.map.centerline_world
        idx = jnp.int32(prog_idx) % self.n_waypoints
        nxt = (idx + 1) % self.n_waypoints
        tangent = cl[nxt] - cl[idx]
        track_angle = jnp.arctan2(tangent[1], tangent[0])
        rel_heading = _wrap_angle(theta - track_angle)

        # signed lateral offset from centerline
        to_car = jnp.array([x, y]) - cl[idx]
        normal = jnp.array([-tangent[1], tangent[0]])
        normal = normal / (jnp.linalg.norm(normal) + 1e-8)
        lat_offset = jnp.dot(to_car, normal)

        return jnp.concatenate([ego_lidar, jnp.array([speed, rel_heading, lat_offset])])

    # ---- dynamics -----------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action, key):
        """
        state  : [x, y, θ, speed, prog_idx, step_count]
        action : [steer, accel]
        returns: (new_state, obs, reward, done, new_key)
        """
        x, y, theta, speed, prog_idx, step_count = state

        steer = jnp.clip(action[0], -self.MAX_STEER, self.MAX_STEER)
        accel = jnp.clip(action[1], self.MAX_DECEL, self.MAX_ACCEL)

        new_speed = jnp.clip(speed + accel * self.DT, self.MIN_SPEED, self.MAX_SPEED)
        dtheta = (new_speed * jnp.tan(steer) / self.WHEELBASE) * self.DT
        new_theta = _wrap_angle(theta + dtheta)
        new_x = x + new_speed * jnp.cos(new_theta) * self.DT
        new_y = y + new_speed * jnp.sin(new_theta) * self.DT

        # collision check via distance transform
        row, col = self._world_to_px(new_x, new_y)
        h, w = self.map.occupied.shape
        in_bounds = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        r_c = jnp.clip(jnp.int32(jnp.round(row)), 0, h - 1)
        c_c = jnp.clip(jnp.int32(jnp.round(col)), 0, w - 1)
        wall_dist = self.map.dt[r_c, c_c] * self.map.resolution
        collision = (~in_bounds) | (wall_dist < self.map.resolution * 1.5)

        # progress reward
        cl = self.map.centerline_world
        dists_sq = ((cl - jnp.array([new_x, new_y])) ** 2).sum(axis=1)
        new_prog_idx = jnp.float32(jnp.argmin(dists_sq))

        old_d = self.cum_dist[jnp.int32(prog_idx) % self.n_waypoints]
        new_d = self.cum_dist[jnp.int32(new_prog_idx) % self.n_waypoints]
        delta = new_d - old_d
        delta = jnp.where(
            delta < -self.track_length / 2,
            delta + self.track_length,
            delta,
        )
        delta = jnp.where(
            delta > self.track_length / 2,
            delta - self.track_length,
            delta,
        )

        reward = delta + jnp.where(collision, self.COLLISION_PENALTY, 0.0)

        new_step_count = step_count + 1
        done = collision | (new_step_count >= self.MAX_STEPS)

        new_state = jnp.array(
            [
                new_x,
                new_y,
                new_theta,
                new_speed,
                new_prog_idx,
                new_step_count,
            ]
        )

        key, subkey = jax.random.split(key)
        new_state = jax.lax.cond(done, lambda: self._reset(subkey), lambda: new_state)
        obs = self._get_obsv(new_state)
        return new_state, obs, reward, done, key

    # ---- reset --------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, key):
        cl = self.map.centerline_world
        idx = jax.random.randint(key, (), 0, self.n_waypoints)
        nxt = (idx + 1) % self.n_waypoints
        pos = cl[idx]
        tangent = cl[nxt] - cl[idx]
        theta = jnp.arctan2(tangent[1], tangent[0])
        return jnp.array([pos[0], pos[1], theta, 0.0, jnp.float32(idx), 0.0])

    def reset(self, key):
        key, subkey = jax.random.split(key)
        state = self._reset(subkey)
        obs = self._get_obsv(state)
        return state, obs, key

    # ---- batched interface --------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def batch_reset(self, key, num_envs: int = 1024):
        keys = jax.random.split(key, num_envs + 1)
        states = jax.vmap(self._reset)(keys[1:])
        obs = jax.vmap(self._get_obsv)(states)
        return states, obs, keys[0]

    @partial(jax.jit, static_argnums=(0,))
    def batch_step(self, states, actions, keys):
        return jax.vmap(self.step)(states, actions, keys)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(
    yaml_path: Path,
    num_envs: int = 1024,
    seed: int = 42,
    lidar_range: float = 20.0,
    n_beams: int = 108,
    fov_deg: float = 270.0,
    n_lookup_angles: int = 360,
):
    init("jaxoracer", spawn=True)
    env = Environment(
        yaml_path,
        lidar_range,
        n_beams=n_beams,
        fov_deg=fov_deg,
        n_lookup_angles=n_lookup_angles,
        seed=seed,
    )

    key = jax.random.PRNGKey(seed)
    states, obs, key = env.batch_reset(key, num_envs)

    for _ in range(1000):
        key, act_key = jax.random.split(key)
        actions = jax.random.uniform(act_key, (num_envs, 2), minval=-1.0, maxval=1.0)
        actions = actions.at[:, 0].multiply(env.MAX_STEER)
        actions = actions.at[:, 1].multiply(env.MAX_ACCEL)
        states, obs, rewards, dones, keys = env.batch_step(
            states, actions, jax.random.split(key, num_envs)
        )


if __name__ == "__main__":
    run(main)
