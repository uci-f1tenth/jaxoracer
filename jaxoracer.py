from collections import deque
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from cv2 import IMREAD_GRAYSCALE, imread
from jax import lax
from rerun import Image, LineStrips2D, init, log
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from typer import run
from yaml import safe_load

OCC_THRESH = 230
G = 9.81
N_RAY_ITERS = 64
N_SUBSTEPS = 4
COLLISION_REWARD = -10.0

X, Y, DELTA, V, PSI, DPSI, BETA, PROG = range(8)

VEHICLE_PARAMS = SimpleNamespace(
    tire=SimpleNamespace(p_dy1=1.0489, p_ky1=-4.9488),
    steering=SimpleNamespace(max=0.4189, min=-0.4189, v_max=3.2, v_min=-3.2),
    longitudinal=SimpleNamespace(a_max=9.51, v_max=20.0, v_min=-5.0, v_switch=7.319),
    a=0.15875,
    b=0.17145,
    h_s=0.074,
    m=3.74,
    I_z=0.04712,
)


class StepResult(NamedTuple):
    state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    key: jnp.ndarray


def wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def iround(x):
    return jnp.int32(jnp.round(x))


def clamp_steer_rate(angle, rate, p):
    c = jnp.clip(rate, p.v_min, p.v_max)
    at = ((angle <= p.min) & (c <= 0)) | ((angle >= p.max) & (c >= 0))
    return jnp.where(at, 0.0, c)


def clamp_accel(v, a, p):
    v_safe = jnp.where(v > p.v_switch, v, 1.0)
    pl = jnp.where(v > p.v_switch, p.a_max * p.v_switch / v_safe, p.a_max)
    c = jnp.clip(a, -p.a_max, pl)
    at = ((v <= p.v_min) & (c <= 0)) | ((v >= p.v_max) & (c >= 0))
    return jnp.where(at, 0.0, c)


def dynamics_ks_cog(x, us, ua, p):
    lwb = p.a + p.b
    beta = jnp.arctan(jnp.tan(x[DELTA]) * p.b / lwb)
    return jnp.array(
        [
            x[V] * jnp.cos(beta + x[PSI]),
            x[V] * jnp.sin(beta + x[PSI]),
            us,
            ua,
            x[V] * jnp.cos(beta) * jnp.tan(x[DELTA]) / lwb,
        ]
    )


def dynamics_st(x, u_raw, p):
    mu = p.tire.p_dy1
    C = -p.tire.p_ky1 / p.tire.p_dy1
    lf, lr, h, m, I = p.a, p.b, p.h_s, p.m, p.I_z
    lwb = lf + lr
    us = clamp_steer_rate(x[DELTA], u_raw[0], p.steering)
    ua = clamp_accel(x[V], u_raw[1], p.longitudinal)
    use_kin = jnp.abs(x[V]) < 0.1

    # kinematic branch
    f = dynamics_ks_cog(x, us, ua, p)
    db = (lr * us) / (
        lwb * jnp.cos(x[DELTA]) ** 2 * (1 + (jnp.tan(x[DELTA]) ** 2 * lr / lwb) ** 2)
    )
    ddp = (1 / lwb) * (
        ua * jnp.cos(x[BETA]) * jnp.tan(x[DELTA])
        - x[V] * jnp.sin(x[BETA]) * db * jnp.tan(x[DELTA])
        + x[V] * jnp.cos(x[BETA]) * us / jnp.cos(x[DELTA]) ** 2
    )
    kin = jnp.array([f[0], f[1], f[2], f[3], f[4], ddp, db])

    v_safe = jnp.where(use_kin, 1.0, x[V])
    Fnf = G * lr - ua * h
    Fnr = G * lf + ua * h
    dyn = jnp.array(
        [
            x[V] * jnp.cos(x[BETA] + x[PSI]),
            x[V] * jnp.sin(x[BETA] + x[PSI]),
            us,
            ua,
            x[DPSI],
            -mu * m / (v_safe * I * lwb) * (lf**2 * C * Fnf + lr**2 * C * Fnr) * x[DPSI]
            + mu * m / (I * lwb) * (lr * C * Fnr - lf * C * Fnf) * x[BETA]
            + mu * m / (I * lwb) * lf * C * Fnf * x[DELTA],
            (mu / (v_safe**2 * lwb) * (C * Fnr * lr - C * Fnf * lf) - 1) * x[DPSI]
            - mu / (v_safe * lwb) * (C * Fnr + C * Fnf) * x[BETA]
            + mu / (v_safe * lwb) * C * Fnf * x[DELTA],
        ]
    )

    return jnp.where(use_kin, kin, dyn)


def cast_ray(dt_map, row, col, dc, dr, max_steps, n_iters):
    h, w = dt_map.shape

    def step(state, _):
        t, done = state
        r, c = iround(row + t * dr), iround(col + t * dc)
        ib = (0 <= r) & (r < h) & (0 <= c) & (c < w)
        d = jnp.where(ib, dt_map[jnp.clip(r, 0, h - 1), jnp.clip(c, 0, w - 1)], 0.0)
        hit = (d < 1.0) | ~ib
        nt = jnp.minimum(t + jnp.maximum(d, 1.0), max_steps)
        nd = done | hit
        return (jnp.where(done, t, jnp.where(nd, t, nt)), nd), None

    (dist, _), _ = lax.scan(step, (jnp.float32(0.0), False), None, length=n_iters)
    return dist


def scan_lidar(dt_map, row, col, angles, max_px, n_iters, res):
    cast = partial(cast_ray, dt_map, row, col, max_steps=max_px, n_iters=n_iters)
    return jax.vmap(cast, (0, 0))(jnp.cos(angles), -jnp.sin(angles)) * res


class Map:
    def __init__(self, path: Path, lidar_range: float, n_ray_iters: int) -> None:
        with open(path, "r") as f:
            self.meta = safe_load(f)
        raw = imread(str(path.parent / self.meta["image"]), IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(path.parent / self.meta["image"])
        occ = raw < OCC_THRESH
        self.occupied = jnp.asarray(occ)
        self.res = float(self.meta["resolution"])
        self.ox, self.oy = float(self.meta["origin"][0]), float(self.meta["origin"][1])
        self.h, self.w = occ.shape
        self.dt = jnp.asarray(distance_transform_edt(~occ).astype(np.float32))
        self.max_range_px = lidar_range / self.res
        self.n_ray_iters = n_ray_iters
        self._compute_centerline(raw)
        self._build_prog_lut()

    def _compute_centerline(self, raw, smooth_window=51):
        skel = skeletonize(raw > OCC_THRESH)
        fg = set(map(tuple, np.argwhere(skel).tolist()))
        if len(fg) < 3:
            raise ValueError("Skeleton too small")

        def adj(p):
            return [
                (p[0] + dr, p[1] + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (p[0] + dr, p[1] + dc) in fg
            ]

        orow = self.h - 1 + self.oy / self.res
        ocol = -self.ox / self.res
        start = min(fg, key=lambda p: (p[0] - orow) ** 2 + (p[1] - ocol) ** 2)
        nbrs = adj(start)
        if len(nbrs) < 2:
            raise ValueError(f"Start {start} has < 2 neighbors")

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
        if found is None:
            raise ValueError("No loop in skeleton")

        path = [start]
        p = found
        while p != src:
            path.append(p)
            p = parent[p]
        path.append(src)
        path.reverse()

        world = np.array(
            [
                [c * self.res + self.ox, (self.h - 1 - r) * self.res + self.oy]
                for r, c in path
            ],
            dtype=np.float64,
        )
        w = min(smooth_window, len(world) // 2 * 2 - 1)
        if w >= 5:
            world = np.column_stack(
                [
                    savgol_filter(world[:, 0], w, 3, mode="wrap"),
                    savgol_filter(world[:, 1], w, 3, mode="wrap"),
                ]
            )

        self.centerline = jnp.asarray(world.astype(np.float32))
        self.centerline_px = jnp.column_stack(
            [
                (self.centerline[:, 0] - self.ox) / self.res,
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
            ]
        )
        diffs = jnp.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = jnp.arctan2(diffs[:, 1], diffs[:, 0])
        seg_lens = jnp.sqrt((diffs**2).sum(axis=1))
        self.cum_dist = jnp.cumsum(seg_lens)
        self.track_length = float(self.cum_dist[-1])
        self.n_waypoints = self.centerline.shape[0]

    def _build_prog_lut(self):
        cl_px = np.asarray(self.centerline_px)
        tree = KDTree(cl_px[:, ::-1])
        rows, cols = np.mgrid[0 : self.h, 0 : self.w]
        pts = np.column_stack([rows.ravel(), cols.ravel()])
        _, idxs = tree.query(pts)
        self.prog_lut = jnp.asarray(idxs.reshape(self.h, self.w).astype(np.int32))

    def w2px(self, x, y):
        return self.h - 1 - (y - self.oy) / self.res, (x - self.ox) / self.res

    def log(self, raw):
        log("map", Image(raw))
        log("dt", Image(np.asarray(self.dt)))
        log(
            "map/centerline",
            LineStrips2D([np.asarray(self.centerline_px)], colors=[[255, 0, 0]]),
        )


class Environment:
    def __init__(
        self,
        path: Path,
        lidar_range: float = 20.0,
        n_beams: int = 108,
        fov_deg: float = 270.0,
        num_envs: int = 1024,
        dt: float = 1 / 60,
        n_substeps: int = N_SUBSTEPS,
        n_ray_iters: int = N_RAY_ITERS,
        params: SimpleNamespace = VEHICLE_PARAMS,
    ):
        self.num_envs = num_envs
        self.sim_dt = dt
        self.n_substeps = n_substeps
        self.params = params
        self.map = Map(path, lidar_range, n_ray_iters)
        self.beam_offsets = jnp.linspace(
            -jnp.radians(fov_deg) / 2, jnp.radians(fov_deg) / 2, n_beams
        )

    def _get_obs(self, state):
        x, y, spd, psi = state[X], state[Y], state[V], state[PSI]
        row, col = self.map.w2px(x, y)
        r, c = (
            jnp.clip(iround(row), 0, self.map.h - 1),
            jnp.clip(iround(col), 0, self.map.w - 1),
        )
        lidar = scan_lidar(
            self.map.dt,
            r,
            c,
            psi + self.beam_offsets,
            self.map.max_range_px,
            self.map.n_ray_iters,
            self.map.res,
        )
        prog = self.map.prog_lut[r, c]
        ta = self.map.angles[prog]
        to_car = jnp.array([x, y]) - self.map.centerline[prog]
        lat = jnp.dot(to_car, jnp.array([-jnp.sin(ta), jnp.cos(ta)]))
        return jnp.concatenate([lidar, jnp.array([spd, lat, wrap(psi - ta)])])

    def _rk4(self, s, a):
        sub_dt = self.sim_dt / self.n_substeps
        p = self.params

        def body(_, s):
            k1 = dynamics_st(s, a, p)
            k2 = dynamics_st(s + (sub_dt / 2) * k1, a, p)
            k3 = dynamics_st(s + (sub_dt / 2) * k2, a, p)
            k4 = dynamics_st(s + sub_dt * k3, a, p)
            return s + (sub_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return lax.fori_loop(0, self.n_substeps, body, s)

    def _step(self, state, action, key):
        old_prog = jnp.int32(state[PROG])
        dyn = state[:7]
        new_dyn = self._rk4(dyn, action)
        new_dyn = new_dyn.at[PSI].set(wrap(new_dyn[PSI]))
        new_dyn = new_dyn.at[DELTA].set(
            jnp.clip(new_dyn[DELTA], self.params.steering.min, self.params.steering.max)
        )

        row, col = self.map.w2px(new_dyn[X], new_dyn[Y])
        r, c = (
            jnp.clip(iround(row), 0, self.map.h - 1),
            jnp.clip(iround(col), 0, self.map.w - 1),
        )
        hit = self.map.occupied[r, c]

        frozen = jnp.array([dyn[X], dyn[Y], dyn[DELTA], 0.0, dyn[PSI], 0.0, 0.0])
        safe = jnp.where(hit, frozen, new_dyn)

        sr, sc = self.map.w2px(safe[X], safe[Y])
        sr, sc = (
            jnp.clip(iround(sr), 0, self.map.h - 1),
            jnp.clip(iround(sc), 0, self.map.w - 1),
        )
        new_prog = self.map.prog_lut[sr, sc]
        new_state = jnp.concatenate([safe, jnp.array([jnp.float32(new_prog)])])

        delta = self.map.cum_dist[new_prog] - self.map.cum_dist[old_prog]
        tl = self.map.track_length
        delta = jnp.where(
            delta < -tl / 2, delta + tl, jnp.where(delta > tl / 2, delta - tl, delta)
        )
        reward = jnp.where(hit, COLLISION_REWARD, delta)

        return new_state, reward, hit

    def step_and_reset(self, state, action, key):
        def single(s, a, k):
            ns, r, d = self._step(s, a, k)

            obs = self._get_obs(ns)

            k, sk = jax.random.split(k)
            idx = jax.random.randint(sk, (), 0, self.map.n_waypoints)
            pos = self.map.centerline[idx]
            rs = jnp.array(
                [
                    pos[0],
                    pos[1],
                    0.0,
                    0.0,
                    self.map.angles[idx],
                    0.0,
                    0.0,
                    jnp.float32(idx),
                ]
            )
            ro = self._get_obs(rs)

            final_s = jnp.where(d, rs, ns)
            final_obs = jnp.where(d, ro, obs)

            return final_s, final_obs, r, d, k

        return jax.vmap(single)(state, action, key)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        keys = jax.random.split(key, self.num_envs)

        def single(k):
            k, sk = jax.random.split(k)
            idx = jax.random.randint(sk, (), 0, self.map.n_waypoints)
            pos = self.map.centerline[idx]
            s = jnp.array(
                [
                    pos[0],
                    pos[1],
                    0.0,
                    0.0,
                    self.map.angles[idx],
                    0.0,
                    0.0,
                    jnp.float32(idx),
                ]
            )
            return s, self._get_obs(s), k

        states, obs, keys = jax.vmap(single)(keys)
        return states, obs, keys

    # @partial(jax.jit, static_argnums=(0,))
    # def step(self, states, actions, keys):
    #     keys = jax.vmap(lambda k: jax.random.split(k)[0])(keys)

    #     def single(s, a, k):
    #         ns, r, d = self._step(s, a, k)
    #         obs = self._get_obs(ns)
    #         return StepResult(ns, obs, r, d, k)

    #     return jax.vmap(single)(states, actions, keys)

    # @partial(jax.jit, static_argnums=(0,))
    # def auto_reset(self, result):
    #     def single(s, o, r, d, k):
    #         k, sk = jax.random.split(k)
    #         idx = jax.random.randint(sk, (), 0, self.map.n_waypoints)
    #         pos = self.map.centerline[idx]
    #         rs = jnp.array(
    #             [
    #                 pos[0],
    #                 pos[1],
    #                 0.0,
    #                 0.0,
    #                 self.map.angles[idx],
    #                 0.0,
    #                 0.0,
    #                 jnp.float32(idx),
    #             ]
    #         )
    #         ro = self._get_obs(rs)
    #         return jnp.where(d, rs, s), jnp.where(d, ro, o), k

    #     states, obs, keys = jax.vmap(single)(
    #         result.state, result.obs, result.reward, result.done, result.key
    #     )
    #     return states, obs, keys


@partial(jax.jit, static_argnums=(0, 4))
def rollout(self, states, obs, keys, n_steps):
    def body(carry, _):
        states, obs, keys, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        actions = jnp.column_stack(
            [
                jax.random.uniform(k1, (self.num_envs,), minval=-3.2, maxval=3.2),
                jax.random.uniform(k2, (self.num_envs,), minval=-9.51, maxval=9.51),
            ]
        )
        states, obs, rewards, dones, keys = self.step_and_reset(states, actions, keys)
        return (states, obs, keys, key), rewards

    (states, obs, keys, _), rewards = lax.scan(
        body, (states, obs, keys, jax.random.PRNGKey(0)), None, length=n_steps
    )
    return states, obs, keys, rewards


def benchmark(env, key, num_envs, warmup=10, timed=200):
    states, obs, keys = env.reset(key)
    k = jax.random.PRNGKey(99)

    states, obs, keys, rewards = rollout(env, states, obs, keys, warmup)

    jax.block_until_ready(states)

    t0 = perf_counter()
    states, obs, keys, rewards = rollout(env, states, obs, keys, timed)
    jax.block_until_ready(states)
    elapsed = perf_counter() - t0

    total_steps = timed * num_envs
    print(f"\n{'=' * 50}")
    print(f"  {num_envs} envs × {timed} steps = {total_steps:,} total steps")
    print(f"  {elapsed:.3f}s elapsed")
    print(f"  {total_steps / elapsed:,.0f} steps/s")
    print(f"  {timed / elapsed:,.1f} batches/s")
    print(f"{'=' * 50}\n")


def main(
    yaml_path: Path,
    seed: int = 42,
    num_envs: int = 1024,
    lidar_range: float = 20.0,
    n_beams: int = 108,
    fov_deg: float = 270.0,
    steps: int = 1000,
    log_every: int = 100,
):
    init("jaxoracer", spawn=True)
    env = Environment(yaml_path, lidar_range, n_beams, fov_deg, num_envs)
    raw = imread(str(yaml_path.parent / env.map.meta["image"]), IMREAD_GRAYSCALE)
    env.map.log(raw)

    key = jax.random.PRNGKey(seed)
    benchmark(env, key, num_envs)

    # states, obs, keys = env.reset(key)
    # total_reward = jnp.zeros(num_envs)

    # for t in range(1, steps + 1):
    #     key, k1, k2 = jax.random.split(key, 3)
    #     actions = jnp.column_stack(
    #         [
    #             jax.random.uniform(k1, (num_envs,), minval=-3.2, maxval=3.2),
    #             jax.random.uniform(k2, (num_envs,), minval=-9.51, maxval=9.51),
    #         ]
    #     )
    #     result = env.step(states, actions, keys)
    #     total_reward += result.reward
    #     if t % log_every == 0:
    #         print(
    #             f"step {t:>5d} | avg reward/step: {float(jnp.mean(total_reward) / t):+.4f} | collisions: {int(jnp.sum(result.done))}"
    #         )
    #     states, obs, keys = env.auto_reset(result)


if __name__ == "__main__":
    run(main)
