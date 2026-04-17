"""Single-file PPO training for the JAX F1Tenth-style racing environment."""

from collections import deque
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import NamedTuple, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from cv2 import IMREAD_GRAYSCALE, imread
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import lax
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


def wrap_angle(a):
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
        self.obs_dim = n_beams + 3  # lidar + [speed, lateral_err, heading_err]
        self.act_dim = 2  # [steer_rate, accel]
        self.lidar_range = lidar_range

    def _get_obs(self, state):
        x, y, spd, psi = state[X], state[Y], state[V], state[PSI]
        row, col = self.map.w2px(x, y)
        r = jnp.clip(iround(row), 0, self.map.h - 1)
        c = jnp.clip(iround(col), 0, self.map.w - 1)
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
        return jnp.concatenate([lidar, jnp.array([spd, lat, wrap_angle(psi - ta)])])

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
        new_dyn = new_dyn.at[PSI].set(wrap_angle(new_dyn[PSI]))
        new_dyn = new_dyn.at[DELTA].set(
            jnp.clip(new_dyn[DELTA], self.params.steering.min, self.params.steering.max)
        )

        row, col = self.map.w2px(new_dyn[X], new_dyn[Y])
        r = jnp.clip(iround(row), 0, self.map.h - 1)
        c = jnp.clip(iround(col), 0, self.map.w - 1)
        hit = self.map.occupied[r, c]

        frozen = jnp.array([dyn[X], dyn[Y], dyn[DELTA], 0.0, dyn[PSI], 0.0, 0.0])
        safe = jnp.where(hit, frozen, new_dyn)

        sr, sc = self.map.w2px(safe[X], safe[Y])
        sr = jnp.clip(iround(sr), 0, self.map.h - 1)
        sc = jnp.clip(iround(sc), 0, self.map.w - 1)
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


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # ── Actor ──
        a = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        a = act_fn(a)
        a = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            a
        )
        a = act_fn(a)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(a)
        # Learnable per-action log standard deviation
        actor_logstd = self.param(
            "actor_logstd", nn.initializers.zeros, (self.action_dim,)
        )
        pi = distrax.MultivariateNormalDiag(
            loc=actor_mean, scale_diag=jnp.exp(actor_logstd)
        )

        # ── Critic ──
        c = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        c = act_fn(c)
        c = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            c
        )
        c = act_fn(c)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)

        return pi, jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


# Action bounds
STEER_RATE_MAX = 3.2
ACCEL_MAX = 9.51
ACTION_HIGH = jnp.array([STEER_RATE_MAX, ACCEL_MAX])
ACTION_LOW = -ACTION_HIGH


def scale_action(raw_action):
    """Squash network output → environment action range via tanh."""
    return jnp.tanh(raw_action) * ACTION_HIGH


def make_train(config, env: Environment):
    """Build a fully-jittable PPO train function for the custom Environment."""

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    obs_dim = env.obs_dim
    act_dim = env.act_dim
    lidar_range = env.lidar_range

    obs_scale = jnp.concatenate(
        [
            jnp.full((obs_dim - 3,), 1.0 / lidar_range),  # lidar
            jnp.array([1.0 / 20.0, 1.0 / 2.0, 1.0 / jnp.pi]),  # speed, lat, heading
        ]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        network = ActorCritic(action_dim=act_dim, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(obs_dim)
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        states, obs, env_keys = env.reset(_rng)

        def _update_step(runner_state, unused):
            train_state, states, obs, env_keys, rng = runner_state

            def _env_step(carry, unused):
                train_state, states, obs, env_keys, rng = carry

                normed_obs = obs * obs_scale

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, normed_obs)
                raw_action = pi.sample(seed=_rng)  # (num_envs, 2)
                log_prob = pi.log_prob(raw_action)  # (num_envs,)

                env_action = scale_action(raw_action)

                states_new, obs_new, reward, done, env_keys_new = env.step_and_reset(
                    states, env_action, env_keys
                )

                transition = Transition(
                    done=done,
                    action=raw_action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=normed_obs,
                )
                carry = (train_state, states_new, obs_new, env_keys_new, rng)
                return carry, transition

            carry, traj_batch = lax.scan(
                _env_step,
                (train_state, states, obs, env_keys, rng),
                None,
                length=config["NUM_STEPS"],
            )
            train_state, states, obs, env_keys, rng = carry

            normed_last = obs * obs_scale
            _, last_val = network.apply(train_state.params, normed_last)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Policy loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(_rng, batch_size)

                # Flatten (NUM_STEPS, NUM_ENVS, ...) → (batch_size, ...)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            avg_reward = traj_batch.reward.mean()
            collision_frac = traj_batch.done.mean()
            metric = {"avg_reward": avg_reward, "collision_frac": collision_frac}

            if config.get("DEBUG"):
                jax.debug.callback(
                    lambda m: print(
                        f"  avg_reward/step={float(m['avg_reward']):+.4f}  "
                        f"collision_frac={float(m['collision_frac']):.4f}"
                    ),
                    metric,
                )

            runner_state = (train_state, states, obs, env_keys, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, states, obs, env_keys, _rng)
        runner_state, metrics = lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def main(
    yaml_path: Path,
    seed: int = 42,
    num_envs: int = 1024,
    lidar_range: float = 20.0,
    n_beams: int = 108,
    fov_deg: float = 270.0,
    total_timesteps: int = 10_000_000,
    num_steps: int = 128,
    num_minibatches: int = 32,
    update_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.001,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    anneal_lr: bool = True,
    debug: bool = True,
):
    print(f"Building environment from {yaml_path} …")
    env = Environment(
        yaml_path,
        lidar_range=lidar_range,
        n_beams=n_beams,
        fov_deg=fov_deg,
        num_envs=num_envs,
    )
    print(
        f"  map {env.map.h}×{env.map.w}  |  "
        f"track_length={env.map.track_length:.1f}m  |  "
        f"obs_dim={env.obs_dim}  act_dim={env.act_dim}"
    )

    config = {
        "LR": lr,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "TOTAL_TIMESTEPS": total_timesteps,
        "UPDATE_EPOCHS": update_epochs,
        "NUM_MINIBATCHES": num_minibatches,
        "GAMMA": gamma,
        "GAE_LAMBDA": gae_lambda,
        "CLIP_EPS": clip_eps,
        "ENT_COEF": ent_coef,
        "VF_COEF": vf_coef,
        "MAX_GRAD_NORM": max_grad_norm,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": anneal_lr,
        "DEBUG": debug,
    }

    num_updates = int(total_timesteps // num_steps // num_envs)
    total_real = num_updates * num_steps * num_envs
    print(
        f"  {num_envs} envs × {num_steps} steps × {num_updates} updates "
        f"= {total_real:,} total timesteps"
    )

    print("Compiling PPO (this may take a few minutes on first run) …")
    rng = jax.random.PRNGKey(seed)
    train_fn = jax.jit(make_train(config, env))

    t0 = perf_counter()
    out = train_fn(rng)
    jax.block_until_ready(out)
    elapsed = perf_counter() - t0

    metrics = out["metrics"]
    avg_r = np.asarray(metrics["avg_reward"])
    col_f = np.asarray(metrics["collision_frac"])

    print(f"\n{'=' * 60}")
    print(f"  Training completed in {elapsed:.1f}s")
    print(f"  {total_real / elapsed:,.0f} env steps/s")
    print(f"  Final avg_reward/step : {avg_r[-1]:+.4f}")
    print(f"  Final collision_frac  : {col_f[-1]:.4f}")
    print(f"  Best  avg_reward/step : {avg_r.max():+.4f}")
    print(f"{'=' * 60}\n")

    final_params = out["runner_state"][0].params
    save_path = yaml_path.parent / "ppo_params.npz"
    flat_params = {
        "/".join(map(str, k)): np.asarray(v)
        for k, v in jax.tree_util.tree_leaves_with_path(final_params)
    }
    np.savez(str(save_path), **flat_params)
    print(f"Saved parameters to {save_path}")


if __name__ == "__main__":
    run(main)
