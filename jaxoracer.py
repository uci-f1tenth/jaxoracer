"""learnrl_jax.py — F1tenth RL in pure JAX (kinematic bicycle model).

  • lax.scan compiles the T-step rollout into one XLA program.
  • jax.vmap parallelises all N environments natively on the GPU.
  • Kinematic bicycle model: 5 states, single RK4 step per tick.

Usage (CPU):
  uv run --with "jax[cpu]" --with optax --with scipy --with scikit-image \
         --with pillow --with pyyaml --with tqdm --with tyro \
         scripts/learnrl_jax.py

Usage (GPU/CUDA):
  uv run --with "jax[cuda12]" --with optax --with scipy --with scikit-image \
         --with pillow --with pyyaml --with tqdm --with tyro \
         scripts/learnrl_jax.py
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import optax
import scipy.ndimage as ndi
import scipy.signal
import skimage.morphology
import tyro
import tqdm
import yaml as pyyaml
from PIL import Image
from scipy.spatial import KDTree

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")


# ── CLI config ─────────────────────────────────────────────────────────────────

@dataclass
class Args:
    yaml: str = "maps/my_map.yaml"
    num_envs: int = 1024
    num_steps: int = 128
    total_timesteps: int = 200_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    seed: int = 42
    max_steps: int = 10_000
    dr_frac: float = 0.2
    save_interval: int = 50
    save_dir: str = "checkpoints"
    scan_steps: int = 30
    """Fixed raycast march iterations (EDT sphere-tracing)."""


# ── Physics constants ──────────────────────────────────────────────────────────

STEER_MIN      = np.float32(-0.5236)
STEER_MAX      = np.float32( 0.5236)
STEER_VEL_MAX  = np.float32(3.2)
V_MIN          = np.float32(-5.0)
V_MAX          = np.float32(20.0)
WIDTH          = np.float32(0.27)
LENGTH         = np.float32(0.50)
CD_A_RHO_HALF  = np.float32(0.5 * 0.3 * 0.04 * 1.225)
PI             = np.float32(np.pi)
DT             = np.float32(1. / 60.)

N_BEAMS = 108
FOV      = np.float32(270.0 * np.pi / 180.0)
MIN_RANGE = np.float32(0.06)
MAX_RANGE = np.float32(10.0)
N_LOOK   = 10

# Kinematic bicycle model parameters
P_DEF: dict[str, np.float32] = {k: np.float32(v) for k, v in {
    "lf": 0.15532, "lr": 0.16868, "mass": 3.906, "a_max": 9.51}.items()}
RAND_KEYS = ("lf", "lr", "mass", "a_max")


# ── MAP LOADING ────────────────────────────────────────────────────────────────

class MapData:
    """Holds map arrays as JAX tensors for use inside JIT-compiled code."""

    def __init__(self, yaml_path: str,
                 edt_thresh_frac: float = 0.1,
                 sg_window: int = 11, sg_degree: int = 3):
        meta = pyyaml.safe_load(open(yaml_path))
        img_path = Path(yaml_path).parent / meta["image"]
        img = np.array(Image.open(img_path).convert("L"))
        res = float(meta["resolution"])
        ox = float(meta["origin"][0])
        oy = float(meta["origin"][1])
        otheta = float(meta["origin"][2]) if len(meta["origin"]) > 2 else 0.0
        h, w = img.shape

        free = img >= 210
        edt_np = ndi.distance_transform_edt(free).astype(np.float32) * res

        edt_px = ndi.distance_transform_edt(free)
        wide_free = free & (edt_px >= float(edt_px.max() * edt_thresh_frac))
        thin = skimage.morphology.skeletonize(wide_free)
        ordered = _extract_loop(thin, ox, oy, res, otheta, h)
        assert len(ordered) >= 2, "Could not extract skeleton loop from map"
        smoothed = np.array(
            scipy.signal.savgol_filter(
                np.asarray(ordered, dtype=np.float64),
                sg_window, sg_degree, axis=0, mode="wrap"),
            dtype=np.float32)

        tree = KDTree(smoothed)
        ys_g, xs_g = np.mgrid[0:h, 0:w]
        world_x = xs_g.ravel().astype(np.float32) * res + ox
        world_y = (h - 1 - ys_g.ravel()).astype(np.float32) * res + oy
        _, lut = tree.query(np.c_[world_x, world_y])

        angles = np.linspace(-FOV / 2, FOV / 2, N_BEAMS, dtype=np.float32)
        beam_sc = np.stack([np.sin(angles), np.cos(angles)], axis=1)

        seg = np.linalg.norm(smoothed[1:] - smoothed[:-1], axis=1)
        look_step = max(1, round(1.0 / seg.mean()))

        self.edt      = jnp.array(edt_np)
        self.skeleton = jnp.array(smoothed)
        self.skel_lut = jnp.array(lut.astype(np.int32))
        self.beam_sc  = jnp.array(beam_sc)
        self.w, self.h = w, h
        self.res      = np.float32(res)
        self.inv_res  = np.float32(1.0 / res)
        self.ox       = np.float32(ox)
        self.oy       = np.float32(oy)
        self.n_wps    = int(len(smoothed))
        self.look_step = int(look_step)
        self.obs_dim  = N_BEAMS + 3 + 2 + N_LOOK * 2  # lidar(108) + [v,δ,ψ̇](3) + [heading,lat](2) + lookahead(20)


def _nbrs8(p: tuple, pts: set) -> list:
    x, y = p
    return [(x + dx, y + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0) and (x + dx, y + dy) in pts]


def _largest_component(pts: set) -> set:
    visited: set = set()
    best: list = []
    for p in pts:
        if p in visited:
            continue
        comp: list = []
        q = [p]
        visited.add(p)
        for cur in q:
            comp.append(cur)
            for n in _nbrs8(cur, pts):
                if n not in visited:
                    visited.add(n)
                    q.append(n)
        if len(comp) > len(best):
            best = comp
    return set(best)


def _extract_loop(thin_img: np.ndarray, ox, oy, res, otheta, h) -> list:
    ys, xs = np.where(thin_img)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if not pts:
        raise RuntimeError("Empty skeleton")
    pts = _largest_component(pts)
    for _ in range(500):
        endpoints = [p for p in pts if len(_nbrs8(p, pts)) < 2]
        if not endpoints:
            break
        pts -= set(endpoints)
    if not pts:
        raise RuntimeError("Skeleton collapsed to empty after pruning")
    origin_px = (-ox / res, (h - 1) + oy / res)
    start = min(pts, key=lambda p: (p[0] - origin_px[0]) ** 2
                + (p[1] - origin_px[1]) ** 2)
    adj = _nbrs8(start, pts)
    if len(adj) < 2:
        raise RuntimeError("Start node has fewer than 2 neighbours")
    src, targets = adj[0], set(adj[1:])
    parent: dict = {src: src}
    queue = [src]
    found = None
    for cur in queue:
        for n in _nbrs8(cur, pts):
            if n == start or n in parent:
                continue
            parent[n] = cur
            if n in targets:
                found = n
                break
            queue.append(n)
        if found:
            break
    path = [start]
    if found:
        p = found
        while p != src:
            path.append(p)
            p = parent[p]
        path.append(src)
    path.reverse()
    world = [[px * res + ox, (h - 1 - py) * res + oy] for px, py in path]
    if len(world) >= 2:
        dx0, dy0 = world[1][0] - world[0][0], world[1][1] - world[0][1]
        err = ((np.arctan2(dy0, dx0) - otheta + 3 * np.pi) % (2 * np.pi)) - np.pi
        if abs(err) > np.pi / 2:
            world[1:] = world[1:][::-1]
    return world


# ── Kinematic bicycle model ────────────────────────────────────────────────────
# State: [x, y, delta, v, psi]  (5 floats)

def _kin_dyn(s, accel, sv, p):
    """Kinematic bicycle model — continuous-time derivatives."""
    x, y, delta, v, psi = s
    lwb = p["lf"] + p["lr"]
    drag = CD_A_RHO_HALF * v * jnp.abs(v) / p["mass"]
    return jnp.array([
        v * jnp.cos(psi),
        v * jnp.sin(psi),
        sv,
        accel - drag,
        v / lwb * jnp.tan(delta),
    ])


# ── JAX environment functions (closed over a fixed MapData) ───────────────────

def make_sim_fns(map_data: MapData, max_steps: int, dr_frac: float,
                 dt: float = float(DT), scan_steps: int = 30):
    """Return JIT-compiled batched environment functions.

    Returns:
        batched_step : (state, actions) → (new_state, (reward, term, trunc))
        batched_obs  : state → (N, obs_dim)
        batched_init : rng_keys(N,2) → state
    """
    edt      = map_data.edt
    skeleton = map_data.skeleton
    skel_lut = map_data.skel_lut
    beam_sc  = map_data.beam_sc
    W, H     = map_data.w, map_data.h
    res, inv_res = map_data.res, map_data.inv_res
    ox, oy   = map_data.ox, map_data.oy
    n_wps    = jnp.int32(map_data.n_wps)
    look_step = jnp.int32(map_data.look_step)
    dt_f     = jnp.float32(dt)
    Wf, Hf   = jnp.float32(W), jnp.float32(H)
    CIRCUMRADIUS = jnp.float32(np.sqrt((LENGTH / 2.) ** 2 + (WIDTH / 2.) ** 2))

    def _collides(dyn):
        x, y = dyn[0], dyn[1]
        px_f = (x - ox) * inv_res
        py_f = (Hf - 1.) - (y - oy) * inv_res
        not_finite = ~jnp.isfinite(x) | ~jnp.isfinite(y)
        in_map = (px_f >= 0.) & (px_f < Wf) & (py_f >= 0.) & (py_f < Hf)
        pxi = jnp.int32(jnp.clip(px_f, 0., Wf - 1.))
        pyi = jnp.int32(jnp.clip(py_f, 0., Hf - 1.))
        edt_val = jnp.where(in_map, edt[pyi, pxi], 0.)
        return not_finite | (edt_val < CIRCUMRADIUS)

    def _compute_obs(dyn, wp_idx, p):
        x, y, delta, v, psi = dyn
        sh, ch = jnp.sin(psi), jnp.cos(psi)
        lwb = p["lf"] + p["lr"]
        yaw_rate = v / lwb * jnp.tan(delta)
        lx, ly = x + p["lf"] * ch, y + p["lf"] * sh

        sa, ca = beam_sc[:, 0], beam_sc[:, 1]
        bdx = ch * ca - sh * sa
        bdy = sh * ca + ch * sa
        px0 = (lx - ox) * inv_res
        py0 = (Hf - 1.) - (ly - oy) * inv_res
        dpx = bdx * inv_res
        dpy = -bdy * inv_res
        min_step = jnp.float32(res * 0.5)

        def ray_step(carry, _):
            t, result, done = carry
            pxf = px0 + t * dpx
            pyf = py0 + t * dpy
            ib = (pxf >= 0.) & (pxf < Wf) & (pyf >= 0.) & (pyf < Hf)
            pxi = jnp.int32(jnp.clip(pxf, 0., Wf - 1.))
            pyi = jnp.int32(jnp.clip(pyf, 0., Hf - 1.))
            d = jnp.where(ib, edt[pyi, pxi], 0.)
            hit = (~done) & (~ib | (d < min_step))
            new_result = jnp.where(hit, t, result)
            new_done = done | hit
            new_t = jnp.where(new_done, t, t + jnp.maximum(d, min_step))
            return (new_t, new_result, new_done), None

        t0 = jnp.full(N_BEAMS, MIN_RANGE)
        r0 = jnp.full(N_BEAMS, MAX_RANGE)
        d0 = jnp.zeros(N_BEAMS, dtype=jnp.bool_)
        (_, scans, _), _ = lax.scan(ray_step, (t0, r0, d0), None, length=scan_steps)

        wp_pt   = skeleton[wp_idx]
        wp_next = skeleton[(wp_idx + 1) % n_wps]
        cth = jnp.arctan2(wp_next[1] - wp_pt[1], wp_next[0] - wp_pt[0])
        heading_err = ((cth - psi) + PI) % (2. * PI) - PI
        lat_err = -(x - wp_pt[0]) * jnp.sin(cth) + (y - wp_pt[1]) * jnp.cos(cth)

        look_idx = (wp_idx + jnp.arange(1, N_LOOK + 1, dtype=jnp.int32) * look_step) % n_wps
        look_pts = skeleton[look_idx]
        dx_w, dy_w = look_pts[:, 0] - x, look_pts[:, 1] - y
        look_local = jnp.stack([dx_w * ch + dy_w * sh, -dx_w * sh + dy_w * ch],
                                axis=1).ravel()

        return jnp.concatenate([scans, jnp.array([v, delta, yaw_rate]),
                                 jnp.array([heading_err, lat_err]), look_local])

    def _random_state(rng):
        rng_new, rng_draw = jax.random.split(rng)
        ri = jax.random.randint(
            jax.random.fold_in(rng_draw, jnp.uint32(0)), (), 0, n_wps)
        pt      = skeleton[ri]
        pt_next = skeleton[(ri + 1) % n_wps]
        th = jnp.arctan2(pt_next[1] - pt[1], pt_next[0] - pt[0])
        p = dict(P_DEF)
        for i, k in enumerate(RAND_KEYS):
            u = jax.random.uniform(jax.random.fold_in(rng_draw, jnp.uint32(i + 1)))
            p[k] = P_DEF[k] * (1. + jnp.float32(dr_frac) * (u * 2. - 1.))
        dyn = jnp.array([pt[0], pt[1], 0., 0., th])
        return dyn, ri, p, rng_new

    def _single_step(state, action):
        dyn, wp_idx, steps, rng, p = (
            state["dyn"], state["wp_idx"], state["steps"],
            state["rng"], state["p"])

        sv     = jnp.clip(action[0], -1., 1.) * STEER_VEL_MAX
        sv     = jnp.where((sv < 0.) & (dyn[2] <= STEER_MIN), 0., sv)
        sv     = jnp.where((sv > 0.) & (dyn[2] >= STEER_MAX), 0., sv)
        accel  = jnp.clip(action[1], -1., 1.) * p["a_max"]

        # Single RK4 step
        f  = lambda s: _kin_dyn(s, accel, sv, p)
        k1 = f(dyn)
        k2 = f(dyn + k1 * (dt_f / 2))
        k3 = f(dyn + k2 * (dt_f / 2))
        k4 = f(dyn + k3 * dt_f)
        dyn_new = dyn + dt_f / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

        x, y, delta, v, psi = dyn_new
        dyn_new = jnp.array([x, y,
                              jnp.clip(delta, STEER_MIN, STEER_MAX),
                              jnp.clip(v, V_MIN, V_MAX),
                              (psi + PI) % (2. * PI) - PI])

        px_new = jnp.int32(jnp.clip((x - ox) * inv_res, 0., Wf - 1.))
        py_new = jnp.int32(jnp.clip((Hf - 1.) - (y - oy) * inv_res, 0., Hf - 1.))
        new_wi  = skel_lut[py_new * W + px_new]
        d_wp    = jnp.float32(new_wi - wp_idx)
        d_wp    = jnp.where(d_wp >  jnp.float32(n_wps) / 2., d_wp - jnp.float32(n_wps), d_wp)
        d_wp    = jnp.where(d_wp < -jnp.float32(n_wps) / 2., d_wp + jnp.float32(n_wps), d_wp)

        edt_val    = edt[py_new, px_new]
        terminated = _collides(dyn_new)
        truncated  = (steps + jnp.int32(1)) >= jnp.int32(max_steps)
        reward = (d_wp / jnp.float32(n_wps) * 100. * (1. + jnp.maximum(v, 0.) / 10.)
                  - 0.1 * jnp.exp(-3. * edt_val)
                  - jnp.where(terminated, 100., 0.))

        reset_dyn, reset_ri, reset_p, new_rng = _random_state(rng)
        done = terminated | truncated
        final_dyn   = jnp.where(done, reset_dyn, dyn_new)
        final_wi    = jnp.where(done, reset_ri, new_wi)
        final_steps = jnp.where(done, jnp.int32(0), steps + jnp.int32(1))
        final_p     = jax.tree.map(lambda a, b: jnp.where(done, a, b), reset_p, p)

        new_state = {"dyn": final_dyn, "wp_idx": final_wi,
                     "steps": final_steps, "rng": new_rng, "p": final_p,
                     "obs": _compute_obs(final_dyn, final_wi, final_p)}
        return new_state, (reward, terminated, truncated)

    def _single_obs(state):
        return _compute_obs(state["dyn"], state["wp_idx"], state["p"])

    def _init_single(rng):
        dyn, ri, p, rng = _random_state(rng)
        return {"dyn": dyn, "wp_idx": ri, "steps": jnp.int32(0), "rng": rng, "p": p,
                "obs": _compute_obs(dyn, ri, p)}

    batched_step = jax.jit(jax.vmap(_single_step))
    batched_obs  = jax.jit(jax.vmap(_single_obs))
    batched_init = jax.jit(jax.vmap(_init_single))
    return batched_step, batched_obs, batched_init


# ── Policy (hand-rolled MLP) ──────────────────────────────────────────────────

LOG_STD_MIN, LOG_STD_MAX = jnp.float32(-5.), jnp.float32(0.5)


def init_policy(key, obs_dim: int, hidden: int, act_dim: int = 2):
    def layer(k, i, o, std=float(np.sqrt(2))):
        return {"w": jax.random.normal(k, (i, o)) * float(std / np.sqrt(i)),
                "b": jnp.zeros(o)}
    ks = jax.random.split(key, 6)
    return {
        "actor":      [layer(ks[0], obs_dim, hidden),
                       layer(ks[1], hidden,  hidden),
                       layer(ks[2], hidden,  act_dim, std=0.01)],
        "actor_logstd": jnp.full((act_dim,), -0.5, dtype=jnp.float32),
        "critic":     [layer(ks[3], obs_dim, hidden),
                       layer(ks[4], hidden,  hidden),
                       layer(ks[5], hidden,  1, std=1.)],
    }


def _mlp(layers, x):
    for lay in layers[:-1]:
        x = jax.nn.tanh(x @ lay["w"] + lay["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def _policy_forward(params, obs, key):
    mean    = _mlp(params["actor"], obs)
    log_std = jnp.clip(params["actor_logstd"], LOG_STD_MIN, LOG_STD_MAX)
    std     = jnp.exp(log_std)
    action  = mean + std * jax.random.normal(key, mean.shape)
    lp      = (-0.5 * ((action - mean) / std) ** 2
               - jnp.log(std) - 0.5 * jnp.log(2. * jnp.pi)).sum(-1)
    value   = _mlp(params["critic"], obs).squeeze(-1)
    return action, lp, value


def _policy_evaluate(params, obs, action):
    mean    = _mlp(params["actor"], obs)
    log_std = jnp.clip(params["actor_logstd"], LOG_STD_MIN, LOG_STD_MAX)
    std     = jnp.exp(log_std)
    lp      = (-0.5 * ((action - mean) / std) ** 2
               - jnp.log(std) - 0.5 * jnp.log(2. * jnp.pi)).sum(-1)
    entropy = (jnp.log(std) + 0.5 * jnp.log(2. * jnp.pi * jnp.e)).sum(-1)
    value   = _mlp(params["critic"], obs).squeeze(-1)
    return lp, entropy, value


# ── PPO rollout + update (fully JIT-compiled) ─────────────────────────────────

def make_ppo_fns(batched_step, batched_obs, optimizer, args: Args):
    N      = args.num_envs
    T      = args.num_steps
    B      = T * N
    mnb    = args.num_minibatches
    mb     = B // mnb
    epochs = args.update_epochs
    gamma  = jnp.float32(args.gamma)
    lam    = jnp.float32(args.gae_lambda)
    clip_c = jnp.float32(args.clip_coef)
    ent_c  = jnp.float32(args.ent_coef)
    vf_c   = jnp.float32(args.vf_coef)
    env_ids = jnp.arange(N, dtype=jnp.uint32)

    @jax.jit
    def collect_rollout(params, state, rng):
        def step_fn(carry, _):
            state, rng = carry
            obs = state["obs"]
            rng, key = jax.random.split(rng)
            action, lp, value = jax.vmap(
                lambda o, i: _policy_forward(params, o, jax.random.fold_in(key, i))
            )(obs, env_ids)
            new_state, (rew, term, trunc) = batched_step(state, action)
            done = (term | trunc).astype(jnp.float32)
            return (new_state, rng), (obs, action, rew, done, lp, value)

        (final_state, rng), traj = lax.scan(step_fn, (state, rng), None, length=T)
        return final_state, rng, traj

    @jax.jit
    def update(params, opt_state, traj, last_value, rng):
        obs, actions, rewards, dones, old_lp, values = traj

        def gae_step(gae_acc, idx):
            v_next = jnp.where(idx < T - 1,
                               values[jnp.minimum(idx + 1, T - 1)], last_value)
            delta   = rewards[idx] + gamma * v_next * (1. - dones[idx]) - values[idx]
            new_gae = delta + gamma * lam * (1. - dones[idx]) * gae_acc
            return new_gae, new_gae

        _, advs_rev = lax.scan(gae_step, jnp.zeros(N, dtype=jnp.float32),
                               jnp.arange(T - 1, -1, -1, dtype=jnp.int32))
        advs    = advs_rev[::-1]
        returns = advs + values

        def fl(x): return x.reshape(B, *x.shape[2:])
        obs_f = fl(obs);  act_f = fl(actions)
        ret_f = returns.ravel();  lp_f = old_lp.ravel()
        adv_f = advs.ravel()
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        def epoch_fn(carry, _):
            params, opt_state, rng = carry
            rng, sk = jax.random.split(rng)
            idx = jax.random.permutation(sk, B)

            def mb_fn(carry, mb_i):
                params, opt_state = carry
                sel   = lax.dynamic_slice(idx, (mb_i * mb,), (mb,))
                b_obs = obs_f[sel]; b_act = act_f[sel]
                b_ret = ret_f[sel]; b_adv = adv_f[sel]; b_lp = lp_f[sel]

                def loss_fn(p):
                    new_lp, ent, new_v = jax.vmap(
                        lambda o, a: _policy_evaluate(p, o, a))(b_obs, b_act)
                    ratio = jnp.exp(new_lp - b_lp)
                    pg = -jnp.minimum(ratio * b_adv,
                                      jnp.clip(ratio, 1. - clip_c, 1. + clip_c) * b_adv).mean()
                    vl = 0.5 * jnp.mean((new_v - b_ret) ** 2)
                    return pg + vf_c * vl + ent_c * (-ent.mean())

                loss, grads = jax.value_and_grad(loss_fn)(params)
                updates, new_opt = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), new_opt), loss

            (params, opt_state), losses = lax.scan(
                mb_fn, (params, opt_state), jnp.arange(mnb, dtype=jnp.int32))
            return (params, opt_state, rng), losses.mean()

        (params, opt_state, rng), ep_losses = lax.scan(
            epoch_fn, (params, opt_state, rng), None, length=epochs)
        return params, opt_state, rng, ep_losses.mean()

    return collect_rollout, update


# ── Training loop ──────────────────────────────────────────────────────────────

def main():
    args = tyro.cli(Args)
    B = args.num_envs * args.num_steps
    num_iters = args.total_timesteps // B
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Backend : {jax.default_backend()}"
          f"  |  envs={args.num_envs}  T={args.num_steps}  batch={B}")

    print("Loading map …")
    t0 = time.time()
    map_data = MapData(args.yaml)
    print(f"  skeleton={map_data.n_wps} pts  obs_dim={map_data.obs_dim}"
          f"  ({time.time()-t0:.1f}s)")

    batched_step, batched_obs, batched_init = make_sim_fns(
        map_data, args.max_steps, args.dr_frac, scan_steps=args.scan_steps)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_key, policy_key = jax.random.split(rng, 3)

    state  = batched_init(jax.random.split(init_key, args.num_envs))
    params = init_policy(policy_key, map_data.obs_dim, args.hidden)

    num_iters_total = max(1, num_iters)
    lr_schedule = optax.linear_schedule(
        args.learning_rate, 0.,
        num_iters_total * args.update_epochs * args.num_minibatches)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5))
    opt_state = optimizer.init(params)

    collect_rollout, update = make_ppo_fns(batched_step, batched_obs, optimizer, args)

    print("Compiling (first iteration may be slow) …")
    ep_rew_buf: list[float] = []
    sps_list:   list[float] = []
    global_step = 0

    for iteration in tqdm.trange(1, num_iters + 1):
        t_iter = time.time()

        rng, roll_key = jax.random.split(rng)
        state, rng, traj = collect_rollout(params, state, roll_key)
        jax.block_until_ready(traj)

        last_value = jax.vmap(
            lambda o: _mlp(params["critic"], o).squeeze())(state["obs"])

        rng, upd_key = jax.random.split(rng)
        params, opt_state, rng, loss = update(
            params, opt_state, traj, last_value, upd_key)
        jax.block_until_ready(loss)

        elapsed = time.time() - t_iter
        sps_list.append(B / elapsed)
        global_step += B

        _, rewards, dones, *_ = traj
        ep_rew_buf.extend(float(r) for r, d in
                          zip(rewards.ravel().tolist(), dones.ravel().tolist()) if d)

        if iteration % 10 == 0:
            mean_rew = float(np.mean(ep_rew_buf[-500:])) if ep_rew_buf else float("nan")
            print(f"  iter={iteration:6d}  steps={global_step/1e6:.1f}M  "
                  f"SPS={int(np.mean(sps_list[-20:])):,}  "
                  f"ep_rew={mean_rew:.1f}  loss={float(loss):.4f}")

        if iteration % args.save_interval == 0:
            path = os.path.join(args.save_dir, f"ckpt_{iteration}.npz")
            leaves, _ = jax.tree_util.tree_flatten(params)
            np.savez(path, *[np.array(x) for x in leaves])
            print(f"  ✓ checkpoint → {path}")

    print(f"\nDone. Mean SPS (last 20 iters): {int(np.mean(sps_list[-20:])):,}")


if __name__ == "__main__":
    main()
