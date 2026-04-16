"""learnrl_jax.py — F1tenth PPO · pure JAX · dynamic bicycle model.

lax.scan compiles the T-step rollout; vmap parallelises N envs on-device.
State: [x, y, δ, v_x, v_y, ψ, ψ̇] — linearised Pacejka tires, one RK4 step at 60 Hz.

Usage:
    uv run --with "jax[cpu]" --with optax --with "wandb[media]" --with av \\
           --with scipy --with scikit-image --with pillow \\
           --with pyyaml --with tqdm --with tyro scripts/learnrl_jax.py
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import av
import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.ndimage as ndi
import scipy.signal
import skimage.morphology
import tqdm
import tyro
import yaml as pyyaml
from jax import lax
from PIL import Image, ImageDraw
from scipy.spatial import KDTree

import wandb

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class Args:
    yaml: str = "maps/my_map.yaml"
    exp_name: str = "learnrl_jax"
    seed: int = 42
    num_envs: int = 0
    """Parallel environments (0 = auto-detect: 1024 on GPU/TPU, 256 on CPU)."""
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
    max_steps: int = 10_000
    dr_frac: float = 0.2
    save_interval: int = 50
    save_dir: str = "checkpoints"
    scan_steps: int = 30
    track: bool = False
    """Enable wandb logging."""
    capture_video: bool = False
    """Record top-down eval video and upload to wandb (requires --track)."""
    video_interval: int = 25
    """Record a video every N iterations (0 = disabled)."""
    video_max_steps: int = 3_600
    """Max frames per eval episode."""
    video_crf: int = 23
    """H.264 CRF quality (18–28 typical)."""


# ── Constants ──────────────────────────────────────────────────────────────────

STEER_MIN, STEER_MAX = np.float32(-0.5236), np.float32(0.5236)
STEER_VEL_MAX = np.float32(3.2)
V_MIN, V_MAX = np.float32(-5.0), np.float32(20.0)
DPSI_MAX = np.float32(10.0)  # yaw-rate clip (rad/s)
WIDTH, LENGTH = np.float32(0.27), np.float32(0.50)
DRAG = np.float32(0.5 * 0.3 * 0.04 * 1.225)  # ½ρCdA
G = np.float32(9.81)
PI, DT = np.float32(np.pi), np.float32(1.0 / 60.0)
N_BEAMS, N_LOOK = 108, 10
FOV = np.float32(270.0 * np.pi / 180.0)
MIN_RANGE, MAX_RANGE = np.float32(0.06), np.float32(10.0)
# Dynamic bicycle model parameters (F1tenth gym defaults)
P_DEF = {
    k: np.float32(v)
    for k, v in {
        "lf": 0.15875,  # front axle distance from CG  (m)
        "lr": 0.17145,  # rear axle distance from CG   (m)
        "m": 3.74,  # mass                          (kg)
        "I_z": 0.04712,  # yaw moment of inertia         (kg·m²)
        "mu": 0.523,  # tyre–road friction coefficient
        "C_Sf": 4.718,  # front cornering stiffness factor
        "C_Sr": 5.4562,  # rear cornering stiffness factor
        "a_max": 9.51,  # max longitudinal acceleration (m/s²)
    }.items()
}
RAND_KEYS = ("lf", "lr", "m", "a_max")  # parameters to randomise


# ── Map preprocessing ──────────────────────────────────────────────────────────


def _nbrs8(p, pts):
    x, y = p
    return [
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0) and (x + dx, y + dy) in pts
    ]


def _largest_cc(pts):
    seen, best = set(), []
    for p in pts:
        if p in seen:
            continue
        comp, q = [], [p]
        seen.add(p)
        for c in q:
            comp.append(c)
            for n in _nbrs8(c, pts):
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        if len(comp) > len(best):
            best = comp
    return set(best)


def _skeleton_loop(thin, ox, oy, res, otheta, h):
    pts = set(zip(*[a.tolist() for a in np.where(thin)[::-1]]))  # (x,y) pixels
    if not pts:
        raise RuntimeError("Empty skeleton")
    pts = _largest_cc(pts)
    for _ in range(500):
        ends = [p for p in pts if len(_nbrs8(p, pts)) < 2]
        if not ends:
            break
        pts -= set(ends)
    if not pts:
        raise RuntimeError("Skeleton collapsed")
    start = min(
        pts, key=lambda p: (p[0] + ox / res) ** 2 + (p[1] - (h - 1) - oy / res) ** 2
    )
    adj = _nbrs8(start, pts)
    if len(adj) < 2:
        raise RuntimeError("Start node < 2 neighbours")
    src, targets = adj[0], set(adj[1:])
    parent, queue, found = {src: src}, [src], None
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
        if (
            abs(((np.arctan2(dy0, dx0) - otheta + 3 * np.pi) % (2 * np.pi)) - np.pi)
            > np.pi / 2
        ):
            world[1:] = world[1:][::-1]
    return world


class MapData:
    """Preprocessed map: EDT, skeleton LUT, beam directions, and RGB image."""

    def __init__(
        self,
        yaml_path: str,
        edt_thresh_frac: float = 0.1,
        sg_window: int = 11,
        sg_degree: int = 3,
    ):
        meta = pyyaml.safe_load(open(yaml_path))
        raw = np.array(Image.open(Path(yaml_path).parent / meta["image"]).convert("L"))
        res = float(meta["resolution"])
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        otheta = float(meta["origin"][2]) if len(meta["origin"]) > 2 else 0.0
        h, w = raw.shape

        free = raw >= 210
        edt_np = ndi.distance_transform_edt(free).astype(np.float32) * res
        edt_px = ndi.distance_transform_edt(free)
        loop = _skeleton_loop(
            skimage.morphology.skeletonize(
                free & (edt_px >= edt_px.max() * edt_thresh_frac)
            ),
            ox,
            oy,
            res,
            otheta,
            h,
        )
        assert len(loop) >= 2
        skel = np.array(
            scipy.signal.savgol_filter(
                np.array(loop, np.float64), sg_window, sg_degree, axis=0, mode="wrap"
            ),
            dtype=np.float32,
        )

        ys, xs = np.mgrid[0:h, 0:w]
        wx = (xs.ravel() * res + ox).astype(np.float32)
        wy = ((h - 1 - ys.ravel()) * res + oy).astype(np.float32)
        _, lut = KDTree(skel).query(np.c_[wx, wy])

        angles = np.linspace(-FOV / 2, FOV / 2, N_BEAMS, dtype=np.float32)
        self.edt = jnp.array(edt_np)
        self.skeleton = jnp.array(skel)
        self.skel_lut = jnp.array(lut.astype(np.int32))
        self.beam_sc = jnp.array(np.stack([np.sin(angles), np.cos(angles)], 1))
        self.img_rgb = np.stack([raw] * 3, -1)  # H×W×3 uint8 for rendering
        self.w, self.h = w, h
        self.res = np.float32(res)
        self.inv_res = np.float32(1.0 / res)
        self.ox, self.oy = np.float32(ox), np.float32(oy)
        self.n_wps = len(skel)
        seg_len = np.linalg.norm(skel[1:] - skel[:-1], axis=1).mean()
        self.look_step = max(1, round(1.0 / seg_len)) if seg_len > 0 else 1
        self.obs_dim = N_BEAMS + 3 + 2 + N_LOOK * 2  # lidar + [v,δ,ψ̇] + [he,le] + look


# ── Dynamic bicycle model ──────────────────────────────────────────────────────
# State s = [x, y, δ, v_x, v_y, ψ, ψ̇]


def _dyn(s, accel, sv, p):
    """Single-track dynamic bicycle model with linearised Pacejka tyre forces."""
    x, y, delta, vx, vy, psi, dpsi = s
    lwb = p["lf"] + p["lr"]
    safe_vx = jnp.maximum(
        jnp.abs(vx), jnp.float32(0.05)
    )  # guard against divide-by-zero

    # Slip angles — arctan is sufficient since safe_vx > 0 (no quadrant ambiguity)
    alpha_f = jnp.arctan((dpsi * p["lf"] + vy) / safe_vx) - delta
    alpha_r = jnp.arctan((dpsi * p["lr"] - vy) / safe_vx)

    # Lateral tyre forces (linearised Pacejka, weight-distributed)
    mg = p["m"] * G
    F_yf = -p["mu"] * mg * (p["lr"] / lwb) * p["C_Sf"] * alpha_f
    F_yr = -p["mu"] * mg * (p["lf"] / lwb) * p["C_Sr"] * alpha_r

    # Longitudinal force + aero drag on vx
    F_x = p["m"] * accel - DRAG * vx * jnp.abs(vx)

    return jnp.array(
        [
            vx * jnp.cos(psi) - vy * jnp.sin(psi),  # ẋ
            vx * jnp.sin(psi) + vy * jnp.cos(psi),  # ẏ
            sv,  # δ̇
            (F_x - F_yf * jnp.sin(delta) + p["m"] * vy * dpsi) / p["m"],  # v̇_x
            (F_yr + F_yf * jnp.cos(delta) - p["m"] * vx * dpsi) / p["m"],  # v̇_y
            dpsi,  # ψ̇
            (F_yf * p["lf"] * jnp.cos(delta) - F_yr * p["lr"]) / p["I_z"],  # ψ̈
        ]
    )


# ── Simulation ─────────────────────────────────────────────────────────────────


def make_sim_fns(
    map_data: MapData,
    max_steps: int,
    dr_frac: float,
    dt: float = float(DT),
    scan_steps: int = 30,
):
    """Return (batched_step, batched_init) — JIT+vmap compiled."""
    edt, skel, lut, bsc = (
        map_data.edt,
        map_data.skeleton,
        map_data.skel_lut,
        map_data.beam_sc,
    )
    W, H = map_data.w, map_data.h
    res, inv = map_data.res, map_data.inv_res
    ox, oy = map_data.ox, map_data.oy
    n_wps = jnp.int32(map_data.n_wps)
    look_step = jnp.int32(map_data.look_step)
    dt_f = jnp.float32(dt)
    Wf, Hf = jnp.float32(W), jnp.float32(H)
    COLR = jnp.float32(np.sqrt((LENGTH / 2) ** 2 + (WIDTH / 2) ** 2))  # circumradius

    def _to_px(x, y):
        return (
            jnp.int32(jnp.clip((x - ox) * inv, 0.0, Wf - 1.0)),
            jnp.int32(jnp.clip((Hf - 1.0) - (y - oy) * inv, 0.0, Hf - 1.0)),
        )

    def _collides(s):
        x, y = s[0], s[1]
        pxf, pyf = (x - ox) * inv, (Hf - 1.0) - (y - oy) * inv
        in_map = (pxf >= 0.0) & (pxf < Wf) & (pyf >= 0.0) & (pyf < Hf)
        d = jnp.where(
            in_map,
            edt[
                jnp.int32(jnp.clip(pyf, 0.0, Hf - 1.0)),
                jnp.int32(jnp.clip(pxf, 0.0, Wf - 1.0)),
            ],
            0.0,
        )
        return (~jnp.isfinite(x)) | (~jnp.isfinite(y)) | (d < COLR)

    def _obs(s, wi, p):
        x, y, delta, vx, vy, psi, dpsi = s
        sh, ch = jnp.sin(psi), jnp.cos(psi)
        # Lidar: EDT sphere-tracing, all N_BEAMS rays in parallel via lax.scan
        sa, ca = bsc[:, 0], bsc[:, 1]
        bdx, bdy = ch * ca - sh * sa, sh * ca + ch * sa
        px0 = (x + p["lf"] * ch - ox) * inv
        py0 = (Hf - 1.0) - (y + p["lf"] * sh - oy) * inv
        dpx, dpy = bdx * inv, -bdy * inv
        minstep = jnp.float32(res * 0.5)

        def _ray(carry, _):
            t, r, done = carry
            pxf, pyf = px0 + t * dpx, py0 + t * dpy
            ib = (pxf >= 0.0) & (pxf < Wf) & (pyf >= 0.0) & (pyf < Hf)
            d = jnp.where(
                ib,
                edt[
                    jnp.int32(jnp.clip(pyf, 0.0, Hf - 1.0)),
                    jnp.int32(jnp.clip(pxf, 0.0, Wf - 1.0)),
                ],
                0.0,
            )
            hit = (~done) & (~ib | (d < minstep))
            r = jnp.where(hit, t, r)
            done = done | hit
            t = jnp.where(done, t, t + jnp.maximum(d, minstep))
            return (t, r, done), None

        (_, scans, _), _ = lax.scan(
            _ray,
            (
                jnp.full(N_BEAMS, MIN_RANGE),
                jnp.full(N_BEAMS, MAX_RANGE),
                jnp.zeros(N_BEAMS, jnp.bool_),
            ),
            None,
            length=scan_steps,
        )
        # Waypoint error
        wp, wpn = skel[wi], skel[(wi + 1) % n_wps]
        cth = jnp.arctan2(wpn[1] - wp[1], wpn[0] - wp[0])
        he = ((cth - psi) + PI) % (2.0 * PI) - PI
        le = -(x - wp[0]) * jnp.sin(cth) + (y - wp[1]) * jnp.cos(cth)
        # Lookahead waypoints in local frame
        idx = (wi + jnp.arange(1, N_LOOK + 1, dtype=jnp.int32) * look_step) % n_wps
        dxy = skel[idx] - jnp.array([x, y])
        look = jnp.stack(
            [dxy[:, 0] * ch + dxy[:, 1] * sh, -dxy[:, 0] * sh + dxy[:, 1] * ch], 1
        ).ravel()
        return jnp.concatenate(
            [scans, jnp.array([vx, delta, dpsi]), jnp.array([he, le]), look]
        )

    def _reset(rng):
        rng, sub = jax.random.split(rng)
        ri = jax.random.randint(jax.random.fold_in(sub, jnp.uint32(0)), (), 0, n_wps)
        pt, ptn = skel[ri], skel[(ri + 1) % n_wps]
        # Start from defaults; randomise only RAND_KEYS
        p = {k: P_DEF[k] for k in P_DEF}
        for i, k in enumerate(RAND_KEYS):
            u = jax.random.uniform(jax.random.fold_in(sub, jnp.uint32(i + 1)))
            p[k] = P_DEF[k] * (1.0 + jnp.float32(dr_frac) * (u * 2.0 - 1.0))
        th = jnp.arctan2(ptn[1] - pt[1], ptn[0] - pt[0])
        # 7-state: [x, y, δ, v_x, v_y, ψ, ψ̇]
        s = jnp.array(
            [
                pt[0],
                pt[1],
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(0.0),
                th,
                jnp.float32(0.0),
            ]
        )
        return s, ri, p, rng

    def _step(state, action):
        s, wi, steps, rng, p = (
            state["dyn"],
            state["wp_idx"],
            state["steps"],
            state["rng"],
            state["p"],
        )
        sv = jnp.clip(action[0], -1.0, 1.0) * STEER_VEL_MAX
        sv = jnp.where((sv < 0.0) & (s[2] <= STEER_MIN), 0.0, sv)
        sv = jnp.where((sv > 0.0) & (s[2] >= STEER_MAX), 0.0, sv)
        accel = jnp.clip(action[1], -1.0, 1.0) * p["a_max"]
        f = lambda s: _dyn(s, accel, sv, p)
        k1 = f(s)
        k2 = f(s + k1 * (dt_f / 2))
        k3 = f(s + k2 * (dt_f / 2))
        k4 = f(s + k3 * dt_f)
        s = s + dt_f / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x, y, delta, vx, vy, psi, dpsi = s
        s = jnp.array(
            [
                x,
                y,
                jnp.clip(delta, STEER_MIN, STEER_MAX),
                jnp.clip(vx, V_MIN, V_MAX),
                jnp.clip(vy, -V_MAX, V_MAX),
                (psi + PI) % (2.0 * PI) - PI,
                jnp.clip(dpsi, -DPSI_MAX, DPSI_MAX),
            ]
        )
        px, py = _to_px(x, y)
        new_wi = lut[py * W + px]
        d_wp = jnp.float32(new_wi - wi)
        d_wp = jnp.where(
            d_wp > jnp.float32(n_wps) / 2.0, d_wp - jnp.float32(n_wps), d_wp
        )
        d_wp = jnp.where(
            d_wp < -jnp.float32(n_wps) / 2.0, d_wp + jnp.float32(n_wps), d_wp
        )
        term = _collides(s)
        trunc = (steps + jnp.int32(1)) >= jnp.int32(max_steps)
        reward = (
            d_wp / jnp.float32(n_wps) * 100.0 * (1.0 + jnp.maximum(vx, 0.0) / 10.0)
            - 0.1 * jnp.exp(-3.0 * edt[py, px])
            - jnp.where(term, 100.0, 0.0)
        )
        rs, rwi, rp, rng = _reset(rng)
        done = term | trunc
        s = jnp.where(done, rs, s)
        wi = jnp.where(done, rwi, new_wi)
        p = jax.tree.map(lambda a, b: jnp.where(done, a, b), rp, p)
        st = jnp.where(done, jnp.int32(0), steps + jnp.int32(1))
        ns = {
            "dyn": s,
            "wp_idx": wi,
            "steps": st,
            "rng": rng,
            "p": p,
            "obs": _obs(s, wi, p),
        }
        return ns, (reward, term, trunc)

    def _init(rng):
        s, ri, p, rng = _reset(rng)
        return {
            "dyn": s,
            "wp_idx": ri,
            "steps": jnp.int32(0),
            "rng": rng,
            "p": p,
            "obs": _obs(s, ri, p),
        }

    return jax.jit(jax.vmap(_step)), jax.jit(jax.vmap(_init))


# ── Policy ─────────────────────────────────────────────────────────────────────

LOG_STD_MIN, LOG_STD_MAX = jnp.float32(-5.0), jnp.float32(0.5)


def init_policy(key, obs_dim: int, hidden: int, act_dim: int = 2):
    def layer(k, i, o, std=float(np.sqrt(2))):
        return {
            "w": jax.random.normal(k, (i, o)) * float(std / np.sqrt(i)),
            "b": jnp.zeros(o),
        }

    ks = jax.random.split(key, 6)
    return {
        "actor": [
            layer(ks[0], obs_dim, hidden),
            layer(ks[1], hidden, hidden),
            layer(ks[2], hidden, act_dim, std=0.01),
        ],
        "actor_logstd": jnp.full((act_dim,), -0.5, dtype=jnp.float32),
        "critic": [
            layer(ks[3], obs_dim, hidden),
            layer(ks[4], hidden, hidden),
            layer(ks[5], hidden, 1, std=1.0),
        ],
    }


def _mlp(layers, x):
    for l in layers[:-1]:
        x = jax.nn.tanh(x @ l["w"] + l["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def _forward(params, obs, key):
    mean = _mlp(params["actor"], obs)
    log_std = jnp.clip(params["actor_logstd"], LOG_STD_MIN, LOG_STD_MAX)
    std = jnp.exp(log_std)
    act = mean + std * jax.random.normal(key, mean.shape)
    lp = (
        -0.5 * ((act - mean) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)
    ).sum(-1)
    return act, lp, _mlp(params["critic"], obs).squeeze(-1)


def _evaluate(params, obs, act):
    mean = _mlp(params["actor"], obs)
    log_std = jnp.clip(params["actor_logstd"], LOG_STD_MIN, LOG_STD_MAX)
    std = jnp.exp(log_std)
    lp = (
        -0.5 * ((act - mean) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)
    ).sum(-1)
    ent = (jnp.log(std) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(-1)
    return lp, ent, _mlp(params["critic"], obs).squeeze(-1)


# ── PPO ────────────────────────────────────────────────────────────────────────


def make_ppo_fns(batched_step, optimizer, args: Args):
    N, T, B = args.num_envs, args.num_steps, args.num_envs * args.num_steps
    mnb, mb = args.num_minibatches, B // args.num_minibatches
    gamma = jnp.float32(args.gamma)
    lam = jnp.float32(args.gae_lambda)
    clip_c = jnp.float32(args.clip_coef)
    ent_c = jnp.float32(args.ent_coef)
    vf_c = jnp.float32(args.vf_coef)
    env_ids = jnp.arange(N, dtype=jnp.uint32)

    @jax.jit
    def rollout(params, state, rng):
        def step(carry, _):
            state, rng = carry
            obs = state["obs"]
            rng, key = jax.random.split(rng)
            act, lp, val = jax.vmap(
                lambda o, i: _forward(params, o, jax.random.fold_in(key, i))
            )(obs, env_ids)
            state, (rew, term, trunc) = batched_step(state, act)
            return (state, rng), (
                obs,
                act,
                rew,
                (term | trunc).astype(jnp.float32),
                lp,
                val,
            )

        (state, rng), traj = lax.scan(step, (state, rng), None, length=T)
        return state, rng, traj

    @jax.jit
    def update(params, opt_state, traj, last_val, rng):
        obs, acts, rews, dones, old_lp, vals = traj

        def gae(acc, i):
            vn = jnp.where(i < T - 1, vals[jnp.minimum(i + 1, T - 1)], last_val)
            delta = rews[i] + gamma * vn * (1.0 - dones[i]) - vals[i]
            new = delta + gamma * lam * (1.0 - dones[i]) * acc
            return new, new

        _, adv_r = lax.scan(
            gae, jnp.zeros(N, jnp.float32), jnp.arange(T - 1, -1, -1, dtype=jnp.int32)
        )
        advs = adv_r[::-1]
        rets = advs + vals
        fl = lambda x: x.reshape(B, *x.shape[2:])
        obs_f = fl(obs)
        act_f = fl(acts)
        ret_f = rets.ravel()
        lp_f = old_lp.ravel()
        adv_f = advs.ravel()
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        def epoch(carry, _):
            params, opt, rng = carry
            rng, sk = jax.random.split(rng)
            idx = jax.random.permutation(sk, B)

            def mb_fn(carry, mbi):
                params, opt = carry
                sel = lax.dynamic_slice(idx, (mbi * mb,), (mb,))
                bo, ba, br, bad, blp = (
                    obs_f[sel],
                    act_f[sel],
                    ret_f[sel],
                    adv_f[sel],
                    lp_f[sel],
                )

                def loss(p):
                    nlp, ent, nv = jax.vmap(lambda o, a: _evaluate(p, o, a))(bo, ba)
                    r = jnp.exp(nlp - blp)
                    pg = -jnp.minimum(
                        r * bad, jnp.clip(r, 1.0 - clip_c, 1.0 + clip_c) * bad
                    ).mean()
                    return (
                        pg
                        + vf_c * 0.5 * jnp.mean((nv - br) ** 2)
                        + ent_c * (-ent.mean())
                    )

                g, grads = jax.value_and_grad(loss)(params)
                ups, opt = optimizer.update(grads, opt, params)
                return (optax.apply_updates(params, ups), opt), g

            (params, opt), ls = lax.scan(
                mb_fn, (params, opt), jnp.arange(mnb, dtype=jnp.int32)
            )
            return (params, opt, rng), ls.mean()

        (params, opt_state, rng), ep_ls = lax.scan(
            epoch, (params, opt_state, rng), None, length=args.update_epochs
        )
        return params, opt_state, rng, ep_ls.mean()

    return rollout, update


# ── Rendering ──────────────────────────────────────────────────────────────────


def _render_frame(map_data: MapData, dyn: np.ndarray) -> np.ndarray:
    """Top-down RGB frame: map background + car rectangle."""
    img = Image.fromarray(map_data.img_rgb.copy())
    draw = ImageDraw.Draw(img)
    x, y = float(dyn[0]), float(dyn[1])
    psi = float(dyn[5])  # ψ is state[5] in 7-state model
    px = float((x - map_data.ox) * map_data.inv_res)
    py = float((map_data.h - 1) - (y - map_data.oy) * map_data.inv_res)
    cp, sp = float(np.cos(psi)), float(np.sin(psi))
    hl, hw = float(LENGTH / 2 * map_data.inv_res), float(WIDTH / 2 * map_data.inv_res)
    pts = [
        (px + lx * cp - ly * sp, py - lx * sp - ly * cp)
        for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]
    draw.polygon(pts, fill=(220, 50, 50))
    frame = np.array(img)
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]  # H.264 requires even dims


def record_eval_video(params: dict, map_data: MapData, args: Args):
    """Run a deterministic eval episode and return (wandb.Video|None, total_reward)."""
    bstep, binit = make_sim_fns(
        map_data, args.max_steps, args.dr_frac, scan_steps=args.scan_steps
    )
    state = binit(jax.random.split(jax.random.PRNGKey(args.seed + 999), 1))
    out = os.path.join(args.save_dir, "eval_video.mp4")
    total_rew, n_frames, container, stream = 0.0, 0, None, None
    try:
        for _ in range(args.video_max_steps):
            obs = state["obs"][0]
            action = jnp.clip(_mlp(params["actor"], obs), -1.0, 1.0)[None]
            state, (rew, term, trunc) = bstep(state, action)
            total_rew += float(rew[0])
            frame = _render_frame(map_data, np.array(state["dyn"][0]))
            if container is None:
                h, w = frame.shape[:2]
                container = av.open(out, mode="w")
                stream = container.add_stream("h264", rate=60)
                stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
                stream.options = {"crf": str(args.video_crf), "preset": "ultrafast"}
            for pkt in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(pkt)
            n_frames += 1
            if bool(term[0]) or bool(trunc[0]):
                break
    finally:
        if container:
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
    return (wandb.Video(out, format="mp4") if n_frames else None), total_rew


# ── Training ───────────────────────────────────────────────────────────────────


def main():
    args = tyro.cli(Args)
    if args.num_envs <= 0:
        args.num_envs = 256 if jax.default_backend() == "cpu" else 1024
    B = args.num_envs * args.num_steps
    num_iters = args.total_timesteps // B
    os.makedirs(args.save_dir, exist_ok=True)

    if args.track:
        wandb.init(
            project="f1tenth-ppo-jax",
            name=f"{args.exp_name}__{args.seed}",
            config=vars(args),
            save_code=True,
        )

    print(
        f"Backend: {jax.default_backend()}  envs={args.num_envs}  T={args.num_steps}  B={B}"
    )
    t0 = time.time()
    map_data = MapData(args.yaml)
    print(
        f"  map: {map_data.n_wps} wps  obs_dim={map_data.obs_dim}  ({time.time() - t0:.1f}s)"
    )

    bstep, binit = make_sim_fns(
        map_data, args.max_steps, args.dr_frac, scan_steps=args.scan_steps
    )
    rng = jax.random.PRNGKey(args.seed)
    rng, ki, kp = jax.random.split(rng, 3)
    state = binit(jax.random.split(ki, args.num_envs))
    params = init_policy(kp, map_data.obs_dim, args.hidden)

    n_iters = max(1, num_iters)
    sched = optax.linear_schedule(
        args.learning_rate, 0.0, n_iters * args.update_epochs * args.num_minibatches
    )
    opt = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm), optax.adam(sched, eps=1e-5)
    )
    opt_state = opt.init(params)
    rollout, update = make_ppo_fns(bstep, opt, args)

    video_iters = (
        set(range(1, num_iters + 1, args.video_interval)) | {num_iters}
        if args.capture_video and args.video_interval > 0
        else set()
    )

    print("Compiling first iteration… (may take 1–5 min on CPU)")
    ep_rews: list[float] = []
    ep_rew_acc: np.ndarray = np.zeros(args.num_envs, dtype=np.float64)
    sps_buf: list[float] = []
    global_step = 0

    for it in tqdm.trange(1, num_iters + 1):
        t_it = time.time()
        rng, rk = jax.random.split(rng)
        state, rng, traj = rollout(params, state, rk)
        jax.block_until_ready(traj)

        last_val = jax.vmap(lambda o: _mlp(params["critic"], o).squeeze())(state["obs"])
        rng, uk = jax.random.split(rng)
        params, opt_state, rng, loss = update(params, opt_state, traj, last_val, uk)
        jax.block_until_ready(loss)

        sps = B / (time.time() - t_it)
        sps_buf.append(sps)
        global_step += B

        # Accumulate proper per-episode returns from the trajectory.
        # traj = (obs, act, rew, done, lp, val) — T×N each.
        obs_t, act_t, rews_t, dones_t, lp_t, val_t = traj
        rews_np = np.array(rews_t)  # T × N
        dones_np = np.array(dones_t)  # T × N  (1.0 = episode ended)
        for t in range(args.num_steps):
            ep_rew_acc += rews_np[t]
            for i in np.where(dones_np[t] >= 0.5)[0]:
                ep_rews.append(float(ep_rew_acc[i]))
                ep_rew_acc[i] = 0.0

        mean_rew = float(np.mean(ep_rews[-500:])) if ep_rews else float("nan")
        lr_now = float(sched(opt_state[1][1].count))

        if it % 10 == 0:
            print(
                f"  iter={it:6d}  steps={global_step / 1e6:.1f}M  "
                f"SPS={int(np.mean(sps_buf[-20:])):,}  "
                f"ep_rew={mean_rew:.1f}  loss={float(loss):.4f}"
            )

        if args.track:
            log = {
                "charts/episode_return": mean_rew,
                "charts/SPS": sps,
                "charts/learning_rate": lr_now,
                "losses/total": float(loss),
                "perf/global_step": global_step,
            }
            if it in video_iters:
                print(f"\n[iter {it}] Recording eval video…")
                vid, eval_rew = record_eval_video(params, map_data, args)
                log["charts/eval_return"] = eval_rew
                if vid is not None:
                    log["media/eval_video"] = vid
                    print(f"[iter {it}] Captured  eval_return={eval_rew:.2f}")
                else:
                    print(f"[iter {it}] WARNING: no frames captured")
            wandb.log(log, step=global_step)

        if it % args.save_interval == 0:
            ckpt = os.path.join(args.save_dir, f"ckpt_{it}.npz")
            leaves, _ = jax.tree_util.tree_flatten(params)
            np.savez(ckpt, *[np.array(x) for x in leaves])
            print(f"  ✓ {ckpt}")

    print(f"\nDone. Mean SPS: {int(np.mean(sps_buf[-20:])):,}")
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
