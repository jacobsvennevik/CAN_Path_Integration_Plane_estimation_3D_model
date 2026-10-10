from dataclasses import dataclass, asdict
from copy import deepcopy

import numpy as np

from model.metrics import torus_error, tracking_score
from model.path_integration import PathIntegrator, build_rotation_matrix
from config import world_to_flat_bins
from model.network.QAN3D import Torus3DQAN


@dataclass
class Trajectory:
    """Walk plus optional plane schedule. ``__iter__`` yields the legacy triple."""
    world_pos: np.ndarray
    v_body_seq: np.ndarray
    torus_gt: np.ndarray
    n_true_seq: np.ndarray = None
    segment_starts: np.ndarray = None
    plane_id: np.ndarray = None
    flight_of: np.ndarray = None
    route_table: list = None
    flight_table: list = None
    n_route_draws: int = 0

    def __iter__(self):
        yield self.world_pos
        yield self.v_body_seq
        yield self.torus_gt


def world_to_torus_gt(world_pos, scale):
    """Ground-truth torus coordinates for a world path, in radians.

    The metres-to-radians conversion is a plain scaling. PeriodicEuclidean lays
    the torus axes directly on the world axes, so there is no basis change to
    apply.

    The + pi centres the path in the middle of the torus rather than at the
    corner, so a walk starting at the world origin does not sit on the wrap.
    """
    return (np.pi + np.asarray(world_pos) * scale) % (2 * np.pi)


def torus_gt_from_velocity(v_body_seq, n_true_seq, scale, d):
    """Path-dependent GT: (π + scale · cumsum((R(n_t) v_t)[:d])) mod 2π.

    On a flat floor (n ≡ ẑ) this is bit-identical to world_to_torus_gt.
    """
    v_body_seq = np.asarray(v_body_seq, dtype=float)
    n_true_seq = np.asarray(n_true_seq, dtype=float)
    T = v_body_seq.shape[0]
    d = int(d)
    drive = np.empty((T, d))
    if T == 0:
        return drive
    chg = np.empty(T, dtype=bool)
    chg[0] = True
    chg[1:] = np.any(n_true_seq[1:] != n_true_seq[:-1], axis=1)
    bounds = np.append(np.flatnonzero(chg), T)
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        R = build_rotation_matrix(n_true_seq[t0])
        drive[t0:t1] = (v_body_seq[t0:t1] @ R.T)[:, :d]
    return (np.pi + scale * np.cumsum(drive, axis=0)) % (2 * np.pi)


def bounce_square(uv, heading, speed, limit):
    """One in-plane step at `heading`, reflected so it stays in the ±limit square.

    The new position is the old position plus the (possibly reflected) step.
    No random turn: the caller updates the heading first.
    """
    heading = float(heading)
    duv = speed * np.array([np.cos(heading), np.sin(heading)])
    new_uv = np.asarray(uv, dtype=float) + duv
    for dim in range(2):
        if new_uv[dim] > limit or new_uv[dim] < -limit:
            heading = (np.pi - heading) if dim == 0 else (-heading)
            duv = speed * np.array([np.cos(heading), np.sin(heading)])
            new_uv = np.asarray(uv, dtype=float) + duv
    return new_uv, heading, duv


def plane_step(uv, heading, speed, step_turn_std, limit, rng):
    """One persistent-walk step inside the ±limit square."""
    heading = float(heading) + float(rng.normal(0.0, step_turn_std))
    return bounce_square(uv, heading, speed, limit)


def planar_walk(rng, n_steps, speed, step_turn_std, limit, uv0=None, heading=None):
    """Persistent random walk in plane coordinates. Reflection stays in-plane.

    step_turn_std is the standard deviation of the heading change per step.
    uv0 is the in-plane start (default origin). heading is the initial direction
    in that plane; omitted, it is drawn at random.

    Returns
    -------
    uv : (n_steps, 2) in-plane positions
    duv : (n_steps, 2) in-plane steps. duv[0] = 0, and uv[t] = uv[t-1] + duv[t].
    """
    uv = np.zeros((n_steps, 2))
    duv = np.zeros((n_steps, 2))
    if uv0 is not None:
        uv[0] = np.asarray(uv0, dtype=float)
    if heading is None:
        heading = float(rng.uniform(0, 2 * np.pi))
    for t in range(1, n_steps):
        uv[t], heading, duv[t] = plane_step(
            uv[t - 1], heading, speed, step_turn_std, limit, rng)
    return uv, duv


@dataclass
class WalkParams:
    n_steps: int
    rng: object
    scale: float
    dt: float
    world_speed: float
    step_turn_std: float
    limit: float


def walk_params(experiment_cfg, dt, turn_std=None, n_steps=None, seed=None,
                scale=None):
    """Shared walk numbers: speed, heading diffusion, box limit.

    The speed formula target_speed · dt / scale lives only here.
    ``scale`` is required: it is no longer stored on the experiment config.
    """
    if scale is None:
        raise ValueError("walk_params requires scale")
    n_steps = experiment_cfg.n_steps if n_steps is None else int(n_steps)
    seed = experiment_cfg.seed if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    scale = float(scale)
    dt = float(dt)
    torus_inc = experiment_cfg.target_speed_rad_per_time * dt   # rad/step
    world_speed = torus_inc / scale              # m/step
    omega_std = experiment_cfg.omega_std if turn_std is None else float(turn_std)
    step_turn_std = omega_std * np.sqrt(dt)
    # Reflect at boundaries ±(env_size/2)
    limit = experiment_cfg.env_size / 2
    return WalkParams(
        n_steps=n_steps, rng=rng, scale=scale, dt=dt,
        world_speed=world_speed, step_turn_std=step_turn_std, limit=limit,
    )


@dataclass
class ExperimentResult:
    torus_gt:          np.ndarray
    theta_hist:        np.ndarray
    n_hat_hist:        np.ndarray
    n_held_hist:       np.ndarray  # sample-and-hold normal that actually drives E
    error_rad:         np.ndarray
    tracking_score:    float
    params:            dict
    n_sheet:           int
    refresh_mask:      np.ndarray
    ratemap_sums:      np.ndarray = None
    ratemap_counts:    np.ndarray = None
    sub_idx: np.ndarray = None   # neuron subsample indices (3-D path)
    shuf_sums:         np.ndarray = None   # (n_shuffle, bins, ..., n_sub) shuffle maps
    active_mask:       np.ndarray = None   # (n_sub,) bool — neurons with non-trivial activity
    kappa_hist:        np.ndarray = None  # vMF κ
    warmup_volume:     np.ndarray = None
    n_gravity_flips:   int = 0
    field_pos:         np.ndarray = None   # (n_samples, 3) metres, volumetric IFD
    field_act:         np.ndarray = None   # (n_samples, n_sub) activity trace
    field_stride:      int = 1


class BaseExperiment:
    condition_label = "base"
    ratemap_ndim = 2

    def __init__(self, config):
        self.qan = Torus3DQAN.from_config(config.network)
        self.config = config
        e = config.experiment
        self.ratemap_n_sub = e.ratemap_n_sub
        self.ratemap_n_shuffle = e.ratemap_n_shuffle
        self.ratemap_n_null = int(getattr(e, "ratemap_n_null", 300))
        self.ratemap_seed = e.ratemap_seed
        self.ratemap_active_thresh = e.ratemap_active_thresh

        print(
            f"[{self.condition_label}] "
            f"neurons {self.qan.cans[0].S.shape[0]:,} "
            f"arena {e.env_size} m | grid_period {e.grid_spacing} m | "
            f"periods {e.env_size / e.grid_spacing:.2f}"
        )

    def generate_trajectory(self, turn_std: float = None, n_steps=None, seed=None):
        """Required hook: return a Trajectory (or a 3-tuple via Trajectory.__iter__).

        n_steps and seed default to config.experiment when omitted.
        """
        raise NotImplementedError("Should be implemented by subclass")

    def _walk_params(self, turn_std: float = None, n_steps=None, seed=None):
        return walk_params(
            self.config.experiment, self.config.network.dt,
            turn_std=turn_std, n_steps=n_steps, seed=seed,
            scale=self.config.scale,
        )

    def _settle_steps(self):
        s = self.config.network.settle_steps
        if s is None:
            raise ValueError(
                "NetworkConfig.settle_steps is None. Set it on the experiment; "
                "there is no fallback formula."
            )
        return int(s)

    def _integrator(self, plane):
        N_neurons = self.qan.cans[0].S.shape[0]
        decode_chunk = max(64, int(256e6 / (4 * N_neurons)))
        kappa_v = plane.kappa_v
        return PathIntegrator(
            qan=self.qan,
            kappa_v=kappa_v,
            scale=self.config.scale,
            # plane_mode: flip the filter on/off when given; otherwise PlaneConfig.plane_mode
            plane_mode=plane.plane_mode,
            decode_chunk=decode_chunk,
            kappa_w=plane.kappa_w,
            kappa_sens=plane.kappa_sens,
            tau_refresh=plane.tau_refresh,
            seed=plane.seed,
            refresh_settle_steps=plane.refresh_settle_steps,
            reset_prior_at_segment_start=plane.reset_prior_at_segment_start,
        )

    def run(self, world_pos, v_body_seq, torus_gt, g_vec,
            n_true_seq=None, segment_starts=None, plane=None) -> ExperimentResult:
        plane = self.config.plane if plane is None else plane
        bins = self.config.experiment.ratemap_bins
        env_size = self.config.experiment.env_size
        ndim = self.ratemap_ndim
        T = len(world_pos)

        flat = world_to_flat_bins(world_pos, env_size, bins, ndim=ndim)

        integrator = self._integrator(plane)
        integrator.reset(torus_gt[0])
        settle = self._settle_steps()
        integrator.warmup(n_steps=settle)
        warmup_volume = integrator.backend.current_volume()

        refresh_mask = integrator.default_refresh_mask(
            T, timing=plane.refresh_timing, segment_starts=segment_starts,
        )

        N = integrator.backend.S.shape[1]
        rng = np.random.default_rng(self.ratemap_seed)

        # neuron subsample + shuffle lags (one RNG, sequential draws)
        # ratemap_n_sub <= 0 keeps every cell. The shuffle buffer uses this
        # same column order; score_2d_from_map indexes both with one scored_idx.
        sub_idx = None
        if self.ratemap_n_sub > 0:
            sub_idx = np.sort(rng.choice(N, size=min(self.ratemap_n_sub, N),
                                         replace=False))

        lags = None
        n_shuffle = self.ratemap_n_shuffle
        if n_shuffle > 0:
            min_lag = max(1, int(T * 0.1))
            max_lag = max(min_lag + 1, int(T * 0.9))
            if min_lag < max_lag:
                lags = rng.integers(min_lag, max_lag, size=n_shuffle)
            else:
                n_shuffle = 0          # trajectory too short for shuffles

        # Chance cells for the full sheet, drawn after the lags so the lags stay put.
        null_idx = None
        if sub_idx is None and n_shuffle > 0 and self.ratemap_n_null > 0:
            null_idx = np.sort(rng.choice(
                N, size=min(self.ratemap_n_null, N), replace=False))

        analysis = getattr(self.config, "analysis", None)
        field_stride = getattr(analysis, "field_stride", 10)
        max_trace_bytes = getattr(analysis, "max_trace_bytes", 1024 ** 3)
        theta_hist = integrator.run(
            v_body_seq, g_vec,
            flat_indices=flat,
            ratemap_bins=bins,
            ratemap_ndim=ndim,
            sub_idx=sub_idx,
            n_shuffle=n_shuffle if lags is not None else 0,
            lags=lags,
            null_idx=null_idx,
            n_true_seq=n_true_seq,
            refresh_mask=refresh_mask,
            segment_starts=segment_starts,
            world_pos=world_pos if ndim == 3 else None,
            field_stride=field_stride,
            max_trace_bytes=max_trace_bytes,
        )

        kappa = np.asarray(integrator.history["kappa"], dtype=float)
        n_hat_hist = np.asarray(integrator.history["n_hat"])
        n_held_hist = np.asarray(integrator.history["n_held"])
        n_sheet = int(integrator.backend.n)
        torus_err = torus_error(theta_hist, torus_gt)
        score = tracking_score(theta_hist, torus_gt)

        active_mask = None
        if integrator.ratemap_sums is not None and integrator.ratemap_counts is not None:
            sums_flat   = integrator.ratemap_sums.reshape(-1, integrator.ratemap_sums.shape[-1])  # (total_bins, n_sub)
            counts_flat = integrator.ratemap_counts.reshape(-1)                                    # (total_bins,)
            valid       = counts_flat > 0                                                          # unvisited bins → nan
            rate_flat = np.full_like(sums_flat, np.nan)
            rate_flat[valid] = sums_flat[valid] / counts_flat[valid, np.newaxis]
            span = np.nanmax(rate_flat, axis=0) - np.nanmin(rate_flat, axis=0)
            active_mask = span > self.ratemap_active_thresh * (np.nanmax(span) + 1e-12)

        params = deepcopy(asdict(self.config))
        params["plane"] = asdict(plane)
        params["scale"] = float(self.config.scale)
        params["device"] = str(integrator.backend.device)

        return ExperimentResult(
            ratemap_sums=integrator.ratemap_sums,
            ratemap_counts=integrator.ratemap_counts,
            torus_gt=torus_gt,
            theta_hist=theta_hist,
            n_hat_hist=n_hat_hist,
            n_held_hist=n_held_hist,
            error_rad=torus_err,
            tracking_score=float(score),
            params=params,
            n_sheet=n_sheet,
            refresh_mask=np.asarray(refresh_mask, dtype=bool),
            sub_idx=sub_idx,
            shuf_sums=integrator.ratemap_shuf_sums,
            active_mask=active_mask,
            kappa_hist=kappa,
            warmup_volume=warmup_volume,
            n_gravity_flips=int(getattr(integrator, "n_gravity_flips", 0)),
            field_pos=getattr(integrator, "field_pos", None),
            field_act=getattr(integrator, "field_act", None),
            field_stride=int(getattr(integrator, "field_stride", None) or 1),
        )
